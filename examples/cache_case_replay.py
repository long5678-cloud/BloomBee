"""Replay explicit finite cache cases through one normally loaded BloomBee model.

The plan/JSONL belongs to the load driver. Predictors receive only the public
OrdinaryClient completion history; the worker never reads this plan or queue.
Model loading and source preparation are outside request measurements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import threading
import time


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def process_record(child, command):
    fields = Path('/proc', str(child.pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=child.pid, pgid=os.getpgid(child.pid), uid=os.getuid(),
                start_ticks=int(fields[19]), hostname=socket.gethostname(), command=command,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.output_root.mkdir(parents=True, exist_ok=False)
    save(args.output_root/'effective-plan.json', plan)
    import torch
    import httpx
    from bloombee import AutoDistributedModelForCausalLM
    from bloombee.kv_cache.network.ordinary import OrdinaryClient
    from bloombee.kv_cache.telemetry import query, PublisherRef
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        plan['model_path'], initial_peers=[plan['peer']], dht_prefix=plan['dht_prefix'],
        local_files_only=True, torch_dtype=torch.float16, device_map='cpu',
        max_retries=1, request_timeout=600, connect_timeout=20, update_period=5).eval()
    try:
        with httpx.Client(trust_env=False, timeout=120) as control:
            for case in plan['cases']:
                root = args.output_root/case['case_id']
                rows = [json.loads(line) for line in Path(case['requests_jsonl']).read_text().splitlines() if line.strip()]
                if not rows:
                    raise ValueError('nonempty finite case required')
                for row in rows:
                    ids = row['token_ids']
                    if not ids or any(type(t) is not int or not 0 <= t < model.config.vocab_size for t in ids):
                        raise ValueError('actual vocabulary bound violated')
                    if type(row['new_tokens']) is not int or row['new_tokens'] <= 0 or len(ids)+row['new_tokens'] > plan['max_session_tokens']:
                        raise ValueError('complete request exceeds admitted service limit')
                config = json.loads(Path(case['client_config']).read_text())
                job = json.loads(Path(case['job']).read_text()) if case.get('job') else {}
                worker = None
                if case.get('worker_argv'):
                    log = Path(case['worker_log'])
                    log.parent.mkdir(parents=True, exist_ok=True)
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
                    worker = subprocess.Popen(case['worker_argv'], env=environment, stdin=subprocess.DEVNULL,
                        stdout=log.open('xb'), stderr=subprocess.STDOUT, start_new_session=True)
                    save(log.with_name('worker-launch.json'), process_record(worker, case['worker_argv']))
                    ready = Path(case['worker_ready'])
                    start = time.monotonic()
                    while not ready.exists():
                        if worker.poll() is not None:
                            raise RuntimeError('owned native worker exited before readiness: '+str(log))
                        if time.monotonic()-start > 180:
                            raise TimeoutError('native worker startup receipt unavailable; process retained: '+str(log))
                        time.sleep(.1)
                with OrdinaryClient(model, output_root=root, context=case.get('context'), config=config, job=job) as client:
                    if case.get('group_id'):
                        for route, value in [('configure_group', dict(group_id=case['group_id'])),
                                             ('begin_formal', dict(mode=case['mode'], clear_history=True))]:
                            response = control.post(plan['control_endpoint']+'/'+route, json=value)
                            save(root/(route+'.json'), dict(status=response.status_code, body=response.json()))
                            response.raise_for_status()
                    offered = queue.Queue()
                    t0 = time.monotonic_ns()
                    first_stamp = rows[0].get('trace_timestamp', 0)
                    def arrivals():
                        with (root/'arrivals.jsonl').open('x') as log:
                            for index, row in enumerate(rows):
                                scheduled = t0+int((row.get('trace_timestamp', first_stamp)-first_stamp)*1e9)
                                delay = (scheduled-time.monotonic_ns())/1e9
                                if delay > 0:
                                    time.sleep(delay)
                                event = dict(index=index, request_id=row['request_id'], scheduled_ns=scheduled,
                                             actual_offered_ns=time.monotonic_ns(), original_timestamp=row.get('trace_timestamp'))
                                log.write(json.dumps(event)+'\n'); log.flush()
                                offered.put((row, event))
                    producer = threading.Thread(target=arrivals, daemon=True, name='trace-arrivals')
                    producer.start()
                    for index in range(len(rows)):
                        row, arrival = offered.get()
                        ids = torch.tensor([row['token_ids']], dtype=torch.long)
                        captures = []; forward_times = []
                        def capture(module, inputs, kwargs, output):
                            forward_times.append(time.monotonic_ns())
                            if len(captures) < 3:
                                logits = output.get('logits') if isinstance(output, dict) else output.logits
                                captures.append(logits[:, -1:, :].detach().cpu().clone())
                        hook = model.register_forward_hook(capture, with_kwargs=True)
                        start = time.monotonic_ns(); output = terminal = error = None
                        try:
                            output, terminal = client.generate(ids, max_new_tokens=row['new_tokens'],
                                request_id=row['request_id'], priorities=row.get('priorities'), purpose=row.get('purpose'),
                                min_new_tokens=row['new_tokens'], do_sample=False, attention_mask=torch.ones_like(ids),
                                eos_token_id=None, pad_token_id=0, logits_to_keep=1)
                        except Exception as caught:
                            error = caught
                            terminal = getattr(caught, 'terminal', getattr(caught, 'kv_terminal', None))
                        finally:
                            end = time.monotonic_ns(); hook.remove()
                        score_name = 'scores_'+str(index)+'.pt'
                        torch.save(captures, root/score_name)
                        result = dict(index=index, request_id=row['request_id'], trace_line=row.get('trace_line'),
                            purpose=(terminal or {}).get('purpose', row.get('purpose', 'measured')),
                            requested_input_tokens=len(row['token_ids']), requested_output_tokens=row['new_tokens'],
                            output_ids=output.reshape(-1).tolist() if output is not None else None,
                            model_terminal=terminal, error=dict(type=type(error).__name__, message=str(error)) if error else None,
                            arrival=arrival, service_started_ns=start, service_ended_ns=end,
                            service_seconds=(end-start)/1e9, offered_to_terminal_seconds=(end-arrival['actual_offered_ns'])/1e9,
                            forward_count=len(forward_times), first_forward_ready_ns=forward_times[0] if forward_times else None,
                            last_forward_ready_ns=forward_times[-1] if forward_times else None,
                            first_logits_seconds=(forward_times[0]-start)/1e9 if forward_times else None,
                            first_logits_note='compute-side first logits, not streamed token delivery; API returns full output at terminal',
                            decode_tokens_per_second=((len(forward_times)-1)*1e9/(forward_times[-1]-forward_times[0])) if len(forward_times)>1 else None,
                            scores=score_name, history_count=len(client.history),
                            timing_scope='actual ordinary consumer request including cache prediction/transfer/commit; excludes model load/source prep')
                        save(root/('request_'+str(index)+'.json'), result)
                        print(json.dumps(dict(case=case['case_id'], request=row['request_id'], purpose=result['purpose'],
                            status=(terminal or {}).get('status'), seconds=result['service_seconds'], error=result['error'])), flush=True)
                        if error is not None:
                            raise error
                    producer.join()
                    if case.get('group_id'):
                        response = control.get(plan['control_endpoint']+'/status'); response.raise_for_status()
                        save(root/'model-state-after.json', response.json())
                if config.get('distributed_telemetry', False):
                    receipt = json.loads((root/'telemetry_publisher.json').read_text())
                    refs = [PublisherRef(**value) for value in receipt['publishers']]
                    report = query(model.transformer.h.sequence_manager.dht, namespace=receipt['namespace'],
                        publishers=refs, start=receipt['start'], end=receipt['end'])
                    save(root/'telemetry-query.json', report)
                    if plan.get('telemetry_observer'):
                        response = control.post(plan['telemetry_observer']+'/query', json=dict(
                            namespace=receipt['namespace'], publishers=receipt['publishers'],
                            start=receipt['start'], end=receipt['end'], case_id=case['case_id']))
                        save(root/'telemetry-query-other-host.json', dict(status=response.status_code, body=response.json()))
                        response.raise_for_status()
                if worker is not None:
                    code = worker.wait(timeout=60)
                    save(root/'worker-exit.json', dict(exit_code=code))
                    if code:
                        raise RuntimeError('owned native worker exit '+str(code))
                save(root/'case-complete.json', dict(case=case['case_id'], requests=len(rows)))
        save(args.output_root/'complete.json', dict(cases=len(plan['cases']), status='finite_replay_complete'))
    finally:
        model.transformer.h.sequence_manager.shutdown()


if __name__ == '__main__':
    main()
