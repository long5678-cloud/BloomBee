"""Independent original-time ingress and serial ordinary model service.

One finite group per process. No ranker code can read this process's offered
request queue: the predictor receives terminal history through an explicit IPC
adapter, never the stream file or the next request.
"""
from pathlib import Path
import copy
import hashlib
import http.server
import importlib.util
import json
import os
import queue
import re
import sys
import threading
import time
import traceback

P = Path(__file__).resolve().parent
ASSET = None  # Model assets are supplied explicitly in front_config.json.


def save(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, indent=2)


def append(path, value):
    raw = (json.dumps(value, separators=(',', ':'))+'\n').encode()
    with Path(path).open('ab', buffering=0) as output:
        output.write(raw)
        os.fsync(output.fileno())


def mapping(row, vocab):
    tokens = []
    chain = hashlib.sha256(b'bloombee-trace-map-v1')
    assert len(row['hash_ids']) == (row['input_length']+15)//16
    for block in row['hash_ids']:
        chain.update(str(block).encode()+b':')
        seed = chain.digest()
        tokens.extend(int.from_bytes(hashlib.sha256(seed+bytes([i])).digest()[:8], 'big') % vocab
                      for i in range(min(16, row['input_length']-len(tokens))))
    assert len(tokens) == row['input_length']
    return tokens


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    global P
    P = args.run_dir.resolve()
    job = json.loads((P/'job_config.json').read_text())
    cfg = json.loads((P/'front_config.json').read_text())
    # Historical harness opted in; the public adapter itself defaults off.
    cfg.setdefault('distributed_telemetry', True)
    result_dir = P/cfg['result_dir']
    result_dir.mkdir()
    import torch
    from bloombee import AutoDistributedModelForCausalLM
    from bloombee.kv_cache.retention_completion import prefix_key
    import httpx
    torch.set_num_threads(4)
    peer = cfg['peer']
    model = AutoDistributedModelForCausalLM.from_pretrained(
        cfg['model_path'], local_files_only=True, initial_peers=[peer],
        dht_prefix=cfg.get('model_dht_prefix', job['dht_prefix']), torch_dtype=torch.float16, device_map='cpu',
        max_retries=1, request_timeout=3600, connect_timeout=20, update_period=5).eval()
    from bloombee.kv_cache.network.client_compat import install_logit_argument_forwarding
    logit_adapter = install_logit_argument_forwarding(model)
    # Metadata-only CPU-stage timing. No tensor copies or additional model
    # forward pass; long-batch arithmetic remains in the same batch wall.
    stage_rows = []
    stage_started = {}
    def stage_pre(name):
        def begin(module, args, kwargs):
            tensor = args[0] if args and hasattr(args[0], 'shape') else None
            stage_started[name] = (time.monotonic_ns(), list(tensor.shape) if tensor is not None else None)
        return begin
    def stage_post(name):
        def end(module, args, kwargs, output):
            begin, shape = stage_started.pop(name)
            stage_rows.append(dict(stage=name, start_ns=begin, end_ns=time.monotonic_ns(),
                input_shape=shape, output_shape=list(output.shape) if hasattr(output, 'shape') else None,
                dtype=str(module.weight.dtype) if hasattr(module, 'weight') else None,
                device=str(module.weight.device) if hasattr(module, 'weight') else None,
                chunked_forward=getattr(module, 'use_chunked_forward', None)))
        return end
    for name, module in [('cpu_embedding', model.get_input_embeddings()), ('cpu_lm_head', model.get_output_embeddings())]:
        module.register_forward_pre_hook(stage_pre(name), with_kwargs=True)
        module.register_forward_hook(stage_post(name), with_kwargs=True)
    generation = copy.deepcopy(model.generation_config)
    generation.eos_token_id = None
    generation.forced_eos_token_id = None
    offered = queue.Queue()
    accepted = []
    terminals = []
    failures = []
    ingress_lock = threading.Lock()
    active = False
    first_ingress = None
    last_terminal = None
    stopped = threading.Event()
    client = httpx.Client(trust_env=False, timeout=120)
    control_endpoint = cfg.get('control_endpoint', 'http://127.0.0.1:'+str(job['control_port']))
    ledger = Path(cfg.get('terminal_ledger', job['terminal_ledger']))
    # The harness creates a NEW node-local ledger before starting this process.
    history = [json.loads(line) for line in ledger.read_text().splitlines()]
    adapter = None
    if cfg.get('completion_adapter'):
        if cfg['completion_adapter'] == 'builtin':
            from bloombee.kv_cache.network.completion import Adapter
        else:
            spec = importlib.util.spec_from_file_location('group_completion_adapter', cfg['completion_adapter'])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            Adapter = module.Adapter
        adapter = Adapter(model=model, job=job, config=cfg, output_root=result_dir)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def do_POST(self):
            nonlocal first_ingress
            try:
                assert self.path == '/infer'
                n = int(self.headers.get('Content-Length', '0'))
                assert 0 < n < 2*1024**2
                value = json.loads(self.rfile.read(n))
                row, index = value['row'], value['index']
                assert type(index) is int and 0 <= index < cfg['count']
                with ingress_lock:
                    assert index == len(accepted), 'original order changed'
                    now = time.monotonic_ns()
                    if first_ingress is None:
                        first_ingress = now
                    entry = dict(index=index, actual_ingress_ns=now,
                                 queued=offered.qsize(),
                                 worker_active=active, row_sha256=value['row_sha256'],
                                 offered_monotonic_ns=value['offered_monotonic_ns'])
                    assert entry['row_sha256'] == hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    accepted.append(entry)
                    append(result_dir/'ingress.jsonl', entry)
                    offered.put((index, row, entry))
                raw = json.dumps(dict(accepted=True, ingress=entry)).encode()
                self.send_response(202)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except Exception as error:
                self.send_error(409, str(error))

    if cfg.get('build_source_library'):
        # Only the frozen visible history is supplied here. These short source
        # builds are preparation, never original trace demands or predictor H.
        history_rows = [json.loads(line) for line in (P/cfg['history_input']).read_text().splitlines()]
        seen = set()
        for row_index, historical in enumerate(history_rows):
            known = mapping(historical, model.config.vocab_size)
            key = tuple(known[:16])
            if key in seen:
                continue
            seen.add(key)
            prompt = known[:17]
            attempt = cfg['group']+'_build_'+str(len(seen))
            provider_log = Path(cfg['provider_log'])
            offset = provider_log.stat().st_size
            value = torch.tensor([prompt], dtype=torch.long)
            start_build = time.monotonic_ns()
            with torch.inference_mode(), model.inference_session(max_length=20,
                    dynamo_cache=dict(context=job['context'], token_ids=prompt,
                                      priorities=[100], attempt_id=attempt)):
                result = model.generate(value,max_new_tokens=3,min_new_tokens=3,
                                        do_sample=False,generation_config=generation,
                                        eos_token_id=None,pad_token_id=0,attention_mask=torch.ones_like(value))
            assert result.shape[-1] == 20
            with provider_log.open('rb') as file:
                file.seek(offset)
                body = file.read()
            builds = []
            for line in body.splitlines():
                _, sep, payload = line.partition(b'DYNAMO_SERVICE ')
                if sep:
                    event = json.loads(payload)
                    if event.get('telemetry_attempt_id') == attempt:
                        builds.append(event)
            assert len(builds) == 32
            publications = [r.get('native_bundle_publication') for event in builds
                            for r in event.get('remote_publication',{}).get('records',[])
                            if (r.get('native_bundle_publication') or {}).get('status') == 'published']
            assert len(publications) == 1, 'exact32layer native source publication not acknowledged'
            save(result_dir/('source_build_'+str(len(seen))+'.json'),dict(
                attempt_id=attempt, original_history_row=row_index, input_ids=prompt,
                output_ids=result.reshape(-1).tolist(), rows=builds,
                native_publication=publications[0], elapsed_ns=time.monotonic_ns()-start_build,
                purpose='known_history_cache_build_not_original_H_or_benchmark'))
            # Legacy Dynamo source retention is not a terminal demand learner.
            response = client.post(control_endpoint+'/terminal',json=dict(attempt_id=attempt,status='complete_success'))
            response.raise_for_status()
        save(result_dir/'source_library_build_done.json',dict(prefixes=len(seen),original_trace_requests=0,
                                                            predictor_history_events=0))

    server = http.server.ThreadingHTTPServer(('127.0.0.1', cfg['ingress_port']), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    save(result_dir/'ready.json', dict(pid=os.getpid(), model_class=str(type(model)),
                                      layers=len(model.transformer.h), peer=peer, logit_adapter=logit_adapter,
                                      ingress_port=cfg['ingress_port'], cuda_initialized=torch.cuda.is_initialized()))
    for _ in range(cfg['count']):
        if getattr(getattr(adapter, 'prefetch', None), 'owns_page_terminal', False):
            adapter.resume_pending_completion()
        index, row, ingress = offered.get()
        if index == 0 and cfg.get('restore_initial_history'):
            original = Path(cfg['restore_source_ledger']).read_bytes()
            source = [json.loads(line) for line in original.splitlines()]
            assert len(source) == cfg.get('initial_history_count', 2) and not history and ledger.stat().st_size == 0
            last_source = source[-1]['completed_monotonic']
            anchor = ingress['actual_ingress_ns']/1e9
            common = []
            boundary = cfg.get('prefetch_controller', {}).get('history_boundary')
            for raw in source:
                historical_event = copy.deepcopy(raw)
                historical_event['completed_monotonic'] = anchor - cfg['initial_last_age_s'] - (last_source-raw['completed_monotonic'])
                historical_event['initialization_provenance'] = dict(source_ledger=cfg['restore_source_ledger'],
                    source_ledger_sha256=hashlib.sha256(original).hexdigest(),
                    actual_source_completed_monotonic=raw['completed_monotonic'],
                    original_first_actual_ingress_ns=ingress['actual_ingress_ns'],
                    relative_age_restored=True, new_model_demand=False)
                append(ledger, historical_event); history.append(historical_event)
                if boundary is not None:
                    ref = Path(historical_event['terminal_ref'])
                    common.append(dict(request_id=historical_event['request_id'],attempt_id=historical_event['attempt_id'],
                        token_ids=historical_event['token_ids'],outcome='processed_success',
                        consumer_instance_id=boundary['consumer_id'],
                        completion_event_seq=historical_event['completion_event_seq'],
                        disclosed_ns=int(historical_event['completed_monotonic']*1e9),
                        clock_domain_id=boundary['clock_domain_id'],
                        terminal_receipt_ref=dict(path=str(ref),sha256=hashlib.sha256(ref.read_bytes()).hexdigest()),
                        history_source='restored_real_initial_history',visibility_scope='consumer',
                        initialization_provenance=historical_event['initialization_provenance']))
            if boundary is not None:cfg['prefetch_controller']['initial_history'] = common
            save(result_dir/'initial_history_restoration.json',dict(actual_ingress_ns=ingress['actual_ingress_ns'],
                last_age_s=cfg['initial_last_age_s'],events=history,
                source_sha256=hashlib.sha256(original).hexdigest(),common_history=common))
        request_id = cfg['group']+'_r'+str(index+1)
        attempt_id = request_id+'_a1'
        ids = mapping(row, model.config.vocab_size)
        requested = row['output_length']
        captures = []
        stage_rows.clear()
        capture_errors = []
        forward_calls = 0
        token_times = []
        class TokenClock:
            def put(self, value):
                token_times.append(dict(monotonic_ns=time.monotonic_ns(),
                    shape=list(value.shape),numel=value.numel(),prompt=not token_times))
            def end(self):
                pass
        token_clock = TokenClock()

        def capture(module, args, kwargs, output):
            nonlocal forward_calls
            forward_calls += 1
            if forward_calls <= 3:
                try:
                    raw = output.get('logits') if isinstance(output, dict) else output.logits
                    captures.append(raw[:, -1:, :].detach().cpu().clone())
                except Exception as error:
                    capture_errors.append(str(error))

        hook = model.register_forward_hook(capture, with_kwargs=True)
        with ingress_lock:
            active = True
        start = time.monotonic_ns()
        output = None
        terminal = dict(request_id=request_id, attempt_id=attempt_id, index=index,
                        actual_ingress_ns=ingress['actual_ingress_ns'], service_start_ns=start,
                        requested_input_tokens=len(ids), requested_output_tokens=requested,
                        input_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest())
        try:
            tensor = torch.tensor([ids], dtype=torch.long)
            priorities = [100] + [0]*max(0, len(ids)//16-1)
            options = dict(max_length=len(ids)+requested)
            if cfg.get('cache_enabled', True):
                options['dynamo_cache'] = dict(context=job['context'], token_ids=ids,
                                               priorities=priorities, attempt_id=attempt_id)
            if adapter is not None and hasattr(adapter, 'before_request'):
                terminal['prefetch_claim'] = adapter.before_request(attempt_id=attempt_id, token_ids=ids,
                    actual_ingress_ns=ingress['actual_ingress_ns'],
                    clock_domain=cfg.get('ingress_clock_domain'))
            with torch.inference_mode(), model.inference_session(**options):
                output = model.generate(tensor, max_new_tokens=requested, min_new_tokens=requested,
                                        do_sample=False, generation_config=generation, logits_to_keep=1, streamer=token_clock,
                                        eos_token_id=None, pad_token_id=0, attention_mask=torch.ones_like(tensor))
            out = output.reshape(-1).tolist()
            actual = len(out)-len(ids)
            assert out[:len(ids)] == ids and actual == requested
            done = time.monotonic_ns()
            terminal.update(status='complete_success', actual_output_tokens=actual,
                            model_complete_ns=done, output_ids=out,
                            output_sha256=hashlib.sha256(json.dumps(out, separators=(',', ':')).encode()).hexdigest())
            ref = result_dir/('terminal_'+str(index+1)+'.json')
            observation = (adapter.observe_terminal(terminal=terminal, input_ids=ids, output_ids=output)
                           if adapter is not None else None)
            terminal['actual_reuse_observation'] = observation
            event = dict(request_id=request_id, attempt_id=attempt_id,
                         kv_reuse_success=observation['kv_reuse_success'] if observation is not None else False,
                         completion_event_seq=len(history)+1, status='complete_success',
                         actual_output_tokens=actual, requested_output_tokens=requested,
                         terminal_ref=str(ref), token_ids=ids, prefix_key=prefix_key(ids),
                         completed_monotonic=done/1e9)
            save(ref, terminal)
            append(ledger, event)
            history.append(event)
        except Exception as error:
            if terminal.get('status') == 'complete_success':
                # An evidence/observer problem must never relabel an already
                # completed model output as an inference failure or zero tokens.
                terminal.update(evidence_error_type=type(error).__name__, evidence_error=str(error),
                                evidence_traceback=traceback.format_exc())
            else:
                terminal.update(status='failed', actual_output_tokens=0,
                                model_complete_ns=time.monotonic_ns(), error_type=type(error).__name__,
                                error=str(error), traceback=traceback.format_exc())
                failures.append(terminal)
                append(cfg.get('failed_ledger',job['failed_ledger']), dict(attempt_id=attempt_id, status='failed'))
            target = result_dir/('terminal_'+str(index+1)+'.json')
            if not target.exists():
                save(target, terminal)
            else:
                save(result_dir/('evidence_error_'+str(index+1)+'.json'), terminal)
        finally:
            hook.remove()
        # Metadata acknowledgement and telemetry belong to this same full
        # generation. No per-layer or prefill-only callback advances demand H.
        if not getattr(getattr(adapter, 'prefetch', None), 'owns_page_terminal', False):
            try:
                response = client.post(control_endpoint+'/terminal',
                                       json=dict(attempt_id=attempt_id, status=terminal['status']))
                response.raise_for_status()
                save(result_dir/('control_'+str(index+1)+'.json'), response.json())
            except Exception as error:
                terminal['control_error'] = str(error)
        if adapter is not None:
            try:
                terminal['completion_adapter'] = adapter.finish(terminal=terminal, input_ids=ids,
                                                               output_ids=output, history=tuple(history))
            except Exception as error:
                terminal['completion_adapter_error'] = str(error)
        if captures:
            with (result_dir/('raw_'+str(index+1)+'.pt')).open('xb') as file:
                torch.save(captures, file)
        save(result_dir/('token_times_'+str(index+1)+'.json'),dict(attempt_id=attempt_id,rows=token_times,scope='sampler streamer callback; first record is original prompt'))
        save(result_dir/('cpu_stages_'+str(index+1)+'.json'),dict(attempt_id=attempt_id,rows=stage_rows))
        terminal['capture_errors'] = capture_errors
        terminal['settled_ns'] = time.monotonic_ns()
        # Batch timing ends at the actual final generate terminal. Earlier
        # completion/control work still delays subsequent service and therefore
        # remains inside the batch wall clock. Tail bookkeeping is costed apart.
        last_terminal = terminal['model_complete_ns']
        terminals.append(terminal)
        append(result_dir/'terminals.jsonl', terminal)
        with ingress_lock:
            active = False
    seconds = (last_terminal-first_ingress)/1e9
    tokens = sum(t['actual_output_tokens'] for t in terminals if t['status'] == 'complete_success')
    save(result_dir/'done.json', dict(group=cfg['group'], accepted=len(accepted), terminals=len(terminals),
                                      completed=len(terminals)-len(failures), failed=len(failures),
                                      complete_output_tokens=tokens, first_ingress_ns=first_ingress,
                                      last_terminal_ns=last_terminal, batch_wall_seconds=seconds,
                                      tokens_per_second=tokens/seconds, failures=failures,
                                      correctness='separate raw/full-output audit required',
                                      evidence_errors=[t['attempt_id'] for t in terminals if any(k in t for k in ('evidence_error','control_error','completion_adapter_error'))],
                                      valid_for_comparison=not any(any(k in t for k in ('evidence_error','control_error','completion_adapter_error')) for t in terminals),
                                      actual_model_worker_count=1, original_arrival_ingress_independent=True))
    # Explicit post-batch dense call drains previously queued page frees on
    # the model thread. Its token is excluded; its cost is a separately logged
    # tail, never removed from any earlier request's in-batch wall time.
    if cfg.get('drain_after', True):
        tail_start = time.monotonic_ns()
        prompt = ids[:17]
        tensor = torch.tensor([prompt], dtype=torch.long)
        with torch.inference_mode(), model.inference_session(max_length=len(prompt)+1):
            drained = model.generate(tensor, max_new_tokens=1, min_new_tokens=1,
                do_sample=False, generation_config=generation, eos_token_id=None,
                pad_token_id=0, attention_mask=torch.ones_like(tensor))
        status = client.get(control_endpoint+'/status'); status.raise_for_status()
        actual = status.json()
        assert len(actual['states']) == 32 and all(r['active_sessions'] == 0 for r in actual['states'])
        save(result_dir/'tail_drain.json', dict(start_ns=tail_start,
            end_ns=time.monotonic_ns(), output_tokens=1, measured_tokens=0,
            purpose='post_batch_model_thread_release_not_predictor_demand', status=actual))
    if adapter is not None:
        adapter.close()
    server.shutdown()
    client.close()
    model.transformer.h.sequence_manager.shutdown()
    save(result_dir/'settled.json', dict(wall=time.time(),monotonic_ns=time.monotonic_ns(),ordinary_front_complete=True))


if __name__ == '__main__':
    main()
