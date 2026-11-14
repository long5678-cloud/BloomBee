"""CPU integration check with archived ordinary model outputs and two live DHT peers."""
import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryFile
from uuid import uuid4

from hivemind import DHT
from bloombee.kv_cache.telemetry import Telemetry, query
from bloombee.kv_cache.identity import canonical_digest

from bloombee.kv_cache import telemetry_paged as mod
ROOT = Path(__file__).resolve().parent

def receipt(name):
    row=json.loads((ROOT/'fixtures/paged'/(name+'.json')).read_text())
    assert len(row['events'])==32
    output=row['ids'][0]
    return row['events'],output[:row['input_tokens']],output

def test_archived_paged_evidence_against_packaged_adapter():
    peers = []
    checks = []
    opt = dict(start=True, host_maddrs=['/ip4/127.0.0.1/tcp/0'], use_relay=False, client_mode=False)
    try:
        writer = DHT(**opt); peers.append(writer)
        reader = DHT(initial_peers=writer.get_visible_maddrs(), **opt); peers.append(reader)
        ns = 'paged-' + uuid4().hex
        def publisher(suffix):
            return Telemetry(writer, namespace=ns + suffix, node_id='actual-consumer-A', history_start=0, clock=lambda:100)
        pub = publisher('-history')
        rows, prompt, output = receipt('P_remote')
        uids = sorted([r['uid'] for r in rows],key=lambda u:int(u.rsplit('.',1)[1]))
        sink = mod.PagedTelemetry(pub, layer_uids=uids)
        facts = {}
        for i,name in enumerate(['P_remote','P_gpu','Q_cpu_restore','Q_partial_remote']):
            rs, inp, out = receipt(name)
            args = dict(request_id=name, attempt_id=name, terminal='completed', completed_at=20+i,
                        input_ids=inp, output_ids=out, max_new_tokens=3,
                        rows=mod.BoundRows(name, tuple(rs)), evidence='historical')
            status = sink.finish(**args)
            assert status['recorded'] and status['kind']=='reuse_success'
            assert not sink.finish(**args)['recorded']
            facts[name] = status['kv_id']
        assert facts['P_remote']==facts['P_gpu'] and facts['Q_cpu_restore']==facts['Q_partial_remote'] and len(set(facts.values()))==2
        pub.publish()
        old = query(reader, namespace=pub.namespace, publishers=[pub.ref], start=10, end=100, clock=lambda:100)
        assert old['complete'] and old['total_uses']==4 and [r['total'] for r in old['ranked']]==[2,2]
        checks.append('four archived ordinary generations: remote, GPU, CPU and mixed; matching group and exact DHT totals')

        pub = publisher('-controls')
        sink = mod.PagedTelemetry(pub,layer_uids=uids)
        tagged = [dict(r,telemetry_attempt_id='final') for r in rows]
        defaults = dict(request_id='req',attempt_id='final',terminal='completed',completed_at=60,
            input_ids=prompt,output_ids=output,max_new_tokens=3,rows=tagged,evidence='test')
        def finish(**overrides):return sink.finish(**dict(defaults,**overrides))
        def rejected(label,**overrides):
            try:finish(**overrides)
            except (ValueError,TypeError):checks.append(label)
            else:raise AssertionError(label)
        others = [dict(r,telemetry_attempt_id='abandoned') for r in rows]
        assert finish(rows=others+tagged)['recorded'];assert not finish()['recorded']
        checks.append('interleaved attempts filtered; identical logical completion idempotent')
        rejected('conflicting logical terminal', terminal='cancelled')
        rejected('wrong complete output',request_id='truncated',output_ids=output[:-1])
        rejected('one missing layer',request_id='missing',rows=tagged[:-1])
        rejected('duplicate layer',request_id='dup',rows=tagged+[tagged[0]])
        rejected('untagged ordinary logs cannot be matched by order',request_id='untagged',rows=rows)
        rejected('wrong identity for same length',request_id='bad-id',input_ids=prompt[::-1],output_ids=prompt[::-1]+output[-3:])
        damaged = copy.deepcopy(tagged); damaged[0]['gpu_block_sources'][6]['generation'] += 1
        rejected('page slot generation must match adopted page',request_id='bad-page',rows=damaged)
        damaged = copy.deepcopy(tagged); damaged[0]['native_import_exact']=False
        rejected('inexact import',request_id='bad-copy',rows=damaged)
        for terminal in ['cancelled','failed']:
            st=finish(request_id=terminal,terminal=terminal,output_ids=None)
            assert st['kind']=='request_failed'
        for purpose in ['warmup','prefetch']:
            assert finish(request_id=purpose,purpose=purpose)['event'] is None
        cold = [dict(r,mode='cold',skipped_tokens=0,computed_tokens=len(prompt),gpu_hit_tokens=0,
                     cpu_hit_tokens=0,remote_hit_tokens=0,remote_errors=[]) for r in tagged]
        assert finish(request_id='cold',rows=cold)['kind']=='cold_recompute'
        fallback=[dict(r,mode='fallback') for r in cold]
        assert finish(request_id='fallback',rows=fallback)['kind']=='fallback'
        assert finish(request_id='partial',rows=cold[:1]+tagged[1:])['kind']=='fallback'
        # Locally produced pages have no remote receipts; backend must supply actual block identity.
        local = copy.deepcopy(tagged)
        for r in local:
            final_source = r['gpu_block_sources'][6]
            r['telemetry_identity'] = final_source['cache_identity']
            r['gpu_block_sources'] = [{k:s[k] for k in ('logical_start','slot','generation')} for s in r['gpu_block_sources']]
            r.update(remote_records=[],gpu_hit_tokens=112,remote_hit_tokens=0,native_import_exact=False)
        assert finish(request_id='local',rows=local)['kv_id']==facts['P_remote']
        missing_identity=copy.deepcopy(local)
        for row in missing_identity:row['telemetry_identity']=None
        rejected('local reuse without any identity is unknown',request_id='no-identity',rows=missing_identity)
        pub.publish()
        result = query(reader,namespace=pub.namespace,publishers=[pub.ref],start=10,end=100,clock=lambda:100)
        assert result['complete'] and result['stage_counts']==dict(request_failed=2,cold_recompute=1,fallback=2,reuse_success=2)
        checks.append('negative and local-only controls, independent reader checks stage and node counts')
        report = dict(status='PASS', historical_model_calls=4, new_gpu_calls=0,
                      historical_counts=old['ranked'], historical_kinds=old['stage_counts'], control_kinds=result['stage_counts'],
                      checks=checks, new_code_sha256=__import__('hashlib').sha256(Path(mod.__file__).read_bytes()).hexdigest())
    finally:
        for p in reversed(peers):
            p.shutdown();p.join(timeout=10)
        assert all(not p.is_alive() and p.exitcode==0 for p in peers)
    report['peer_exit_codes']=[p.exitcode for p in peers]
    print(json.dumps(report,indent=2))

