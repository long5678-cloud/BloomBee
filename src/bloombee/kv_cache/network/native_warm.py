"""Explicit pre-inference warm; official geometry registration + coordinator.

Does not instantiate a BloomBee model or submit inference/SDK retrieval.
Receipt is tied to target registration and exact L1 chunk placements.
"""
import argparse,json,time,hashlib,uuid
from dataclasses import replace
from pathlib import Path
import httpx
from bloombee.kv_cache.network.snapshot_bridge import SnapshotLayout,identity_api
from bloombee.kv_cache.network.clock_domain import current as clock_domain

def layout_from_config(config,tokens):
    n=int(config.get('prefix_length',16));template=identity_api.CacheIdentity.from_dict(config['identity_template'])
    layers=int(config.get('native_layers',32));tokens=list(tokens)
    if len(tokens)<n:raise ValueError('insufficient actual prefix tokens')
    identities=[replace(template,layer_start=i,layer_end=i+1,prefix_length=n,prefix_tokens_digest=identity_api.token_ids_digest(tokens[:n]),position_digest=identity_api.canonical_digest(list(range(n)))).to_dict() for i in range(layers)]
    return SnapshotLayout(identities,tokens[:n],max_bytes=config.get('capacity_bytes',256*1024**2))

def l1_state(client,config,layout):
    coordinator=config['coordinator_url'];target=config['lmcache_target']['instance_id']
    r=client.get(coordinator+'/instances');r.raise_for_status()
    matches=[i for i in r.json()['instances'] if i['instance_id']==target]
    if len(matches)!=1:raise RuntimeError('target instance registration missing or ambiguous')
    registration=matches[0]
    r=client.get(coordinator+'/directory/keys',params=dict(limit=10000));r.raise_for_status();directory=r.json()
    if directory['total']>10000:raise RuntimeError('directory truncated; cannot prove empty or full target L1')
    chunks=[];target_total=0
    for row in directory['keys']:
        key=row['key']
        target_total+=sum(p['instance_id']==target and p['tier']=='l1' for p in row['placements'])
        if key['model_name']!=layout.model_name or key.get('cache_salt','')!=layout.cache_salt:continue
        for p in row['placements']:
            if p['instance_id']==target and p['tier']=='l1':chunks.append(dict(chunk_hash=key['chunk_hash_hex'],incarnation=p['incarnation'],size_bytes=p['size_bytes']))
    return dict(target_total_l1_chunks=target_total,registration=registration,chunks=sorted(chunks,key=lambda x:x['chunk_hash']),observed_ns=time.monotonic_ns())

def receipt_digest(receipt):
    return identity_api.canonical_digest({k:v for k,v in receipt.items() if k!='receipt_sha256'})

def validate_ready(client,config,layout,receipt,arrival_ns):
    """Conservative current-directory evidence, not a promise of zero network."""
    if not receipt:return dict(ready=False,reason='no_receipt')
    if receipt.get('receipt_sha256')!=receipt_digest(receipt):return dict(ready=False,reason='receipt_digest')
    if (receipt.get('model_name')!=layout.model_name or receipt.get('cache_salt')!=layout.cache_salt or receipt.get('target')!=config['lmcache_target'] or receipt.get('coordinator_url')!=config['coordinator_url']):return dict(ready=False,reason='identity_or_endpoint')
    domain=clock_domain(config)
    if domain is None or domain!=receipt.get('clock_domain'):return dict(ready=False,reason='clock_domain_unknown_or_mismatched',ordering='causal_handoff_only; no monotonic comparison')
    if receipt.get('completed_ns',arrival_ns)>=arrival_ns:return dict(ready=False,reason='not_completed_before_arrival')
    if time.time()-receipt.get('completed_wall_time',0)>config.get('warm_receipt_max_age_s',300):return dict(ready=False,reason='receipt_expired')
    current=l1_state(client,config,layout)
    if current['registration']!=receipt['after']['registration']:return dict(ready=False,reason='target_registration_changed',current=current)
    if len(current['chunks'])!=len(layout.tokens)//16 or current['chunks']!=receipt['after']['chunks']:return dict(ready=False,reason='current_L1_missing_or_changed',current=current)
    return dict(ready=True,reason='matched_completed_receipt_and_current_directory_L1',receipt_sha256=receipt['receipt_sha256'],current=current,network_proof='native trace and TCP backend counters still required')

def warm_registered(client,config,layout,*,require_empty=False,reason='explicit_before_first_inference'):
    before=l1_state(client,config,layout)
    if require_empty and before['target_total_l1_chunks']:raise RuntimeError('target prefix L1 already populated before explicit warm')
    start=time.monotonic_ns();r=client.post(config['coordinator_url']+'/cache/prefetches',json=dict(instance_id=config['lmcache_target']['instance_id'],model_name=layout.model_name,world_size=1,token_ids=layout.tokens,cache_salt=layout.cache_salt));r.raise_for_status();submitted=r.json()
    if not submitted.get('request_id'):raise RuntimeError('native warm noop is not completed prefix residency')
    deadline=time.monotonic()+config.get('native_timeout_s',60)
    while time.monotonic()<deadline:
        r=client.get(config['coordinator_url']+f"/cache/prefetches/{config['lmcache_target']['instance_id']}/{submitted['request_id']}");r.raise_for_status();status=r.json()
        if status.get('status')=='completed':break
        time.sleep(.01)
    else:raise TimeoutError('native warm completion timeout')
    if status.get('found_keys')!=len(layout.tokens)//16:raise RuntimeError('native warm did not find all prefix chunks')
    while time.monotonic()<deadline:
        after=l1_state(client,config,layout)
        if len(after['chunks'])==len(layout.tokens)//16:break
        time.sleep(.02)
    else:raise TimeoutError('coordinator L1 placements did not become complete')
    receipt=dict(schema=1,clock_domain=clock_domain(config),request_id=submitted['request_id'],reason=reason,model_name=layout.model_name,cache_salt=layout.cache_salt,target=config['lmcache_target'],coordinator_url=config['coordinator_url'],prefix_tokens_digest=identity_api.token_ids_digest(layout.tokens),started_ns=start,completed_ns=time.monotonic_ns(),completed_wall_time=time.time(),submitted=submitted,status=status,before=before,after=after,require_empty=require_empty,target_inference_requests_submitted_by_this_client=0)
    receipt['receipt_sha256']=receipt_digest(receipt);return receipt

def explicit_warm(config,tokens,output):
    from bloombee.kv_cache.network.native_probe import Geometry
    layout=layout_from_config(config,tokens);geometry=None
    with httpx.Client(trust_env=False,timeout=config.get('native_timeout_s',60)) as client:
        try:
            geometry=Geometry(config['lmcache_target'],layout.model_name,len(layout.identities),layout.heads,layout.dim,layout.dtype)
            receipt=warm_registered(client,config,layout,require_empty=True)
            with Path(output).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
            return receipt
        finally:
            if geometry is not None:geometry.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--tokens-json',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    r=explicit_warm(json.loads(Path(a.config).read_text()),json.loads(Path(a.tokens_json).read_text()),a.out)
    print(json.dumps(dict(completed=True,receipt_sha256=r['receipt_sha256'],receipt_path=a.out,target_inference_requests=0)))
