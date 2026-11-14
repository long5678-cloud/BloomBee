"""Read-only qualification BEFORE ranking/warm. LMCache v0.5.7 only.

Uses official /directory/lookup and /status, never SDK.lookup/retrieve or warm.
Scope: world_size1, object_group0, full aligned prefix from position0.
Directory freshness is bounded evidence, NOT physical availability guarantee.
"""
import argparse,json,math,time
from pathlib import Path
import httpx
from bloombee.kv_cache.network.strategy_contract import digest
from bloombee.kv_cache.network.native_warm import layout_from_config

def stream_state(status,instance,incarnation=None,*,max_event_age_s=60,max_heartbeat_age_s=10):
    # max_event_age_s is the legacy keyword for the batch delivery limit.
    def count(value):
        return type(value) is int and value >= 0

    def duration(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    try:
        ingest = status['ingest']
        stream = ingest['streams'][instance]
        heartbeat = status['registry']['seconds_since_heartbeat'][instance]
        if not all(count(stream[k]) for k in ('incarnation', 'last_seq', 'missing_batches', 'events_dropped')):
            return 'unknown', 'invalid_stream_counter'
        if stream['last_seq'] < 1 or type(stream['gap_detected']) is not bool:
            return 'unknown', 'stream_frontier_unconfirmed'
        if stream['gap_detected'] or stream['missing_batches'] or stream['events_dropped']:
            return 'stale', 'event_gap'
        if incarnation is not None and (not count(incarnation) or stream['incarnation'] != incarnation):
            return 'stale', 'placement_incarnation_mismatch'
        delay = stream['event_delivery_delay_seconds']
        if not all(duration(v) for v in (stream['seconds_since_last_event'], delay, heartbeat,
                                            max_event_age_s, max_heartbeat_age_s)):
            return 'unknown', 'invalid_timing_evidence'
        if delay > max_event_age_s or heartbeat > max_heartbeat_age_s:
            return 'stale', 'event_delivery_delay_or_heartbeat_age'
        applied = ingest['batches_applied']
        consumers = status['consumers']
        if not count(applied) or applied < 1 or not isinstance(consumers, dict) or 'KeyDirectory' not in consumers:
            return 'unknown', 'directory_consumer_state_missing'
        for consumer in consumers.values():
            if not all(count(consumer[k]) for k in ('batches_delivered', 'consume_failures', 'fence_failures')):
                return 'unknown', 'invalid_consumer_counter'
            if consumer['batches_delivered'] != applied or consumer['consume_failures'] or consumer['fence_failures']:
                return 'unknown', 'consumer_delivery_or_apply_failure'
        return 'fresh', 'bounded_official_status'
    except (KeyError, TypeError):
        return 'unknown', 'status_fields_or_membership_missing'


def matches_rule(placement,rule):
    return all(placement.get(k)==v for k,v in rule.items())

class HistoryCandidateFilter:
    def __init__(self,config,client=None):
        self.config=config;self.client=client or httpx.Client(trust_env=False,timeout=config.get('native_timeout_s',60));self.calls=[]
        if not config.get('candidate_remote_placements'):raise ValueError('explicit remote placement rules required')
        for rule in config['candidate_remote_placements']:
            if not {'instance_id','tier'}<=rule.keys():raise ValueError('remote rule requires instance_id and tier')
            if rule['tier']=='l1' and rule['instance_id']==config['lmcache_target']['instance_id']:raise ValueError('target L1 cannot be a remote source')
            if rule['tier']=='l2' and not rule.get('backend'):raise ValueError('remote L2 needs explicit network backend identity')
    def request(self,method,path,body=None):
        if (method,path) not in [('GET','/status'),('POST','/directory/lookup')]:raise ValueError('qualification only permits read-only official metadata APIs')
        self.calls.append(dict(method=method,path=path));url=self.config['coordinator_url']+path
        r=self.client.get(url) if method=='GET' else self.client.post(url,json=body)
        r.raise_for_status();return r.json()
    def filter(self,history):
        history=tuple(history)
        cfg=self.config;target=cfg['lmcache_target']['instance_id'];consumer=cfg.get('candidate_consumer_id',target);rows=[];unique={}
        # Only this consumer's SUCCESSFULLY processed token history is enumerable.
        # Producer-only or future trace rows cannot enter by directory enumeration.
        for event in tuple(history):
            row=dict(history_id=event.get('request_id'),eligible=False)
            rows.append(row)
            if event.get('consumer_instance_id')!=consumer:row['reason']='consumer_history_owner_mismatch';continue
            cutoff=cfg.get('candidate_history_before_row')
            if cutoff is not None and (type(event.get('trace_row_index')) is not int or event['trace_row_index']>=cutoff):row['reason']='future_or_unknown_trace_row';continue
            if not event.get('history_source') or not event.get('visibility_scope'):row['reason']='history_provenance_missing';continue
            row.update(history_source=event['history_source'],visibility_scope=event['visibility_scope'],trace_row_index=event.get('trace_row_index'))
            if event.get('outcome')!='processed_success':row['reason']='not_successfully_processed_history';continue
            try:
                tokens=event['token_ids'];blocks=event['blocks'];n=int(cfg.get('prefix_length',16))
                if len(tokens)<=n or len(blocks)<n//16:row['reason']='short_history_or_block_range';continue
                layout=layout_from_config(cfg,tokens)
                if event.get('identities') is not None and event['identities']!=layout.identities:row['reason']='full_identity_or_model_layer_range_mismatch';continue
                if event.get('model_name',layout.model_name)!=layout.model_name:row['reason']='model_namespace_mismatch';continue
                if event.get('cache_salt',layout.cache_salt)!=layout.cache_salt:row['reason']='prefix_identity_salt_mismatch';continue
                key=layout.cache_salt
                row.update(identity_scope='exact model/execution/layout/full-layer range and full-prefix salt; other identities never substituted',model_name=layout.model_name,cache_salt=key,prefix_length=n,complete_chunks=n//16,ignored_trailing_tokens=len(tokens)-n,blocks=list(blocks[:n//16]))
                if key in unique:
                    row['reason']='duplicate_processed_prefix';row['duplicate_of']=unique[key][1]['history_id'];continue
                unique[key]=(layout,row,event)
            except (KeyError,TypeError,ValueError) as e:row.update(reason='invalid_history_identity',error=str(e))
        started=time.monotonic_ns()
        try:status=self.request('GET','/status')
        except (httpx.HTTPError,ValueError) as e:
            for _,row,_ in unique.values():row.update(reason='directory_unknown',error=str(e))
            return dict(candidates=[],decisions=rows,metadata_calls=self.calls,status='unknown',history_snapshot_sha256=digest(history))
        thresholds=dict(max_event_age_s=cfg.get('candidate_max_delivery_delay_s',cfg.get('candidate_max_event_age_s',60)),max_heartbeat_age_s=cfg.get('candidate_max_heartbeat_age_s',10))
        target_health=stream_state(status,target,**thresholds)
        remote_health={rule['instance_id']:stream_state(status,rule['instance_id'],**thresholds) for rule in cfg['candidate_remote_placements']}
        candidates=[]
        for layout,row,event in unique.values():
            try:
                response=self.request('POST','/directory/lookup',dict(model_name=layout.model_name,world_size=1,token_ids=layout.tokens,cache_salt=layout.cache_salt))
                row['directory_response']=response
                count=len(layout.tokens)//16;resolved=response.get('results',[])
                if response.get('chunks')!=count or len(resolved)!=count:row['reason']='unsupported_or_incomplete_key_fanout';continue
                hashes=set();local=[];remote=[];unknown=[];key_evidence=[]
                for index,item in enumerate(resolved):
                    key=item['key']
                    if key['model_name']!=layout.model_name or key.get('cache_salt','')!=layout.cache_salt or key['object_group_id']!=0 or key['kv_rank']!=16777472:
                        raise ValueError('resolved key model/salt/rank/object group mismatch')
                    if key['chunk_hash_hex'] in hashes:raise ValueError('duplicate resolved chunk key')
                    hashes.add(key['chunk_hash_hex']);local_ok=False;remote_ok=False;uncertain=False;proof=[]
                    for placement in item['placements']:
                        islocal=placement['instance_id']==target and placement['tier']=='l1'
                        isremote=any(matches_rule(placement,r) for r in cfg['candidate_remote_placements'])
                        if not islocal and not isremote:
                            proof.append(dict(placement=placement,classification='unapproved_location'));continue
                        health=stream_state(status,placement['instance_id'],placement['incarnation'],**thresholds)
                        proof.append(dict(placement=placement,classification='target_l1' if islocal else 'remote_allowlisted',freshness=health))
                        if health[0]!='fresh':uncertain=True;continue
                        if placement['size_bytes']<=0:uncertain=True;continue
                        local_ok|=islocal;remote_ok|=isremote
                    local.append(local_ok);remote.append(remote_ok);unknown.append(uncertain)
                    key_evidence.append(dict(chunk_index=index,token_range=[index*16,(index+1)*16],local_complete_key=local_ok,remote_complete_key=remote_ok,uncertain_placement=uncertain,placements=proof))
                row.update(keys=key_evidence,local_chunks=sum(local),remote_chunks=sum(remote),target_status=target_health,remote_authority_status=remote_health,partial_semantics='ALL complete prefix chunks required; no partial-prefix shortening')
                if target_health[0]!='fresh':row['reason']='target_directory_'+target_health[0]
                elif all(local):row['reason']='target_l1_complete'
                elif any(unknown):row['reason']='directory_stale_or_unknown_placement'
                elif not all(remote) and any(v[0]!='fresh' for v in remote_health.values()):row['reason']='remote_directory_stale_or_unknown'
                elif not all(remote):row['reason']='remote_absent_for_exact_model_salt_keys' if not any(remote) else 'remote_partial_prefix'
                else:
                    row.update(eligible=True,reason='remote_complete_target_l1_not_complete')
                    candidates.append(dict(history_id=row['history_id'],prefix_token_ids=list(layout.tokens),prefix_length=len(layout.tokens),token_ids=list(event['token_ids']),blocks=list(event['blocks']),identities=layout.identities,model_name=layout.model_name,cache_salt=layout.cache_salt))
            except (httpx.HTTPError,KeyError,TypeError,ValueError) as e:row.update(reason='directory_unknown_or_identity_mismatch',error=str(e))
        # Two official status observations bound metadata races. Do not call a
        # moving emitter slice an atomic H/R/L snapshot.
        changed=[];status_after=None
        try:
            status_after=self.request('GET','/status')
            relevant={target,*remote_health}
            for owner in sorted(relevant):
                before=status.get('ingest',{}).get('streams',{}).get(owner)
                after=status_after.get('ingest',{}).get('streams',{}).get(owner)
                fields=('incarnation','last_seq','gap_detected')
                if (before is None or after is None or any(before.get(k)!=after.get(k) for k in fields)
                        or stream_state(status_after,owner,**thresholds)[0]!='fresh'):
                    changed.append(owner)
        except (httpx.HTTPError,ValueError) as error:changed=['status_after_unknown'];status_after=dict(error=str(error))
        if changed:
            for _,row,_ in unique.values():
                row['reason_before_concurrency_check']=row.get('reason')
                row.update(eligible=False,reason='directory_concurrent_change_or_unknown',changed_emitters=changed)
            candidates=[]
        return dict(candidates=candidates,history_snapshot_sha256=digest(history),history_boundary=dict(input_events=len(tuple(history)),cutoff_exclusive=cfg.get('candidate_history_before_row'),scope='current consumer visible successfully completed history only'),decisions=rows,metadata_calls=self.calls,official_status=status,official_status_after=status_after,concurrent_changed_emitters=changed,started_ns=started,completed_ns=time.monotonic_ns(),scope='read-only metadata qualification, not ranking/warm/download or physical availability proof')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--history-json',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    f=HistoryCandidateFilter(json.loads(Path(a.config).read_text()));result=f.filter(json.loads(Path(a.history_json).read_text()))
    with Path(a.out).open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(dict(candidates=[x['history_id'] for x in result['candidates']],output=a.out)))
