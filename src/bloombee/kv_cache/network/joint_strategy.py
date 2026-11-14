"""Page-codec namespace adapter for the existing causal H / two score options.

Only terminal H is enumerated. Registry entries, byte hashes and source native
placements are evidence, never demand events. No test/front queue reader.
"""
import copy,time
from pathlib import Path
from bloombee.kv_cache.network.bundle_contract import digest,validate_manifest,layout_metadata
from bloombee.kv_cache.network.native_page_bridge import codec_layout

def layout_for_tokens(config,tokens):
    from bloombee.kv_cache.identity import CacheIdentity,canonical_digest,token_ids_digest
    from dataclasses import replace
    tokens=list(tokens[:16])
    if len(tokens)!=16:raise ValueError('unrepresentable short completed page demand')
    templates=config['page_identity_template_by_layer']
    identities=[replace(CacheIdentity.from_dict(i),prefix_length=16,prefix_tokens_digest=token_ids_digest(tokens),position_digest=canonical_digest(dict(start=0,end=16))).to_dict() for i in templates]
    return codec_layout([identities],tokens)

def joint_context(config,report,history,boundary,decision_id,causal_cutoff_ns):
    from bloombee.kv_cache.network.strategy_contract import candidate_key,identity_scope,validate_context
    from bloombee.kv_cache.network.strategy_runner import validate_demand_ledger
    validate_demand_ledger(history,boundary,causal_cutoff_ns)
    if boundary['consumer_id']!=config.get('candidate_consumer_id',config['native_endpoint']['instance_id']):raise ValueError('consumer boundary differs from actual target')
    if report['history_snapshot_sha256']!=digest(history):raise ValueError('joint C/H snapshot differs')
    candidates=[]
    for item in report['candidates']:
        row=dict(item);row.update(candidate_key=candidate_key(row),identity_scope=identity_scope(row));candidates.append(row)
    events=[];seen=set();scope=None
    for raw in history:
        if raw['request_id'] in seen:continue
        seen.add(raw['request_id']);layout=layout_for_tokens(config,raw['token_ids'])
        row=dict(model_name=layout.model_name,cache_salt=layout.cache_salt,prefix_length=16,prefix_token_ids=layout.tokens)
        scope=identity_scope(row)
        events.append(dict(candidate_key=candidate_key(row),identity_scope=scope,completion_event_seq=raw['completion_event_seq'],request_id=raw['request_id'],disclosed_ns=raw['disclosed_ns'],consumer_id=raw['consumer_instance_id'],clock_domain_id=raw['clock_domain_id'],outcome=raw['outcome'],terminal_receipt_ref=raw['terminal_receipt_ref'],history_source=raw['history_source'],visibility_scope=raw['visibility_scope']))
    if scope is None:scope=identity_scope(dict(model_name=config['page_model_name'],prefix_length=16))
    return validate_context(dict(schema_version=1,decision_id=decision_id,decision_seq=events[-1]['completion_event_seq'] if events else boundary['initial_completion_seq'],identity_scope=scope,candidates=candidates,completed_history=events,causal_cutoff_ns=causal_cutoff_ns,history_boundary=copy.deepcopy(boundary),qualification_status='unknown' if report.get('unknown') else 'qualified',qualification_sha256=digest(report),history_snapshot_sha256=digest(history)))

class PageCandidateFilter:
    def __init__(self,config,bank,local_page_probe=None):self.config=config;self.bank=bank;self.local_page_probe=local_page_probe
    def filter(self,history,*,exclude_inflight=False):
        import httpx
        from bloombee.kv_cache.network.history_candidates import stream_state
        cfg=self.config;target=cfg['native_endpoint']['instance_id'];source=cfg['source_instance'];decisions=[];candidates=[];seen=set();unknown=False
        registry={};bank_before=self.bank.inventory()
        # Front may attach the disclosed publication manifest to past terminal
        # events. Never scan source-private files or future registry rows.
        for event in history:
            manifest=event.get('source_bundle_manifest')
            if manifest:
                validate_manifest(manifest)
                if manifest['run_id']!=cfg.get('source_run_id',cfg['run_id']) or manifest['source_instance']!=source:raise ValueError('wrong arm/source manifest in disclosed H')
                registry[(manifest['layout']['model_name'],manifest['layout']['cache_salt'])]=manifest
        limits=dict(max_event_age_s=cfg.get('candidate_max_delivery_delay_s',cfg.get('candidate_max_event_age_s',60)),max_heartbeat_age_s=cfg.get('candidate_max_heartbeat_age_s',10))
        with httpx.Client(trust_env=False,timeout=cfg.get('native_timeout_s',120)) as client:
            r=client.get(cfg['coordinator_url']+'/status');r.raise_for_status();before=r.json()
            for event in history:
                layout=layout_for_tokens(cfg,event['token_ids']);key=(layout.model_name,layout.cache_salt);row=dict(request_id=event['request_id'],model_name=key[0],cache_salt=key[1],eligible=False)
                decisions.append(row)
                if key in seen:row['reason']='duplicate_history_prefix';continue
                seen.add(key);manifest=registry.get(key)
                if manifest is not None:validate_manifest(manifest,layout)
                r=client.post(cfg['coordinator_url']+'/directory/lookup',json=dict(model_name=key[0],cache_salt=key[1],world_size=1,token_ids=layout.tokens));r.raise_for_status();body=r.json();row['directory']=body
                if body.get('chunks')!=1 or len(body.get('results',[]))!=1:row['reason']='native_key_fanout_unknown';unknown=True;continue
                result=body['results'][0];encoded=result['key']
                if encoded['model_name']!=key[0] or encoded.get('cache_salt')!=key[1] or encoded.get('kv_rank')!=16777472 or encoded.get('object_group_id')!=0:raise ValueError('joint lookup resolved wrong complete key')
                statuses={owner:stream_state(before,owner,**limits) for owner in (source,target)};row['status']=statuses
                if any(s[0]!='fresh' for s in statuses.values()):row['reason']='native_directory_unknown_or_stale';unknown=True;continue
                remote=False;temporary=False
                for p in result['placements']:
                    state=stream_state(before,p['instance_id'],p['incarnation'],**limits)
                    if state[0]!='fresh':unknown=True;continue
                    remote|=(p['instance_id']==source and p['tier']=='l2' and p['backend']=='valkey' and p['size_bytes']==layout.tensor_bytes)
                    temporary|=(p['instance_id']==target and p['tier']=='l1' and p['size_bytes']==layout.tensor_bytes)
                local=self.local_page_probe(layout,manifest or dict(layout=layout_metadata(layout))) if self.local_page_probe else dict(known=False,reason='actual32layer_page_pool_state_not_supplied')
                owned=[o for o in bank_before['objects'] if o['source_manifest']['layout']['model_name']==key[0] and o['source_manifest']['layout']['cache_salt']==key[1]]
                bank_boundary=any(o['state']!='ready' or not o['claimable'] or o['sending'] or o.get('deferred_remove_status') for o in owned)
                bank_ready=any(o['state']=='ready' and o['claimable'] and o['validated_complete_32layer_sha'] and manifest is not None and o['object_id']==manifest['object_id'] for o in owned)
                row.update(remote_complete=remote,native_transfer_l1_present=temporary,managed_bank_present=bank_ready,bank_object_evidence=owned,actual_page_pool_state=local)
                if manifest is None:
                    if remote:row['reason']='remote_bundle_missing_disclosed_manifest';unknown=True
                    else:row['reason']='source_remote_absent_no_manifest'
                    continue
                if bank_boundary:row['reason']='bank_object_lifecycle_boundary_unknown';unknown=True
                elif not local.get('known'):row['reason']='actual_page_pool_state_unknown';unknown=True
                elif exclude_inflight and type(local.get('demand_inflight')) is not bool:row['reason']='target_demand_inflight_unknown';unknown=True
                elif exclude_inflight and local.get('demand_inflight') is True:row['reason']='target_demand_inflight'
                elif local.get('complete') or row['managed_bank_present'] or temporary:row['reason']='target_already_has_complete_page_or_transfer'
                elif not remote:row['reason']='source_exact_bundle_not_in_native_L2'
                else:
                    row.update(eligible=True,reason='disclosed_H_remote_complete_actual_target_missing')
                    candidates.append(dict(model_name=key[0],cache_salt=key[1],prefix_length=16,prefix_token_ids=layout.tokens,source_manifest=manifest,history_id=event['request_id']))
            r=client.get(cfg['coordinator_url']+'/status');r.raise_for_status();after=r.json()
        changed=[]
        for owner in (source,target):
            old=before.get('ingest',{}).get('streams',{}).get(owner,{});new=after.get('ingest',{}).get('streams',{}).get(owner,{})
            if any(old.get(k)!=new.get(k) for k in ('incarnation','last_seq','gap_detected')):changed.append(owner)
            if stream_state(after,owner,**limits)[0]!='fresh':unknown=True
        bank_after=self.bank.inventory()
        bank_changed=bank_before['bank_epoch']!=bank_after['bank_epoch'] or bank_before['objects']!=bank_after['objects']
        if changed or bank_changed:unknown=True
        return dict(bank_inventory_before=bank_before,bank_inventory_after=bank_after,bank_concurrent_changed=bank_changed,candidates=[] if unknown else candidates,decisions=decisions,unknown=unknown,concurrent_changed_emitters=changed,official_status=before,official_status_after=after,history_snapshot_sha256=digest(history),scope='joint exact page identities; native temporary L1, bank and actual page CPU/GPU kept distinct')


def make_joint_runner(config,bridge,bank,emit,local_page_probe=None):
    from bloombee.kv_cache.network.strategy_runner import StrategyRunner,validate_demand_ledger
    from bloombee.kv_cache.network.strategy_contract import validate_context,validate_result
    import threading
    class JointRunner(StrategyRunner):
        def qualify(self,history):return PageCandidateFilter(config,bank,local_page_probe).filter(history)
        def warm_selected(self,context,selected_result,history):
            with self.warm_lock:
                validate_context(context);validate_result(context,selected_result);validate_demand_ledger(history,context['history_boundary'],context['causal_cutoff_ns'])
                if digest(history)!=context['history_snapshot_sha256'] or selected_result['strategy']!=config['prefetch_strategy']:raise ValueError('selected joint context/arm differs')
                result=copy.deepcopy(selected_result);result.update(ranked_selected_key=result['selected_key'],warm_attempts=0)
                if result['selected_key'] is None:return result
                recheck=PageCandidateFilter(config,bank,local_page_probe).filter(history,exclude_inflight=True);fresh=joint_context(config,recheck,history,context['history_boundary'],context['decision_id'],context['causal_cutoff_ns']);eligible={c['candidate_key']:c for c in fresh['candidates']}
                if fresh['qualification_status']!='qualified' or result['selected_key'] not in eligible:return dict(result,selected_key=None,abstain_reason='joint_selected_no_longer_qualified')
                candidate=eligible[result['selected_key']];manifest=candidate['source_manifest'];info=manifest['layout'];layout=codec_layout(info['identities_by_block'],info['tokens'])
                path=Path(config['strategy_attempt_dir']);path.mkdir(parents=True,exist_ok=True)
                try:
                    with (path/(digest(dict(run_id=config['run_id'],decision_id=context['decision_id']))+'.json')).open('x') as stream:__import__('json').dump(dict(context_sha256=digest(context),selected_key=result['selected_key'],object_id=manifest['object_id']),stream)
                except FileExistsError:return dict(result,selected_key=None,abstain_reason='joint_decision_already_attempted')
                result['warm_attempts']=1;result['official_warm_posts']=None
                link=dict(decision_id=context['decision_id'],strategy=config['prefetch_strategy'],candidate_key=result['selected_key'],context_sha256=digest(context),qualification_sha256=digest(recheck),target_instance=config['native_endpoint']['instance_id'],run_id=config['run_id'])
                emit('strategy_warm_submit',decision_id=context['decision_id'],joint=True,selected_key=result['selected_key'],object_id=manifest['object_id'])
                try:
                    result['joint_ready']=bridge.warm_to_bank(layout,manifest,link)
                    result['official_warm_posts']=result['joint_ready'].get('official_warm_posts')
                except Exception as error:result.update(warm_error_type=type(error).__name__,warm_error=str(error))
                return result
        def close(self):bridge.close()
    return JointRunner(config,emit)
