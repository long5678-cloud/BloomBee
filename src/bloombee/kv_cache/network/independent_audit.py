"""独立只读 E/C 执行门禁；不调用候选过滤器、SDK、warm 或票据 claim。

从完整终态 H 独立重建四张量 page codec 身份。元数据缺失、身份冲突、
快照变化均 unknown；unknown 与空集合是不同结果。仅 metadata HTTP。
"""
import copy
import hashlib
import ipaddress
import json
import math
import re
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler

NATIVE_BYTES = 16 * 1024**2
VARIABLE = {'layer_start','layer_end','prefix_tokens_digest','prefix_length','position_digest'}
IDENTITY_FIELDS = set(('model_id model_revision config_digest tokenizer_digest adapter_digest '
    'layer_start layer_end prefix_tokens_digest prefix_length position_digest attention_digest '
    'rope_digest layout dtype num_attention_heads num_key_value_heads head_dim execution_digest '
    'schema_version position_start batch_size').split())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def candidate_key(row):
    return digest({k:row[k] for k in ('model_name','cache_salt','prefix_length','prefix_token_ids')})


def scope(row):
    return digest(dict(model_name=row['model_name'],prefix_length=16,world_size=1,object_group_id=0))


def identity(value):
    require(isinstance(value,dict) and set(value)==IDENTITY_FIELDS,'identity_schema_mismatch')
    for key,val in value.items():
        if key.endswith('_digest'):
            require(isinstance(val,str) and re.fullmatch('[0-9a-f]{64}',val),'identity_digest_invalid')
    require(re.fullmatch('[0-9a-f]{40}',value['model_revision']) is not None,'model_revision_not_full_sha')
    require(value['schema_version']==1 and value['position_start']==0 and value['batch_size']==1,
        'identity_schema_position_batch')
    require(value['dtype']=='float16' and value['layout']=='bhsd_v1' and
        value['num_attention_heads']==32 and value['head_dim']==128,'frozen_four_tensor_geometry')
    require(type(value['num_key_value_heads']) is int and value['num_key_value_heads']>0 and
        32%value['num_key_value_heads']==0,'invalid_kv_heads')
    return value


def reconstruct(config, tokens):
    """独立复现已冻结 codec 的 namespace/salt 公式，不调用31过滤器。"""
    require(isinstance(tokens,list) and len(tokens)>=16 and
        all(type(t) is int and t>=0 for t in tokens),'short_or_invalid_terminal_tokens')
    tokens=list(tokens[:16]);templates=config['page_identity_template_by_layer']
    require(len(templates)==32,'missing_32_identity_templates')
    ids=[];base=None
    for layer,template in enumerate(templates):
        template=identity(template)
        require(template['layer_start']==layer and template['layer_end']==layer+1,'template_layer_order')
        item=dict(template,prefix_length=16,prefix_tokens_digest=digest(tokens),
            position_digest=digest(dict(start=0,end=16)))
        current={k:v for k,v in item.items() if k not in VARIABLE}
        if base is None:base=current
        require(current==base,'cross_layer_static_identity_mismatch')
        ids.append(item)
    descriptor=dict(format='bloombee_paged_kv_and_aux_lmcache_v1',base=base,
        logical_layers=32,block_size=16,storage_layers=64)
    model='bb_page_'+digest(descriptor)
    salt=digest(dict(model_name=model,blocks=[[digest(i) for i in ids]]))
    info=dict(model_name=model,cache_salt=salt,tokens=tokens,block_size=16,
        logical_layers=32,storage_layers=64,shape=[2,64,16,4096],dtype='torch.float16',
        identities_by_block=[ids])
    row=dict(model_name=model,cache_salt=salt,prefix_length=16,prefix_token_ids=tokens)
    row.update(candidate_key=candidate_key(row),identity_scope=scope(row))
    require(config.get('page_model_name',model)==model,'configured_codec_namespace_mismatch')
    return row,info


def manifest_check(manifest, info, config):
    require(isinstance(manifest,dict),'source_manifest_missing')
    require(manifest.get('format')=='bb_joint_page_bundle_v1' and manifest.get('layout')==info,
        'source_manifest_codec_full_identity_mismatch')
    for field in ('sha_by_block','wire_bytes_by_block'):
        grid=manifest.get(field)
        require(isinstance(grid,list) and len(grid)==1 and len(grid[0])==32,'source_manifest_grid_incomplete')
    require(all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in manifest['sha_by_block'][0]),
        'source_wire_sha_invalid')
    require(all(type(v)is int and v>0 for v in manifest['wire_bytes_by_block'][0]),'source_wire_sizes_invalid')
    require(manifest.get('native_tensor_bytes')==NATIVE_BYTES and
        isinstance(manifest.get('native_value_sha256'),str) and
        re.fullmatch('[0-9a-f]{64}',manifest['native_value_sha256']),'native_value_identity_invalid')
    obj=digest({k:manifest[k] for k in ('format','layout','sha_by_block','native_value_sha256','wire_bytes_by_block')})
    require(manifest.get('object_id')==obj,'manifest_object_digest_mismatch')
    require(manifest.get('source_instance')==config['source_instance'] and
        manifest.get('run_id')==config['source_run_id'],'source_manifest_instance_or_run_mismatch')
    require(manifest.get('source_readback_exact') is True,'source_native_readback_unproved')
    registration=manifest.get('native_registration',{})
    require(registration.get('model_layers')==32 and registration.get('storage_layers')==64 and
        registration.get('endpoint',{}).get('instance_id')==config['source_instance'],
        'source_native_registration_unproved')
    return obj


def history_check(context, qualification, history, config):
    require(isinstance(history,list),'raw_terminal_history_required')
    require(context.get('schema_version')==1,'context_schema')
    cutoff=context['causal_cutoff_ns'];boundary=context['history_boundary']
    require(type(cutoff)is int and cutoff<=time.monotonic_ns(),'future_or_invalid_cutoff')
    require(boundary.get('truncated') is False and
        boundary.get('completion_sequence_semantics')=='unique_successful_full_request_disclosure',
        'complete_history_boundary_required')
    require(boundary['consumer_id']==config.get('candidate_consumer_id',config['native_endpoint']['instance_id']),
        'boundary_consumer_mismatch')
    require(context.get('history_snapshot_sha256')==digest(history)==qualification.get('history_snapshot_sha256'),
        'raw_H_context_qualification_digest_mismatch')
    require(context.get('qualification_sha256')==digest(qualification),'qualification_digest_mismatch')
    prior=boundary['initial_completion_seq'];last=-1;seen=set();events=[];objects={}
    require(type(prior)is int and prior==0,'nonzero_initial_history_boundary_unproved')
    for event in history:
        rid=event['request_id'];require(isinstance(rid,str) and rid and rid not in seen,'duplicate_terminal_request')
        seen.add(rid)
        require(event.get('outcome')=='processed_success' and event.get('consumer_instance_id')==boundary['consumer_id']
            and event.get('clock_domain_id')==boundary['clock_domain_id'],'terminal_owner_clock_outcome_mismatch')
        require(type(event.get('completion_event_seq'))is int and event['completion_event_seq']==prior+1,
            'noncontiguous_terminal_H')
        require(type(event.get('disclosed_ns'))is int and last<=event['disclosed_ns']<=cutoff,
            'future_or_unordered_terminal_H')
        require(event.get('history_source') and event.get('visibility_scope'),'terminal_provenance_missing')
        ref=event.get('terminal_receipt_ref');require(isinstance(ref,dict) and ref.get('path') and ref.get('sha256'),
            'terminal_file_receipt_required')
        raw=Path(ref['path']).read_bytes();require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'terminal_file_sha_mismatch')
        proof=json.loads(raw)
        require(proof.get('request_id')==rid and proof.get('status') in ('complete_success','completed'),
            'terminal_file_request_or_status_mismatch')
        if 'actual_output_tokens' in proof:
            require(type(proof['actual_output_tokens'])is int and proof['actual_output_tokens']==proof.get('requested_output_tokens'),
                'terminal_model_output_incomplete')
        elif 'actual_output_length' in proof:
            require(type(proof['actual_output_length'])is int and proof['actual_output_length']==proof.get('expected_output_length'),
                'terminal_model_output_incomplete')
        else:raise ValueError('terminal_full_output_proof_missing')
        if event.get('attempt_id') is not None:require(proof.get('attempt_id')==event['attempt_id'],'terminal_attempt_mismatch')
        if 'clock_domain_id' in proof:require(proof['clock_domain_id']==boundary['clock_domain_id'],'terminal_file_clock_mismatch')
        if 'completed_monotonic' in proof:
            require(type(proof['completed_monotonic']) in (int,float) and math.isfinite(proof['completed_monotonic'])
                and int(proof['completed_monotonic']*1e9)<=event['disclosed_ns'],'terminal_file_after_disclosure')
        row,info=reconstruct(config,event['token_ids'])
        flat=dict(candidate_key=row['candidate_key'],identity_scope=row['identity_scope'],
            completion_event_seq=event['completion_event_seq'],request_id=rid,disclosed_ns=event['disclosed_ns'],
            consumer_id=event['consumer_instance_id'],clock_domain_id=event['clock_domain_id'],outcome=event['outcome'],
            terminal_receipt_ref=ref,history_source=event['history_source'],visibility_scope=event['visibility_scope'])
        events.append(flat);entry=objects.setdefault(row['candidate_key'],dict(descriptor=row,layout=info,manifests=[]))
        if event.get('source_bundle_manifest') is not None:entry['manifests'].append(event['source_bundle_manifest'])
        prior=event['completion_event_seq'];last=event['disclosed_ns']
    require(context.get('completed_history')==events and context.get('decision_seq')==prior,
        'context_flattened_H_differs_from_complete_raw_H')
    require(context.get('identity_scope')==scope(dict(model_name=config['page_model_name'])),
        'context_static_scope_mismatch')
    return objects


def _request(method, endpoint, route, body, config):
    """唯一网络入口；只允许三类 metadata API，不使用环境代理。"""
    require((method,route) in {('GET','/status'),('GET','/instances'),('POST','/status'),
        ('POST','/directory/lookup'),('POST','/page_probe')},'non_metadata_operation_forbidden')
    parsed=urlsplit(endpoint)
    require(parsed.scheme=='http' and parsed.hostname and not parsed.username and not parsed.password and
        parsed.path in ('','/') and not parsed.query and not parsed.fragment,'invalid_metadata_endpoint')
    headers={'Content-Type':'application/json'}
    if endpoint.rstrip('/')==config.get('ticket_endpoint','').rstrip('/') and config.get('ticket_control_token'):
        headers['X-Bridge-Token']=config['ticket_control_token']
    req=Request(endpoint.rstrip('/')+route,data=None if body is None else json.dumps(body).encode(),
        headers=headers,method=method)
    with build_opener(ProxyHandler({})).open(req,timeout=config.get('audit_metadata_timeout_s',10)) as response:
        raw=response.read(16*1024**2+1)
        require(len(raw)<=16*1024**2,'metadata_response_size_exceeded')
        return json.loads(raw)


def frontier(status, owners, config):
    def natural(number):
        return type(number) is int and number >= 0

    def seconds(number):
        return type(number) in (int, float) and math.isfinite(number) and number >= 0

    result = {}
    ingest = status['ingest']
    delay_limit = config.get('candidate_max_delivery_delay_s', config.get('candidate_max_event_age_s', 60))
    heartbeat_limit = config.get('candidate_max_heartbeat_age_s', 10)
    require(seconds(delay_limit) and seconds(heartbeat_limit), 'native_timing_limit_unknown')
    for owner in owners:
        s = ingest['streams'][owner]
        require(natural(s['incarnation']) and natural(s['last_seq']) and s['last_seq'] > 0
                and s['gap_detected'] is False, 'native_frontier_gap_or_identity_unknown')
        for key in ('missing_batches', 'events_dropped'):
            require(natural(s[key]) and s[key] == 0, 'native_event_loss_or_count_unknown')
        require(seconds(s['seconds_since_last_event']), 'native_idle_duration_unknown')
        delay = s['event_delivery_delay_seconds']
        require(seconds(delay) and delay <= delay_limit, 'native_event_delivery_stale_or_unknown')
        age = status['registry']['seconds_since_heartbeat'][owner]
        require(seconds(age) and age <= heartbeat_limit, 'native_heartbeat_stale')
        result[owner] = {k:s[k] for k in ('incarnation','last_seq','gap_detected')}
    consumers = status['consumers']
    require(isinstance(consumers, dict) and 'KeyDirectory' in consumers, 'native_directory_consumer_unknown')
    applied = ingest['batches_applied']
    require(natural(applied) and applied > 0, 'native_applied_count_unknown')
    for entry in consumers.values():
        require(natural(entry['batches_delivered']) and entry['batches_delivered'] == applied,
                'native_consumer_delivery_mismatch')
        for key in ('consume_failures', 'fence_failures'):
            require(natural(entry[key]) and entry[key] == 0, 'native_consumer_apply_failure')
    return result


def bank_semantic(body):
    inv=body.get('inventory',body)
    require(inv.get('schema_version')==1 and type(inv.get('bank_epoch'))is int and
        type(inv.get('actual_monotonic_ns'))is int and isinstance(inv.get('objects'),list),
        'bank_readonly_inventory_missing')
    require(len({o['ticket_id'] for o in inv['objects']})==len(inv['objects']),'duplicate_bank_ticket')
    for obj in inv['objects']:
        expiry=obj.get('expires_at')
        require(type(expiry) in (int,float) and math.isfinite(expiry),'bank_expiry_unknown')
        require(type(obj.get('expired'))is bool and obj['expired']==(inv['actual_monotonic_ns']/1e9>=expiry),
            'bank_expiry_boundary_inconsistent')
    return {k:copy.deepcopy(v) for k,v in inv.items() if k!='actual_monotonic_ns'}


def page_semantic(body,info):
    require(body.get('snapshot_atomic') is True and body.get('snapshot_lock')=='model_metadata_guard_v1'
        and type(body.get('page_epoch'))is int,'page_atomic_snapshot_epoch_unproved')
    require(body.get('known') is True and type(body.get('complete'))is bool and
        type(body.get('demand_inflight'))is bool,'page_pool_snapshot_unknown')
    evidence=body.get('evidence');require(isinstance(evidence,list) and len(evidence)==32,'page_pool_32_layer_evidence_missing')
    require(sorted(r['layer'] for r in evidence)==list(range(32)),'page_pool_layer_coverage')
    for row in evidence:
        require(row.get('identity')==info['identities_by_block'][0][row['layer']],'page_pool_identity_mismatch')
        require(type(row.get('cpu_complete'))is bool and type(row.get('gpu_complete'))is bool,'page_pool_completeness_unknown')
        if row['gpu_complete']:require(type(row.get('slot'))is int and type(row.get('generation'))is int,'GPU_slot_generation_unknown')
        if row['cpu_complete']:require(isinstance(row.get('cpu_digest'),str) and bool(row['cpu_digest']),'CPU_block_digest_unknown')
    actual=all(r['cpu_complete'] or r['gpu_complete'] for r in evidence)
    require(body['complete']==actual,'page_complete_does_not_match_real_layers')
    require(type(body.get('runtime_pid'))is int,'page_probe_runtime_identity_unknown')
    return {k:copy.deepcopy(v) for k,v in body.items() if k not in ('monotonic_ns','actual_monotonic_ns')}


def reader_semantic(body):
    require(body.get('snapshot_atomic') is True and body.get('snapshot_lock')=='model_metadata_guard_v1'
        and type(body.get('page_epoch'))is int,'page_status_atomic_epoch_unproved')
    require(type(body.get('pid'))is int and body.get('clock_domain') and len(body.get('states',[]))==32,
        'page_runtime_status_unknown')
    require(all(row.get('active_sessions')==0 for row in body['states']),'actual_model_mutation_inflight')
    # prepared payload is not visible to page_probe; inspect its independent registry.
    registry=body.get('bundle_registry')
    require(isinstance(registry,dict),'reader_lease_inventory_missing')
    require(registry.get('schema')==1 and type(registry.get('epoch'))is int and
        registry.get('runtime_pid')==body['pid'] and isinstance(registry.get('leases'),list),
        'reader_lease_version_unknown')
    require(not registry['leases'],'reader_prepared_or_active_lease')
    return {k:copy.deepcopy(v) for k,v in body.items() if k not in ('monotonic_ns','actual_monotonic_ns')}


def audit(context, qualification, history, config, decision_dir):
    """返回可JSON序列化门禁；未知/误收/漏收/重复均不准execute。"""
    result=dict(schema_version=1,ec_match=False,unknown=[],false_positive=[],false_negative=[],
        duplicate_C=[],E=None,C=None,H=None,R=None,L=None,started_ns=time.monotonic_ns(),
        scope='independent full terminal H; exact codec; source ValkeyL2 minus usable native/bank/page local',
        demand_inflight_excluded=False,metadata_only=True)
    evidence=[]
    def query(label,method,endpoint,route,body=None):
        response=_request(method,endpoint,route,body,config)
        evidence.append(dict(label=label,method=method,endpoint=endpoint,route=route,
            request=body,response=response,response_sha256=digest(response),observed_ns=time.monotonic_ns()))
        return response
    try:
        objects=history_check(context,qualification,history,config)
        result['H']=sorted(objects)
        source=config['source_instance'];target=config['native_endpoint']['instance_id']
        require(source!=target,'source_and_target_instance_equal')
        owners=(source,target);coordinator=config['coordinator_url'];ticket=config['ticket_endpoint'];page=config['page_control_url']
        statuses=[];banks=[];reader_states=[];object_snapshots=[]
        for phase in ('before','after'):
            status=query(phase+'_coordinator','GET',coordinator,'/status');f=frontier(status,owners,config);statuses.append(f)
            registrations=query(phase+'_instances','GET',coordinator,'/instances')
            regs={}
            for owner in owners:
                found=[r for r in registrations['instances'] if r['instance_id']==owner]
                require(len(found)==1,'native_instance_membership_unknown');regs[owner]=found[0]
            bank=query(phase+'_bank','POST',ticket,'/status',{});banks.append(bank_semantic(bank))
            runtime=query(phase+'_page_status','GET',page,'/status');reader_states.append(reader_semantic(runtime))
            snap={}
            for key,item in objects.items():
                row=item['descriptor'];info=item['layout']
                lookup=query(phase+'_directory_'+key,'POST',coordinator,'/directory/lookup',
                    dict(model_name=row['model_name'],cache_salt=row['cache_salt'],world_size=1,token_ids=row['prefix_token_ids']))
                require(lookup.get('chunks')==1 and len(lookup.get('results',[]))==1,'directory_key_fanout_unknown')
                found=lookup['results'][0];encoded=found['key']
                require(encoded['model_name']==row['model_name'] and encoded.get('cache_salt')==row['cache_salt'] and
                    encoded.get('kv_rank')==16777472 and encoded.get('object_group_id')==0 and
                    re.fullmatch('[0-9a-f]{64}',encoded['chunk_hash_hex']),'directory_full_key_identity_mismatch')
                remote=[];native=[]
                for placement in found['placements']:
                    owner=placement.get('instance_id')
                    relevant=(owner==source and placement.get('tier')=='l2' and placement.get('backend')=='valkey') or (owner==target and placement.get('tier')=='l1')
                    if not relevant:continue
                    require(placement.get('incarnation')==f[owner]['incarnation'],'placement_instance_incarnation_mismatch')
                    require(placement.get('size_bytes')==NATIVE_BYTES,'placement_native_value_size_mismatch')
                    (remote if owner==source else native).append(placement)
                require(len(remote)<=1 and len(native)<=1,'duplicate_native_placement')
                manifests=item['manifests'];manifest=None;obj=None
                if manifests:
                    for m in manifests:
                        obj=manifest_check(m,info,config)
                        if manifest is not None:require(obj==manifest['object_id'],'immutable_source_value_conflict')
                        manifest=m
                if remote:require(manifest is not None,'remote_present_but_disclosed_full_manifest_missing')
                if remote:
                    expected_inc=config.get('audit_source_incarnation',manifest.get('source_incarnation'))
                    require(type(expected_inc)is int and expected_inc==f[source]['incarnation'],
                        'source_manifest_publication_incarnation_unbound')
                    registered_endpoint=manifest['native_registration']['endpoint']
                    endpoint=urlsplit(registered_endpoint['http_url'])
                    require(endpoint.scheme=='http' and not endpoint.username and not endpoint.password and
                        endpoint.port==regs[source]['http_port'] and
                        (endpoint.hostname==regs[source]['ip'] or ipaddress.ip_address(endpoint.hostname).is_loopback),
                        'source_native_registration_endpoint_changed')
                probe=query(phase+'_page_'+key,'POST',page,'/page_probe',
                    dict(tokens=row['prefix_token_ids'],identities_by_block=info['identities_by_block']))
                probe=page_semantic(probe,info)
                require(probe['runtime_pid']==runtime['pid'],'page_probe_runtime_changed')
                require(probe['page_epoch']==runtime['page_epoch'],'page_mutated_inside_snapshot_scan')
                bank_local=False
                for bankobj in banks[-1]['objects']:
                    bank_info=(bankobj.get('source_manifest') or {}).get('layout',{})
                    if bank_info.get('model_name')!=row['model_name'] or bank_info.get('cache_salt')!=row['cache_salt']:continue
                    require(manifest is not None,'bank_present_but_manifest_missing')
                    bank_manifest=bankobj.get('source_manifest')
                    require(manifest_check(bank_manifest,info,config)==obj and bankobj.get('object_id')==obj,
                        'bank_full_object_identity_mismatch')
                    require(bankobj.get('manifest_sha256')==digest(bank_manifest) and
                        bankobj.get('validated_complete_32layer_sha') is True,'bank_wire_validation_unproved')
                    require(bankobj.get('state')=='ready' and bankobj.get('claimable') is True and
                        bankobj.get('expired') is False and not bankobj.get('sending') and
                        not bankobj.get('deferred_remove_status'),'bank_nonconsumable_or_transitioning')
                    require(bankobj.get('local_payload_bytes')==sum(manifest['wire_bytes_by_block'][0]),'bank_payload_incomplete')
                    require((bankobj.get('native_target') or {}).get('instance_id')==target,
                        'bank_native_target_identity_unknown')
                    registration=bankobj.get('native_registration') or {}
                    require(all(registration.get(k)==regs[target].get(k) for k in
                        ('instance_id','ip','http_port','registration_time')),'bank_native_registration_changed')
                    chunks=bankobj.get('native_chunk_incarnations')
                    require(isinstance(chunks,list) and len(chunks)==1 and
                        chunks[0].get('incarnation')==f[target]['incarnation'] and
                        chunks[0].get('size_bytes')==NATIVE_BYTES,'bank_native_incarnation_or_size_unknown')
                    require(bankobj.get('expected_key')==encoded,'bank_native_full_key_mismatch')
                    bank_local=True
                snap[key]=dict(remote=bool(remote),native_l1=bool(native),bank=bank_local,
                    page=probe['complete'],directory=found,page_probe=probe,registrations=regs)
            object_snapshots.append(snap)
        require(statuses[0]==statuses[1],'native_frontier_changed_during_independent_audit')
        for label in ('official_status','official_status_after'):
            require(frontier(qualification[label],owners,config)==statuses[0],'worker_and_audit_native_snapshot_differ')
        require(banks[0]==banks[1],'bank_inventory_changed_during_independent_audit')
        for label in ('bank_inventory_before','bank_inventory_after'):
            require(bank_semantic(qualification[label])==banks[0],'worker_and_audit_bank_snapshot_differ')
        require(reader_states[0]==reader_states[1],'page_runtime_or_reader_changed_during_audit')
        require(object_snapshots[0]==object_snapshots[1],'per_object_native_or_page_snapshot_changed')
        original_probes={}
        for item in qualification.get('decisions',[]):
            if 'actual_page_pool_state' in item:original_probes[(item['model_name'],item['cache_salt'])]=item['actual_page_pool_state']
        for key,item in objects.items():
            pair=(item['descriptor']['model_name'],item['descriptor']['cache_salt'])
            require(pair in original_probes,'worker_same_snapshot_page_evidence_missing')
            require(page_semantic(original_probes[pair],item['layout'])==object_snapshots[0][key]['page_probe'],
                'worker_and_audit_page_snapshot_differ')
        H=set(objects);R={k for k,v in object_snapshots[0].items() if v['remote']}
        L={k for k,v in object_snapshots[0].items() if v['native_l1'] or v['bank'] or v['page']};E=(H&R)-L
        rawC=qualification.get('candidates');ctxC=context.get('candidates')
        require(isinstance(rawC,list) and isinstance(ctxC,list),'C_not_full_list')
        C=[]
        for row in rawC:
            key=candidate_key(row);C.append(key)
            independent_row,info=reconstruct(config,row['prefix_token_ids'])
            require({k:row[k] for k in ('model_name','cache_salt','prefix_length','prefix_token_ids')}==
                {k:independent_row[k] for k in ('model_name','cache_salt','prefix_length','prefix_token_ids')},'C_exact_descriptor_mismatch')
            object_id=manifest_check(row.get('source_manifest'),info,config)
            if key in objects:
                require(all(m['object_id']==object_id for m in objects[key]['manifests']),
                    'C_manifest_differs_from_disclosed_source_value')
        reported=[]
        for row in ctxC:
            key=candidate_key(row);require(row.get('candidate_key')==key and row.get('identity_scope')==scope(row),'context_C_identity_mismatch');reported.append(key)
        duplicates=sorted({k for k in C+reported if C.count(k)>1 or reported.count(k)>1})
        require(C==reported,'context_C_differs_from_qualification_full_C')
        for raw,wrapped in zip(rawC,ctxC):
            require({k:v for k,v in raw.items() if k not in ('candidate_key','identity_scope')}==
                {k:v for k,v in wrapped.items() if k not in ('candidate_key','identity_scope')},
                'context_C_manifest_or_payload_differs')
        result.update(H=sorted(H),R=sorted(R),L=sorted(L),E=sorted(E),C=sorted(set(C)),
            false_positive=sorted(set(C)-E),false_negative=sorted(E-set(C)),duplicate_C=duplicates,
            independent_objects=object_snapshots[0],history_snapshot_sha256=digest(history),context_sha256=digest(context))
        if qualification.get('unknown') or context.get('qualification_status')!='qualified':result['unknown'].append('worker_qualification_unknown')
        result['ec_match']=not result['unknown'] and not result['false_positive'] and not result['false_negative'] and not duplicates
    except Exception as error:
        result['unknown'].append(type(error).__name__+':'+str(error))
    result['completed_ns']=time.monotonic_ns()
    try:
        directory=Path(decision_dir);path=directory/('audit_ec_evidence_'+uuid.uuid4().hex+'.json')
        data=json.dumps(dict(schema_version=1,observations=evidence,result=result),sort_keys=True,
            ensure_ascii=False,allow_nan=False).encode()
        with path.open('xb') as f:f.write(data)
        result['evidence_ref']=dict(path=str(path),sha256=hashlib.sha256(data).hexdigest())
    except Exception as error:
        result['ec_match']=False;result['unknown'].append('audit_evidence_write_failed:'+type(error).__name__)
    return result
