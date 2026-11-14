"""Ported 28 independent metadata gate cases plus actual Controller execute IPC.
Synthetic metadata fixtures; not official LMCache or GPU acceptance.
"""
import copy,hashlib,json,time
from pathlib import Path
from bloombee.kv_cache.network import independent_audit as gate

def test_source_audit_cases_and_controller_execute_gate(tmp_path,monkeypatch):
    OUT=tmp_path/'audit';OUT.mkdir()
    def save(p,x):
        with p.open('x') as f:json.dump(x,f,indent=2)
    template=json.loads((Path(__file__).parent/'fixtures/published_identity.json').read_text())['events'][0]['remote_publication']['records'][0]['cache_identity']
    templates=[dict(template,layer_start=i,layer_end=i+1) for i in range(32)]
    cfg=dict(page_identity_template_by_layer=templates,native_endpoint=dict(instance_id='a',http_url='http://127.0.0.1:9001'),source_instance='b',source_run_id='source_library',coordinator_url='http://127.0.0.1:9000',ticket_endpoint='http://127.0.0.1:9002',page_control_url='http://127.0.0.1:9003',candidate_consumer_id='client_a1',audit_source_incarnation=7)
    tokens=list(range(16));desc,layout=gate.reconstruct(cfg,tokens);cfg['page_model_name']=desc['model_name']
    from bloombee.kv_cache.page_bundle import PageBundleLayout
    codec=PageBundleLayout([layout['identities_by_block'][0]],tokens)
    assert (codec.model_name,codec.cache_salt,list(codec.shape),str(codec.dtype),codec.tensor_bytes)==(layout['model_name'],layout['cache_salt'],layout['shape'],layout['dtype'],gate.NATIVE_BYTES)
    manifest=dict(format='bb_joint_page_bundle_v1',layout=layout,sha_by_block=[[gate.digest(dict(layer=i,fixture=True)) for i in range(32)]],wire_bytes_by_block=[[512 for _ in range(32)]],native_value_sha256=gate.digest('fixture_not_model_KV'),native_tensor_bytes=gate.NATIVE_BYTES,source_instance='b',run_id='source_library',source_readback_exact=True,native_registration=dict(model_layers=32,storage_layers=64,endpoint=dict(instance_id='b',http_url='http://127.0.0.1:9004')))
    manifest['object_id']=gate.digest({k:manifest[k] for k in ('format','layout','sha_by_block','native_value_sha256','wire_bytes_by_block')})
    boundary=dict(truncated=False,consumer_id='client_a1',window_id='window_a',completion_sequence_semantics='unique_successful_full_request_disclosure',clock_domain_id='fixture_clock',initial_completion_seq=0)
    proof=OUT/'terminal.json';save(proof,dict(request_id='request_a',attempt_id='attempt_a',status='complete_success',actual_output_tokens=3,requested_output_tokens=3))
    ref=dict(path=str(proof),sha256=hashlib.sha256(proof.read_bytes()).hexdigest())
    now=time.monotonic_ns();event=dict(request_id='request_a',attempt_id='attempt_a',token_ids=tokens,outcome='processed_success',consumer_instance_id='client_a1',clock_domain_id='fixture_clock',completion_event_seq=1,disclosed_ns=now-1000,terminal_receipt_ref=ref,history_source='offline_full_terminal_fixture',visibility_scope='fixture_consumer',source_bundle_manifest=manifest)
    nativekey=dict(model_name=desc['model_name'],cache_salt=desc['cache_salt'],kv_rank=16777472,object_group_id=0,chunk_hash_hex=gate.digest('native_chunk_fixture'))
    status=dict(ingest=dict(batches_applied=10,streams={owner:dict(incarnation=inc,last_seq=5,gap_detected=False,missing_batches=0,events_dropped=0,event_delivery_delay_seconds=0.004,seconds_since_last_event=4572.) for owner,inc in [('a',8),('b',7)]}),registry=dict(seconds_since_heartbeat=dict(a=0.,b=0.)),consumers=dict(KeyDirectory=dict(batches_delivered=10,consume_failures=0,fence_failures=0)))
    bank=dict(schema_version=1,inventory_revision='joint_bank_inventory_a1',bank_epoch=0,actual_monotonic_ns=now,objects=[])
    page=dict(known=True,complete=False,demand_inflight=False,runtime_pid=123,monotonic_ns=now,snapshot_atomic=True,snapshot_lock='model_metadata_guard_v1',page_epoch=4,evidence=[dict(uid=f'layer.{i}',layer=i,identity=layout['identities_by_block'][0][i],gpu_complete=False,slot=None,generation=None,cpu_complete=False,cpu_digest=None,gpu_source=None,cpu_source=None) for i in range(32)])
    runtime=dict(pid=123,clock_domain='boot_fixture',monotonic_ns=now,snapshot_atomic=True,snapshot_lock='model_metadata_guard_v1',page_epoch=4,states=[dict(uid=f'layer.{i}',active_sessions=0) for i in range(32)],bundle_registry=dict(schema=1,epoch=0,runtime_pid=123,leases=[]))
    remote=dict(instance_id='b',tier='l2',backend='valkey',incarnation=7,size_bytes=gate.NATIVE_BYTES)
    registrations=dict(instances=[dict(instance_id='a',ip='127.0.0.1',http_port=9001,registration_time=1),dict(instance_id='b',ip='127.0.0.1',http_port=9004,registration_time=2)])
    def fixture():return dict(history=[copy.deepcopy(event)],C=True,duplicate=False,local_native=False,placements=[copy.deepcopy(remote)],page=copy.deepcopy(page),bank=copy.deepcopy(bank),runtime=copy.deepcopy(runtime),status=copy.deepcopy(status),config=copy.deepcopy(cfg))
    def run(name,mutate,expect,reason=None):
        data=fixture();mutate(data);history=data['history'];candidates=[dict(desc,source_manifest=copy.deepcopy(manifest),history_id='request_a')] if data['C'] else []
        for row in candidates:row.pop('candidate_key',None);row.pop('identity_scope',None)
        if data['duplicate']:candidates=candidates+candidates
        qualification=dict(candidates=candidates,unknown=False,concurrent_changed_emitters=[],official_status=copy.deepcopy(data['status']),official_status_after=copy.deepcopy(data['status']),history_snapshot_sha256=gate.digest(history),bank_inventory_before=copy.deepcopy(data['bank']),bank_inventory_after=copy.deepcopy(data['bank']),decisions=[dict(model_name=desc['model_name'],cache_salt=desc['cache_salt'],actual_page_pool_state=copy.deepcopy(data['page']))])
        flattened=[dict(candidate_key=desc['candidate_key'],identity_scope=desc['identity_scope'],completion_event_seq=h['completion_event_seq'],request_id=h['request_id'],disclosed_ns=h['disclosed_ns'],consumer_id=h['consumer_instance_id'],clock_domain_id=h['clock_domain_id'],outcome=h['outcome'],terminal_receipt_ref=h['terminal_receipt_ref'],history_source=h['history_source'],visibility_scope=h['visibility_scope']) for h in history]
        context=dict(schema_version=1,decision_id=name,decision_seq=flattened[-1]['completion_event_seq'] if flattened else 0,identity_scope=desc['identity_scope'],causal_cutoff_ns=now,history_boundary=copy.deepcopy(boundary),completed_history=flattened,history_snapshot_sha256=gate.digest(history),qualification_sha256=gate.digest(qualification),qualification_status='qualified',candidates=[dict(c,candidate_key=gate.candidate_key(c),identity_scope=gate.scope(c)) for c in candidates])
        counts={}
        def request(method,endpoint,route,body,config):
            counts[endpoint,route]=counts.get((endpoint,route),0)+1;n=counts[endpoint,route]
            if endpoint==cfg['coordinator_url'] and route=='/status':
                r=copy.deepcopy(data['status'])
                if data.get('frontier_change') and n==2:r['ingest']['streams']['b']['last_seq']+=1
                return r
            if route=='/instances':return copy.deepcopy(registrations)
            if route=='/directory/lookup':
                places=copy.deepcopy(data['placements'])
                if data['local_native']:places.append(dict(instance_id='a',tier='l1',backend='dram',incarnation=8,size_bytes=gate.NATIVE_BYTES))
                return dict(chunks=1,results=[dict(key=copy.deepcopy(nativekey),placements=places)])
            if endpoint==cfg['ticket_endpoint']:
                inv=copy.deepcopy(data['bank'])
                if data.get('bank_change') and n==2:inv['bank_epoch']+=1
                return dict(inventory=inv)
            if route=='/page_probe':
                p=copy.deepcopy(data['page'])
                if data.get('page_change') and n==2:p['page_epoch']+=1
                return p
            if endpoint==cfg['page_control_url'] and route=='/status':return copy.deepcopy(data['runtime'])
            raise AssertionError('非metadata路径')
        monkeypatch.setattr(gate,'_request',request);directory=OUT/name;directory.mkdir();result=gate.audit(context,qualification,history,data['config'],directory)
        assert result['ec_match'] is expect,(name,result)
        if reason:assert any(reason in s for s in result['unknown']),(name,result)
        save(directory/'result.json',result)
        if name in ('eligible','C_false_negative','stale_native'):
            from bloombee.kv_cache.network.controller import Controller, write, file_sha
            from bloombee.kv_cache.network.joint_strategy import joint_context
            from bloombee.kv_cache.network.strategy_runner import score_context
            owner=object.__new__(Controller)
            owner.worker_config=dict(data['config'],run_id='cpu_gate',prefetch_strategy='lrfu')
            owner.boundary=copy.deepcopy(boundary);owner.history=copy.deepcopy(history)
            owner.metadata=dict(segment_id='test',run_id='cpu_gate',active_strategy='lrfu',source_id='fixture',
                registry_sha256=gate.digest('registry'),clock_domain_id='fixture_clock',
                freeze_sha256=gate.digest('freeze'),raw_qualification_ref='assigned')
            owner.inbox=directory/'controller';owner.inbox.mkdir()
            owner.root=directory/'controller_private';owner.root.mkdir()
            owner.decisions=[];owner.spec={};owner.config_sha=gate.digest(owner.worker_config)
            owner.ready={'ticket_endpoint':cfg['ticket_endpoint']};owner.audit=gate.audit
            def worker_marker(path):
                request_path=path.parent/'score_request.json'
                incoming=json.loads(request_path.read_text());q=copy.deepcopy(qualification)
                ctx=joint_context(owner.worker_config,q,incoming['history'],incoming['boundary'],
                    incoming['decision_id'],incoming['causal_cutoff_ns'])
                selected=score_context(ctx,'lrfu')
                files={'qualification.json':q,'context.json':ctx,'shadow.json':{},'score_result.json':dict(
                    context_sha256=gate.digest(ctx),selection_sha256=gate.digest(selected),selection=selected)}
                hashes={}
                for filename,value in files.items():hashes[filename]=write(path.parent/filename,value)
                return dict(status='scored',config_sha256=owner.config_sha,
                    score_request_sha256=file_sha(request_path),files=hashes)
            owner._wait=worker_marker
            decision=owner._score(final=False,enabled=True)
            owner._dispatch(decision)
            assert (Path(decision['directory'])/'execute.json').exists() is expect
            from bloombee.kv_cache.network.worker_protocol import Worker
            import concurrent.futures
            from types import SimpleNamespace
            execute_dir=Path(decision['directory'])
            consumer=object.__new__(Worker);consumer.emit=lambda *a,**kw:None
            warm_calls=[]
            consumer.runner=SimpleNamespace(warm_selected=lambda *a:(warm_calls.append(True) or dict(warm_attempts=0,scope='CPU execution-gate boundary; no native call')))
            consumer.pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
            job=dict(decision_id=decision['decision_id'],execute_handled=False,
                result=json.loads((execute_dir/'score_result.json').read_text()),
                context=json.loads((execute_dir/'context.json').read_text()),
                request=json.loads((execute_dir/'score_request.json').read_text()))
            try:
                consumer.execute(execute_dir,job)
                if expect:
                    job['future'].result()
                    assert json.loads((execute_dir/'execute_done.json').read_text())['status']=='accepted_async'
                    assert warm_calls==[True]
                else:assert not job['execute_handled'] and not warm_calls
            finally:consumer.pool.shutdown(wait=True)
            if name=='eligible':
                owner.audit=None
                blocked=owner._score(final=False,enabled=True);owner._dispatch(blocked)
                assert not (Path(blocked['directory'])/'execute.json').exists()
        return dict(case=name,ec_match=result['ec_match'],unknown=result['unknown'],false_positive=result['false_positive'],false_negative=result['false_negative'],duplicate_C=result['duplicate_C'])
    def no_manifest_absent(d):d['history'][0].pop('source_bundle_manifest');d['placements']=[];d['C']=False
    def bank_ready(d):
        d['C']=False;d['bank']['objects']=[dict(ticket_id='ticket',object_id=manifest['object_id'],source_manifest=copy.deepcopy(manifest),manifest_sha256=gate.digest(manifest),validated_complete_32layer_sha=True,state='ready',claimable=True,expired=False,sending=False,deferred_remove_status=None,local_payload_bytes=16384,expires_at=now/1e9+300,native_target=dict(instance_id='a'),native_registration=copy.deepcopy(registrations['instances'][0]),native_chunk_incarnations=[dict(incarnation=8,size_bytes=gate.NATIVE_BYTES)],expected_key=copy.deepcopy(nativekey))]
    def mixed(d):
        d['C']=False;d['page']['complete']=True
        for i,row in enumerate(d['page']['evidence']):
            if i%2:row.update(gpu_complete=True,slot=i,generation=2)
            else:row.update(cpu_complete=True,cpu_digest=gate.digest(i))
    def partial(d):
        mixed(d);d['C']=True;d['page']['complete']=False;d['page']['evidence'][-1]['gpu_complete']=False
    def stale(d):d['status']['registry']['seconds_since_heartbeat']['b']=100
    def missing_manifest(d):d['history'][0].pop('source_bundle_manifest')
    def claimed(d):bank_ready(d);d['bank']['objects'][0].update(state='claimed',claimable=False)
    def prepared(d):d['runtime']['bundle_registry']['leases']=[dict(prepared=True)]
    def no_epoch(d):d['page'].pop('page_epoch')
    def duplicate_h(d):d['history'].append(copy.deepcopy(d['history'][0]))
    cases=[('eligible',lambda d:None,True,None),('absent_remote_missing_manifest',no_manifest_absent,True,None),('native_local',lambda d:d.update(C=False,local_native=True),True,None),('bank_ready',bank_ready,True,None),('mixed_tiers_complete',mixed,True,None),('partial_layers_eligible',partial,True,None),('initial_demand_inflight_allowed',lambda d:d['page'].update(demand_inflight=True),True,None),('remote_manifest_missing',missing_manifest,False,'remote_present'),('stale_native',stale,False,'stale'),('native_incarnation',lambda d:d['placements'][0].update(incarnation=6),False,'incarnation'),('wrong_native_size',lambda d:d['placements'][0].update(size_bytes=8388608),False,'size'),('native_frontier_change',lambda d:d.update(frontier_change=True),False,'changed'),('bank_epoch_change',lambda d:d.update(bank_change=True),False,'bank_inventory_changed'),('page_epoch_change',lambda d:d.update(page_change=True),False,'page_mutated'),('claimed_bank',claimed,False,'nonconsumable'),('prepared_reader',prepared,False,'reader_prepared'),('missing_page_epoch',no_epoch,False,'atomic_snapshot'),('C_false_negative',lambda d:d.update(C=False),False,None),('C_false_positive',lambda d:d.update(placements=[]),False,None),('duplicate_C',lambda d:d.update(duplicate=True),False,None),('duplicate_H',duplicate_h,False,'duplicate_terminal'),('source_inc_pin_missing',lambda d:d['config'].pop('audit_source_incarnation'),False,'incarnation_unbound')]
    cases.extend([
        ('empty_full_history',lambda d:d.update(history=[],C=False),True,None),
        ('future_H',lambda d:d['history'][0].update(disclosed_ns=now+1000000),False,'future'),
        ('failed_H',lambda d:d['history'][0].update(outcome='failed'),False,'outcome'),
        ('wrong_full_manifest_layer',lambda d:d['history'][0]['source_bundle_manifest']['layout']['identities_by_block'][0][31].update(layer_start=30),False,'full_identity'),
        ('expired_ready_bank',lambda d:(bank_ready(d),d['bank']['objects'][0].update(expires_at=now/1e9-1,expired=True,claimable=False)),False,'nonconsumable'),
        ('wrong_source_generation_pin',lambda d:d['config'].update(audit_source_incarnation=6),False,'incarnation_unbound'),
    ])
    cases.append(('terminal_receipt_mismatch',lambda d:d['history'][0]['terminal_receipt_ref'].update(sha256='0'*64),False,'terminal_file_sha_mismatch'))
    results=[run(*case) for case in cases]
    assert len(results)==29
