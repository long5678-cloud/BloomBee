"""Review regressions. Controlled model/log boundaries, real CPU pools and HTTP.

These tests do not claim new model/GPU or official LMCache transfer evidence.
"""
import copy,json,time,threading,http.server,hashlib
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
from threading import RLock
import pytest
import torch
from bloombee.kv_cache.network.ordinary import OrdinaryClient
from bloombee.kv_cache.network.completion import Adapter
from bloombee.kv_cache.network.lifecycle import LifecycleError
from bloombee.kv_cache.dynamo import DynamoCache
from bloombee.server.dynamo_pages import EnginePagePool
from bloombee.kv_cache import page_retention as ret


def archived():
    row=json.loads((Path(__file__).parent/'fixtures/paged/P_remote.json').read_text())
    ids=row['ids'][0];return row['events'],ids[:row['input_tokens']],ids[-3:]

class Model:
    def __init__(self,log,modes=('cold',),stage=None):
        self.log=log;self.modes=iter(modes);self.stage=stage
        events,_,_=archived()
        self.transformer=SimpleNamespace(h=SimpleNamespace(sequence_manager=SimpleNamespace(
            block_uids=[r['uid'] for r in events],dht=object())))
    def prepare_inputs_for_generation(self,*args,**kwargs):return {}
    def inference_session(self,**options):self.options=options;return nullcontext()
    def generate(self,ids,**kwargs):
        mode=next(self.modes)
        if mode=='failed':raise RuntimeError('original model exception')
        attempt=self.options['dynamo_cache']['attempt_id'] if 'dynamo_cache' in self.options else 'no-cache'
        if self.stage:self.stage(attempt,ids.reshape(-1).tolist())
        if mode!='observer_error':
            rows,prompt,_=archived()
            with self.log.open('a') as stream:
                for raw in rows:
                    row=dict(raw,telemetry_attempt_id=attempt,mode='cold',skipped_tokens=0,
                        computed_tokens=len(prompt),gpu_hit_tokens=0,cpu_hit_tokens=0,remote_hit_tokens=0,remote_errors=[])
                    stream.write('DYNAMO_SERVICE '+json.dumps(row)+'\n')
        return torch.cat([ids,torch.tensor([archived()[2]])],dim=1)


def test_missing_priority_preserves_real_cpu_threshold(tmp_path):
    log=tmp_path/'provider.log';log.touch();model=Model(log)
    ids=torch.tensor([list(range(33))])
    for n,hints in enumerate((None,[0,100],[50,0])):
        with OrdinaryClient(model,output_root=tmp_path/str(n),context='ctx') as client:
            # Fresh model mode per call.
            model.modes=iter(['cold']);client.generate(ids,max_new_tokens=3,priorities=hints)
        emitted=model.options['dynamo_cache']
        assert ('priorities' in emitted)==(hints is not None)
        layers={0:dict(key=torch.ones(33,1,2),value=torch.ones(33,1,2))}
        wrapped=DynamoCache(4,16,context='ctx',expected_layers=(0,),host_offload_min_priority=50)
        reference=DynamoCache(4,16,context='ctx',expected_layers=(0,),host_offload_min_priority=50)
        assert wrapped.put_completed(ids[0].tolist(),layers,priorities=emitted.get('priorities'))['new_blocks']==reference.put_completed(ids[0].tolist(),layers,priorities=hints)['new_blocks']

@pytest.mark.parametrize('modes',[('failed','cold','cold'),('failed','failed','cold'),('observer_error','cold','cold')])
def test_every_attempt_has_unique_evidence_and_success_history(tmp_path,modes):
    log=tmp_path/'provider.log';log.touch();model=Model(log,modes)
    ids=torch.tensor([archived()[1]])
    with OrdinaryClient(model,output_root=tmp_path/'client',context='ctx',
                        config=dict(cache_evidence=True,provider_log=str(log))) as client:
        for i,mode in enumerate(modes):
            if mode=='failed':
                with pytest.raises(RuntimeError,match='original model exception'):client.generate(ids,max_new_tokens=3,request_id=str(i))
            elif mode=='observer_error':
                with pytest.raises(LifecycleError):client.generate(ids,max_new_tokens=3,request_id=str(i))
            else:client.generate(ids,max_new_tokens=3,request_id=str(i))
        assert client.attempt_sequence==3
        assert len(client.history)==modes.count('cold')
        assert [e['completion_event_seq'] for e in client.history]==list(range(1,len(client.history)+1))
        assert client.adapter.publisher is None
        assert len(list((tmp_path/'client').glob('layers_*.json')))==3


@pytest.mark.parametrize('failure_mode',['observation','partial_write','flush','fsync','publish'])
def test_observation_failure_aborts_real_pending_refs_and_next_request_works(tmp_path,monkeypatch,failure_mode):
    log=tmp_path/'provider.log';log.touch();root=tmp_path/'client'
    ledger=root/'history.jsonl'
    for k,v in dict(BLOOMBEE_RETENTION_POLICY='lru',BLOOMBEE_HISTORY_TIER='both',
        BLOOMBEE_RETENTION_TERMINALS=str(ledger),BLOOMBEE_RETAINED_GPU_BLOCKS='1').items():monkeypatch.setenv(k,v)
    state=SimpleNamespace(binding='ctx',lock=RLock(),remote_cpu_sources={},remote_gpu_sources={},
        cache=DynamoCache(1,16,context='ctx',expected_layers=2),
        pool=EnginePagePool(3,16,1,2,dtype=torch.float32,device='cpu'))
    def stage(attempt,tokens):
        cfg=dict(attempt_id=attempt,token_ids=tokens)
        ret.begin_prefill(state,cfg);session=state.pool.begin();x=torch.ones(16,1,2)
        session.append(x,x);ret.stage(state,session,cfg,lambda i:{0:dict(key=x,value=x),1:dict(key=x,value=x)})
        session.release()
    if failure_mode != 'observation':
        from bloombee.kv_cache.network import ordinary
        failed = [False]
        original_open = Path.open
        original_fsync = ordinary.os.fsync
        original_replace = ordinary.os.replace
        class FaultFile:
            def __init__(self, stream): self.stream=stream
            def __enter__(self): return self
            def __exit__(self,*args): return self.stream.__exit__(*args)
            def fileno(self): return self.stream.fileno()
            def write(self,payload):
                if failure_mode=='partial_write' and not failed[0]:
                    failed[0]=True;self.stream.write(payload[:7]);return 7
                return self.stream.write(payload)
            def flush(self):
                self.stream.flush()
                if failure_mode=='flush' and not failed[0]:
                    failed[0]=True;raise OSError('injected draft flush failure')
        def opened(path,*args,**kwargs):
            stream=original_open(path,*args,**kwargs)
            return FaultFile(stream) if path.name.startswith('history_draft_') and args==('x',) else stream
        def fsync(fd):
            if failure_mode=='fsync' and not failed[0]:
                failed[0]=True;raise OSError('injected draft fsync failure')
            return original_fsync(fd)
        def replace(source,target):
            if failure_mode=='publish' and not failed[0]:
                failed[0]=True;raise OSError('injected atomic publication failure')
            return original_replace(source,target)
        monkeypatch.setattr(Path,'open',opened)
        monkeypatch.setattr(ordinary.os,'fsync',fsync)
        monkeypatch.setattr(ordinary.os,'replace',replace)
    statuses=[]
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            statuses.append(body)
            try:result=ret.terminal(state,body);code=200
            except Exception as error:result=dict(error=str(error));code=500
            raw=json.dumps(result).encode();self.send_response(code);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever);thread.start()
    try:
        model=Model(log,(('observer_error' if failure_mode=='observation' else 'cold'),'cold','cold'),stage)
        with OrdinaryClient(model,output_root=root,context='ctx',config=dict(cache_evidence=True,
            provider_log=str(log),control_endpoint='http://127.0.0.1:'+str(server.server_port))) as client:
            ids=torch.tensor([archived()[1]])
            with pytest.raises(LifecycleError) as caught:client.generate(ids,max_new_tokens=3,request_id='bad-observation')
            assert caught.value.terminal['status']=='complete_success'
            assert statuses[-1]['status']=='cancelled' and statuses[-1]['model_status']=='complete_success'
            assert not state.retention_pending and sum(p.refs for p in state.pool.pages)==0
            assert not client.history and not ledger.read_text()
            client.generate(ids,max_new_tokens=3,request_id='next')
            assert len(client.history)==1 and not state.retention_pending
            client.generate(ids,max_new_tokens=3,request_id='next-again')
            assert [h['completion_event_seq'] for h in client.history]==[1,2]
            assert [json.loads(line)['completion_event_seq'] for line in ledger.read_text().splitlines()]==[1,2]
            assert not state.retention_pending and all(v['status']=='complete_success' for v in statuses[1:])
            assert sum(p.refs for p in state.pool.pages)==0
    finally:server.shutdown();thread.join();server.server_close()


def test_publisher_failure_does_not_block_terminal_disclosure_or_real_close(tmp_path):
    import httpx
    from bloombee.kv_cache.network.joint_runtime import create_joint_runtime
    from bloombee.kv_cache.network import release_l1
    helper=Path(release_l1.__file__)
    runtime=create_joint_runtime(dict(native_endpoint=dict(instance_id='target',url='tcp://127.0.0.1:1',http_url='http://127.0.0.1:1'),
        release_helper=dict(path=str(helper),sha256=hashlib.sha256(helper.read_bytes()).hexdigest())))
    ready=runtime.start();worker_thread=runtime.http_thread;calls=[]
    class Prefetch:
        def terminal(self,**kw):calls.append('terminal')
        def disclose_complete(self,**kw):calls.append('disclose')
        def close(self):calls.append('close');runtime.close()
    model=Model(tmp_path/'unused')
    adapter=Adapter(model=model,job={},config=dict(distributed_telemetry=True,telemetry_namespace='test',consumer_node_id='consumer'),output_root=tmp_path)
    adapter.prefetch=Prefetch();adapter.observations['a']=dict(kind='reuse_success',kv_reuse_success=True)
    original=adapter.publisher.publish
    def outage(**kw):raise RuntimeError('publisher outage')
    adapter.publisher.publish=outage
    try:
        with pytest.raises(LifecycleError):adapter.finish(terminal=dict(attempt_id='a',status='complete_success'),input_ids=[],output_ids=None,history=[])
        assert calls==['terminal','disclose']
        with pytest.raises(LifecycleError):adapter.close()
        assert calls.count('close')==1 and not worker_thread.is_alive()
        with httpx.Client(trust_env=False,timeout=.2) as client:
            with pytest.raises(httpx.TransportError):client.post(ready['ticket_endpoint']+'/status',json={})
        adapter.publisher.publish=lambda **kw:dict(observed_until=time.time())
        adapter.close();adapter.close()
        assert calls.count('close')==1 and adapter._closed
    finally:
        if worker_thread.is_alive():runtime.close()


def test_terminal_error_still_publishes_and_close_retries_cleanup(tmp_path):
    model=Model(tmp_path/'unused');a=Adapter(model=model,job={},config=dict(
        distributed_telemetry=True,telemetry_namespace='test',consumer_node_id='c'),output_root=tmp_path)
    calls=[]
    def terminal(**kw):
        calls.append('terminal')
        if calls.count('terminal')==1:raise RuntimeError('terminal unavailable')
    def close():
        calls.append('close')
        if calls.count('close')==1:raise RuntimeError('partial cleanup')
    a.prefetch=SimpleNamespace(terminal=terminal,disclose_complete=lambda **kw:calls.append('disclose'),close=close)
    a.observations['a']=dict(kind='disabled')
    a.publisher.publish=lambda **kw:(calls.append('publish') or dict(observed_until=time.time()))
    with pytest.raises(LifecycleError):a.finish(terminal=dict(attempt_id='a',status='complete_success'),input_ids=[],output_ids=None,history=[])
    assert calls==['terminal','publish']
    with pytest.raises(LifecycleError):a.close()
    a.close();a.close();assert calls.count('close')==2

@pytest.mark.parametrize('value',['false',0,None])
def test_telemetry_requires_real_boolean(tmp_path,value):
    with pytest.raises(TypeError):OrdinaryClient(Model(tmp_path/'unused'),output_root=tmp_path/'out',config=dict(distributed_telemetry=value))


def test_c_only_requires_no_telemetry_namespace_and_no_publisher(tmp_path,monkeypatch):
    import bloombee.kv_cache.network.completion as module
    def forbidden(*a,**kw):raise AssertionError('disabled B constructed publisher')
    monkeypatch.setattr(module,'Telemetry',forbidden)
    adapter=Adapter(model=Model(tmp_path/'unused'),job={},config=dict(prefetch_adapter='builtin',
        restore_initial_history=True),output_root=tmp_path)
    assert adapter.publisher is None and adapter.prefetch_factory is not None
    adapter.close()


def test_original_model_error_is_not_masked_by_finish_and_close_outages(tmp_path):
    log=tmp_path/'provider.log';log.touch()
    model=Model(log,('failed',))
    client=OrdinaryClient(model,output_root=tmp_path/'client',context='ctx',config=dict(
        provider_log=str(log),distributed_telemetry=True,consumer_node_id='c',telemetry_namespace='test'))
    def outage(**kw):raise RuntimeError('DHT down')
    client.adapter.publisher.publish=outage
    with pytest.raises(RuntimeError,match='original model exception') as caught:
        with client:client.generate(torch.tensor([archived()[1]]),max_new_tokens=3)
    assert caught.value.kv_terminal['status']=='failed'
    assert any(e['stage']=='telemetry.publish' for e in caught.value.kv_lifecycle_errors)
    assert isinstance(caught.value.kv_close_error,LifecycleError)
    assert not client.history


def test_control_post_failure_cannot_skip_prefetch_terminal(tmp_path):
    log=tmp_path/'provider.log';log.touch();model=Model(log)
    with OrdinaryClient(model,output_root=tmp_path/'client',context='ctx',config=dict(cache_evidence=True,
        provider_log=str(log),control_endpoint='http://127.0.0.1:1',control_timeout_s=.1)) as client:
        calls=[]
        client.adapter.prefetch=SimpleNamespace(terminal=lambda **kw:calls.append('terminal'),
            disclose_complete=lambda **kw:calls.append('disclose'),close=lambda:calls.append('close'),
            claim_current=lambda **kw:{'status':'no_ready_fallback'})
        with pytest.raises(LifecycleError):client.generate(torch.tensor([archived()[1]]),max_new_tokens=3)
        assert calls[:2]==['terminal','disclose']


@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('purpose',['warmup','prefetch'])
@pytest.mark.parametrize('rows_mode',['missing','cold','reuse'])
def test_nonmeasurement_purposes_validate_facts_without_publishing(tmp_path,enabled,purpose,rows_mode):
    log=tmp_path/'provider.log';log.touch()
    model=Model(log)
    config=dict(provider_log=str(log),distributed_telemetry=enabled,purpose=purpose,
        telemetry_namespace='test',consumer_node_id='consumer')
    adapter=Adapter(model=model,job={},config=config,output_root=tmp_path)
    prompt=archived()[1];attempt='attempt';terminal=dict(request_id='r',attempt_id=attempt,
        status='complete_success',requested_output_tokens=3)
    adapter.before_request(attempt_id=attempt,token_ids=prompt,actual_ingress_ns=1,clock_domain=None)
    if rows_mode!='missing':
        with log.open('a') as stream:
            for raw in archived()[0]:
                row=dict(raw,telemetry_attempt_id=attempt)
                if rows_mode=='cold':row.update(mode='cold',skipped_tokens=0,computed_tokens=len(prompt),
                    gpu_hit_tokens=0,cpu_hit_tokens=0,remote_hit_tokens=0,remote_errors=[])
                stream.write('DYNAMO_SERVICE '+json.dumps(row)+'\n')
    records=[]
    if enabled:adapter.publisher.record=lambda event:records.append(event) or True
    if rows_mode=='missing':
        with pytest.raises(ValueError,match='missing transformer layers'):
            adapter.observe_terminal(terminal=terminal,input_ids=prompt,
                output_ids=torch.tensor([prompt+archived()[2]]))
        assert attempt not in adapter.observations
    else:
        receipt=adapter.observe_terminal(terminal=terminal,input_ids=prompt,
            output_ids=torch.tensor([prompt+archived()[2]]))
        assert receipt['evidence_kind']==('reuse_success' if rows_mode=='reuse' else 'cold_recompute')
        assert receipt['kv_reuse_success']==(rows_mode=='reuse')
        assert not receipt['recorded']
        assert receipt['kind']==('excluded' if enabled else 'disabled')
    assert not records

@pytest.mark.parametrize('side',['page','ticket'])
@pytest.mark.parametrize('recover_at',['next_generate','close'])
def test_terminal_outage_recovers_original_success_and_ordered_history(tmp_path,side,recover_at):
    from test_native_configuration import deployment_fixture
    from bloombee.kv_cache.network.configuration import generate_bundle
    from bloombee.kv_cache.network.controller import Controller
    meta=tmp_path/'metadata';meta.mkdir()
    cfg=deployment_fixture(meta);cfg['target_config_dir']=str(tmp_path/'bundle')
    bundle=generate_bundle(cfg,tmp_path/'bundle');config=json.loads((bundle/'client.json').read_text())
    owner=tmp_path/'controller';owner.mkdir();controller=Controller({},config,owner)
    controller._connect=lambda:None
    controller.inbox.mkdir(parents=True)
    calls=[];outage=[True]
    def call(which,attempt,status):
        calls.append((which,attempt,status))
        if which==side and outage[0]:
            outage[0]=False;raise RuntimeError('transient '+which+' terminal')
        return dict(ok=True)
    from test_background_dispatch import receipt, reservation
    def page_terminal(route, value):
        call('page', value['attempt_id'], value['status'])
        return dict(release_expected=receipt(value['attempt_id']))
    controller._post=page_terminal
    controller._release_post=lambda attempt, timeout:receipt(attempt)
    controller.ticket=SimpleNamespace(terminal=lambda attempt,status:call('ticket',attempt,status),close=lambda:None)
    controller._wait=lambda path:dict(status='done')
    controller._score=lambda **kw:dict(status='CPU scoring boundary')
    def claim(**kw):
        controller.pending_attempts[kw['attempt_id']]=dict(reserved=reservation(kw['attempt_id']))
        return dict(status='CPU claim boundary')
    controller.claim_current=claim
    log=tmp_path/'provider.log';log.touch()
    client=OrdinaryClient(Model(log,('cold','cold','cold')),output_root=tmp_path/'client',context='ctx',
        config=dict(cache_evidence=True,provider_log=str(log)))
    client.adapter.prefetch=controller
    try:
        with pytest.raises(LifecycleError):
            client.generate(torch.tensor([archived()[1]]),max_new_tokens=3,request_id='r0')
        assert len(client.history)==1 and not controller.history and len(controller.pending_attempts)==1
        original=client.history[0]['attempt_id']
        if recover_at=='next_generate':
            for i in (1,2):
                client.generate(torch.tensor([archived()[1]]),max_new_tokens=3,request_id='r'+str(i))
            assert [h['request_id'] for h in controller.history]==['r0','r1','r2']
            assert [h['completion_event_seq'] for h in controller.history]==[1,2,3]
        client.close()
        assert not controller.pending_attempts and not client.adapter._pending_completions
        assert controller.closed and client.closed
        assert len(controller.history)==len(client.history)
        assert {status for _,attempt,status in calls if attempt==original}=={'complete_success'}
        assert all(h['outcome']=='processed_success' for h in controller.history)
    finally:controller.http.close()


@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('purpose',['warmup','prefetch'])
def test_nonmeasurement_missing_rows_aborts_ordinary_commit(tmp_path,enabled,purpose):
    log=tmp_path/'provider.log';log.touch()
    client=OrdinaryClient(Model(log,('observer_error',)),output_root=tmp_path/'client',context='ctx',
        config=dict(cache_evidence=True,provider_log=str(log),purpose=purpose,
            distributed_telemetry=enabled,telemetry_namespace='test',consumer_node_id='consumer'))
    if enabled:client.adapter.publisher.publish=lambda **kw:dict(observed_until=time.time())
    with pytest.raises(LifecycleError) as caught:
        client.generate(torch.tensor([archived()[1]]),max_new_tokens=3)
    assert caught.value.terminal['status']=='complete_success'
    assert caught.value.terminal['cache_terminal_status']=='cancelled'
    assert not client.history and not client.ledger.read_text()
    assert not client.adapter.observations
    client.close()


@pytest.mark.parametrize('fault',['terminal','disclose'])
def test_persistent_handoff_failure_still_closes_actual_ticket_service(tmp_path,fault):
    import httpx
    from bloombee.kv_cache.network.joint_runtime import create_joint_runtime
    from bloombee.kv_cache.network import release_l1
    helper=Path(release_l1.__file__)
    runtime=create_joint_runtime(dict(native_endpoint=dict(instance_id='target',url='tcp://127.0.0.1:1',http_url='http://127.0.0.1:1'),
        release_helper=dict(path=str(helper),sha256=hashlib.sha256(helper.read_bytes()).hexdigest())))
    ready=runtime.start();thread=runtime.http_thread;calls=[]
    class Prefetch:
        def terminal(self,**kw):
            calls.append(('terminal',kw['terminal']['status']))
            if fault=='terminal':raise RuntimeError('persistent terminal outage')
        def disclose_complete(self,**kw):
            calls.append(('disclose',kw['terminal']['status']))
            if fault=='disclose':raise RuntimeError('persistent disclosure outage')
        def close(self):calls.append('close');runtime.close()
    adapter=Adapter(model=Model(tmp_path/'unused'),job={},config={},output_root=tmp_path)
    adapter.prefetch=Prefetch();adapter.observations['a']=dict(kind='disabled',kv_reuse_success=False)
    try:
        with pytest.raises(LifecycleError):adapter.finish(terminal=dict(attempt_id='a',status='complete_success'),
            input_ids=[],output_ids=None,history=[dict(request_id='r',attempt_id='a')])
        for i in range(2):
            with pytest.raises(LifecycleError):adapter.close()
        assert not thread.is_alive() and calls.count('close')==1
        assert not adapter._closed and adapter._prefetch_closed
        assert len(adapter._pending_completions)==1
        assert adapter._pending_completions[0]['terminal']['status']=='complete_success'
        assert adapter._pending_completions[0]['history']==(dict(request_id='r',attempt_id='a'),)
        with httpx.Client(trust_env=False,timeout=.2) as client:
            with pytest.raises(httpx.TransportError):client.post(ready['ticket_endpoint']+'/status',json={})
        receipt=json.loads((tmp_path/'telemetry_publisher_retry_2.json').read_text())
        assert receipt['pending_completions'][0]['terminal']['status']=='complete_success'
        assert receipt['errors']
        assert all(row[1]=='complete_success' for row in calls if isinstance(row,tuple))
    finally:
        if thread.is_alive():runtime.close()
