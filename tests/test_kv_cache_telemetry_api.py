"""Public imports, real local DHT transport, and explicit test evidence only."""
from collections import Counter, defaultdict
from concurrent.futures import Future
from dataclasses import asdict, replace
import hashlib
import json
import threading
import time
import uuid

import pytest
from hivemind import DHT
from bloombee.kv_cache import Group, Layer
from bloombee.kv_cache.telemetry import Event, IncompleteTelemetry, PublisherRef, Telemetry, query
from bloombee.kv_cache.telemetry_adapter import cache_group_id, record_generation, generate_with_telemetry

A, B = hashlib.sha256(b"test-group-a").hexdigest(), hashlib.sha256(b"test-group-b").hexdigest()


@pytest.fixture(scope="module")
def peers():
    nodes = []
    try:
        a = DHT(start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False, client_mode=False)
        nodes.append(a)
        for _ in range(2):
            nodes.append(DHT(start=True, initial_peers=a.get_visible_maddrs(),
                             host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False, client_mode=False))
        assert len({n.pid for n in nodes}) == len({str(n.peer_id) for n in nodes}) == 3
        yield nodes
    finally:
        for n in reversed(nodes):
            n.shutdown()
            n.join(timeout=10)
            assert not n.is_alive() and n.exitcode == 0


def setup(peers, *, now=100, retention=60, max_events=4096):
    clock = [now]
    ns = uuid.uuid4().hex
    pubs = [Telemetry(n, namespace=ns, node_id="test-node-"+str(i), clock=lambda:clock[0],
                      history_start=90, retention_s=retention, max_events=max_events) for i,n in enumerate(peers[:2])]
    def ask(**kwargs):
        return query(peers[2], namespace=ns, publishers=[p.ref for p in pubs],
                     clock=lambda:clock[0], **kwargs)
    return clock, pubs, ask


def test_two_publishers_independent_ledger_and_windows(peers):
    clock, pubs, ask = setup(peers)
    # Independent source ledger, defined before candidate ingestion.
    ledger = [(0,A,"r1","reuse_success",99), (1,A,"r2","reuse_success",100),
              (0,B,"r1","reuse_success",100), (1,A,"failed","fallback",100),
              (0,A,"unused","prefetch_unused_bytes",100),(1,A,"lookup","lookup",100)]
    for node,kv,req,kind,t in ledger:
        e = Event(kv,req,kind,t,evidence="test",numerical_passed=False)
        assert pubs[node].record(e)
        assert not pubs[node].record(e)
    for p in pubs:p.publish()
    for start,end in [(90,100),(99,100),(98,99)]:
        result = ask(start=start,end=end)
        expected = defaultdict(Counter)
        stages = Counter()
        for node,kv,req,kind,t in ledger:
            if start<t<=end:
                stages[kind]+=1
                if kind=="reuse_success":expected[kv][pubs[node].ref.node_id]+=1
        ranked = [dict(kv_id=k,total=sum(c.values()),nodes={p.ref.node_id:c.get(p.ref.node_id,0) for p in pubs}) for k,c in expected.items()]
        ranked.sort(key=lambda r:(-r['total'],r['kv_id']))
        assert result['ranked']==ranked and result['stage_counts']==dict(stages)
        assert result['complete'] and result['numerical_failed']==result['total_uses']
    one = ask(start=90,end=100,kv_id=A)
    assert len(one['ranked'])==1 and one['ranked'][0]['total']==2
    assert len(pubs[0]._events)==3  # diagnostics are distinct from successful uses.


def test_coverage_missing_expired_and_partial(peers):
    clock,pubs,ask=setup(peers)
    for p in pubs:p.record(Event(A,"r","reuse_success",100,evidence="test"));p.publish()
    clock[0]=101
    with pytest.raises(IncompleteTelemetry) as e:ask(start=90,end=101)
    assert all(i['reason']=="window_not_fully_observed_or_retained" for i in e.value.result['issues'])
    assert not ask(start=90,end=101,allow_partial=True)['complete']
    absent = PublisherRef("absent","session")
    with pytest.raises(IncompleteTelemetry):
        query(peers[2],namespace=pubs[0].namespace,publishers=[pubs[0].ref,absent],start=90,end=100,clock=lambda:101)
    partial=query(peers[2],namespace=pubs[0].namespace,publishers=[pubs[0].ref,absent],start=90,end=100,clock=lambda:101,allow_partial=True)
    assert partial['total_uses']==1 and partial['issues'][0]['publisher']['node_id']=='absent'
    with pytest.raises(ValueError,match='shorten'):pubs[0].publish(ttl_s=.1)
    expired=Telemetry(peers[0],namespace=pubs[0].namespace,node_id='expired',clock=lambda:101,history_start=90)
    expired.record(Event(A,'expire','reuse_success',100,evidence='test'))
    expired.publish(ttl_s=.3)
    time.sleep(.4)
    partial=query(peers[2],namespace=pubs[0].namespace,publishers=[pubs[1].ref,expired.ref],start=90,end=100,clock=lambda:101,allow_partial=True)
    assert not partial['complete'] and partial['total_uses']==1
    assert any(i['reason']=='missing_or_expired' for i in partial['issues'])


def test_retention_capacity_and_stale_retries(peers):
    clock,pubs,ask=setup(peers,retention=10,max_events=1)
    event=Event(A,"r","reuse_success",100,evidence="test")
    for p in pubs:p.record(event)
    with pytest.raises(OverflowError):pubs[0].record(replace(event,request_id='second'))
    with pytest.raises(ValueError,match="conflicting"):pubs[0].record(replace(event,numerical_passed=False))
    clock[0]=111
    for p in pubs:p.publish()
    result=ask(start=101,end=111)
    assert result['complete'] and result['total_uses']==0
    assert all(c['retained_since']==101 for c in result['coverage'])
    with pytest.raises(IncompleteTelemetry):ask(start=100,end=111)
    with pytest.raises(ValueError,match="stale"):pubs[0].record(event)
    clock[0]=110
    with pytest.raises(ValueError,match='backwards'):pubs[0].publish()


def test_incarnations_and_input_validation(peers):
    clock,pubs,ask=setup(peers)
    ns=pubs[0].namespace
    other=Telemetry(peers[1],namespace=ns,node_id=pubs[0].ref.node_id,clock=lambda:100,history_start=90)
    for p in (pubs[0],other):p.record(Event(A,'same-request','reuse_success',100,evidence='test'));p.publish()
    result=query(peers[2],namespace=ns,publishers=[pubs[0].ref,other.ref],start=90,end=100,clock=lambda:100)
    assert result['ranked'][0]['nodes']=={pubs[0].ref.node_id:2}
    with pytest.raises(ValueError):ask(start=0,end=float('nan'))
    with pytest.raises(ValueError):ask(start=90,end=101)
    with pytest.raises(ValueError):ask(start=100,end=100)
    with pytest.raises(ValueError):Event(None,'r','reuse_success',100)


def test_publish_order_is_serialized_record_can_proceed():
    entered,release,second_started,second_done=(threading.Event() for _ in range(4))
    class ControlledDHT:
        def __init__(self):self.calls=[]
        def store(self,key,value,**kwargs):
            self.calls.append((key,value))
            if len(self.calls)==1:
                entered.set()
                assert release.wait(5)
            f=Future();f.set_result(True);return f
    dht=ControlledDHT();now=[100]
    p=Telemetry(dht,namespace='concurrency-test',node_id='test',clock=lambda:now[0],history_start=90)
    p.record(Event(A,'a','reuse_success',100,evidence='test'))
    errors=[]
    def publish(second=False):
        try:
            if second:second_started.set()
            p.publish()
        except BaseException as e:errors.append(e)
        finally:
            if second:second_done.set()
    a=threading.Thread(target=publish);a.start();assert entered.wait(5)
    now[0]=101
    assert p.record(Event(A,'b','reuse_success',101,evidence='test'))
    b=threading.Thread(target=publish,args=(True,));b.start();assert second_started.wait(5)
    assert not second_done.wait(.05) and len(dht.calls)==1
    release.set();a.join(5);b.join(5)
    assert not a.is_alive() and not b.is_alive() and not errors
    indices=[value for key,value in dht.calls if key.endswith(':index')]
    assert [(v['count'],v['observed_until']) for v in indices]==[(1,100),(2,101)]


def test_completion_adapter_group_mapping_and_failure(peers):
    clock,pubs,ask=setup(peers)
    identities=[dict(model='test-only',prefix='anonymous',layer_start=i,layer_end=i+1) for i in range(32)]
    layers=tuple(Layer(json.dumps(row),hashlib.sha256(str(i).encode()).hexdigest(),10,1000,'test') for i,row in enumerate(identities))
    group=Group(layers);kv=cache_group_id(group)
    legacy=hashlib.sha256(json.dumps([hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest() for x in identities],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert kv==legacy and kv!=group.key
    metrics=dict(hit=True,fallback=None,imported_layers=32,skipped_tokens_per_layer=16,
        kv_copy_integrity=[dict(key_exact=True,value_exact=True) for _ in identities],
        actual_backend_steps=[dict(layer=i,computed_tokens=n) for i in range(32) for n in (4,1,1)])
    args=dict(request_id='r',kv_id=kv,metrics=metrics,layer_reuses=identities,expected_layers=32,
              input_length=20,prefix_tokens=16,max_new_tokens=3,output_ids=list(range(23)),completed_at=100,evidence='test',numerical_passed=False)
    assert record_generation(pubs[0],**args)['reuse_completed']
    assert not record_generation(pubs[0],**args)['recorded']
    bad=dict(args,request_id='bad',layer_reuses=identities[:-1])
    assert record_generation(pubs[0],**bad)['kind']=='fallback'
    cold=dict(args,request_id='cold',kv_id=None,prefix_tokens=0,layer_reuses=[])
    assert record_generation(pubs[1],**cold)['kind']=='cold_recompute'
    failed=dict(args,request_id='decode-failed',output_ids=[])
    assert record_generation(pubs[1],**failed)['kind']=='request_failed'
    for p in pubs:p.publish()
    result=ask(start=90,end=100)
    assert result['total_uses']==1 and result['numerical_failed']==1
    assert result['stage_counts']==dict(reuse_success=1,fallback=1,cold_recompute=1,request_failed=1)


def test_native_call_wrapper_without_audit_probes_and_restore(peers):
    from types import SimpleNamespace
    clock,pubs,ask=setup(peers)
    identities=[dict(prefix='test-only',layer_start=i,layer_end=i+1) for i in range(32)]
    previous=[]
    class NativeContractFixture:
        fixture=SimpleNamespace(backends=list(range(32)))
        record_reuse=staticmethod(previous.append)
        def identity_factory(self,ids):return identities
        def generate(self,ids,*,prefix_length,max_new_tokens):
            for identity in identities:self.record_reuse(identity)
            return ids+[7]*max_new_tokens,dict(hit=True,fallback=None,imported_layers=32,skipped_tokens_per_layer=prefix_length)
    g=NativeContractFixture();original=g.record_reuse
    output,metrics,status=generate_with_telemetry(g,pubs[0],list(range(20)),request_id='wrapper',prefix_length=16,max_new_tokens=3,evidence='test')
    assert len(output)==23 and status['reuse_completed'] and status['recorded']
    assert status['event'].copy_integrity is None and status['event'].backend_work is None
    assert len(previous)==32 and g.record_reuse==original
    g.generate=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('fixture decode error'))
    with pytest.raises(RuntimeError,match='decode'):
        generate_with_telemetry(g,pubs[1],list(range(20)),request_id='failed',prefix_length=16,evidence='test')
    assert g.record_reuse==original
    for p in pubs:p.publish()
    result=ask(start=90,end=100)
    assert result['total_uses']==1 and result['stage_counts']['request_failed']==1
