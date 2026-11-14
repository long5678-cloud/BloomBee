"""Real CPU component composition. No model, LMCache service, or speed claim."""
import hashlib
import http.server
import json
from pathlib import Path
import struct
import threading
import time
import uuid
import pytest
from bloombee.kv_cache import Group,Layer
from bloombee.kv_cache.network.api import Options,Components
from bloombee.kv_cache.telemetry import Event,query

@pytest.mark.parametrize('telemetry,prefetch',[(False,None),(True,None),(False,'lrfu'),(True,'transition')])
def test_component_combinations_real_cpu_io(telemetry,prefetch):
    from hivemind import DHT
    import httpx
    nodes=[];components=None
    try:
        if telemetry:
            writer=DHT(start=True,host_maddrs=['/ip4/127.0.0.1/tcp/0'],use_relay=False)
            nodes.append(writer)
            reader=DHT(start=True,initial_peers=writer.get_visible_maddrs(),host_maddrs=['/ip4/127.0.0.1/tcp/0'],use_relay=False)
            nodes.append(reader)
        release=Path(__file__).parents[2]/'src/bloombee/kv_cache/network/release_l1.py'
        # Resolve from the installed package rather than a project-relative path.
        from bloombee.kv_cache.network import release_l1
        release=Path(release_l1.__file__)
        cfg=dict(native_endpoint=dict(instance_id='cpu_target',url='tcp://127.0.0.1:1',http_url='http://127.0.0.1:1'),
                 release_helper=dict(path=str(release),sha256=hashlib.sha256(release.read_bytes()).hexdigest()))
        clock=[100.]
        components=Components(Options('lru',telemetry,prefetch),local_kwargs=dict(capacity_bytes=65536,clock=lambda:100.),
            telemetry_kwargs=dict(dht=writer,namespace=uuid.uuid4().hex,node_id='cpu_consumer',clock=lambda:clock[0],history_start=90.) if telemetry else None,
            prefetch_kwargs=dict(config=cfg) if prefetch else None)
        layers=[];blobs={}
        for i in range(32):
            ident=dict(layer_start=i,layer_end=i+1,prefix='fixture')
            h=json.dumps(dict(identity=ident)).encode();blob=b'KVSLAB01'+struct.pack('!I',len(h))+h+b'data'
            digest=hashlib.sha256(blob).hexdigest();layers.append(Layer(json.dumps(ident),digest,len(blob),1000.,'source'));blobs[digest]=blob
        group=Group(tuple(layers))
        for request in ('first','second'):
            with components.local.begin(group,request) as demand:
                fetch=demand.bind_fetcher(lambda endpoint,digest,**kw:blobs[digest])
                for layer in layers:fetch('source',layer.sha256,allowed_endpoints=['source'])
                receipt=demand.finish(reuse_success=True)
            if components.telemetry:
                components.telemetry.record(Event('a'*64,request,'reuse_success',100.,evidence='test'))
        assert receipt['hit'] and receipt['remote_objects']==0
        if components.telemetry:
            components.telemetry.publish()
            result=query(reader,namespace=components.telemetry.namespace,publishers=[components.telemetry.ref],start=90.,end=100.,clock=lambda:clock[0])
            assert result['complete'] and result['total_uses']==2
        if components.prefetch:
            runtime=components.prefetch
            ready=runtime.start()
            with httpx.Client(trust_env=False) as client:
                response=client.post(ready['ticket_endpoint']+'/status',json={})
                response.raise_for_status()
                assert response.json()['inventory']['objects']==[]
            assert runtime.bank.accounting()['payload_bytes']==0
    finally:
        if components:components.close()
        for n in reversed(nodes):n.shutdown();n.join(timeout=10);assert not n.is_alive()
