"""Runnable CPU component combinations, separate from real LMCache/model acceptance.

A uses real serialized Placement, B uses real local DHT peers, C starts the real
ready-bank lifecycle. Without --native-config, C performs no SDK transfer/model
call. With a supplied native config/context it also executes actual qualification
and validated scoring, never fabricating source publications or history.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import uuid
from bloombee.kv_cache import Group,Layer
from bloombee.kv_cache.network.api import Components,Options
from bloombee.kv_cache.telemetry import Event,query


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modules',choices=['A','B','AB','C','ABC'],required=True)
    parser.add_argument('--policy',choices=['lru','hotness','decay'],default='lru')
    parser.add_argument('--strategy',choices=['lrfu','transition'],default='lrfu')
    parser.add_argument('--native-config',type=Path)
    args=parser.parse_args();nodes=[];components=None
    try:
        b='B' in args.modules;c='C' in args.modules;a='A' in args.modules
        now=time.time()
        telemetry=None
        if b:
            from hivemind import DHT
            writer=DHT(start=True,host_maddrs=['/ip4/127.0.0.1/tcp/0'],use_relay=False);nodes.append(writer)
            reader=DHT(start=True,initial_peers=writer.get_visible_maddrs(),host_maddrs=['/ip4/127.0.0.1/tcp/0'],use_relay=False);nodes.append(reader)
            telemetry=dict(dht=writer,namespace=uuid.uuid4().hex,node_id='example-consumer',history_start=now)
        native=None
        if c:
            from bloombee.kv_cache.network import release_l1
            p=Path(release_l1.__file__)
            config=json.loads(args.native_config.read_text()) if args.native_config else dict(
                native_endpoint=dict(instance_id='example-target',url='tcp://127.0.0.1:1',http_url='http://127.0.0.1:1'))
            config.setdefault('release_helper',dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            native=dict(config=config)
        components=Components(Options(args.policy if a else None,b,args.strategy if c else None),
            local_kwargs=dict(capacity_bytes=65536),telemetry_kwargs=telemetry,prefetch_kwargs=native)
        rows=[]
        if a:
            layers=[];blobs={}
            for i in range(32):
                ident=dict(layer_start=i,layer_end=i+1,prefix='cpu-example')
                head=json.dumps(dict(identity=ident)).encode();blob=b'KVSLAB01'+struct.pack('!I',len(head))+head+b'fixture'
                digest=hashlib.sha256(blob).hexdigest();blobs[digest]=blob
                layers.append(Layer(json.dumps(ident),digest,len(blob),time.time()+60,'fixture-source'))
            group=Group(tuple(layers))
            for rid in ('first','second'):
                with components.local.begin(group,rid) as demand:
                    fetch=demand.bind_fetcher(lambda endpoint,digest,**kw:blobs[digest])
                    for layer in layers:fetch(layer.endpoint,layer.sha256,allowed_endpoints=['fixture-source'])
                    result=demand.finish(reuse_success=True)
                rows.append(dict(request=rid,hit=result['hit'],source_objects=result['remote_objects']))
                if b:components.telemetry.record(Event('a'*64,rid,'reuse_success',time.time(),evidence='test'))
        elif b:
            # Explicit caller-confirmed fixture event; no model-use claim.
            components.telemetry.record(Event('a'*64,'fixture','reuse_success',time.time(),evidence='test'))
        result=dict(scope='CPU fixtures/DHT/component lifecycle; no real-model or native-transfer claim',local=rows)
        if b:
            snapshot=components.telemetry.publish()
            result['telemetry']=query(reader,namespace=components.telemetry.namespace,publishers=[components.telemetry.ref],start=now,end=snapshot['observed_until'])
        if c:result['prefetch_lifecycle']=components.prefetch.start()
        print(json.dumps(result,indent=2))
    finally:
        if components:components.close()
        for node in reversed(nodes):node.shutdown();node.join(timeout=10)

if __name__=='__main__':main()
