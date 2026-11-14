"""Real serialized-group CPU retention; no model/GPU or simulated network claim."""
import argparse
import hashlib
import json
import struct
from bloombee.kv_cache import Group,Layer
from bloombee.kv_cache.network.api import Components,Options


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',choices=['lru','hotness','decay'],default='lru')
    args=parser.parse_args()
    blobs={};layers=[]
    for i in range(32):
        ident=dict(layer_start=i,layer_end=i+1,prefix='example')
        head=json.dumps(dict(identity=ident)).encode()
        blob=b'KVSLAB01'+struct.pack('!I',len(head))+head+b'example fixture'
        digest=hashlib.sha256(blob).hexdigest();blobs[digest]=blob
        layers.append(Layer(json.dumps(ident),digest,len(blob),1000.,'fixture-source'))
    group=Group(tuple(layers))
    c=Components(Options(local=args.policy),local_kwargs=dict(capacity_bytes=group.nbytes,clock=lambda:100.))
    for request in ('first','second'):
        with c.local.begin(group,request) as demand:
            fetch=demand.bind_fetcher(lambda endpoint,digest,**kwargs:blobs[digest])
            for layer in layers:fetch(layer.endpoint,layer.sha256,allowed_endpoints=['fixture-source'])
            event=demand.finish(reuse_success=True)
            print(json.dumps(dict(request=request,hit=event['hit'],source_objects=event['remote_objects'],
                                 success_increment=event['success_increment'],source='local test fixture')))

if __name__=='__main__':main()
