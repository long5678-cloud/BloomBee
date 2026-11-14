"""Real randomly initialized BloomBee blocks consuming public cache options.

Offline CPU model-layer examples, not pretrained language generation or an 8B
numerical certificate. No execution-directory import, mock or transport service.
"""
import hashlib
import json

import torch
from transformers import BloomConfig

from bloombee.kv_cache import (
    CacheKey, RetentionRange, create_cache, move_tensor_tree, tensor_nbytes,
)
from bloombee.models.bloom.block import WrappedBloomBlock
from examples.kv_cache_options import example_group, unpack_example


class TinyModel:
    def __init__(self, seed, layers):
        torch.set_num_threads(1);torch.manual_seed(seed)
        self.seed,self.layers=seed,layers
        self.batch=2 if seed==7 else 1
        self.hidden=16 if seed==7 else 32
        self.heads=4 if seed==7 else 2
        config=BloomConfig(hidden_size=self.hidden,n_head=self.heads,n_layer=layers,
                           hidden_dropout=0.,attention_dropout=0.)
        self.blocks=[WrappedBloomBlock(config,layer_idx=i).eval() for i in range(layers)]
        generator=torch.Generator().manual_seed(seed+1000)
        self.embedding=torch.randn(256,self.batch,self.hidden,generator=generator)
        self.context=f"tiny-bloom/seed={seed}/layers={layers}/batch={self.batch}/hidden={self.hidden}/heads={self.heads}/float32"

    def embed(self,ids):return self.embedding[list(ids)].permute(1,0,2)

    @torch.inference_mode()
    def prefix(self,ids):
        hidden=self.embed(ids);snapshots=[]
        for block in self.blocks:
            hidden,kv=block(hidden,use_cache=True)
            snapshots.append(tuple(t.detach().clone() for t in kv))
        return tuple(snapshots)

    @torch.inference_mode()
    def resume(self,ids,snapshots):
        hidden=self.embed(ids)
        for block,kv in zip(self.blocks,snapshots):hidden,_=block(hidden,layer_past=kv,use_cache=True)
        return hidden

    @torch.inference_mode()
    def full(self,ids):
        hidden=self.embed(ids)
        for block in self.blocks:hidden,_=block(hidden,use_cache=True)
        return hidden

    def key(self,ids):
        return CacheKey(self.context,hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest())


def assert_output(model,prefix,tail,output):
    full=model.full([*prefix,*tail])[:,-len(tail):]
    torch.testing.assert_close(output,full,atol=1e-5,rtol=1e-4)
    return float((output-full).abs().max())


def check_group_option(option,seed):
    model=TinyModel(seed,32);rows=[]
    for length in (2,4,6):
        ids=list(range(1,length+1));tail=[80,81]
        kv=model.prefix(ids)
        tensors=[dict(key=k,value=v) for k,v in kv]
        group,payloads=example_group(model.key(ids).digest,tensors,
                identity_extra=dict(execution=model.context,prefix_tokens=ids,prefix_length=length))
        cache=create_cache(option,capacity_bytes=group.nbytes,clock=lambda:1.)
        expected=model.resume(tail,kv)
        for attempt in range(2):
            calls=[];restored=[]
            def source(endpoint,sha,**kwargs):calls.append(sha);return payloads[sha]
            with cache.begin(group,f"{length}/{attempt}") as request:
                fetch=request.bind_fetcher(source)
                for layer in group.layers:
                    ts=unpack_example(fetch(layer.endpoint,layer.sha256,allowed_endpoints=[layer.endpoint]),layer.identity)
                    restored.append((ts['key'],ts['value']))
                output=model.resume(tail,restored)
                torch.testing.assert_close(output,expected,atol=0.,rtol=0.)
                error=assert_output(model,ids,tail,output)
                row=request.finish(reuse_success=True)
            assert row['hit']==bool(attempt and option!='remote-only')
            assert len(calls)==(0 if row['hit'] else 32)
            rows.append(dict(prefix_length=length,attempt=attempt,hit=row['hit'],source_objects=len(calls),
                             output_exact_to_cached_reference=True,full_max_abs=error))
    return dict(option=option,seed=seed,layers=32,batch=model.batch,hidden=model.hidden,rows=rows)


def check_tiered_option(option,seed):
    model=TinyModel(seed,2 if seed==7 else 3)
    ids={'P':[1,2,3,4],'Q':[11,12,13,14],'R':[21,22,23,24]}
    snapshots={label:model.prefix(tokens) for label,tokens in ids.items()}
    keys={label:model.key(tokens) for label,tokens in ids.items()};labels={v:k for k,v in keys.items()}
    def validate(key,value):
        expected=snapshots[labels[key]]
        if len(value)!=model.layers or any(not torch.equal(a,b) for x,y in zip(value,expected) for a,b in zip(x,y)):
            raise ValueError('model snapshot/schema/content mismatch')
    U=tensor_nbytes(snapshots['P']);rows=[]
    method='LRU' if option=='tiered-lru' else option
    wants={
        'P':{'LRU':[2,2,2,0],'A':[2,2,2,0],'B':[2,2,2,2],'A+B':[2,2,2,1]},
        'Q':{'LRU':[2,2,2,2],'A':[2,2,2,1],'B':[2,2,2,0],'A+B':[2,2,2,0]},
    }
    for last in ('P','Q'):
        cache=create_cache(option,fast_bytes=2*U,slow_bytes=2*U,size=tensor_nbytes,
            move=move_tensor_tree,validate=validate,context=model.context)
        for index,label in enumerate(('Q','P','R',last)):
            with cache.acquire(keys[label],lambda key:snapshots[labels[key]],priority=10 if label=='Q' else None) as lease:
                output=model.resume([90],lease.value)
                error=assert_output(model,ids[label],[90],output)
                assert lease.origin==wants[last][method][index]
                row=dict(vector=last,index=index,origin=lease.origin,full_max_abs=error)
                row['commit']=lease.finish(True)
            rows.append(row)
    return dict(option=option,seed=seed,layers=model.layers,batch=model.batch,hidden=model.hidden,rows=rows)


def to_time_major(snapshots):
    return {i:dict(key=k.permute(2,0,1),value=v.permute(1,0,2)) for i,(k,v) in enumerate(snapshots)}


def from_time_major(layers):
    return tuple((layers[i]['key'].permute(1,2,0),layers[i]['value'].permute(1,0,2)) for i in sorted(layers))


def check_token_option(method,seed,*,online=False,threshold=35):
    model=TinyModel(seed,2 if seed==7 else 3);block=2 if seed==7 else 4
    p=list(range(1,block*4+1));q=[*p[:block],*range(40,40+block*3)]
    pkv,qkv=model.prefix(p),model.prefix(q)
    args=dict(capacity_blocks=4 if method=='F' else 5,context=model.context,
              expected_layers=model.layers,block_size=block,
              slow_capacity_blocks=4 if method=='F' else 0)
    if method=='E' and online:args['e_ranges']=(RetentionRange(0,block,80,1.),)
    if method=='F':args['threshold']=threshold
    cache=create_cache(method,**args)
    cache.put_completed(p,to_time_major(pkv),now=0.)
    cache.put_completed(q,to_time_major(qkv),now=.5)
    rows=[]
    for label,tokens,expected in (('P',p,pkv),('Q',q,qkv)):
        layers,n,event=cache.get(tokens,now=1.1,target_device='cpu')
        try:
            if n:
                restored=from_time_major(layers)
                for (key,value),(oldk,oldv) in zip(restored,expected):
                    assert torch.equal(key,oldk[...,:n]) and torch.equal(value,oldv[:,:n])
                output=model.resume([*tokens[n:],90],restored)
            else:output=model.full([*tokens,90])
            error=assert_output(model,tokens,[90],output[:,-1:])
            rows.append(dict(label=label,reuse_tokens=n,source=event['source'],full_max_abs=error))
        finally:
            if n:cache.release()
    if method=='F':
        assert rows[0]['reuse_tokens']==(len(p) if threshold==35 else 0)
        assert rows[1]['reuse_tokens']==len(q)
    else:assert 0<rows[0]['reuse_tokens']<len(p) and rows[1]['reuse_tokens']==len(q)
    return dict(option=method,seed=seed,layers=model.layers,batch=model.batch,hidden=model.hidden,
                block_size=block,online=online,threshold=threshold,rows=rows)


def run_examples():
    results=[]
    for seed in (7,31):
        for option in ('lru','hotness','decay','remote-only','C','D','FIFO','FILO','MRU'):
            results.append(check_group_option(option,seed))
        for option in ('tiered-lru','A','B','A+B'):results.append(check_tiered_option(option,seed))
        for method,online,threshold in (('E',False,35),('E',True,35),('F',False,35),('F',False,60),('G',False,35),('H',False,35)):
            results.append(check_token_option(method,seed,online=online,threshold=threshold))
    return dict(kind='real random BloomBee blocks on CPU; not 8B pretrained generation',cases=results)


if __name__=='__main__':print(json.dumps(run_examples(),ensure_ascii=False,indent=2))
