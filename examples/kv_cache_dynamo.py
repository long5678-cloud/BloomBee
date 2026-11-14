"""Runnable native-rule CPU fixtures and real BloomBee block continuation.

This demonstrates a synchronous GPU/CPU rule adapter on CPU. It does not run
Dynamo's Rust engine, use CUDA, or claim model-level numerical equivalence.
"""
import json
import torch

from bloombee.kv_cache import create_cache
from examples.kv_cache_model_cpu import TinyModel,to_time_major,from_time_major,assert_output


def _slice(layers,n):return {i:{k:v[:n] for k,v in ts.items()} for i,ts in layers.items()}


def fixture_demo():
    ids=list(range(1,9));layers={i:dict(key=torch.arange(16,dtype=torch.float32).reshape(8,2)+i,
                                     value=torch.ones(8,2)*(i+1)) for i in range(2)}
    rows=[]
    for threshold,priorities,wanted in [(0,None,8),(50,[80,50,0,100],4),
                                        (80,[80,50,100,100],2),(50,None,0),
                                        (50,[50]*4,8),(80,[80]*4,8)]:
        cache=create_cache('dynamo',capacity_blocks=4,block_size=2,context='example/model/schema',
                           expected_layers=2,host_offload_min_priority=threshold)
        put=cache.put_completed(ids,layers,priorities=priorities)
        retained,n,event=cache.get(ids)
        try:
            assert n==wanted
            if n:assert all(torch.equal(retained[i][k],layers[i][k][:n]) for i in layers for k in layers[i])
            assert all(v.device.type=='cpu' for block in cache.entries.values() for ts in block.layers.values() for v in ts.values())
        finally:
            if n:cache.release()
        rows.append(dict(threshold=threshold,priorities=priorities,put=put,skipped_tokens=n,get=event))
        cache.clear();assert cache.resident_bytes()==0
    cache=create_cache('dynamo',capacity_blocks=4,block_size=2,context='example/chunks',expected_layers=2,
                       host_offload_min_priority=50)
    first=cache.put_completed(ids[:4],_slice(layers,4),priorities=[80,0,100,100],request_id='r1',final=False)
    later=cache.put_completed(ids,layers,request_id='r1',final=False)
    assert first['terminated_at_block']==later['terminated_at_block']==1 and later['new_blocks']==0
    independent=cache.put_completed(ids,layers,priorities=[100]*4,request_id='r2')
    still_stopped=cache.put_completed(ids,layers,priorities=[100]*4,request_id='r1',final=False)
    assert independent['new_blocks']==3 and still_stopped['terminated_at_block']==1
    assert cache.finish_request('r1')
    return dict(kind='CPU data/rule example, not GPU/model evidence',cases=rows,
                chunk_case=dict(first=first,later=later,independent=independent,still_stopped=still_stopped))


def model_case(seed):
    model=TinyModel(seed,2 if seed==7 else 3);block=2 if seed==7 else 4
    ids=list(range(1,block*4+1));source=to_time_major(model.prefix(ids));rows=[]
    settings=[(0,None,4*block),(50,[80,50,0,100],2*block),(80,[80,50,100,100],block),
              (50,None,0),(50,[50]*4,4*block),(80,[80]*4,4*block)]
    for threshold,priorities,wanted in settings:
        cache=create_cache('dynamo',capacity_blocks=4,block_size=block,context=model.context,
                           expected_layers=model.layers,host_offload_min_priority=threshold)
        source_refs={i:{k:v.clone() for k,v in ts.items()} for i,ts in source.items()}
        put=cache.put_completed(ids,source,priorities=priorities)
        assert all(torch.equal(source[i][k],source_refs[i][k]) for i in source for k in source[i])
        retained,n,event=cache.get(ids)
        try:
            assert n==wanted
            if n:
                assert all(torch.equal(retained[i][k],source[i][k][:n]) for i in source for k in source[i])
                output=model.resume([*ids[n:],90],from_time_major(retained))
            else:output=model.full([*ids,90])
            error=assert_output(model,ids,[90],output[:,-1:])
        finally:
            if n:cache.release()
        rows.append(dict(threshold=threshold,priorities=priorities,skipped_tokens=n,full_output_max_abs=error,
                         put=put,get=event,source_unchanged=True))
        cache.clear();assert cache.used_blocks==0
    return dict(seed=seed,layers=model.layers,batch=model.batch,hidden=model.hidden,heads=model.heads,
                block_size=block,kind='real random CPU BloomBee blocks, not pretrained LM generation',cases=rows)


if __name__=='__main__':
    print(json.dumps(dict(fixture=fixture_demo(),models=[model_case(7),model_case(31)]),ensure_ascii=False,indent=2))
