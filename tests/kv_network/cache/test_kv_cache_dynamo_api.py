"""Public native-rule entry and actual BloomBee block integration."""
import inspect
from pathlib import Path

import pytest
import torch

from bloombee.kv_cache import DynamoCache,create_cache,describe_options
from examples.kv_cache_dynamo import fixture_demo,model_case
from examples.kv_cache_model_cpu import TinyModel,to_time_major


def test_native_constructor_metadata_preserves_capacity_and_default_rule():
    record=next(r for r in describe_options() if r['name']=='dynamo')
    parameters={p['name']:p for p in record['parameters']}
    assert parameters['capacity_blocks']['required']
    assert parameters['host_offload_min_priority']['default']==0
    assert record['experimental'] and '28f9765d' in record['source']
    for alias in ['dynamo','Dynamo','dynamo-native','native-dynamo']:
        cache=create_cache(alias,capacity_blocks=8,context='model/revision/schema')
        assert type(cache) is DynamoCache and cache.host_offload_min_priority==0
        assert Path(inspect.getfile(type(cache))).resolve().parent==Path(inspect.getfile(__import__('bloombee.kv_cache',fromlist=['Placement']).Placement)).resolve().parent


@pytest.mark.parametrize('kwargs',[
    dict(capacity_blocks=4,context='x',method='F'),dict(capacity_blocks=4,context='x',policy='lru'),
    dict(capacity_blocks=4,context='x',slow_capacity_blocks=4),dict(capacity_blocks=4,context='x',fast_device='cpu')])
def test_native_contract_does_not_silently_accept_other_lifecycles(kwargs):
    with pytest.raises(TypeError):create_cache('dynamo',**kwargs)


def test_public_default_cutoff_equality_and_chunk_request_independence():fixture_demo()


@pytest.mark.parametrize('seed',[7,31])
def test_native_cpu_replicas_feed_real_bloom_blocks(seed):model_case(seed)


def test_public_model_namespace_and_exceptional_lease_cleanup():
    model=TinyModel(31,2);ids=[1,2,3,4];source=to_time_major(model.prefix(ids))
    cache=create_cache('dynamo',capacity_blocks=2,block_size=2,context=model.context,expected_layers=2)
    cache.put_completed(ids,source)
    with pytest.raises(ValueError,match='context'):cache.get(ids,context='wrong-model')
    with pytest.raises(RuntimeError,match='consumer'):
        _,n,_=cache.get(ids)
        try:
            assert n==4
            with pytest.raises(RuntimeError,match='active'):cache.clear()
            raise RuntimeError('consumer failed')
        finally:cache.release()
    cache.clear();assert cache.resident_bytes()==0 and cache.used_blocks==0


def test_known_priority_gap_never_picks_later_high_blocks():
    ids=list(range(8));layers={0:dict(key=torch.arange(8.).reshape(8,1),value=torch.ones(8,1))}
    cache=create_cache('dynamo',capacity_blocks=4,block_size=2,context='known',expected_layers=1,host_offload_min_priority=50)
    event=cache.put_completed(ids,layers,priorities=[80,None,100,100])
    assert event['terminated_at_block']==1 and event['new_blocks']==1
    _,n,_=cache.get(ids)
    try:assert n==2
    finally:cache.release()
