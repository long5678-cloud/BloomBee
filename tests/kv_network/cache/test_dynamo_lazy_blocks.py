"""CPU offload transactions from real per-block tensors, no GPU/RPC claim."""
import pytest
import torch
from bloombee.kv_cache import create_cache


def cache(threshold=0,capacity=8):
    return create_cache("dynamo",capacity_blocks=capacity,block_size=2,
                        context="lazy-source",expected_layers=(0,1),
                        host_offload_min_priority=threshold)


def source(index):
    return {i:{name:torch.full((2,2,4),float(index+i)) for name in ("key","value")}
            for i in (0,1)}


@pytest.mark.parametrize("threshold,hints,want",[(0,None,[0,1,2]),(50,[50,0,100],[0]),
                                                (100,[100,99,100],[0]),(50,None,[])])
def test_lazy_reader_only_new_qualifying_complete_blocks(threshold,hints,want):
    store=cache(threshold); calls=[]
    def read(index):
        calls.append(index)
        return source(index)
    event=store.put_completed_blocks(list(range(7)),read,priorities=hints)
    assert calls==want and event["new_blocks"]==len(want)
    assert event["ignored_tail_tokens"]==1
    before=tuple((k,b.priority,b.returned_tick,b.acquired_tick) for k,b in store.entries.items())
    def forbidden(index):
        raise AssertionError("deduplicated block must not be read")
    store.put_completed_blocks(list(range(7)),forbidden,priorities=hints)
    assert before==tuple((k,b.priority,b.returned_tick,b.acquired_tick) for k,b in store.entries.items())


def test_lazy_chunk_cutoff_survives_new_scheduling_calls_and_can_finish():
    store=cache(50); calls=[]
    def read(index):calls.append(index);return source(index)
    store.put_completed_blocks([1,2],read,priorities=[80,0,100],request_id="r",final=False)
    second=store.put_completed_blocks([1,2,3,4],read,request_id="r",final=False)
    third=store.put_completed_blocks([1,2,3,4,5,6],read,request_id="r",final=True)
    assert calls==[0] and second["terminated_at_block"]==1 and third["terminated_at_block"]==1
    assert not store._requests


def test_matching_does_not_stamp_acquisition_only_new_capacity_does():
    store=cache()
    store.put_completed_blocks([1,2],source)
    key=next(iter(store.entries))
    before=store.entries[key]
    assert before.acquired_tick>0
    store.get([1,2])
    during=store.entries[key]
    assert during.acquired_tick==before.acquired_tick and during.returned_tick==before.returned_tick
    store.release()
    after=store.entries[key]
    assert after.acquired_tick==before.acquired_tick and after.returned_tick>before.returned_tick


def test_lazy_reader_exception_or_inconsistent_schema_is_atomic():
    store=cache(); original_tick=store._tick
    def error(index):
        if index==1:raise ValueError("source unavailable")
        return source(index)
    with pytest.raises(ValueError):store.put_completed_blocks([1,2,3,4],error)
    assert not store.entries and store._tick==original_tick
    def changed(index):
        pair=source(index)
        if index==1:pair[0]["key"]=torch.zeros(2,3,4)
        return pair
    with pytest.raises(ValueError):store.put_completed_blocks([1,2,3,4],changed)
    assert not store.entries and store._schema is None


def test_capacity_refusal_does_not_read_new_source_or_touch_active_blocks():
    store=cache(capacity=1)
    store.put_completed_blocks([1,2],source)
    store.get([1,2])
    before=dict(store.entries)
    def forbidden(index):raise AssertionError("capacity refusal must precede read")
    event=store.put_completed_blocks([3,4],forbidden)
    assert not event["admitted"] and event["copied_bytes"]==0
    assert store.entries==before and store.active
    store.release()
