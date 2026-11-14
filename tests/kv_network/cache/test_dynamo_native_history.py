"""Real CPU tensors/allocator/kernel; no GPU or RPC certification here."""
import asyncio
from dataclasses import replace

import pytest
import torch

from bloombee.kv_cache import create_cache
from bloombee.server.dynamo_gpu import NativeHistory, raw, storage_id
from bloombee.server.dynamo_native_context import native_execution, native_attention_storage
from bloombee.server.dynamo_service import run_dynamo_step
from bloombee.server.dynamo_gpu import register_session, protect_history_before_step
from bloombee.flexgen_utils.pytorch_backend import TorchDevice, TorchTensor
from test_dynamo_service import guard_probe, actual_pipeline_wrapper
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.compression import CompressionConfig
from bloombee.server.memory_cache_manager import KVCacheManager
from bloombee.utils.hivemind_compat import TensorDescriptor


def layers(n=8):
    return {i: {name: torch.arange(n * 8, dtype=torch.float32).reshape(n, 2, 4) + i
                for name in ("key", "value")} for i in (0, 1)}


def test_cpu_range_starts_after_gpu_without_reading_missing_earlier_blocks():
    cache = create_cache("dynamo", capacity_blocks=8, block_size=2, context="test", expected_layers=(0, 1))
    cache.put_completed(list(range(8)), layers())
    cache.entries.pop(cache._path(tuple(range(8)))[0])
    result, n, event = cache.get_range(list(range(8)), start_tokens=2, end_tokens=6)
    assert n == 4 and event["start_tokens"] == 2 and event["end_tokens"] == 6
    assert torch.equal(result[0]["key"], layers()[0]["key"][2:6])
    assert cache.active
    cache.release()
    result, n, _ = cache.get(list(range(8)))
    assert result == {} and n == 0


def test_cpu_range_never_inspects_gpu_owned_earlier_cpu_tensors():
    cache = create_cache("dynamo", capacity_blocks=8, block_size=2, context="test", expected_layers=(0, 1))
    cache.put_completed(list(range(8)), layers())
    # Invalid bytes before the GPU boundary are intentionally ignored. A whole
    # CPU-prefix get would fail integrity; range [2,6) must still work.
    cache.entries[cache._path(tuple(range(8)))[0]].layers[0]["key"].fill_(float("nan"))
    _, n, event = cache.get_range(list(range(8)), start_tokens=2, end_tokens=6)
    assert n == 4 and event["copy_bytes"] == 4 * 2 * 4 * 4 * 4
    cache.release()


def test_cpu_range_stops_at_first_hole_and_does_not_stitch_suffix():
    cache = create_cache("dynamo", capacity_blocks=8, block_size=2, context="test", expected_layers=(0, 1))
    cache.put_completed(list(range(8)), layers())
    cache.entries.pop(cache._path(tuple(range(8)))[2])
    _, n, _ = cache.get_range(list(range(8)), start_tokens=2)
    assert n == 2
    cache.release()


@pytest.mark.parametrize("start,end", [(1, 4), (10, None), (4, 2), (0, 10), (-2, None)])
def test_cpu_range_rejects_invalid_bounds(start, end):
    cache = create_cache("dynamo", capacity_blocks=8, block_size=2, context="test", expected_layers=(0, 1))
    with pytest.raises(ValueError):
        cache.get_range(list(range(8)), start_tokens=start, end_tokens=end)


def retain(pool, start=0, priority=80, handle=1):
    native = (torch.randn(16, 2, 4), torch.randn(16, 2, 4))
    inputs = torch.randn(1, 8, 8)
    output = inputs + 1
    slot = pool.retain(native, list(range(start, start + 8)), inputs, output, [priority]*4, handle)
    return slot, native, inputs


def test_active_session_is_not_hit_or_evicted_and_budget_is_hard():
    pool = NativeHistory("context", 2, 4096, max_slots=1)
    slot, native, inputs = retain(pool)
    assert slot is not None and slot.active
    assert pool.find(list(range(8)), inputs, native[0].shape, native[0].dtype, native[0].device) is None
    other, _, _ = retain(pool, start=100, handle=2)
    assert other is None and pool.slots == [slot] and not pool.evictions
    assert pool.resident_bytes <= pool.budget_bytes
    pool.release(slot, 1)
    assert pool.find(list(range(8)), inputs, native[0].shape, native[0].dtype, native[0].device) == (slot, 8)
    before = slot.returned_tick
    pool.acquire(slot, 2)
    assert slot.returned_tick == before
    with pytest.raises(RuntimeError):
        pool.release(slot, 1)
    pool.release(slot, 2)
    assert slot.returned_tick > before


def test_cpu_history_is_independent_of_native_gpu_reclaim():
    pool = NativeHistory("context", 2, 4096, max_slots=1)
    q, native, inputs = retain(pool)
    cache = create_cache("dynamo", capacity_blocks=8, block_size=2, context="test", expected_layers=(0, 1))
    saved = layers()
    cache.put_completed(list(range(8)), saved)
    pool.release(q, 1)
    r, _, _ = retain(pool, 100, handle=2)
    assert r is not None and q not in pool.slots
    assert pool.evictions[0]["reason"] == "idle_capacity"
    assert pool.evictions[0]["unit"] == "native_dense_slab"
    assert pool.find(list(range(8)), inputs, native[0].shape, native[0].dtype, native[0].device) is None
    restored, n, _ = cache.get(list(range(8)))
    assert n == 8 and torch.equal(restored[0]["key"], saved[0]["key"])
    cache.release()


def test_idle_priority_then_return_order_free_capacity_first():
    pool = NativeHistory("context", 2, 20000, max_slots=2)
    high, _, _ = retain(pool, priority=90)
    pool.release(high, 1)
    low, _, _ = retain(pool, 100, priority=10, handle=2)
    pool.release(low, 2)
    assert not pool.evictions
    third, _, _ = retain(pool, 200, handle=3)
    assert high in pool.slots and low not in pool.slots
    assert pool.evictions[0]["priority"] == 10
    pool.release(third, 3)
    # Replace high priority data only after equal-lowest metadata's earlier
    # returned allocation, not merely because it was queried.
    pool2 = NativeHistory("context", 2, 20000, max_slots=2)
    a, _, _ = retain(pool2, priority=50)
    b, _, _ = retain(pool2, 100, priority=50, handle=2)
    pool2.release(b, 2)
    pool2.release(a, 1)
    retain(pool2, 200, priority=50, handle=3)
    assert a in pool2.slots and b not in pool2.slots


def test_oversize_does_not_evict_existing_and_zero_capacity_no_history():
    pool = NativeHistory("context", 2, 2048)
    slot, _, _ = retain(pool)
    pool.release(slot, 1)
    big = (torch.zeros(1000, 2, 4), torch.zeros(1000, 2, 4))
    assert pool.retain(big, list(range(8)), torch.zeros(1,8,8), torch.zeros(1,8,8), None, 2) is None
    assert pool.slots == [slot] and not pool.evictions
    assert retain(NativeHistory("c", 2, 0))[0] is None


def test_active_slot_can_replace_its_own_history_under_one_slot_limit():
    pool = NativeHistory("context",2,1536,max_slots=1)
    slot,native,inputs = retain(pool)
    assert slot.nbytes == 1536
    updated = pool.retain(native,list(range(8)),inputs,inputs+2,[100]*4,1,existing=slot)
    assert updated is not None and pool.slots == [updated]
    assert updated.native is native and not pool.evictions
    assert pool.resident_bytes == 1536
    pool.release(updated,1)


def test_actual_input_mismatch_and_owner_mismatch_cannot_hit():
    pool = NativeHistory("context", 2, 4096)
    slot, native, inputs = retain(pool)
    pool.release(slot, 1)
    assert pool.find(list(range(8)), inputs+1, native[0].shape, native[0].dtype, native[0].device) is None
    tail_change = inputs.clone()
    tail_change[:,6:] += 1
    assert pool.find(list(range(8)),tail_change,native[0].shape,native[0].dtype,native[0].device,
                     max_tokens=6) == (slot,6)
    pool.pid = -1
    with pytest.raises(RuntimeError):
        pool.acquire(slot, 2)


def test_native_attention_bridge_requires_alias_and_resets_on_exception():
    key, value = torch.randn(16,2,4), torch.randn(16,2,4)
    with native_execution(key, value) as observation:
        k, v = native_attention_storage(key[:4], value[:4], 8)
        assert k is key and v is value and observation["observed"]
        with pytest.raises(RuntimeError):
            native_attention_storage(key[:4].clone(), value[:4], 8)
    k, v = native_attention_storage(key[:4], value[:4], 8)
    assert k.shape[0] == 4


def test_real_flexgen_attention_in_place_matches_concat_without_prefix_copy():
    """Actual existing mha_gen_llama CPU branch, no fake CUDA/RPC labels."""
    torch.manual_seed(44)
    device = TorchDevice("cpu")
    wrap = lambda tensor: TorchTensor.create_from_torch(tensor, device)
    hidden, heads, dim, prefix, suffix = 8, 2, 4, 4, 2
    inputs = torch.randn(1, suffix, hidden)
    weights = [wrap(torch.randn(hidden,hidden)) for _ in range(4)]
    norm = wrap(torch.ones(hidden))
    inv = wrap(1.0 / (10000 ** (torch.arange(0,dim,2).float()/dim)))
    mask = torch.ones(1,suffix,prefix+suffix,dtype=torch.bool)
    mask[0,0,-1] = False
    key, value = torch.randn(12,heads,dim), torch.randn(12,heads,dim)
    key0, value0 = key.clone(), value.clone()
    def run(k,v):
        return device.mha_gen_llama(wrap(inputs.clone()), wrap(mask.clone()), *weights,
                heads, wrap(k), wrap(v), [False,False], 1.0, False, None, norm, inv, None)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as reference_profile:
        expected = run(key[:prefix].clone(), value[:prefix].clone())
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = run(key, value)
    assert torch.allclose(actual[0].data, expected[0].data, atol=1e-5, rtol=1e-5)
    assert torch.equal(key[:prefix], key0[:prefix]) and torch.equal(value[:prefix], value0[:prefix])
    assert not torch.equal(key[prefix:prefix+suffix], key0[prefix:prefix+suffix])
    # RoPE itself uses two small real/imag cats. Only the two history K/V
    # concatenations must disappear; normal RoPE remains unchanged.
    cats = lambda p: sum(e.count for e in p.key_averages() if e.key == "aten::cat")
    assert cats(reference_profile) - cats(profile) == 2
    # Latest upstream RoPE/metadata operations add small cats; only the two
    # full historical K/V concatenations are the contract under test.
    assert cats(profile) >= 2


def test_real_memory_manager_session_free_transfers_same_native_storage():
    """Actual allocation pipe/free notification and manager select_cache view."""
    backend, info, _ = guard_probe()
    cpu = TorchDevice("cpu")
    env = ExecutionEnv(gpu=cpu,cpu=cpu,disk=None,mixed=None)
    compression = CompressionConfig(num_bits=4,group_size=64,group_dim=0,symmetric=False)
    policy = Policy(gpu_batch_size=1,num_gpu_batches=1,w_gpu_percent=100,w_cpu_percent=0,
        cache_gpu_percent=100,cache_cpu_percent=0,act_gpu_percent=100,act_cpu_percent=0,
        overlap=False,sep_layer=True,pin_weight=False,cpu_cache_compute=False,attn_sparsity=1.,
        compress_weight=False,comp_weight_config=compression,compress_cache=False,comp_cache_config=compression)
    manager = KVCacheManager(64,30,policy,env,backend.config)
    descriptor = TensorDescriptor((1,2,4,16),dtype=torch.float16,device=torch.device("cpu"))
    size = manager.get_allocation_size_tokens(descriptor)
    allocate = lambda: tuple(asyncio.run(manager.cache._schedule_alloc(size,descriptor,timeout=0)))
    first = allocate()
    pool = NativeHistory("ctx",2,4096)
    try:
        with manager.use_cache(*first) as tensors:
            key,value = map(raw,tensors[0])
            key.fill_(3); value.fill_(4)
            slot = pool.retain(tensors[0],list(range(8)),torch.zeros(1,8,8),torch.zeros(1,8,8),None,first[0])
            pointer = storage_id(key)
        manager.cache._dynamo_session_release[first[0]] = lambda: pool.release(slot,first[0])
        assert slot.active
        manager.cache.force_free(*first)
        # Free is a real queued runtime control message, not immediate release.
        assert slot.active
        second = allocate()
        with manager.use_cache(*second):
            pass
        assert not slot.active
        pool.acquire(slot,second[0])
        manager.cache._allocated_tensors[second[0]] = slot.native
        manager.cache._dynamo_session_release[second[0]] = lambda: pool.release(slot,second[0])
        with manager.use_cache(*second) as tensors:
            assert storage_id(raw(tensors[0][0])) == pointer
            k,v,_ = manager.select_cache(8,torch.tensor([0]),requested_rows=1)
            assert k.untyped_storage().data_ptr() == pointer["pointer"]
            assert torch.all(k == 3) and torch.all(v == 4)
        manager.cache.force_free(*second)
        with manager.cache.use_cache():
            pass
        assert not slot.active and manager.cache.current_size_tokens == 0
    finally:
        manager._reorder_executor.shutdown(wait=True)


@pytest.mark.parametrize("change", ["reset", "positions", "prune", "adapter", "rows"])
def test_unsupported_or_opt_out_mutation_invalidates_history_before_write(change):
    from types import SimpleNamespace
    pool = NativeHistory("context",2,4096)
    slot, _, _ = retain(pool)
    memory = SimpleNamespace(_dynamo_active_history={},_dynamo_session_release={})
    register_session(memory,pool,slot,1)
    _, info, _ = guard_probe()
    info = replace(info,cache_handles=(1,),prefix_length=8,dynamo_cache=None)
    protect_history_before_step(memory,info,torch.tensor([0]))
    assert slot in pool.slots  # ordinary decode appends without invalidating
    kwargs = {"reset":{"prefix_length":0}, "positions":{"kv_cache_position_ids":torch.tensor([0])},
              "prune":{"need_pruning":True}, "adapter":{"active_adapter":"other"},
              "rows":{"full_batch_size":2}}[change]
    protect_history_before_step(memory,replace(info,**kwargs),torch.tensor([0]))
    assert not pool.slots and not memory._dynamo_active_history and not memory._dynamo_session_release


def test_service_orchestration_real_allocator_gpu_cpu_mixed_and_output_contract(monkeypatch):
    """Control-flow unit with explicit tensor arithmetic (NOT model/RPC proof).

    Allocator, lifecycle pipe, pool, service function, input guards and copies
    are actual components. The supplied arithmetic callback deliberately is
    NOT a model or transport; the independent real kernel test is above.
    """
    backend, base, _ = guard_probe()
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CONTEXT",base.dynamo_cache["context"])
    monkeypatch.setenv("BLOOMBEE_DYNAMO_BLOCK_SIZE","2")
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CAPACITY_BLOCKS","32")
    monkeypatch.setenv("BLOOMBEE_DYNAMO_HOST_MIN_PRIORITY","50")
    monkeypatch.setenv("BLOOMBEE_DYNAMO_GPU_BUDGET_BYTES","16384")
    monkeypatch.setenv("BLOOMBEE_DYNAMO_GPU_MAX_SLOTS","1")
    backend.module = actual_pipeline_wrapper(torch.nn.Linear(8,8).eval())
    backend.name,backend.dtype = "control-only.0",torch.float32
    backend._ensure_model_on_device = lambda: None
    cpu = TorchDevice("cpu")
    env = ExecutionEnv(gpu=cpu,cpu=cpu,disk=None,mixed=None)
    compression = CompressionConfig(num_bits=4,group_size=64,group_dim=0,symmetric=False)
    policy = Policy(gpu_batch_size=1,num_gpu_batches=1,w_gpu_percent=100,w_cpu_percent=0,
        cache_gpu_percent=100,cache_cpu_percent=0,act_gpu_percent=100,act_cpu_percent=0,
        overlap=False,sep_layer=True,pin_weight=False,cpu_cache_compute=False,attn_sparsity=1.,
        compress_weight=False,comp_weight_config=compression,compress_cache=False,comp_cache_config=compression)
    manager = KVCacheManager(128,30,policy,env,backend.config)
    backend.cache_manager = manager
    descriptor = TensorDescriptor((1,2,4,16),dtype=torch.float32,device=torch.device("cpu"))
    size = manager.get_allocation_size_tokens(descriptor)
    pending = []
    def arithmetic_step(x,hypo,info):
        with manager.use_cache(*info.cache_handles) as native:
            key,value = map(raw,native[0])
            start = info.prefix_length
            key[start:start+x.shape[1]].copy_(x.transpose(0,1).reshape(-1,2,4))
            value[start:start+x.shape[1]].copy_((x+2).transpose(0,1).reshape(-1,2,4))
        return x+1,torch.arange(x.shape[1]).unsqueeze(0)
    def request(ids, priorities=None, finish=True):
        handles = tuple(asyncio.run(manager.cache._schedule_alloc(size,descriptor,timeout=0)))
        pending.append(handles)
        cfg = {"context":base.dynamo_cache["context"],"token_ids":ids,
               "priorities":priorities if priorities is not None else [100]*4}
        info = replace(base,cache_handles=handles,dynamo_cache=cfg)
        inputs = torch.tensor(ids,dtype=torch.float32).view(1,-1,1).expand(1,-1,8).clone()
        output,keep = run_dynamo_step(backend,inputs,torch.tensor([0]),info,arithmetic_step)
        assert torch.equal(output,inputs+1) and torch.equal(keep,torch.arange(len(ids)).unsqueeze(0))
        event = backend._dynamo_service_events[-1]
        if finish:
            manager.cache.force_free(*handles)
            pending.remove(handles)
        return event
    try:
        q = list(range(8)); p = q[:6]+[90,91]; r = list(range(100,108))
        source = request(q)
        hit = request(p)
        assert hit["gpu_hit_tokens"] == 6 and hit["cpu_hit_tokens"] == 0
        assert hit["cpu_read_payload_bytes"] == 0 and hit["gpu_prefix_copy_bytes"] == 0
        assert source["destination_key_storage"] == hit["source_key_storage"] == hit["destination_key_storage"]
        # Repeated same native storage re-publication under MAX_SLOTS=1.
        for ids in (q,p,q,p):
            again = request(ids)
            assert again["gpu_hit_tokens"] == 6 and again["cpu_hit_tokens"] == 0
            assert again["source_key_storage"] == again["destination_key_storage"]
        request(r)
        cpu_hit = request(p)
        assert cpu_hit["gpu_hit_tokens"] == 0 and cpu_hit["cpu_hit_tokens"] == 6
        request(r)
        state = backend._dynamo_service_state
        first = state.cache._path(tuple(q))[0]
        state.cache.entries.pop(first)
        request(q[:2],priorities=[0])
        assert first not in state.cache.entries
        mixed = request(p)
        assert mixed["gpu_hit_tokens"] == 2 and mixed["cpu_hit_tokens"] == 4
        assert mixed["cpu_read_range"] == [2,6]
        assert mixed["computed_tokens"] == 2 and mixed["returned_tokens"] == 8
        # A callback is not an attention certificate: the event must say so.
        assert not mixed["native_attention_in_place"]
    finally:
        for handles in pending:
            manager.cache.force_free(*handles)
        with manager.cache.use_cache():
            pass
        manager._reorder_executor.shutdown(wait=True)
