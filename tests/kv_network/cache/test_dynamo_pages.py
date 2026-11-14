"""Actual CPU tensor pages and ordinary Llama projection/RoPE/attention code.

No CUDA/RPC claim. The production mha_gen_llama and OptimizedLlamaAttention
are exercised directly, never a replacement test model. CUDA stream coverage
is separately delegated to the real GPU owner.
"""
import asyncio

import pytest
import torch

from bloombee.server.dynamo_pages import EnginePagePool, PageCapacityError, paged_attention_value
from bloombee.flexgen_utils.pytorch_backend import TorchDevice, TorchTensor


def pool(capacity=4, block=2, heads=4, dim=4, dtype=torch.float32, **kwargs):
    return EnginePagePool(capacity, block, heads, dim, dtype=dtype, device="cpu", **kwargs)


def seed(store, keys, key, value, priorities=None):
    session = store.begin()
    session.append(key, value)
    session.publish(keys, priorities)
    slots = [slot for slot, _ in session.slots]
    session.release()
    return slots


def test_pressure_reclaims_only_low_q_in_same_allocation_and_p_still_hits():
    store = pool(capacity=2)
    pk, pv = torch.randn(2, 4, 4), torch.randn(2, 4, 4)
    p_slot = seed(store, ["P"], pk, pv, [80])[0]
    q_slot = seed(store, ["Q"], torch.randn_like(pk), torch.randn_like(pv), [10])[0]
    storage_ptr = store.k.untyped_storage().data_ptr()
    r = store.begin()
    r.append(torch.randn_like(pk), torch.randn_like(pv))
    assert r.slots[0][0] == q_slot
    assert store.k.untyped_storage().data_ptr() == storage_ptr
    assert store.matching_tokens(["Q"]) == 0
    assert store.matching_tokens(["P"]) == 2
    assert torch.equal(store.k[p_slot], pk) and torch.equal(store.v[p_slot], pv)
    r.publish(["R"], [50])
    r.release()
    with store.begin(["P"]) as hit:
        assert hit.length == 2 and hit.slots[0][0] == p_slot
        assert next(hit.operands()[0].segments()).data_ptr() == store.k[p_slot].data_ptr()


def test_claim_protects_p_even_if_lower_priority_and_capacity_failure_is_preflight():
    store = pool(capacity=2)
    data = torch.randn(2, 4, 4)
    seed(store, ["P"], data, data, [0])
    seed(store, ["Q"], data + 1, data + 2, [90])
    with store.begin(["P"]) as pinned:
        r = store.begin()
        r.append(data + 3, data + 4)
        r.publish(["R"], [1])
        assert store.matching_tokens(["P"]) == 2 and store.matching_tokens(["Q"]) == 0
        before = store.k.clone()
        failed = store.begin()
        with pytest.raises(PageCapacityError):
            failed.append(data, data)
        assert torch.equal(store.k, before)
        failed.release()
        r.release()


def test_shared_claim_counts_and_only_actual_victim_detaches_not_descendants():
    store = pool(capacity=2)
    data = torch.randn(4, 4, 4)
    seed(store, ["prefix", "prefix+tail"], data, data, [0, 90])
    with store.begin(["prefix", "prefix+tail"]) as first:
        with store.begin(["prefix", "prefix+tail"]) as second:
            assert [p.refs for p in store.pages] == [2, 2]
        assert [p.refs for p in store.pages] == [1, 1]
    r = store.begin()
    r.append(data[:2] + 7, data[:2] + 8)
    assert "prefix" not in store.lookup
    assert "prefix+tail" in store.lookup  # not leaf-only cascade eviction
    r.release()
    # Missing prefix stops normal lookup; restoring it later can reconnect the
    # still-valid downstream physical block instead of having erased it.
    assert store.matching_tokens(["prefix", "prefix+tail"]) == 0
    seed(store, ["prefix"], data[:2], data[:2], [80])
    assert store.matching_tokens(["prefix", "prefix+tail"]) == 4


def test_empty_slots_before_any_cached_victim_and_same_priority_return_order():
    store = pool(capacity=3)
    data = torch.randn(2, 4, 4)
    seed(store, ["P"], data, data, [0])
    seed(store, ["Q"], data, data, [0])
    seed(store, ["R"], data, data, [0])
    assert len([e for e in store.events if e['event'] == 'reclaim']) == 0
    with store.begin(["P"]):
        pass  # P has a new return tick; Q is oldest now.
    seed(store, ["S"], data, data, [0])
    assert "Q" not in store.lookup and "P" in store.lookup


def test_partial_tail_never_publishes_or_leaks_after_release_and_generation_checks():
    store = pool(capacity=2)
    session = store.begin()
    session.append(torch.randn(3, 4, 4), torch.randn(3, 4, 4))
    session.publish(["full"])
    operand = session.operands()[0]
    session.release()
    assert store.matching_tokens(["full"]) == 2
    with pytest.raises(RuntimeError, match="closed"):
        list(operand.segments())
    assert len(store.lookup) == 1
    another = store.begin()
    another.append(torch.randn(2, 4, 4), torch.randn(2, 4, 4))
    assert store.matching_tokens(["full"]) == 2  # used the empty partial-tail slot
    another.release()


def test_diagnostic_events_are_bounded_and_consumable_on_hot_hits():
    store = pool(capacity=2, event_limit=5)
    data = torch.randn(2, 4, 4)
    for _ in range(30):
        s = store.begin()
        s.append(data, data)
        s.release()
    assert len(store.events) == 5
    assert len(store.drain_events()) == 5
    assert len(store.events) == 0


def physical_history(k, v):
    """Make real logical history live at noncontiguous physical slots 0,2,..."""
    count, heads, dim = k.shape
    assert count % 2 == 0
    store = pool(capacity=count // 2 + 2, heads=heads, dim=dim, dtype=k.dtype)
    session = store.begin()
    session.append(k[:2], v[:2])
    session.publish([("history", 0)], [80])
    session.release()
    seed(store, [("other", 0)], torch.randn_like(k[:2]), torch.randn_like(v[:2]), [10])
    prefix_keys = [("history", i) for i in range(count // 2)]
    session = store.begin(prefix_keys)
    session.append(k[2:], v[2:])
    session.publish(prefix_keys, [80] * len(prefix_keys))
    assert [slot for slot, _ in session.slots] != list(range(count // 2))
    return store, session


@pytest.mark.parametrize("heads,kv_heads,tgt", [(4, 4, 1), (8, 2, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_real_ordinary_llama_projection_rope_paged_attention_matches_full_prefill(heads, kv_heads, tgt, dtype):
    torch.manual_seed(34)
    d = 8
    hidden = heads * d
    history = 4
    device = TorchDevice("cpu")
    tt = lambda tensor: TorchTensor.create_from_torch(tensor, device)
    weights = [torch.randn(rows, hidden, dtype=dtype) * .03
               for rows in [hidden, kv_heads*d, kv_heads*d, hidden]]
    ln = torch.ones(hidden, dtype=dtype)
    inv = 1 / (10000 ** (torch.arange(0, d, 2).float() / d))
    x = torch.randn(1, history+tgt, hidden, dtype=dtype)
    def mask(total, queries=None):
        queries = total if queries is None else queries
        offset = total - queries
        allow = torch.arange(total)[None, :] <= (torch.arange(queries)[:, None]+offset)
        # Ordinary prefill mha_llama expects ADDITIVE scores; bool input there
        # would add 0/1 instead of masking future queries (single-token hides it).
        return torch.zeros(1, 1, queries, total).masked_fill(~allow[None, None], -65504.0)
    def prefill(values):
        return device.mha_llama(tt(values.clone()), tt(mask(values.shape[1])),
                               *(tt(w) for w in weights), heads, [False]*16,
                               False, None, tt(ln), tt(inv),
                               torch.arange(values.shape[1])[None, :])
    full, _, _ = prefill(x)
    _, history_k, history_v = prefill(x[:, :history])
    store, session = physical_history(history_k.data, history_v.data)
    prefix_ptrs = [part.data_ptr() for part in session.operands()[0].segments()]
    kd, vd = session.operands()
    out, new_k, new_v = device.mha_gen_llama(
        tt(x[:, history:].clone()), tt(mask(history+tgt, tgt)),
        *(tt(w) for w in weights), heads, kd, vd, [False]*16, 1., False, None,
        tt(ln), tt(inv), torch.arange(history, history+tgt)[None, :])
    tol = 2e-5 if dtype == torch.float32 else 2e-3
    torch.testing.assert_close(out.data, full.data[:, history:], atol=tol, rtol=tol)
    assert new_k.data.shape == (tgt, heads, d)  # GQA expanded to attention heads
    assert new_v.data.shape == new_k.data.shape
    event = next(e for e in reversed(store.events) if e['event'] == 'attention_direct_pages')
    assert event['key_ptrs'] == prefix_ptrs
    assert event['history_kv_gather_bytes'] == 0
    session.append(new_k.data, new_v.data)
    assert session.length == history+tgt
    session.release()


def test_attention_uses_global_softmax_not_independent_page_softmax():
    store = pool(capacity=3, heads=2, dim=2)
    key = torch.tensor([[[2., 0.], [1., 0.]], [[1., 0.], [0., 1.]],
                        [[-3., 0.], [1., 1.]], [[-4., 0.], [0., -1.]]])
    value = torch.arange(16).float().reshape(4, 2, 2)
    _, session = physical_history(key, value)
    q = torch.ones(2, 1, 2)
    new_k, new_v = key[:1] * .1, value[:1] * .1
    result = paged_attention_value(q, *session.operands(), new_k, new_v,
                                   torch.ones(1, 1, 5, dtype=torch.bool), batch=1, heads=2)
    dense_k = torch.cat([key, new_k])
    dense_v = torch.cat([value, new_v])
    scores = torch.bmm(q, dense_k.permute(1, 2, 0)) / (2**.5)
    expected = torch.bmm(torch.softmax(scores, -1), dense_v.permute(1, 0, 2)).view(1, 2, 1, 2)
    torch.testing.assert_close(result, expected)
    session.release()


def test_manager_actual_allocation_binding_reads_pages_and_appends_without_dense_rewrite():
    from transformers import LlamaConfig
    from bloombee.server.memory_cache_manager import KVCacheManager
    from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
    from bloombee.flexgen_utils.policy import Policy
    from bloombee.flexgen_utils.compression import CompressionConfig
    from bloombee.utils.hivemind_compat import TensorDescriptor
    config = LlamaConfig(hidden_size=16, num_attention_heads=2, num_key_value_heads=1)
    cpu = TorchDevice("cpu")
    env = ExecutionEnv(gpu=cpu, cpu=cpu, disk=None, mixed=None)
    compression = CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False)
    policy = Policy(gpu_batch_size=1, num_gpu_batches=1, w_gpu_percent=100, w_cpu_percent=0,
                    cache_gpu_percent=100, cache_cpu_percent=0, act_gpu_percent=100,
                    act_cpu_percent=0, overlap=False, sep_layer=True, pin_weight=False,
                    cpu_cache_compute=False, attn_sparsity=1., compress_weight=False,
                    comp_weight_config=compression, compress_cache=False, comp_cache_config=compression)
    manager = KVCacheManager(64, 30, policy, env, config)
    descriptor = TensorDescriptor((1, 2, 8, 16), dtype=torch.float32, device=torch.device("cpu"))
    handles = asyncio.run(manager.cache._schedule_alloc(manager.get_allocation_size_tokens(descriptor), descriptor, timeout=0))
    store = pool(capacity=4, heads=2, dim=8)
    data = torch.randn(4, 2, 8)
    session = store.begin()
    session.append(data, data+1)
    with manager.bind_dynamo_pages(handles, session):
        with manager.use_cache(*handles) as allocated:
            dense_k = allocated[0][0].data.clone()
            k, v, reorder = manager.select_cache(4, hypo_ids=torch.tensor([0]), requested_rows=1)
            assert k.session is session and k.shape == (1, 2, 4, 8)
            new_k, new_v = torch.randn(2, 8, 1), torch.randn(2, 1, 8)
            manager.update_cache((new_k, new_v), 4)
            assert session.length == 5
            assert torch.equal(allocated[0][0].data, dense_k)
            with pytest.raises(ValueError, match="prefix"):
                manager.select_cache(4, requested_rows=1)
    assert manager.dynamo_page_session() is None
    session.release()
    manager.cache.force_free(*handles)


def test_real_production_optimized_llama_attention_module_consumes_pages():
    """Production attention class, not a test-specific model or patched forward."""
    from transformers import LlamaConfig
    from bloombee.models.llama.block import OptimizedLlamaAttention
    from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
    from bloombee.flexgen_utils.policy import Policy
    from bloombee.flexgen_utils.compression import CompressionConfig
    from bloombee.flexgen_utils.utils import ValueHolder
    torch.manual_seed(49)
    heads, kv_heads, d, history, suffix = 4, 2, 8, 4, 3
    hidden = heads*d
    cpu = TorchDevice("cpu")
    env = ExecutionEnv(gpu=cpu, cpu=cpu, disk=None, mixed=None)
    compression = CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False)
    policy = Policy(gpu_batch_size=1, num_gpu_batches=1, w_gpu_percent=100, w_cpu_percent=0,
                    cache_gpu_percent=100, cache_cpu_percent=0, act_gpu_percent=100,
                    act_cpu_percent=0, overlap=False, sep_layer=True, pin_weight=False,
                    cpu_cache_compute=False, attn_sparsity=1., compress_weight=False,
                    comp_weight_config=compression, compress_cache=False, comp_cache_config=compression)
    config = LlamaConfig(hidden_size=hidden, num_attention_heads=heads, num_key_value_heads=kv_heads)
    module = OptimizedLlamaAttention(config=config, env=env, policy=policy, layer_id=0).eval()
    tt = lambda data: TorchTensor.create_from_torch(data, cpu)
    weights = [torch.randn(rows, hidden)*.02 for rows in [hidden, kv_heads*d, kv_heads*d, hidden]]
    ln = torch.ones(hidden)
    inv = 1 / (10000**(torch.arange(0, d, 2).float()/d))
    x = torch.randn(1, history+suffix, hidden)
    def causal(total, rows):
        allow = torch.arange(total)[None, :] <= torch.arange(total-rows, total)[:, None]
        return torch.zeros(1, 1, rows, total).masked_fill(~allow[None, None], -65504.)
    def prefill(values):
        n = values.shape[1]
        return cpu.mha_llama(tt(values.clone()), tt(causal(n, n)), *(tt(w) for w in weights),
                             heads, [False]*16, False, None, tt(ln), tt(inv), torch.arange(n)[None])
    full, _, _ = prefill(x)
    _, hk, hv = prefill(x[:, :history])
    store, session = physical_history(hk.data, hv.data)
    hidden_holder, read_holder, weight_holder, write_holder, mask_holder = [ValueHolder() for _ in range(5)]
    hidden_holder.store(tt(x[:, history:].clone()))
    read_holder.store(session.operands())
    weight_holder.store(tuple((tt(t), False) for t in weights+[ln, inv]))
    mask_holder.store(tt(causal(history+suffix, suffix)))
    positions = torch.arange(history, history+suffix)[None]
    output, _, _ = module(hidden_holder, read_holder, weight_holder, write_holder,
                          attention_mask=mask_holder, position_ids=positions,
                          rotary_position_ids=positions)
    torch.testing.assert_close(output.data, full.data[:, history:], atol=2e-5, rtol=2e-5)
    assert any(e['event']=='attention_direct_pages' for e in store.events)
    session.release()


def test_invalid_append_publish_and_rename_do_not_corrupt_other_pages():
    store = pool(capacity=2)
    data = torch.randn(2, 4, 4)
    seed(store, ["original"], data, data)
    with store.begin(["original"]) as session:
        before = store.k.clone()
        with pytest.raises(ValueError, match="schema"):
            session.append(torch.ones(1, 3, 4), torch.ones(1, 3, 4))
        with pytest.raises(ValueError, match="rename"):
            session.publish(["wrong"])
        with pytest.raises(ValueError, match="0..100"):
            session.publish(["original"], [101])
        assert session.length == 2 and store.matching_tokens(["original"]) == 2
        assert torch.equal(store.k.view(torch.uint8), before.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fp32_page_arithmetic_matches_sdpa_on_identical_qkv(dtype):
    import torch.nn.functional as F
    torch.manual_seed(31)
    heads, dim, n, queries = 4, 16, 4, 3
    key, value = torch.randn(n, heads, dim, dtype=dtype), torch.randn(n, heads, dim, dtype=dtype)
    _, session = physical_history(key, value)
    q = torch.randn(heads, queries, dim, dtype=dtype)
    new_k, new_v = torch.randn(queries, heads, dim, dtype=dtype), torch.randn(queries, heads, dim, dtype=dtype)
    mask = (torch.arange(n+queries)[None] <= torch.arange(n, n+queries)[:, None])[None, None]
    actual = paged_attention_value(q, *session.operands(), new_k, new_v, mask, batch=1, heads=heads)
    # Dense concat is only the independent reference, never the candidate path.
    reference = F.scaled_dot_product_attention(q[None], torch.cat([key, new_k]).permute(1, 0, 2)[None],
                                              torch.cat([value, new_v]).permute(1, 0, 2)[None], attn_mask=mask)
    torch.testing.assert_close(actual, reference, atol=1e-3 if dtype==torch.float16 else 2e-6,
                               rtol=1e-3 if dtype==torch.float16 else 2e-6)
    session.release()


def test_logical_block_interface_reports_live_slot_generation_and_partial_tail():
    store = pool(capacity=3)
    data = torch.randn(3, 4, 4)
    session = store.begin()
    session.append(data, data+1)
    first, tail = session.block(0), session.block(1)
    assert first['length'] == 2 and tail['length'] == 1
    assert first['generation'] == store.pages[first['slot']].generation
    assert first['key'].data_ptr() == store.k[first['slot']].data_ptr()
    assert torch.equal(tail['key'], data[2:])
    with pytest.raises(IndexError):
        session.block(2)
    session.release()
    with pytest.raises(RuntimeError, match='closed'):
        session.block(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="13G must run on its admitted GPU")
def test_cuda_producer_reader_reclaimer_streams_preserve_page_read():
    # Optional real GPU check, never claimed as covered by CPU skips. Producer,
    # attention reader and new owner use different streams on the SAME device.
    device = torch.device('cuda', torch.cuda.current_device())
    store = EnginePagePool(1, 8, 2, 16, dtype=torch.float32, device=device)
    producer, reader = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    with torch.cuda.stream(producer):
        first = store.begin()
        first.append(torch.full((8, 2, 16), 2., device=device),
                     torch.full((8, 2, 16), 4., device=device))
        first.publish(['first'])
        first.release()
    with torch.cuda.stream(reader):
        lease = store.begin(['first'])
        actual = paged_attention_value(torch.ones(2, 1, 16, device=device), *lease.operands(),
                                       torch.ones(1, 2, 16, device=device),
                                       torch.full((1, 2, 16), 4., device=device),
                                       torch.ones(1, 1, 9, device=device, dtype=torch.bool),
                                       batch=1, heads=2)
        lease.release()
    with torch.cuda.stream(producer):
        replacement = store.begin()
        replacement.append(torch.full((8, 2, 16), 9., device=device),
                           torch.full((8, 2, 16), 9., device=device))
        replacement.publish(['replacement'])
        replacement.release()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, torch.full_like(actual, 4.))
    assert len(store.pages) == 1 and store.pages[0].completion is not None


def test_service_can_publish_only_reusable_prefix_and_leave_full_working_tail_private():
    store = pool(capacity=3)
    data = torch.randn(6, 4, 4)
    session = store.begin()
    session.append(data, data)
    session.publish(['historical-first'], [80])
    private_slots = [slot for slot, _ in session.slots[1:]]
    assert all(store.pages[slot].refs == 1 for slot in private_slots)
    session.release()
    assert store.matching_tokens(['historical-first']) == 2
    assert all(store.pages[slot].valid == 0 for slot in private_slots)
