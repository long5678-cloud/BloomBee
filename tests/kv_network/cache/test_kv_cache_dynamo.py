"""Native-rule component vectors with actual CPU tensors and real package import.

These are storage/filtering/lifecycle tests, not a model or CUDA certificate.
Expected priorities and return order follow the frozen Dynamo binding sources,
not the preexisting A/B/F research implementations.
"""

import pytest
import torch

from bloombee.kv_cache.dynamo import DynamoCache


CONTEXT = "model/revision/execution/test-shape-v1"


def cache(capacity=16, threshold=0, *, expected_layers=2, block_size=2):
    return DynamoCache(capacity, block_size, context=CONTEXT,
                       expected_layers=expected_layers,
                       host_offload_min_priority=threshold)


def tensors(ids, layer_ids=(0, 1), *, heads=2, dimension=3):
    # Distinct real tensor values depend on the complete cumulative token chain.
    values = torch.tensor(ids, dtype=torch.float32).cumsum(0).reshape(-1, 1, 1)
    features = torch.arange(heads * dimension, dtype=torch.float32).reshape(1, heads, dimension)
    return {i: {"key": (values + features + i * 100).clone(),
                "value": (values - features - i * 10).clone()} for i in layer_ids}


def snapshot(store):
    entries = tuple((key, block.priority, block.returned_tick, block.acquired_tick,
                     block.digest, tuple((i, name, tensor.data_ptr(), tensor.tolist())
                                         for i, pair in block.layers.items()
                                         for name, tensor in pair.items()))
                    for key, block in store.entries.items())
    return entries, dict(store._requests), store._tick, store._schema, tuple(store._active_keys)


def assert_prefix(actual, source, length):
    assert set(actual) == set(source)
    for i in source:
        for name in ("key", "value"):
            torch.testing.assert_close(actual[i][name], source[i][name][:length], rtol=0, atol=0)


@pytest.mark.parametrize("threshold,priorities,expected", [
    (0, None, 4), (0, [0, 0], 4), (0, [None, None], 4),
    (50, [50, 80], 4), (50, [80, 49], 2), (50, [49, 100], 0),
    (80, [80, 100], 4), (80, [100, 79], 2), (80, [None, 100], 0),
    (100, [100, 100], 4), (1, None, 0), (50, [80], 2),
])
def test_native_contiguous_threshold_not_independent_block_filter(threshold, priorities, expected):
    store = cache(threshold=threshold)
    ids = [1, 2, 3, 4]
    source = tensors(ids)
    event = store.put_completed(ids, source, priorities=priorities)
    assert event["admitted"]
    assert store.used_blocks == expected // 2
    if expected < len(ids):
        assert event["terminated_at_block"] == expected // 2
    actual, length, get = store.get(ids)
    assert length == expected and get["hit"] == bool(expected)
    if length:
        assert_prefix(actual, source, length)
    store.release()


def test_constructor_defaults_missing_priority_and_immediate_independent_cpu_copy():
    store = DynamoCache(4, context=CONTEXT, expected_layers=2)
    assert store.host_offload_min_priority == 0 and store.block_size == 16
    ids = list(range(32))
    source = tensors(ids)
    event = store.put_completed(ids, source)
    assert event["new_blocks"] == 2 and event["copied_bytes"] == store.resident_bytes() > 0
    assert all(block.priority == 0 for block in store.entries.values())
    # Source ownership is retained; no source eviction or source modification.
    for block in store.entries.values():
        for i in source:
            for name in source[i]:
                assert block.layers[i][name].device.type == "cpu"
                assert block.layers[i][name].data_ptr() != source[i][name].data_ptr()
    before = source[0]["key"].clone()
    actual, n, _ = store.get(ids)
    assert n == 32
    assert_prefix(actual, source, n)
    for pair in actual.values():
        for tensor in pair.values():
            tensor.zero_()
    assert torch.equal(source[0]["key"], before)
    store.release()
    source[0]["key"].zero_()
    again, n, _ = store.get(ids)
    assert torch.equal(again[0]["key"], before)
    assert store.used_blocks == 2
    store.release()


def test_future_hints_survive_chunks_and_late_high_block_cannot_cross_cut():
    store = cache(threshold=50)
    ids = list(range(8))
    source = tensors(ids)
    first = store.put_completed(ids[:2], tensors(ids[:2]), priorities=[80, 20, 100, 100],
                                request_id="r", final=False)
    assert first["new_blocks"] == 1
    second = store.put_completed(ids[:4], tensors(ids[:4]), request_id="r", final=False)
    assert second["terminated_at_block"] == 1 and second["new_blocks"] == 0
    third = store.put_completed(ids, source, priorities=[100, 100, 100, 100], request_id="r")
    assert third["terminated_at_block"] == 1 and third["new_blocks"] == 0
    assert "r" not in store._requests
    actual, n, _ = store.get(ids)
    assert n == 2
    assert_prefix(actual, source, n)
    store.release()
    # A distinct request is independent, despite the same token prefix.
    fresh = store.put_completed(ids, source, priorities=[80] * 4, request_id="other")
    assert fresh["new_blocks"] == 3 and fresh["terminated_at_block"] is None
    assert store.get(ids)[1] == 8
    store.release()


def test_stored_future_priorities_enable_later_chunk_without_new_hints():
    store = cache(threshold=80)
    ids = [1, 2, 3, 4, 5, 6]
    store.put_completed(ids[:2], tensors(ids[:2]), priorities=[80, 90, 100],
                        request_id="r", final=False)
    event = store.put_completed(ids, tensors(ids), request_id="r")
    assert event["new_blocks"] == 2 and event["terminated_at_block"] is None
    assert [block.priority for block in store.entries.values()] == [80, 90, 100]


def test_chunk_truncation_persists_even_if_accepted_old_block_was_recycled():
    store = cache(capacity=1, threshold=50)
    store.put_completed([1, 2, 3, 4], tensors([1, 2, 3, 4]), priorities=[50, 0],
                        request_id="cut", final=False)
    store.put_completed([8, 9], tensors([8, 9]), priorities=[80])
    assert store.get([1, 2])[1] == 0
    event = store.put_completed([1, 2, 3, 4, 5, 6], tensors([1, 2, 3, 4, 5, 6]),
                                priorities=[100] * 3, request_id="cut")
    assert event["new_blocks"] == 0 and event["terminated_at_block"] == 1
    assert store.get([1, 2, 3, 4, 5, 6])[1] == 0


def test_incomplete_tail_is_deferred_without_losing_later_priority():
    store = cache(threshold=50)
    event = store.put_completed([1, 2, 3], tensors([1, 2, 3]), priorities=[50, 80],
                                request_id="r", final=False)
    assert event["ignored_tail_tokens"] == 1 and store.used_blocks == 1
    event = store.put_completed([1, 2, 3, 4], tensors([1, 2, 3, 4]), request_id="r")
    assert event["new_blocks"] == 1 and store.used_blocks == 2
    assert store.get([1, 2, 3, 4, 5])[1] == 4
    store.release()


def test_same_suffix_different_prefix_or_context_is_not_reused():
    store = cache()
    ids = [1, 2, 7, 8]
    store.put_completed(ids, tensors(ids))
    assert store.get([9, 10, 7, 8])[1] == 0
    assert store.get([1, 2, 99, 100])[1] == 2
    store.release()
    with pytest.raises(ValueError, match="context"):
        store.get(ids, context="different-revision")
    with pytest.raises(ValueError, match="context"):
        store.put_completed(ids, tensors(ids), context="different-revision")
    other = DynamoCache(4, 2, context="other", expected_layers=2)
    assert other.get(ids)[1] == 0


def test_recycle_lowest_priority_idle_block_not_oldest_high_priority():
    store = cache(capacity=2)
    store.put_completed([1, 2], tensors([1, 2]), priorities=[80])
    store.put_completed([3, 4], tensors([3, 4]), priorities=[10])
    event = store.put_completed([5, 6], tensors([5, 6]))
    assert event["evicted"] == [(CONTEXT, (3, 4))]
    assert store.get([1, 2])[1] == 2
    store.release()


def test_equal_priority_order_is_actual_return_order_not_get_time():
    store = cache(capacity=3)
    for ids in ([1, 2], [3, 4], [5, 6]):
        store.put_completed(ids, tensors(ids), priorities=[50])
    p = (CONTEXT, (1, 2))
    old_return = store.entries[p].returned_tick
    store.get([1, 2])
    assert store.entries[p].returned_tick == old_return
    assert store.entries[p].acquired_tick > 0
    store.release()
    assert store.entries[p].returned_tick > old_return
    event = store.put_completed([7, 8], tensors([7, 8]), priorities=[50])
    assert event["evicted"] == [(CONTEXT, (3, 4))]


def test_free_slot_is_used_before_any_idle_recycle():
    store = cache(capacity=2)
    store.put_completed([1, 2], tensors([1, 2]))
    event = store.put_completed([3, 4], tensors([3, 4]))
    assert event["evicted"] == [] and store.used_blocks == 2


def test_active_read_protected_but_other_idle_block_is_recycled():
    store = cache(capacity=2)
    store.put_completed([1, 2], tensors([1, 2]))
    store.put_completed([3, 4], tensors([3, 4]), priorities=[100])
    actual, n, _ = store.get([1, 2])
    assert n == 2
    event = store.put_completed([5, 6], tensors([5, 6]))
    assert event["evicted"] == [(CONTEXT, (3, 4))]
    assert_prefix(actual, tensors([1, 2]), 2)
    with pytest.raises(RuntimeError, match="release"):
        store.get([1, 2])
    with pytest.raises(RuntimeError, match="active"):
        store.clear()
    store.release()
    store.clear()
    assert not store.used_blocks and store.resident_bytes() == 0


def test_no_idle_capacity_preserves_request_progress_and_allows_same_input_retry():
    store = cache(capacity=1)
    store.put_completed([1, 2], tensors([1, 2]))
    store.get([1, 2])
    before = snapshot(store)
    event = store.put_completed([3, 4], tensors([3, 4]), request_id="pending", final=False)
    assert not event["admitted"] and event["copied_bytes"] == event["offloaded_tokens"] == 0
    assert snapshot(store) == before
    store.release()
    assert store.put_completed([3, 4], tensors([3, 4]), request_id="pending")["admitted"]


def test_batch_over_capacity_does_not_keep_only_unusable_suffix():
    store = cache(capacity=1)
    before = snapshot(store)
    event = store.put_completed([1, 2, 3, 4], tensors([1, 2, 3, 4]))
    assert not event["admitted"] and snapshot(store) == before
    zero = cache(capacity=0)
    assert not zero.put_completed([1, 2], tensors([1, 2]))["admitted"]
    assert zero.resident_bytes() == 0


def test_duplicate_retains_first_content_priority_and_return_order():
    store = cache()
    ids = [1, 2]
    store.put_completed(ids, tensors(ids), priorities=[20])
    before = snapshot(store)
    changed = tensors(ids)
    changed[0]["key"].add_(1)
    event = store.put_completed(ids, changed, priorities=[100])
    assert event["new_blocks"] == event["copied_bytes"] == 0
    assert event["deduplicated_blocks"] == 1
    assert snapshot(store) == before
    actual, n, _ = store.get(ids)
    assert_prefix(actual, tensors(ids), n)
    store.release()


def test_missing_first_block_does_not_reassemble_resident_later_block():
    store = cache(capacity=2)
    store.put_completed([1, 2, 3, 4], tensors([1, 2, 3, 4]), priorities=[0, 100])
    store.put_completed([8, 9], tensors([8, 9]), priorities=[100])
    assert (CONTEXT, (1, 2, 3, 4)) in store.entries
    assert store.get([1, 2, 3, 4])[1] == 0


def test_bad_chunk_identity_schema_and_hints_do_not_advance_state():
    store = cache(threshold=50)
    store.put_completed([1, 2], tensors([1, 2]), priorities=[80], request_id="r", final=False)
    before = snapshot(store)
    with pytest.raises(ValueError, match="extend"):
        store.put_completed([2, 1, 3, 4], tensors([2, 1, 3, 4]), request_id="r")
    with pytest.raises(ValueError, match="schema"):
        store.put_completed([1, 2, 3, 4], tensors([1, 2, 3, 4], dimension=4), request_id="r")
    with pytest.raises(ValueError, match="priority"):
        store.put_completed([1, 2, 3, 4], tensors([1, 2, 3, 4]), priorities=[80, 101], request_id="r")
    assert snapshot(store) == before
    assert store.finish_request("r") and not store.finish_request("r")


def test_copy_integrity_failure_is_atomic_and_retry_uses_same_chunk():
    class FaultOnce(DynamoCache):
        fail = True

        def _copy_block(self, layers, start, end):
            copied = super()._copy_block(layers, start, end)
            if self.fail:
                copied[self.expected_layers[0]]["key"].add_(1)
            return copied

    store = FaultOnce(4, 2, context=CONTEXT, expected_layers=2)
    before = snapshot(store)
    with pytest.raises(ValueError, match="integrity"):
        store.put_completed([1, 2], tensors([1, 2]), priorities=[80], request_id="r", final=False)
    assert snapshot(store) == before
    store.fail = False
    assert store.put_completed([1, 2], tensors([1, 2]), priorities=[80], request_id="r")["admitted"]
    assert store.get([1, 2])[1] == 2
    store.release()


def test_cached_tensor_corruption_or_invalid_target_leaves_no_active_reference():
    store = cache()
    store.put_completed([1, 2], tensors([1, 2]))
    before = snapshot(store)
    with pytest.raises((ValueError, RuntimeError)):
        store.get([1, 2], target_device="invalid-device")
    assert snapshot(store) == before and not store.active
    block = next(iter(store.entries.values()))
    original = block.layers[0]["key"].clone()
    block.layers[0]["key"].zero_()
    with pytest.raises(ValueError, match="integrity"):
        store.get([1, 2])
    assert not store.active
    block.layers[0]["key"].copy_(original)
    assert store.get([1, 2])[1] == 2
    store.release()


@pytest.mark.parametrize("layer_ids", [(0, 1), (2, 4, 8), tuple(range(32))])
def test_actual_multilayer_tensor_bytes_and_partial_prefix(layer_ids):
    store = cache(expected_layers=layer_ids)
    ids = [1, 2, 3, 4, 5]
    source = tensors(ids, layer_ids)
    event = store.put_completed(ids, source)
    expected = sum(tensor[:4].numel() * tensor.element_size()
                   for pair in source.values() for tensor in pair.values())
    assert store.resident_bytes() == event["copied_bytes"] == expected
    actual, n, _ = store.get(ids)
    assert n == 4
    assert_prefix(actual, source, n)
    store.release()


@pytest.mark.parametrize("threshold", [-1, 101, True, 1.5, "50"])
def test_invalid_host_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        cache(threshold=threshold)


def test_empty_and_short_prefix_finalization_does_not_create_partial_blocks():
    store = cache()
    event = store.put_completed([], tensors([]), request_id="empty")
    assert event["new_blocks"] == 0 and store.get([])[1] == 0
    event = store.put_completed([1], tensors([1]))
    assert event["ignored_tail_tokens"] == 1 and store.used_blocks == 0
