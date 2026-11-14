"""Local service components and real Bloom block outputs; not mocked RPC proof.

Actual P2P/InferenceSession/server execution is separately owned by 13G.
Guard probes below are metadata unit tests, not substitute model backends.
"""

from dataclasses import replace
from types import SimpleNamespace
import asyncio

import pytest
import torch
from transformers import LlamaConfig

from bloombee.client.dynamo_config import normalize_dynamo_config
from bloombee.data_structures import InferenceMetadata
from bloombee.kv_cache import create_cache
from bloombee.server.block_functions import _dynamo_step_metadata
from bloombee.server.backend import TransformerBackend
from bloombee.server.dynamo_service import (
    UnsupportedReuse, copy_prefix_to_slab, restore_complete_output, validate_mode, _State, _emit,
    _suffix_metadata,
    _binding,
    native_slab,
)
from bloombee.server.microbatch import build_inference_metadata_batch
from bloombee.utils.hivemind_compat import MSGPackSerializer
from bloombee.utils.convert_block import convert_block, QuantType
from bloombee.server.memory_cache_manager import KVCacheManager
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.pytorch_backend import TorchDevice
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.compression import CompressionConfig
from bloombee.utils.hivemind_compat import TensorDescriptor
from examples.kv_cache_model_cpu import TinyModel, to_time_major


def config(ids=(1, 2, 3, 4)):
    return {"context": "immutable-model/rev/schema", "token_ids": list(ids), "priorities": [80, 50]}


def test_actual_wire_metadata_roundtrip_and_task_metadata_preserve_opt_in():
    metadata = {"session_id": "real-metadata-serialization-test", "dynamo_cache": config()}
    restored = MSGPackSerializer.loads(MSGPackSerializer.dumps(metadata))
    assert restored == metadata
    infos = build_inference_metadata_batch(["layer.0", "layer.1"], [(1, 2), (3, 4)], 0, None,
                                           full_batch_size=1, dynamo_cache=restored["dynamo_cache"])
    assert all(info.dynamo_cache == config() for info in infos)
    assert all(info.prefix_length == 0 for info in infos)
    legacy = build_inference_metadata_batch(["layer.0"], [(1, 2)], 0, None)
    assert legacy[0].dynamo_cache is None


def test_input_identity_metadata_is_copied_not_aliased():
    original = config()
    normalized = normalize_dynamo_config(original)
    original["token_ids"][0] = 999
    original["priorities"][0] = 0
    assert normalized == config()


@pytest.mark.parametrize("bad", [None, {}, {"context": "x"}, {"context": "", "token_ids": [1]},
                                  {"context": "x", "token_ids": [True]},
                                  {"context": "x", "token_ids": [-1]},
                                  {"context": "x", "token_ids": [1], "priorities": [101]},
                                  {"context": "x", "token_ids": [1], "other": 1}])
def test_malformed_client_opt_in_is_rejected(bad):
    with pytest.raises(ValueError):
        normalize_dynamo_config(bad)


def guard_probe():
    # This is a request-mode validator fixture, not an RPC/model implementation.
    backend = SimpleNamespace(
        config=LlamaConfig(hidden_size=8, num_attention_heads=2, num_key_value_heads=2),
        _layer_type_for_this_block=lambda: "full_attention",
        module=torch.nn.Linear(8, 8).eval().requires_grad_(False),
        cache_manager=SimpleNamespace(
            offloading_policy=SimpleNamespace(compress_cache=False, cpu_cache_compute=False,
                                             cache_gpu_percent=100, cache_cpu_percent=0, cache_disk_percent=0),
            _paged_kv_enabled=False,
        ),
    )
    info = InferenceMetadata("layer.0", 0, (1, 2), None, full_batch_size=1,
                             micro_batch_size=1, dynamo_cache=config())
    return backend, info, torch.randn(1, 4, 8)


def test_mode_guard_requires_server_namespace_and_exact_token_length(monkeypatch):
    backend, info, inputs = guard_probe()
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CONTEXT", config()["context"])
    assert validate_mode(backend, inputs, torch.tensor([0]), info) == config()
    with pytest.raises(UnsupportedReuse):
        validate_mode(backend, inputs[:, :3], torch.tensor([0]), info)
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CONTEXT", "another-model")
    with pytest.raises(UnsupportedReuse, match="namespace"):
        validate_mode(backend, inputs, torch.tensor([0]), info)


@pytest.mark.parametrize("changes", [
    dict(prefix_length=1), dict(active_adapter="adapter"), dict(is_spec_dec=True),
    dict(need_pruning=True), dict(full_batch_size=2), dict(batch_offset=1),
    dict(tree_attention_mask=torch.tensor([[1, 0, 1, 1]])), dict(draft_tokens=torch.ones(1, 4)),
    dict(keep_indices=torch.tensor([[3, 2, 1, 0]])), dict(prefill_length=torch.tensor([4])),
])
def test_unsupported_requests_are_explicitly_rejected_for_original_fallback(monkeypatch, changes):
    backend, info, inputs = guard_probe()
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CONTEXT", config()["context"])
    with pytest.raises(UnsupportedReuse):
        validate_mode(backend, inputs, torch.tensor([0]), replace(info, **changes))


def test_deep_prompts_do_not_silently_enter_cache_mode(monkeypatch):
    backend, info, inputs = guard_probe()
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CONTEXT", config()["context"])
    tagged = _dynamo_step_metadata({"dynamo_cache": config()}, [torch.ones(1, 1, 8)])
    with pytest.raises(UnsupportedReuse, match="deep prompt"):
        validate_mode(backend, inputs, torch.tensor([0]), replace(info, dynamo_cache=tagged))
    assert _dynamo_step_metadata({}, [torch.ones(1, 1, 8)]) is None
    assert _dynamo_step_metadata({"dynamo_cache": config()}, [None]) == config()


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
def test_real_ordinary_llama_defaults_allow_only_unpadded_mask_and_matching_ids(monkeypatch, dtype):
    backend, info, inputs = guard_probe()
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CONTEXT", config()["context"])
    info = replace(info, tree_attention_mask=torch.ones(1, 4, dtype=dtype),
                   draft_tokens=torch.tensor([config()["token_ids"]], dtype=torch.int64))
    assert validate_mode(backend, inputs, torch.tensor([0]), info) == config()
    suffix = _suffix_metadata(info, 2, 2, inputs.device)
    assert suffix.prefix_length == 2
    assert suffix.tree_attention_mask.shape == (1, 4)
    assert torch.equal(suffix.tree_attention_mask, info.tree_attention_mask)
    assert suffix.draft_tokens is None and suffix.kv_cache_position_ids is None
    assert suffix.dynamo_cache is None
    assert torch.equal(suffix.keep_indices, torch.tensor([[0, 1]]))


@pytest.mark.parametrize("mask", [torch.ones(1, 3), torch.ones(1, 1, 4, 4),
                                  torch.ones(1, 4, 4).tril(), torch.full((1, 4), 2.),
                                  torch.tensor([[1., 1., float('nan'), 1.]])])
def test_custom_or_misaligned_masks_still_fallback(monkeypatch, mask):
    backend, info, inputs = guard_probe()
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CONTEXT", config()["context"])
    with pytest.raises(UnsupportedReuse, match="mask"):
        validate_mode(backend, inputs, torch.tensor([0]), replace(info, tree_attention_mask=mask))


@pytest.mark.parametrize("ids", [torch.tensor([[1, 2, 3, 99]]), torch.tensor([[1., 2., 3., 4.]]),
                                 torch.tensor([1, 2, 3, 4]), torch.tensor([[1, 2, 3]])])
def test_present_draft_payload_must_be_exact_actual_ids(monkeypatch, ids):
    backend, info, inputs = guard_probe()
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CONTEXT", config()["context"])
    with pytest.raises(UnsupportedReuse, match="draft payload"):
        validate_mode(backend, inputs, torch.tensor([0]), replace(info, draft_tokens=ids))


def test_even_consecutive_explicit_kv_positions_remain_unsupported(monkeypatch):
    backend, info, inputs = guard_probe()
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CONTEXT", config()["context"])
    with pytest.raises(UnsupportedReuse, match="cache-position"):
        validate_mode(backend, inputs, torch.tensor([0]),
                      replace(info, kv_cache_position_ids=torch.tensor([[0, 1, 2, 3]])))


@pytest.mark.parametrize("skipped", [0, 2, 7])
def test_native_causal_mask_with_full_ones_keys_is_exact_suffix_of_original(skipped):
    # Actual native stateless mask implementation, no RPC/model replacement.
    total = 8
    causal = TransformerBackend._create_causal_attention_mask(None, 1, total, skipped, torch.device("cpu"))
    whole = TransformerBackend._create_causal_attention_mask(None, 1, total, 0, torch.device("cpu"))
    key_mask = torch.ones(1, total, dtype=torch.bool)
    assert causal.shape == (1, total - skipped, total)
    assert torch.equal(causal & key_mask.unsqueeze(1), whole[:, skipped:, :])


def test_copy_to_native_shaped_tensors_checks_both_shapes_before_writing():
    key = torch.zeros(8, 2, 4)
    value = torch.zeros_like(key)
    with pytest.raises(UnsupportedReuse):
        copy_prefix_to_slab(key, value, {"key": torch.ones(4, 2, 4), "value": torch.ones(4, 1, 4)})
    assert torch.count_nonzero(key) == torch.count_nonzero(value) == 0
    assert copy_prefix_to_slab(key, value, {"key": torch.ones(4, 2, 4), "value": torch.full((4, 2, 4), 2.)}) == 4
    assert torch.equal(key[:4], torch.ones(4, 2, 4))
    assert torch.equal(value[:4], torch.full((4, 2, 4), 2.))
    assert torch.count_nonzero(key[4:]) == 0


@pytest.mark.parametrize("complete", [False, True])
@torch.inference_mode()
def test_real_bloom_outputs_are_complete_after_actual_kv_copy_and_resume(complete):
    # Real randomly initialized BloomBee block, real public DynamoCache. This
    # tests tensor/output contracts; Bloom is not enabled by the Llama RPC guard.
    model = TinyModel(31, 1)
    prefix = [1, 2, 3, 4]
    tail = [] if complete else [5, 6]
    inputs = model.embed(prefix)
    output = model.full(prefix)
    kv = to_time_major(model.prefix(prefix))[0]
    pool = create_cache("dynamo", capacity_blocks=4, block_size=2,
                        context=model.context, expected_layers=(0, 1))
    event = pool.put_completed(prefix, {0: kv, 1: {"key": output.transpose(0, 1),
                                                  "value": inputs.transpose(0, 1)}})
    assert event["admitted"]
    retained, count, _ = pool.get(prefix + tail)
    try:
        assert count == 4
        assert torch.equal(retained[1]["value"].transpose(0, 1), inputs)
        k_slab = torch.empty(8, model.batch * model.heads, model.hidden // model.heads)
        v_slab = torch.empty_like(k_slab)
        copy_prefix_to_slab(k_slab, v_slab, retained[0])
        snapshot = [(k_slab[:count].permute(1, 2, 0), v_slab[:count].permute(1, 0, 2))]
        suffix = model.resume(tail, snapshot) if tail else output[:, :0]
        restored = restore_complete_output(retained[1]["key"].transpose(0, 1), suffix, len(prefix + tail))
        assert restored.shape[1] == len(prefix + tail)
        torch.testing.assert_close(restored, model.full(prefix + tail), atol=1e-5, rtol=1e-4)
    finally:
        pool.release()
    assert pool.used_blocks == 2


def test_output_contract_cannot_drop_prefix_or_suffix():
    with pytest.raises(RuntimeError, match="every requested"):
        restore_complete_output(torch.ones(1, 2, 8), torch.ones(1, 2, 8), 5)


def test_cache_event_records_the_actual_public_factory_class(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_DYNAMO_HOST_MIN_PRIORITY", "50")
    state = _State("test/immutable-binding")
    assert state.cache.host_offload_min_priority == 50
    holder = SimpleNamespace(name="metadata-probe.0", _dynamo_service_state=state)
    _emit(holder, mode="component-metadata-check")
    event = holder._dynamo_service_events[-1]
    assert event["cache_class"] == "bloombee.kv_cache.dynamo.DynamoCache"
    assert event["backend_class"] == "types.SimpleNamespace"


def actual_pipeline_wrapper(block, freeze=True):
    """The real production convert_block wrapper, not a stand-in class."""
    return convert_block(block, 0, LlamaConfig(hidden_size=8, num_attention_heads=2),
                         [torch.device("cpu")], torch.device("cpu"), QuantType.NONE,
                         freeze=freeze)


@pytest.mark.parametrize("training", [False, True])
def test_real_cli_pipeline_shards_preserve_dropout_guard(monkeypatch, training):
    backend, info, inputs = guard_probe()
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CONTEXT", config()["context"])
    block = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(0.5)).train(training)
    backend.module = actual_pipeline_wrapper(block)
    assert not hasattr(backend.module, "modules")
    assert backend.module.module_shards[0] is block
    if training:
        with pytest.raises(UnsupportedReuse, match="stochastic"):
            validate_mode(backend, inputs, torch.tensor([0]), info)
    else:
        assert validate_mode(backend, inputs, torch.tensor([0]), info) == config()


def test_real_cli_pipeline_binding_uses_delegated_parameter_versions(monkeypatch):
    backend, info, inputs = guard_probe()
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CONTEXT", config()["context"])
    block = torch.nn.Linear(8, 8).eval()
    backend.module = actual_pipeline_wrapper(block)
    backend.name = "actual-wrapper-component.0"
    backend.dtype = torch.float32
    first = _binding(backend)
    assert _binding(backend) == first
    with torch.no_grad():
        block.weight.add_(1)
    assert _binding(backend) != first
    assert list(backend.module.named_buffers()) == list(block.named_buffers())


def test_trainable_pipeline_shard_is_not_silently_cached(monkeypatch):
    backend, info, inputs = guard_probe()
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CONTEXT", config()["context"])
    backend.module = actual_pipeline_wrapper(torch.nn.Linear(8, 8).eval(), freeze=False)
    with pytest.raises(UnsupportedReuse, match="immutable"):
        validate_mode(backend, inputs, torch.tensor([0]), info)


def test_unknown_wrapper_cannot_bypass_module_inspection(monkeypatch):
    backend, info, inputs = guard_probe()
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CONTEXT", config()["context"])
    backend.module = object()
    with pytest.raises(UnsupportedReuse, match="cannot inspect"):
        validate_mode(backend, inputs, torch.tensor([0]), info)


def test_real_pipeline_wrapper_reaches_binding_and_actual_cpu_native_slab(monkeypatch):
    """Real manager allocation/control pipe and wrapper; no RPC/model claim."""
    backend, info, inputs = guard_probe()
    monkeypatch.setenv("BLOOMBEE_DYNAMO_CONTEXT", config()["context"])
    backend.module = actual_pipeline_wrapper(torch.nn.Linear(8, 8).eval())
    backend.name = "native-slab-component.0"
    backend.dtype = torch.float16
    cpu = TorchDevice("cpu")
    env = ExecutionEnv(gpu=cpu, cpu=cpu, disk=None, mixed=None)
    compression = CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False)
    policy = Policy(gpu_batch_size=1, num_gpu_batches=1, w_gpu_percent=100, w_cpu_percent=0,
                    cache_gpu_percent=100, cache_cpu_percent=0,
                    act_gpu_percent=100, act_cpu_percent=0, overlap=False, sep_layer=True,
                    pin_weight=False, cpu_cache_compute=False, attn_sparsity=1.,
                    compress_weight=False, comp_weight_config=compression,
                    compress_cache=False, comp_cache_config=compression)
    manager = KVCacheManager(64, 30, policy, env, backend.config)
    backend.cache_manager = manager
    descriptor = TensorDescriptor((1, 2, 4, 16), dtype=torch.float16, device=torch.device("cpu"))
    size = manager.get_allocation_size_tokens(descriptor)
    # Use the actual allocation scheduler and pipe consumed by use_cache, without
    # substituting cache tensors or pretending to be a connection handler.
    handles = asyncio.run(manager.cache._schedule_alloc(size, descriptor, timeout=0))
    info = replace(info, cache_handles=tuple(handles))
    try:
        assert validate_mode(backend, inputs, torch.tensor([0]), info) == config()
        assert len(_binding(backend)) == 64
        with native_slab(backend, info, 4) as (key, value):
            assert key.device.type == "cpu" and key.shape[1:] == (2, 4)
            key.zero_()
            value.zero_()
            source = {"key": torch.ones(4, 2, 4, dtype=key.dtype),
                      "value": torch.full((4, 2, 4), 2., dtype=value.dtype)}
            assert copy_prefix_to_slab(key, value, source) == 4
        with native_slab(backend, info, 4) as (key, value):
            assert torch.equal(key[:4], source["key"])
            assert torch.equal(value[:4], source["value"])
    finally:
        manager.cache.force_free(*handles)
        with manager.cache.use_cache():
            pass
        manager._reorder_executor.shutdown(wait=True)
