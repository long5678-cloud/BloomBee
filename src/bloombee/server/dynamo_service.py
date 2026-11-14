"""Opt-in ordinary RPC prefill caching, preserving every returned activation.

The server owns a bounded CPU history per backend. Native K/V is layer 0;
auxiliary layer 1 retains output/input activations to restore the full ordinary
RPC output and verify actual incoming layer inputs. It is not a second model
layer. All copies count toward resident payload bytes.
"""

from collections import deque
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
import sys
import threading
from time import perf_counter

import torch

from bloombee.client.dynamo_config import normalize_dynamo_config
from bloombee.kv_cache import create_cache
from bloombee.utils.hivemind_compat import get_logger
from bloombee.utils.misc import is_dummy
from bloombee.server.dynamo_gpu import NativeHistory, storage_id, register_session, invalidate_session
from bloombee.server.dynamo_native_context import native_execution

logger = get_logger(__name__)


class UnsupportedReuse(ValueError):
    """A request cannot use this bounded first-prefill adapter safely."""


def _present(value):
    return value is not None and (not torch.is_tensor(value) or not is_dummy(value))


def _enabled_flag(value):
    if torch.is_tensor(value):
        return bool(value.numel() and value.bool().any().item())
    return bool(value)


def _raw(value):
    tensor = value if isinstance(value, torch.Tensor) else getattr(value, "data", None)
    if not isinstance(tensor, torch.Tensor):
        raise UnsupportedReuse("non-dense native cache representation")
    return tensor


def _module_tree(wrapper):
    """Inspect real nn.Module shards, including the CLI's plain Python wrapper.

    convert_block.PipelineParallelWrapper deliberately exposes module_shards and
    named_parameters/named_buffers but is not itself an nn.Module. Unknown
    representations cannot silently bypass the stochastic-execution check.
    """
    roots = [wrapper] if isinstance(wrapper, torch.nn.Module) else []
    shards = getattr(wrapper, "module_shards", None)
    if shards is not None:
        shards = tuple(shards)
        if not shards or not all(isinstance(shard, torch.nn.Module) for shard in shards):
            raise UnsupportedReuse("cannot inspect actual module shards")
        roots.extend(shards)
    if not roots:
        raise UnsupportedReuse("cannot inspect module stochastic execution")
    seen = set()
    for root in roots:
        for module in root.modules():
            if id(module) not in seen:
                seen.add(id(module))
                yield module


def _validate_immutable_execution(wrapper):
    dropout_classes = (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d,
                       torch.nn.Dropout3d, torch.nn.AlphaDropout, torch.nn.FeatureAlphaDropout)
    for module in _module_tree(wrapper):
        if isinstance(module, dropout_classes) and module.training and module.p:
            raise UnsupportedReuse("stochastic training layers are unsupported")
        # Standard attention can apply functional dropout instead of registering
        # a Dropout child. This matters for Llama attention implementations.
        attention_dropout = getattr(module, "attention_dropout", 0)
        if module.training and isinstance(attention_dropout, (int, float)) and attention_dropout > 0:
            raise UnsupportedReuse("stochastic attention training is unsupported")
        if any(parameter.requires_grad for parameter in module.parameters(recurse=False)):
            raise UnsupportedReuse("cache reuse requires immutable inference parameters")


def validate_mode(backend, inputs, hypo_ids, info):
    if isinstance(info.dynamo_cache, dict) and info.dynamo_cache.get("_server_has_prompts"):
        raise UnsupportedReuse("deep prompt requests are unsupported")
    config = normalize_dynamo_config(info.dynamo_cache)
    namespace = os.environ.get("BLOOMBEE_DYNAMO_CONTEXT", "")
    if not namespace or config["context"] != namespace:
        raise UnsupportedReuse("server model/execution namespace mismatch")
    if inputs.ndim != 3 or inputs.shape[0] != 1 or inputs.shape[1] != len(config["token_ids"]):
        raise UnsupportedReuse("requires exact first-prefill IDs and batch one")
    if info.prefix_length != 0 or info.batch_offset or info.full_batch_size not in (0, 1):
        raise UnsupportedReuse("only initial unpartitioned prefill is supported")
    if info.micro_batch_size not in (0, 1) or info.active_adapter:
        raise UnsupportedReuse("microbatch multiplexing/adapters are unsupported")
    if _enabled_flag(info.is_spec_dec) or _enabled_flag(info.need_pruning):
        raise UnsupportedReuse("speculative/pruning requests are unsupported")
    if _present(info.tree_attention_mask):
        mask = info.tree_attention_mask
        # DistributedLlamaModel.forward passes the ordinary HF padding mask in
        # this field. The backend intersects its causal mask with these keys.
        # Only the exact unpadded shape/value is equivalent to ordinary causality.
        if (not torch.is_tensor(mask) or tuple(mask.shape) != (1, inputs.shape[1])
                or not bool(torch.all(mask == 1).item())):
            raise UnsupportedReuse("padding or custom attention masks are unsupported")
    if _present(info.kv_cache_position_ids):
        # Even non-spec execution takes select_cache_without_reorder when this
        # is nonempty. A consecutive-looking tensor is not proof of equivalence
        # to an empty initial cache, so retain the original safe fallback.
        raise UnsupportedReuse("explicit cache-position selection is unsupported")
    if _present(info.draft_tokens):
        ids = info.draft_tokens
        expected_ids = torch.tensor(config["token_ids"], dtype=torch.int64).unsqueeze(0)
        if (not torch.is_tensor(ids) or ids.dtype != torch.int64
                or tuple(ids.shape) != tuple(expected_ids.shape)
                or not torch.equal(ids.cpu(), expected_ids)):
            raise UnsupportedReuse("draft payload differs from actual ordinary request IDs")
    if _present(info.keep_indices):
        expected = torch.arange(inputs.shape[1], dtype=torch.int64).unsqueeze(0)
        if not torch.equal(info.keep_indices.cpu(), expected):
            raise UnsupportedReuse("nonidentity output selection is unsupported")
    if _enabled_flag(info.prefill_length):
        raise UnsupportedReuse("nonzero custom prefill positions are unsupported")
    if _present(hypo_ids) and not torch.equal(hypo_ids.cpu(), torch.zeros(1, dtype=torch.int64)):
        raise UnsupportedReuse("hypothesis/row reordering is unsupported")
    if getattr(backend.config, "model_type", None) != "llama":
        raise UnsupportedReuse("only standard full-attention Llama is enabled")
    if backend._layer_type_for_this_block() in ("linear_attention", "moe", "mlp"):
        raise UnsupportedReuse("this block is not standard attention")
    manager = backend.cache_manager
    policy = manager.offloading_policy
    if (policy.compress_cache or policy.cpu_cache_compute
            or policy.cache_gpu_percent != 100 or policy.cache_cpu_percent
            or policy.cache_disk_percent or getattr(manager, "_paged_kv_enabled", False)):
        raise UnsupportedReuse("requires dense non-offloaded/nonpaged native KV")
    _validate_immutable_execution(backend.module)
    return config


def _suffix_metadata(info, count, suffix_length, device):
    """Normalize ONLY a previously validated ordinary prefill for continuation.

    A 2D key mask covers all original keys: prefix + suffix. Keep its full width
    because backend.py uses src_len=cache_len+seq_len and intersects with a
    suffix-query causal mask. Cropping to suffix width would lose key alignment.
    The validated non-spec draft payload carries the original IDs only; it has
    no generation/pruning meaning here and is cleared. Explicit KV positions
    cannot reach this helper through validate_mode.
    """
    keep = torch.arange(suffix_length, device=device).unsqueeze(0)
    return replace(info, prefix_length=count, keep_indices=keep, draft_tokens=None,
                   kv_cache_position_ids=None, dynamo_cache=None)


@contextmanager
def native_slab(backend, info, tokens):
    manager = backend.cache_manager
    manager.wait_for_pending_reorder()
    with manager.use_cache(*info.cache_handles) as cache_tensors:
        if len(cache_tensors) != 1 or len(cache_tensors[0]) != 2:
            raise UnsupportedReuse("requires one native K/V pair")
        key, value = map(_raw, cache_tensors[0])
        heads = backend.config.num_attention_heads
        if (key.ndim != 3 or key.shape != value.shape or key.shape[0] < tokens
                or key.shape[1] != heads or key.dtype != value.dtype):
            raise UnsupportedReuse("native time-major KV shape is incompatible")
        yield key, value


def copy_prefix_to_slab(key, value, prefix):
    """Actual dense K/V copy with integrity verification; no control-plane stub."""
    count = prefix["key"].shape[0]
    for destination, name in ((key, "key"), (value, "value")):
        source = prefix[name]
        if source.shape != destination[:count].shape or source.dtype != destination.dtype:
            raise UnsupportedReuse("cached/native K/V schema mismatch")
    for destination, name in ((key, "key"), (value, "value")):
        source = prefix[name]
        destination[:count].copy_(source)
        if not torch.equal(destination[:count], source.to(destination.device)):
            raise UnsupportedReuse("native K/V import copy failed")
    return count


def restore_complete_output(prefix_output, suffix_output, requested_tokens):
    output = torch.cat((prefix_output, suffix_output), dim=1)
    if output.ndim != 3 or output.shape[1] != requested_tokens:
        raise RuntimeError("cache adapter must return every requested activation")
    return output


def _emit(backend, **event):
    state = getattr(backend, "_dynamo_service_state", None)
    actual_cache = getattr(state, "cache", None)
    event = dict(event="dynamo_service", uid=str(backend.name),
                 backend_class=type(backend).__module__ + "." + type(backend).__qualname__,
                 cache_class=(type(actual_cache).__module__ + "." + type(actual_cache).__qualname__
                              if actual_cache is not None else None), **event)
    if not hasattr(backend, "_dynamo_service_events"):
        backend._dynamo_service_events = deque(maxlen=1024)
    backend._dynamo_service_events.append(event)
    # This is the structured completion protocol consumed by OrdinaryClient,
    # not a human verbosity message. Keep it available with a WARNING root
    # logger, and flush before the model RPC completion is observed.
    sys.stdout.write("DYNAMO_SERVICE " + json.dumps(event, sort_keys=True) + "\n")
    sys.stdout.flush()


def _binding(backend):
    # The operator pins the deployed model revision/execution namespace. Track
    # registered parameter mutations as well; FlexGen's external weights still
    # require an immutable deployment namespace and are not training-supported.
    config = backend.config.to_dict()
    versions = [(name, tuple(value.shape), str(value.dtype), value._version)
                for name, value in backend.module.named_parameters()]
    record = dict(namespace=os.environ["BLOOMBEE_DYNAMO_CONTEXT"], uid=str(backend.name),
                  config=config, dtype=str(backend.dtype), versions=versions, owner=os.getpid())
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


class _State:
    def __init__(self, binding):
        self.binding = binding
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.cache = create_cache(
            "dynamo", capacity_blocks=int(os.environ.get("BLOOMBEE_DYNAMO_CAPACITY_BLOCKS", "16")),
            block_size=int(os.environ.get("BLOOMBEE_DYNAMO_BLOCK_SIZE", "16")), context=binding,
            expected_layers=(0, 1),
            host_offload_min_priority=int(os.environ.get("BLOOMBEE_DYNAMO_HOST_MIN_PRIORITY", "0")),
        )
        self.gpu = NativeHistory(binding, self.cache.block_size,
            int(os.environ.get("BLOOMBEE_DYNAMO_GPU_BUDGET_BYTES", "0")),
            int(os.environ.get("BLOOMBEE_DYNAMO_GPU_MAX_SLOTS", "1")))


def _retain(backend, info, state, config, inputs, output):
    put_called = False
    try:
        count = inputs.shape[1]
        with native_slab(backend, info, count) as (key, value):
            layers = {0: {"key": key[:count], "value": value[:count]},
                      1: {"key": output.transpose(0, 1), "value": inputs.transpose(0, 1)}}
            put_called = True
            result = state.cache.put_completed(config["token_ids"], layers, priorities=config.get("priorities"))
        return {"stored": result["admitted"], "cache_put_called": put_called,
                "new_blocks": result["new_blocks"],
                "retained_payload_bytes": result["resident_bytes"],
                "new_payload_copy_bytes": result["copied_bytes"],
                "terminated_at_block": result["terminated_at_block"],
                "cpu_evictions": [{"prefix_tokens": len(item["key"][1]),
                    "prefix_sha256": hashlib.sha256(json.dumps(item["key"][1]).encode()).hexdigest(),
                    "priority": item["priority"], "returned_tick": item["returned_tick"],
                    "acquired_tick": item["acquired_tick"]}
                    for item in result.get("evicted_metadata", [])]}
    except Exception as error:
        # A failed retention attempt cannot discard successful model outputs.
        return {"stored": False, "cache_put_called": put_called, "store_error": type(error).__name__}


def _fallback(backend, inputs, hypo_ids, info, uncached, reason):
    result = uncached(inputs, hypo_ids, replace(info, dynamo_cache=None))
    requested = int(inputs.shape[1]) if inputs.ndim == 3 else 0
    returned = int(result[0].shape[1]) if result[0].ndim == 3 else 0
    attempt_id = info.dynamo_cache.get('attempt_id') if isinstance(info.dynamo_cache, dict) else None
    _emit(backend, mode="fallback", telemetry_attempt_id=attempt_id, reason=str(reason), requested_tokens=requested,
          computed_tokens=requested, skipped_tokens=0, returned_tokens=returned,
          native_layers_imported=0)
    return result


def run_dynamo_step(backend, inputs, hypo_ids, info, uncached):
    """Called by the actual TransformerBackend.inference_step entry point."""
    requested = int(inputs.shape[1]) if inputs.ndim == 3 else 0
    try:
        config = validate_mode(backend, inputs, hypo_ids, info)
        backend._ensure_model_on_device()
        binding = _binding(backend)
        state = getattr(backend, "_dynamo_service_state", None)
        if state is None or state.binding != binding or state.pid != os.getpid():
            state = _State(binding)
            backend._dynamo_service_state = state
        with native_slab(backend, info, requested) as (key, value):
            execution_device, native_shape, native_dtype = key.device, key.shape, key.dtype
    except Exception as error:
        return _fallback(backend, inputs, hypo_ids, info, uncached, error)

    with state.lock:
        count = gpu_count = cpu_count = 0
        gpu_slot = None
        retained = {}
        history_output = inputs[:, :0].to(execution_device)
        source_storage = source_value_storage = source_slot_id = None
        memory = backend.cache_manager.cache
        handle, = info.cache_handles
        event_eviction_start = len(state.gpu.evictions)
        # Leave a full final block for actual forward/logits, as in the fixed
        # connector's host-onboarding path. No auxiliary-only output hit.
        max_skip = ((requested - 1) // state.cache.block_size) * state.cache.block_size
        get_seconds = import_seconds = 0.0
        try:
            started = perf_counter()
            match = state.gpu.find(config["token_ids"], inputs, native_shape, native_dtype, execution_device,
                                   max_tokens=max_skip)
            if match is not None:
                gpu_slot, gpu_count = match
                gpu_count = min(gpu_count, max_skip)
                if gpu_count:
                    source_storage = storage_id(_raw(gpu_slot.native[0]))
                    source_value_storage = storage_id(_raw(gpu_slot.native[1]))
                    source_slot_id = gpu_slot.serial
                    state.gpu.acquire(gpu_slot, handle)
                    # The new ordinary handle now owns THE SAME native tensors.
                    # Its empty allocation is dropped; no prefix copy occurs.
                    memory._allocated_tensors[handle] = gpu_slot.native
                    history_output = gpu_slot.outputs[:, :gpu_count]
                    register_session(memory, state.gpu, gpu_slot, handle)
                else:
                    gpu_slot = None
            retained, cpu_count, cpu_event = state.cache.get_range(
                config["token_ids"], start_tokens=gpu_count, end_tokens=max_skip,
                target_device="cpu")
            count = gpu_count + cpu_count
            get_seconds = perf_counter() - started
            if cpu_count:
                previous_input = retained[1]["value"].transpose(0, 1)
                if previous_input.dtype != inputs.dtype or not torch.equal(previous_input.to(inputs.device), inputs[:, gpu_count:count]):
                    raise UnsupportedReuse("actual incoming layer prefix differs from retained input")
                started = perf_counter()
                with native_slab(backend, info, requested) as (key, value):
                    for destination, name in ((key, "key"), (value, "value")):
                        source = retained[0][name]
                        if source.shape != destination[gpu_count:count].shape or source.dtype != destination.dtype:
                            raise UnsupportedReuse("cached/native K/V schema mismatch")
                    # Only the absent interval is copied from CPU. Verification
                    # reads that interval back, never the resident GPU prefix.
                    for destination, name in ((key, "key"), (value, "value")):
                        destination[gpu_count:count].copy_(retained[0][name])
                # Check persistence through the actual manager, not just a temporary view.
                with native_slab(backend, info, count) as (key, value):
                    if not torch.equal(key[gpu_count:count].cpu(), retained[0]["key"]) or not torch.equal(value[gpu_count:count].cpu(), retained[0]["value"]):
                        raise UnsupportedReuse("native KV copy did not persist")
                history_output = torch.cat((history_output, retained[1]["key"].transpose(0, 1).to(execution_device)), dim=1)
                import_seconds = perf_counter() - started
        except Exception as error:
            if state.cache.active:
                state.cache.release()
            if gpu_slot is not None and gpu_slot in state.gpu.slots:
                # Failed input/import cannot leave a stale historical prefix.
                state.gpu.slots.remove(gpu_slot)
                invalidate_session(memory, handle)
            return _fallback(backend, inputs, hypo_ids, info, uncached, error)

        try:
            started = perf_counter()
            if count:
                suffix = inputs[:, count:]
                changed = _suffix_metadata(info, count, suffix.shape[1], inputs.device)
                with native_slab(backend, info, requested) as (key, value):
                    with native_execution(key, value) as attention:
                        suffix_output, _ = uncached(suffix, hypo_ids, changed)
                output = restore_complete_output(history_output.to(suffix_output.device), suffix_output, requested)
                keep = torch.arange(requested, device=output.device).unsqueeze(0)
                backend._last_keep_indices = keep
                result = (output, keep)
            else:
                result = uncached(inputs, hypo_ids, replace(info, dynamo_cache=None))
                output = result[0]
                attention = {"observed": False}
            if output.shape != inputs.shape:
                raise RuntimeError("ordinary RPC cache changed the output tensor contract")
            compute_restore_seconds = perf_counter() - started
        finally:
            if state.cache.active:
                state.cache.release()
            # Any partial model failure invalidates acquired history until a
            # successful retain republishes it. The session still owns its slab.
            if gpu_slot is not None and gpu_slot in state.gpu.slots:
                state.gpu.slots.remove(gpu_slot)
                invalidate_session(memory, handle)
        started = perf_counter()
        stored = _retain(backend, info, state, config, inputs, output)
        gpu_stored = None
        try:
            with backend.cache_manager.use_cache(*info.cache_handles) as native:
                gpu_stored = state.gpu.retain(native[0], config["token_ids"], inputs, output,
                                              config.get("priorities"), handle)
            if gpu_stored is not None:
                register_session(memory, state.gpu, gpu_stored, handle)
        except Exception as error:
            stored["gpu_store_error"] = type(error).__name__ + ": " + str(error)
        put_seconds = perf_counter() - started
        with native_slab(backend, info, requested) as (key, value):
            destination_storage = storage_id(key)
            destination_value_storage = storage_id(value)
        cpu_bytes = state.cache._bytes(retained) if retained else 0
        kv_cpu_bytes = state.cache._bytes({0: retained[0]}) if retained else 0
        _emit(backend, mode="reuse" if count else "cold", requested_tokens=requested,
              computed_tokens=requested-count, skipped_tokens=count, returned_tokens=output.shape[1],
              native_import_exact=bool(cpu_count), cpu_blocks=state.cache.used_blocks,
              native_layers_imported=int(bool(cpu_count)),
              gpu_hit_tokens=gpu_count, cpu_hit_tokens=cpu_count,
              cpu_read_range=[gpu_count, count], cpu_read_payload_bytes=cpu_bytes,
              cpu_kv_h2d_bytes=kv_cpu_bytes, cpu_import_verification_d2h_bytes=kv_cpu_bytes,
              gpu_prefix_copy_bytes=0, native_attention_in_place=bool(attention["observed"]),
              source_key_storage=source_storage, destination_key_storage=destination_storage,
              source_value_storage=source_value_storage, destination_value_storage=destination_value_storage,
              source_gpu_slot_id=source_slot_id,
              gpu_slot_id=gpu_stored.serial if gpu_stored is not None else None,
              gpu_resident_payload_bytes=state.gpu.resident_bytes,
              gpu_budget_bytes=state.gpu.budget_bytes,
              active_session_reserved_tokens=memory.current_size_tokens,
              active_session_token_budget=memory.max_size_tokens,
              gpu_evictions=state.gpu.evictions[event_eviction_start:],
              gpu_reclamation_unit="native_dense_slab", tail_forward_required=True,
              cache_methods=["get_range"] + (["put_completed"] if stored["cache_put_called"] else []),
              full_cache_hit=(count == requested), cache_get_seconds=get_seconds,
              native_import_seconds=import_seconds, compute_restore_seconds=compute_restore_seconds,
              cache_put_seconds=put_seconds,
              auxiliary_activation_layer=True, **stored)
        return result
