"""Synchronous CPU-replica adaptation of Dynamo v1.6.2's native rules.

Reference commit: 28f9765d98ded77149a94d0defee6d54e97f6741.
The rule sources match v1.6.1, the original model validation reference.
The binding's slot.rs stores per-block priorities, stops at the first below-
threshold block, and keeps that termination across scheduling chunks. Its managed
pool orders inactive BasicMetadata by priority, returned_tick, acquired_tick.

This module ports those rules, not Dynamo's Rust/NIXL runtime. Completed GPU
blocks remain caller-owned; independent CPU copies may coexist immediately.
Tensor layout, explicit compatibility contexts, synchronous transactions, and
SHA integrity checks are BloomBee adaptations. No disk or object-store tier.
"""

from dataclasses import dataclass, replace
import hashlib

import torch

__all__ = ["DynamoCache"]


@dataclass(frozen=True)
class _Block:
    layers: dict
    priority: int
    returned_tick: int
    acquired_tick: int
    digest: str
    nbytes: int


@dataclass(frozen=True)
class _Request:
    ids: tuple
    evaluated_blocks: int
    priorities: tuple
    terminated_at: object


@dataclass(frozen=True)
class _CompletedBlocks:
    reader: object


def _integer(value, name, *, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in the supported range")
    return value


class DynamoCache:
    """Blockwise proactive CPU replicas, with explicit request and read leases.

    capacity_blocks counts fixed-size token blocks across all expected layers.
    context must identify model, revision, execution configuration and schema.
    IDs are actual ordered nonnegative token IDs, not standalone suffix hashes.
    Each put supplies the entire computed prefix as time-major dense tensors.
    Unfinished token tails are reported and deferred to the next cumulative put.

    With a request_id, final=False retains filtering progress and future hints;
    subsequent puts must extend exactly the same prefix. A first priority below
    the configured threshold permanently ends offload for that request. Missing
    hints are zero. Priorities on duplicate CPU blocks keep their first value,
    matching this adapter's first-copy retention choice.

    One get lease may be active; writes may still use free capacity or reclaim
    other idle blocks. Calling release returns matched blocks to the pool in
    prefix order, updating returned_tick. Match/get does not stamp acquired_tick;
    acquiring new/free physical capacity during put stamps acquired_tick.
    Thread concurrency, asynchronous mutation of supplied tensors,
    and in-place mutation of retained internal snapshots are unsupported.

    CPU capacity excludes the caller's GPU/slab, returned working copies,
    verification temporaries and allocator overhead. No claim of pinned memory
    or whole-process memory limits. Tensor/content checks do not prove that a
    caller supplied the correct context for its real model.
    """

    experimental = True
    numerical_status = "native-rule adapter; model numerical validation is a separate gate"

    def __init__(self, capacity_blocks, block_size=16, *, context,
                 expected_layers=32, host_offload_min_priority=0):
        self.capacity_blocks = _integer(capacity_blocks, "capacity_blocks")
        self.block_size = _integer(block_size, "block_size")
        if not self.block_size:
            raise ValueError("block_size must be positive")
        if not isinstance(context, str) or not context:
            raise ValueError("nonempty model/execution compatibility context is required")
        self.context = context
        if type(expected_layers) is int:
            if expected_layers <= 0:
                raise ValueError("expected_layers must be positive")
            self.expected_layers = tuple(range(expected_layers))
        else:
            self.expected_layers = tuple(expected_layers)
            if not self.expected_layers or any(type(i) is not int or i < 0 for i in self.expected_layers):
                raise ValueError("expected_layers must identify nonnegative layer IDs")
            if len(set(self.expected_layers)) != len(self.expected_layers):
                raise ValueError("duplicate layer IDs")
        self.host_offload_min_priority = _integer(
            host_offload_min_priority, "host_offload_min_priority", maximum=100)
        self.entries = {}
        self.events = []
        self._requests = {}
        self._active_keys = ()
        self._schema = None
        self._tick = 0

    @property
    def used_blocks(self):
        return len(self.entries)

    @property
    def active(self):
        return bool(self._active_keys)

    def resident_bytes(self):
        return sum(block.nbytes for block in self.entries.values())

    def _context(self, context):
        if context is not None and context != self.context:
            raise ValueError("model/execution context mismatch")

    @staticmethod
    def _ids(ids):
        result = tuple(ids)
        if any(type(token) is not int or token < 0 for token in result):
            raise ValueError("token IDs must be nonnegative Python integers")
        return result

    def _path(self, ids):
        return tuple((self.context, ids[:end])
                     for end in range(self.block_size, len(ids) + 1, self.block_size))

    def _validate_layers(self, ids, layers):
        if not isinstance(layers, dict) or set(layers) != set(self.expected_layers):
            raise ValueError("layers must exactly match expected_layers")
        schema = {}
        for layer_id in self.expected_layers:
            pair = layers[layer_id]
            if not isinstance(pair, dict) or set(pair) != {"key", "value"}:
                raise ValueError("each layer must contain key and value tensors")
            schema[layer_id] = {}
            for name, tensor in pair.items():
                if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
                    raise ValueError("dense torch tensors are required")
                if tensor.device.type not in ("cpu", "cuda"):
                    raise ValueError("only CPU/CUDA input tensors are supported")
                if (tensor.ndim < 2 or tensor.shape[0] != len(ids)
                        or any(dimension <= 0 for dimension in tensor.shape[1:])):
                    raise ValueError("time-major tensor shape must match all supplied IDs")
                schema[layer_id][name] = (tuple(tensor.shape[1:]), tensor.dtype)
        if self._schema is not None and schema != self._schema:
            raise ValueError("tensor schema changed within compatibility context")
        return schema

    def _digest(self, layers):
        digest = hashlib.sha256()
        for layer_id in self.expected_layers:
            for name in ("key", "value"):
                tensor = layers[layer_id][name].detach().to("cpu").contiguous()
                digest.update(repr((layer_id, name, tuple(tensor.shape), str(tensor.dtype))).encode())
                digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _bytes(layers):
        return sum(tensor.numel() * tensor.element_size()
                   for pair in layers.values() for tensor in pair.values())

    def _slice(self, layers, start, end):
        return {i: {name: layers[i][name][start:end] for name in ("key", "value")}
                for i in self.expected_layers}

    def _copy_block(self, layers, start, end):
        return {i: {name: layers[i][name][start:end].detach().to("cpu", copy=True).contiguous()
                    for name in ("key", "value")} for i in self.expected_layers}

    @staticmethod
    def _request_id(request_id):
        if request_id is not None and (not isinstance(request_id, str) or not request_id):
            raise ValueError("request_id must be a nonempty string or None")

    @staticmethod
    def _priorities(previous, priorities):
        result = list(previous)
        if priorities is not None:
            if isinstance(priorities, (str, bytes, dict)):
                raise ValueError("priorities must be a position sequence or None")
            try:
                updates = tuple(priorities)
            except TypeError as error:
                raise ValueError("priorities must be a position sequence or None") from error
            for index, priority in enumerate(updates):
                value = 0 if priority is None else _integer(priority, "block priority", maximum=100)
                if index < len(result):
                    result[index] = value
                else:
                    result.append(value)
        return tuple(result)

    def put_completed(self, ids, layers, *, priorities=None, request_id=None,
                      final=True, context=None):
        """Proactively copy newly complete qualifying blocks to the CPU pool.

        Whole-batch capacity/copy failure leaves both pool and request progress
        unchanged. On safe-capacity refusal, admitted=False permits an identical
        retry after active reads finish. Otherwise a first failing priority is
        remembered even if later scheduling chunks contain high priorities.
        """
        self._context(context)
        self._request_id(request_id)
        if type(final) is not bool or (request_id is None and not final):
            raise ValueError("nonfinal chunk requires request_id; final must be bool")
        ids = self._ids(ids)
        block_source = isinstance(layers, _CompletedBlocks)
        schema = self._schema if block_source else self._validate_layers(ids, layers)
        previous = self._requests.get(request_id, _Request((), 0, (), None))
        if len(ids) < len(previous.ids) or ids[:len(previous.ids)] != previous.ids:
            raise ValueError("request chunks must extend the original complete prefix")
        hints = self._priorities(previous.priorities, priorities)
        path = self._path(ids)
        count = len(path)
        terminated_at = previous.terminated_at
        candidates = []
        if terminated_at is None:
            for index in range(previous.evaluated_blocks, count):
                priority = hints[index] if index < len(hints) else 0
                if self.host_offload_min_priority > 0 and priority < self.host_offload_min_priority:
                    terminated_at = index
                    break
                candidates.append((index, path[index], priority))
        protected = set(self._active_keys) | {key for _, key, _ in candidates}
        incoming = [(index, key, priority) for index, key, priority in candidates if key not in self.entries]
        needed = max(0, len(self.entries) + len(incoming) - self.capacity_blocks)
        from bloombee.kv_cache.retention_completion import order, snapshot
        import time
        retention_now = time.monotonic()
        retention_view = snapshot(retention_now)
        ranked = sorted(
            (key for key in self.entries if key not in protected),
            key=lambda key: order(key, (self.entries[key].priority, self.entries[key].returned_tick,
                             self.entries[key].acquired_tick, key), view=retention_view, now=retention_now),
        )
        event = dict(event="offload", admitted=False, request_id=request_id,
                     new_blocks=0, deduplicated_blocks=len(candidates) - len(incoming),
                     offloaded_tokens=0, eligible_tokens=len(candidates) * self.block_size,
                     terminated_at_block=terminated_at, ignored_tail_tokens=len(ids) % self.block_size,
                     evicted=[], copied_bytes=0, resident_bytes=self.resident_bytes())
        if needed > len(ranked):
            event["reason"] = "no_idle_capacity"
            return event
        evicted = ranked[:needed]
        evicted_metadata = [{"key": key, "priority": self.entries[key].priority,
                             "returned_tick": self.entries[key].returned_tick,
                             "acquired_tick": self.entries[key].acquired_tick} for key in evicted]
        planned = {key: block for key, block in self.entries.items() if key not in evicted}
        tick = self._tick
        copied_bytes = 0
        for index, key, priority in incoming:
            start, end = index * self.block_size, (index + 1) * self.block_size
            source = layers.reader(index) if block_source else self._slice(layers, start, end)
            source_schema = self._validate_layers(ids[start:end], source)
            if schema is not None and source_schema != schema:
                raise ValueError("new block schema changed inside one transaction")
            schema = source_schema
            expected_digest = self._digest(source)
            copied = self._copy_block(source, 0, self.block_size)
            self._validate_layers(ids[start:end], copied)
            if self._digest(copied) != expected_digest:
                raise ValueError("CPU offload copy integrity mismatch")
            tick += 1
            acquired_tick = tick
            tick += 1
            nbytes = self._bytes(copied)
            planned[key] = _Block(copied, priority, tick, acquired_tick, expected_digest, nbytes)
            copied_bytes += nbytes
        if len(planned) > self.capacity_blocks:
            raise AssertionError("CPU block capacity exceeded")
        # Nothing fallible remains: publish pool, filtering progress, and schema.
        self.entries = planned
        self._tick = tick
        self._schema = schema
        if request_id is not None:
            if final:
                self._requests.pop(request_id, None)
            else:
                self._requests[request_id] = _Request(ids, count, hints, terminated_at)
        event.update(admitted=True, new_blocks=len(incoming), evicted=evicted,
                     new_block_indices=[index for index,key,priority in incoming],
                     evicted_metadata=evicted_metadata,
                     copied_bytes=copied_bytes, resident_bytes=self.resident_bytes(),
                     offloaded_tokens=len(candidates) * self.block_size)
        self.events.append(event)
        return event

    def put_completed_blocks(self, ids, read_block, *, priorities=None, request_id=None,
                             final=True, context=None):
        """Lazy complete-block source for native noncontiguous GPU pages.

        read_block(index) returns exactly one complete time-major block across
        expected_layers. It is invoked only for qualifying blocks not already
        retained on CPU, after capacity planning. No historical GPU K/V gather
        is required. The caller holds its source lease until this call returns.
        Reader/copy/schema failure leaves the CPU transaction unpublished.
        """
        if not callable(read_block):
            raise ValueError("read_block must be callable")
        return self.put_completed(ids, _CompletedBlocks(read_block), priorities=priorities,
                                  request_id=request_id, final=final, context=context)

    def get(self, ids, *, target_device="cpu", context=None):
        return self.get_range(ids, start_tokens=0, target_device=target_device, context=context)

    def get_range(self, ids, *, start_tokens=0, end_tokens=None, target_device="cpu", context=None):
        """Copy the longest compatible contiguous CPU prefix to the caller.

        A successful nonempty get pins its CPU blocks until release(). GPU input
        and returned GPU tensors are working copies, never the retained CPU pool.
        Integrity/transfer failure occurs before pins or metadata are published.
        """
        self._context(context)
        _integer(start_tokens, "start_tokens")
        if start_tokens % self.block_size:
            raise ValueError("CPU onboarding must start at a complete block boundary")
        if self.active:
            raise RuntimeError("release the current prefix before another get")
        ids = self._ids(ids)
        if start_tokens > len(ids):
            raise ValueError("CPU onboarding starts beyond the request")
        end_tokens = len(ids) if end_tokens is None else _integer(end_tokens, "end_tokens")
        if end_tokens < start_tokens or end_tokens > len(ids):
            raise ValueError("invalid CPU onboarding end")
        target = torch.device(target_device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("target device must be CPU or CUDA")
        keys = []
        for key in self._path(ids)[start_tokens // self.block_size:end_tokens // self.block_size]:
            if key not in self.entries:
                break
            block = self.entries[key]
            if self._digest(block.layers) != block.digest:
                raise ValueError("retained CPU block integrity mismatch")
            keys.append(key)
        skipped = len(keys) * self.block_size
        event = dict(event="get", hit=bool(keys), blocks=len(keys), skipped_tokens=skipped,
                     copy_bytes=0, source="cpu", start_tokens=start_tokens,
                     end_tokens=start_tokens + skipped)
        if not keys:
            self.events.append(event)
            return {}, 0, event
        cpu_prefix = {i: {name: torch.cat([self.entries[key].layers[i][name] for key in keys], dim=0)
                          for name in ("key", "value")} for i in self.expected_layers}
        layers = {i: {name: tensor.to(target, copy=True).contiguous()
                      for name, tensor in pair.items()} for i, pair in cpu_prefix.items()}
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        if self._digest(layers) != self._digest(cpu_prefix):
            raise ValueError("onboard copy integrity mismatch")
        self._active_keys = tuple(keys)
        event["copy_bytes"] = self._bytes(layers)
        self.events.append(event)
        return layers, skipped, event

    def release(self):
        """Return active CPU references, refreshing returned_tick in prefix order."""
        keys = self._active_keys
        for key in keys:
            self._tick += 1
            self.entries[key] = replace(self.entries[key], returned_tick=self._tick)
        self._active_keys = ()
        event = dict(event="release", released_blocks=len(keys))
        self.events.append(event)
        return event

    def finish_request(self, request_id):
        """Forget scheduling progress after completion/cancellation, retaining KV."""
        self._request_id(request_id)
        return self._requests.pop(request_id, None) is not None

    def clear(self):
        """Release local pool/request state only when no read references remain."""
        if self.active:
            raise RuntimeError("cannot clear active CPU blocks")
        self.entries = {}
        self._requests = {}
        self._schema = None
        self.events.append(dict(event="clear"))
