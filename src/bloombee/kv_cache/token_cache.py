"""Experimental tensor-backed E/F/G/H options, with explicit caller contracts.

Migrated from the finite real-model TensorPrefixPool, not a claim of numerical
repair: its old strict three-logit equality gate had 45 retained failures. Local
CPU tests verify storage/rules only. No model generation or network occurs here.
"""
from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite

import torch

from .token_rules import TokenRules, normalize_ranges, nonnegative_int, retention_priority


@dataclass(frozen=True)
class _Replica:
    blocks: int
    layers: dict
    priority: int


def _layer_ids(expected_layers):
    if type(expected_layers) is int:
        if expected_layers <= 0:
            raise ValueError("expected_layers must be positive")
        return tuple(range(expected_layers))
    result = tuple(expected_layers)
    if not result or any(type(item) is not int or item < 0 for item in result):
        raise ValueError("expected_layers must be positive count or nonnegative layer IDs")
    if len(set(result)) != len(result):
        raise ValueError("duplicate expected layer ID")
    return result


class TokenCache:
    """Actual owned tensor retention with one active reuse lease.

    E/G/H: independently owned full token blocks and longest cached prefix.
    F: indivisible aligned whole-prefix replicas; fast/slow eviction is LRU,
    priority only gates offload (the tested TensorPrefixPool semantics). Slow
    offload happens only when a completed put needs fast space, never on every
    completion. A slow get copies to target without promoting or freeing slow.

    Tensors are time-major (tokens, ...), exactly matching the supplied token IDs.
    Layer IDs and each layer's K/V feature shapes are explicit; heterogeneous
    layer shapes are supported, but a context's schema cannot change later.
    Capacities count blocks, not bytes; resident_bytes reports owned payload bytes
    excluding working copies, temporary staging, allocator reserved and weights.
    """
    experimental = True
    numerical_status = "old strict logits-equality failures retained; not repaired"

    def __init__(self, method, capacity_blocks, *, context, expected_layers=32,
                 block_size=16, fast_device="cpu", slow_device="cpu",
                 slow_capacity_blocks=0, threshold=35, priority=35, e_ranges=(),
                 h_threshold_tokens=128, h_next_prompt_tokens=16,
                 allow_partial_block=False):
        if not isinstance(method, str) or method.upper() not in ("E", "F", "G", "H"):
            raise ValueError("method must be E, F, G or H")
        self.method = method.upper()
        self.capacity_blocks = nonnegative_int(capacity_blocks, "capacity_blocks")
        self.block_size = nonnegative_int(block_size, "block_size")
        if not self.capacity_blocks or not self.block_size:
            raise ValueError("capacity_blocks and block_size must be positive")
        if not isinstance(context, str) or not context:
            raise ValueError("nonempty compatibility context required")
        self.context = context
        self.expected_layers = _layer_ids(expected_layers)
        self.fast_device = torch.device(fast_device)
        self.slow_device = torch.device(slow_device)
        if self.fast_device.type not in ("cpu", "cuda") or self.slow_device.type != "cpu":
            raise ValueError("fast_device supports CPU/CUDA; F slow_device must be CPU")
        self.slow_capacity_blocks = nonnegative_int(slow_capacity_blocks, "slow_capacity_blocks")
        self.threshold = retention_priority(threshold)
        self.priority = retention_priority(priority)
        self.e_ranges = normalize_ranges(e_ranges, self.block_size)
        self.h_threshold_tokens = nonnegative_int(h_threshold_tokens, "h_threshold_tokens")
        self.h_next_prompt_tokens = nonnegative_int(h_next_prompt_tokens, "h_next_prompt_tokens")
        if type(allow_partial_block) is not bool:
            raise ValueError("allow_partial_block must be bool")
        self.allow_partial_block = allow_partial_block
        if self.method != "F" and (self.slow_capacity_blocks or threshold != 35):
            raise ValueError("slow capacity and offload threshold apply only to F")
        if self.method != "E" and self.e_ranges:
            raise ValueError("e_ranges apply only to E")
        if self.method not in ("E", "F") and priority != 35:
            raise ValueError("priority applies only to E/F")
        if self.method != "H" and (h_threshold_tokens != 128 or h_next_prompt_tokens != 16):
            raise ValueError("H budget and growth apply only to H")
        self.rules = None if self.method == "F" else TokenRules(
            self.method, self.capacity_blocks, self.block_size, default_priority=priority,
            h_threshold_tokens=h_threshold_tokens, h_next_prompt_tokens=h_next_prompt_tokens)
        self.payloads = {}
        self.fast = OrderedDict()
        self.slow = OrderedDict()
        self.events = []
        self.active = False
        self.last_time = float("-inf")
        self._schema = None

    def _time(self, now):
        if isinstance(now, bool) or not isinstance(now, (int, float)):
            raise ValueError("now must be a finite monotone number")
        if not isfinite(now) or now < self.last_time:
            raise ValueError("now must be finite and nondecreasing")

    def _ids(self, ids):
        result = tuple(ids)
        if any(type(item) is not int or item < 0 for item in result):
            raise ValueError("token IDs must be nonnegative Python integers")
        return result

    def _path(self, ids):
        return [(self.context, ids[:end])
                for end in range(self.block_size, len(ids) + 1, self.block_size)]

    def _validate_layers(self, ids, layers):
        if not isinstance(layers, dict) or set(layers) != set(self.expected_layers):
            raise ValueError("layers must exactly match expected_layers")
        schema = {}
        for index in self.expected_layers:
            pair = layers[index]
            if not isinstance(pair, dict) or set(pair) != {"key", "value"}:
                raise ValueError("each layer must contain key and value")
            schema[index] = {}
            for name, tensor in pair.items():
                if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
                    raise ValueError("dense torch tensors required")
                if tensor.ndim < 2 or tensor.shape[0] != len(ids):
                    raise ValueError("every time-major tensor must match the complete ID length")
                schema[index][name] = (tuple(tensor.shape[1:]), tensor.dtype)
        if self._schema is not None and schema != self._schema:
            raise ValueError("tensor schema changed within compatibility context")
        return schema

    def _copy_slice(self, layers, start, end, device):
        return {index: {name: tensor[start:end].detach().to(device, copy=True)
                        for name, tensor in layers[index].items()}
                for index in self.expected_layers}

    @staticmethod
    def _bytes(layers):
        return sum(tensor.numel() * tensor.element_size()
                   for pair in layers.values() for tensor in pair.values())

    def resident_bytes(self):
        """Actual owned tensor payload bytes; includes both F tiers separately."""
        if self.rules is not None:
            return sum(self._bytes(layers) for layers in self.payloads.values())
        return sum(self._bytes(replica.layers)
                   for tier in (self.fast, self.slow) for replica in tier.values())

    def _usage(self, tier):
        return sum(replica.blocks for replica in tier.values())

    def put_completed(self, ids, layers, *, now, ranges=None, priority=None):
        """Admit only a successful immutable completed snapshot, never a live slab.

        Nonaligned puts reject by default. allow_partial_block=True explicitly
        retains only complete blocks and reports the ignored tail in the event.
        E/G/H shared resident blocks keep their first tensor copy; repeated puts
        do not replace it or renew E duration. F replaces a whole identical key.
        """
        if self.active:
            raise RuntimeError("active reuse lease protects retained storage")
        self._time(now)
        ids = self._ids(ids)
        schema = self._validate_layers(ids, layers)
        remainder = len(ids) % self.block_size
        if remainder and not self.allow_partial_block:
            raise ValueError("unaligned completed prefix; opt in to allow_partial_block")
        count = len(ids) // self.block_size
        if count == 0 or count > self.capacity_blocks:
            raise ValueError("completed prefix must have full blocks and fit fast capacity")
        if ranges is not None and self.method != "E":
            raise ValueError("per-request ranges apply only to E")
        if priority is not None and self.method != "F":
            raise ValueError("per-request priority applies only to F; E uses ranges")
        hints = self.e_ranges if ranges is None else normalize_ranges(ranges, self.block_size)
        if any(hint.end > count * self.block_size for hint in hints):
            raise ValueError("retention range exceeds known complete prefix")
        replica_priority = self.priority if priority is None else retention_priority(priority)
        retained_ids = ids[:count * self.block_size]
        before = self.resident_bytes()
        evicted = []
        moved = 0
        if self.rules is not None:
            path = self._path(retained_ids)
            block_hints = [next((hint for hint in hints
                                 if hint.start <= i * self.block_size and
                                 (i + 1) * self.block_size <= hint.end), None)
                           for i in range(count)]
            planned_rules = self.rules.copy()
            evicted = planned_rules.record(path, now, block_hints)
            # Prepare all copies before replacing either decision state or pool.
            incoming = {key: self._copy_slice(layers, i * self.block_size,
                                               (i + 1) * self.block_size, self.fast_device)
                        for i, key in enumerate(path) if key not in self.payloads}
            planned_payloads = {key: value for key, value in self.payloads.items()
                                if key in planned_rules.nodes}
            planned_payloads.update(incoming)
            self.rules, self.payloads = planned_rules, planned_payloads
        else:
            key = (self.context, retained_ids)
            fresh = self._copy_slice(layers, 0, len(retained_ids), self.fast_device)
            planned_fast, planned_slow = OrderedDict(self.fast), OrderedDict(self.slow)
            planned_fast.pop(key, None)
            planned_slow.pop(key, None)
            while self._usage(planned_fast) + count > self.capacity_blocks:
                victim, old = planned_fast.popitem(last=False)
                evicted.append(victim)
                if old.priority >= self.threshold and old.blocks <= self.slow_capacity_blocks:
                    while self._usage(planned_slow) + old.blocks > self.slow_capacity_blocks:
                        planned_slow.popitem(last=False)
                    cpu = self._copy_slice(old.layers, 0, len(victim[1]), self.slow_device)
                    planned_slow[victim] = _Replica(old.blocks, cpu, old.priority)
                    moved += self._bytes(cpu)
            planned_fast[key] = _Replica(count, fresh, replica_priority)
            self.fast, self.slow = planned_fast, planned_slow
        self.last_time = now
        self._schema = schema
        row = dict(event="put", method=self.method, experimental=True,
                   retained_bytes_before=before, retained_bytes_after=self.resident_bytes(),
                   retained_prefix_tokens=len(retained_ids), ignored_tail_tokens=remainder,
                   evicted_prefix_lengths=[len(key[1]) for key in evicted],
                   fast_to_slow_copy_bytes=moved,
                   fast_device=str(self.fast_device), slow_device=str(self.slow_device),
                   fast_blocks=len(self.payloads) if self.rules is not None else self._usage(self.fast),
                   slow_blocks=0 if self.rules is not None else self._usage(self.slow))
        self.events.append(row)
        return row

    def get(self, ids, *, now, target_device=None):
        """Return (independent layers or None, reusable tokens, evidence).

        A hit opens a lease: release in finally after importer/generation. Caller
        must rebuild the identity for the returned token count. F slow hits copy
        but do not promote; misses do not open a lease. Query tail tokens are
        explicitly reported and are never represented by incomplete cache blocks.
        """
        if self.active:
            raise RuntimeError("one active reuse lease at a time")
        self._time(now)
        ids = self._ids(ids)
        target = self.fast_device if target_device is None else torch.device(target_device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("target_device supports CPU/CUDA")
        path = self._path(ids)
        n = 0
        layers = None
        source = "miss"
        copied = 0
        if self.rules is not None:
            usable = []
            for key in path:
                if key not in self.payloads:
                    break
                usable.append(key)
            if usable:
                layers = {index: {name: torch.cat([self.payloads[key][index][name]
                                                   for key in usable], dim=0).to(target)
                                  for name in ("key", "value")}
                          for index in self.expected_layers}
                n = len(usable) * self.block_size
                source = "retained_token_tensor"
                self.rules.touch(usable)
        else:
            key = (self.context, ids[:len(ids) // self.block_size * self.block_size])
            if key in self.fast:
                replica = self.fast[key]
                layers = self._copy_slice(replica.layers, 0, len(key[1]), target)
                self.fast.move_to_end(key)
                source = "retained_whole_fast"
                n = len(key[1])
            elif key in self.slow:
                replica = self.slow[key]
                layers = self._copy_slice(replica.layers, 0, len(key[1]), target)
                source = "retained_whole_slow"
                n = len(key[1])
                copied = self._bytes(layers)
        self.active = bool(n)
        self.last_time = now
        row = dict(event="get", method=self.method, experimental=True, source=source,
                   prefix_tokens=n, query_tail_tokens=len(ids) % self.block_size,
                   slow_to_target_copy_bytes=copied,
                   target_device=str(target),
                   expected_layers=list(self.expected_layers))
        self.events.append(row)
        return layers, n, row

    def release(self):
        if not self.active:
            raise RuntimeError("no active reuse lease")
        self.active = False
