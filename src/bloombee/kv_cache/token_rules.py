"""Experimental E/G/H prefix-tree rules, derived from the tested research adapter.

Only free leaves are evictable. Inputs contain completed/observed history, never
future requests. This module has no model, tensor, network or filesystem imports.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from typing import Optional


def nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def retention_priority(value):
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError("priority must be an integer from 0 to 100")
    return value


@dataclass(frozen=True)
class RetentionRange:
    """Half-open prompt token range; duration is seconds from first admission.

    At expiry priority reverts to the cache's default. This is not KV data TTL.
    Endpoints must align to block_size; overlaps and partial blocks are rejected.
    """
    start: int
    end: int
    priority: int
    duration_s: Optional[float] = None

    def __post_init__(self):
        nonnegative_int(self.start, "range start")
        nonnegative_int(self.end, "range end")
        if self.end <= self.start:
            raise ValueError("range end must exceed start")
        retention_priority(self.priority)
        if self.duration_s is not None:
            if isinstance(self.duration_s, bool) or not isinstance(self.duration_s, (int, float)):
                raise ValueError("duration_s must be a finite nonnegative number or None")
            if not isfinite(self.duration_s) or self.duration_s < 0:
                raise ValueError("invalid duration_s")


def normalize_ranges(ranges, block_size):
    result = []
    for item in ranges:
        if isinstance(item, dict):
            item = RetentionRange(**item)
        if not isinstance(item, RetentionRange):
            raise ValueError("e_ranges must contain RetentionRange objects or mappings")
        if item.start % block_size or item.end % block_size:
            raise ValueError("retention range must align to block_size")
        result.append(item)
    result.sort(key=lambda item: item.start)
    if any(left.end > right.start for left, right in zip(result, result[1:])):
        raise ValueError("overlapping retention ranges are unsupported")
    return tuple(result)


@dataclass
class _Node:
    parent: object
    depth_tokens: int
    history_tokens: int
    last_access: int
    released: int
    hint: object = None
    admitted_at: float = 0.0
    children: set = field(default_factory=set)


class TokenRules:
    """Serial leaf-only rules; capacity is a count of full token blocks.

    G uses reverse order within each completion batch, after older free batches.
    H uses max(observed L + growth - budget, 0); L does not shrink on eviction.
    E uses priority then LRU; an ordinary hit never renews a duration.
    """
    def __init__(self, method, capacity_blocks, block_size, *, default_priority=35,
                 h_threshold_tokens=128, h_next_prompt_tokens=16):
        if method not in ("E", "G", "H"):
            raise ValueError("token rule must be E, G or H")
        self.method = method
        self.capacity = nonnegative_int(capacity_blocks, "capacity_blocks")
        self.block_size = nonnegative_int(block_size, "block_size")
        if not self.capacity or not self.block_size:
            raise ValueError("capacity_blocks and block_size must be positive")
        self.default_priority = retention_priority(default_priority)
        self.h_threshold = nonnegative_int(h_threshold_tokens, "h_threshold_tokens")
        self.h_growth = nonnegative_int(h_next_prompt_tokens, "h_next_prompt_tokens")
        self.nodes = {}
        self.sequence = 0

    def copy(self):
        return deepcopy(self)

    def _priority(self, node, now):
        if node.hint is None:
            return self.default_priority
        duration = node.hint.duration_s
        if duration is not None and now >= node.admitted_at + duration:
            return self.default_priority
        return node.hint.priority

    def _score(self, key, now):
        node = self.nodes[key]
        if self.method == "G":
            return (node.released,)
        if self.method == "E":
            return (self._priority(node, now), node.last_access)
        needed = max(node.history_tokens + self.h_growth - self.h_threshold, 0)
        safe = node.depth_tokens - self.block_size >= needed
        return (-1 if safe else 0, node.last_access)

    def _touch_and_release(self, path):
        self.sequence += 1
        for key in path:
            self.nodes[key].last_access = self.sequence
        for key in reversed(path):
            self.sequence += 1
            self.nodes[key].released = self.sequence

    def touch(self, path):
        """Touch existing usable prefix only; no prediction or history reset."""
        if any(key not in self.nodes for key in path):
            raise ValueError("cannot touch a missing prefix")
        self._touch_and_release(path)

    def record(self, path, now, hints=()):
        """Record a completed request, protecting its whole path during admission.

        All hints are current-known explicit inputs. Shared blocks retain their
        first-admission hint; conflicting hints are rejected instead of guessed.
        """
        if not path or len(path) > self.capacity:
            raise ValueError("completed prefix must fit capacity")
        if hints and len(hints) != len(path):
            raise ValueError("one optional hint per block required")
        hints = hints or (None,) * len(path)
        for key, hint in zip(path, hints):
            if key in self.nodes and self.nodes[key].hint != hint:
                raise ValueError("conflicting hint on resident shared prefix")
        pinned = set(path)
        evicted = []
        for index, (key, hint) in enumerate(zip(path, hints)):
            if key not in self.nodes:
                if len(self.nodes) >= self.capacity:
                    candidates = [candidate for candidate, node in self.nodes.items()
                                  if not node.children and candidate not in pinned]
                    if not candidates:
                        raise RuntimeError("no unprotected leaf available")
                    victim = min(candidates, key=lambda candidate: self._score(candidate, now))
                    node = self.nodes.pop(victim)
                    if node.parent is not None:
                        self.nodes[node.parent].children.remove(victim)
                    evicted.append(victim)
                parent = path[index - 1] if index else None
                self.nodes[key] = _Node(parent, (index + 1) * self.block_size,
                                        len(path) * self.block_size, self.sequence,
                                        self.sequence, hint, now)
                if parent is not None:
                    self.nodes[parent].children.add(key)
            node = self.nodes[key]
            node.history_tokens = max(node.history_tokens, len(path) * self.block_size)
        self._touch_and_release(path)
        return evicted
