"""Whole-prefix historical replicas with byte-bounded exclusive tiers.

Adapted from the independently verified LRU/A/B/A+B research Store. This is
an experimental, serial cache of immutable snapshots, not the active inference
slab. The caller supplies identity/content validation and safe device movement.
No directory lookup, network transport, model loading, or telemetry publication
is performed here.
"""

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional

__all__ = ["CacheKey", "TieredCache", "Lease"]


@dataclass(frozen=True)
class CacheKey:
    """Optional strict key for callers sharing a pool.

    context identifies model/revision, execution schema and tenant namespace.
    digest is a SHA256 of the complete prefix identity AND content/version
    descriptor, never a bare final block hash. These fields are supplied by the
    directory/backend; the cache cannot derive model compatibility from tensors.
    validate(key, value) must still check the actual payload against the key.
    """

    context: str
    digest: str

    def __post_init__(self):
        if not isinstance(self.context, str) or not self.context:
            raise ValueError("key context must be a nonempty string")
        if (
            not isinstance(self.digest, str)
            or len(self.digest) != 64
            or any(char not in "0123456789abcdef" for char in self.digest)
        ):
            raise ValueError("key digest must be a lowercase SHA256")


@dataclass(frozen=True)
class Item:
    value: Any
    nbytes: int
    priority: int
    last: int


class TieredCache:
    """Selectable LRU, tiered LRU (A), priority (B), or tiered priority (A+B).

    size(value) returns positive integer payload bytes. move(value, "fast" or
    "slow") returns a safe target-tier snapshot, without changing its source.
    validate(key, value) raises on incompatible identity/schema or bad content.
    All callbacks must be serial, non-reentrant, and must not mutate the cache.

    Higher explicit priorities are retained longer; ties use successful-access
    LRU. New priorities default to zero; omitted priorities on hits preserve the
    existing value. admission_guard is the separately tested optional extension
    that prevents a low-priority admission from evicting a higher-priority item.

    A/B/A+B/LRU labels preserve the previously tested algorithms. LRU and B have
    no retained slow tier even if slow_bytes is supplied. A and A+B use the extra
    slow_bytes budget. Capacity counts retained payloads, excluding request
    workspace, temporary transfer copies, callback-owned memory and allocator
    overhead. Oversized successful requests may bypass retention.

    Without context, legacy hashable keys are accepted: the caller is entirely
    responsible for full compatibility and immutable identity. With context,
    only CacheKey instances from that exact namespace are accepted. No TTL is
    inferred; external validity changes must be reflected in validate or keys.
    """

    METHODS = ("LRU", "A", "B", "A+B")

    def __init__(
        self,
        method: str,
        fast_bytes: int,
        slow_bytes: int = 0,
        *,
        size: Callable[[Any], int],
        move: Callable[[Any, str], Any],
        validate: Callable[[Hashable, Any], None],
        admission_guard: bool = False,
        context: Optional[str] = None,
    ):
        if method not in self.METHODS:
            raise ValueError("unknown tiered-cache method")
        if any(type(value) is not int or value < 0 for value in (fast_bytes, slow_bytes)):
            raise ValueError("capacities must be nonnegative integer bytes")
        if not all(callable(callback) for callback in (size, move, validate)):
            raise TypeError("size, move and validate must be callable")
        if type(admission_guard) is not bool:
            raise TypeError("admission_guard must be bool")
        if context is not None and (not isinstance(context, str) or not context):
            raise ValueError("context must be a nonempty string or None")
        self.method = method
        self.cap = (fast_bytes, slow_bytes if method in ("A", "A+B") else 0)
        self.priority = method in ("B", "A+B")
        self.guard = admission_guard
        self.context = context
        self.size = size
        self.move = move
        self.validate = validate
        self.tiers = [{}, {}]
        self.pins = {}
        self.seq = 0
        self.stats = dict(
            fast_hit=0, slow_hit=0, source=0, source_bytes=0,
            demote_bytes=0, promote_bytes=0, evictions=0,
        )
        self.peak = [0, 0]
        self.events = []

    def used(self, index: int) -> int:
        if index not in (0, 1):
            raise ValueError("tier index must be 0 (fast) or 1 (slow)")
        return sum(item.nbytes for item in self.tiers[index].values())

    def _check_key(self, key):
        if key is None:
            raise ValueError("a compatible cache key is required")
        hash(key)
        if self.context is not None:
            if not isinstance(key, CacheKey) or key.context != self.context:
                raise ValueError("cache key does not belong to this context")

    def _nbytes(self, value):
        count = self.size(value)
        if type(count) is not int or count <= 0:
            raise ValueError("size must return positive integer bytes")
        return count

    def _move_checked(self, key, value, tier, expected_bytes):
        moved = self.move(value, tier)
        self.validate(key, moved)
        if self._nbytes(moved) != expected_bytes:
            raise ValueError("payload byte size changed during transfer")
        return moved

    def acquire(self, key, loader, priority=None):
        """Return a Lease whose value is ready in the fast/request tier.

        loader(key) is called only on a miss. Slow hits use the RETURNED copied
        fast object, not the original slow snapshot. Retention changes only when
        finish(True) confirms that the request actually consumed this snapshot.
        A callback failure leaves the retained dictionaries and counters intact.
        """
        self._check_key(key)
        if priority is not None and type(priority) is not int:
            raise ValueError("explicit priority must be an integer or None")
        for index, tier in enumerate(self.tiers):
            if key not in tier:
                continue
            item = tier[key]
            self.pins[key] = self.pins.get(key, 0) + 1
            try:
                if index == 0:
                    value = item.value
                    self.validate(key, value)
                    if self._nbytes(value) != item.nbytes:
                        raise ValueError("retained payload byte size changed")
                else:
                    value = self._move_checked(key, item.value, "fast", item.nbytes)
                return Lease(self, key, value, index, priority)
            except BaseException:
                self._unpin(key)
                raise
        value = loader(key)
        self.validate(key, value)
        count = self._nbytes(value)
        moved = self._move_checked(key, value, "fast", count)
        return Lease(self, key, moved, 2, priority)

    def _unpin(self, key):
        self.pins[key] -= 1
        if self.pins[key] == 0:
            del self.pins[key]

    def _commit(self, lease):
        key = lease.key
        self.validate(key, lease.value)
        count = self._nbytes(lease.value)
        if self.pins.get(key, 0):
            return False, count
        tiers = [dict(tier) for tier in self.tiers]
        old = next((tier[key] for tier in tiers if key in tier), None)
        priority = lease.priority
        if priority is None:
            priority = old.priority if old is not None else 0
        if not self.priority:
            priority = 0
        for tier in tiers:
            tier.pop(key, None)
        counters = dict(demote_bytes=0, promote_bytes=0, evictions=0)
        events = []

        def room(index, needed_bytes, incoming_priority):
            tier = tiers[index]
            ranked = sorted(tier, key=lambda k: (tier[k].priority, tier[k].last, repr(k)))
            for victim in ranked:
                if sum(item.nbytes for item in tier.values()) + needed_bytes <= self.cap[index]:
                    break
                item = tier[victim]
                protected = self.guard and item.priority > incoming_priority
                if self.pins.get(victim, 0) or protected:
                    continue
                del tier[victim]
                if index == 0 and item.nbytes <= self.cap[1]:
                    previous_slow = dict(tiers[1])
                    previous_counters = dict(counters)
                    previous_events = len(events)
                    if room(1, item.nbytes, item.priority):
                        moved = self._move_checked(victim, item.value, "slow", item.nbytes)
                        tiers[1][victim] = Item(moved, item.nbytes, item.priority, item.last)
                        counters["demote_bytes"] += item.nbytes
                        events.append(dict(key=repr(victim), direction="fast_to_slow", nbytes=item.nbytes))
                    else:
                        tiers[1] = previous_slow
                        counters.update(previous_counters)
                        del events[previous_events:]
                        counters["evictions"] += 1
                else:
                    counters["evictions"] += 1
            return sum(item.nbytes for item in tiers[index].values()) + needed_bytes <= self.cap[index]

        target = 0 if count <= self.cap[0] else 1
        if count > self.cap[target] or not room(target, count, priority):
            return False, count
        value = lease.value
        if target == 1:
            value = self._move_checked(key, value, "slow", count)
            counters["demote_bytes"] += count
            events.append(dict(key=repr(key), direction="request_to_slow", nbytes=count))
        elif lease.origin == 1:
            counters["promote_bytes"] += count
            events.append(dict(key=repr(key), direction="slow_to_fast", nbytes=count))
        tiers[target][key] = Item(value, count, priority, self.seq + 1)
        if tiers[0].keys() & tiers[1].keys():
            raise AssertionError("the same key cannot occupy both exclusive tiers")
        if any(sum(item.nbytes for item in tier.values()) > self.cap[i] for i, tier in enumerate(tiers)):
            raise AssertionError("retained payload capacity exceeded")
        # All potentially failing callbacks have finished before publication.
        self.tiers = tiers
        self.seq += 1
        for name, value in counters.items():
            self.stats[name] += value
        self.events.extend(events)
        for index in (0, 1):
            self.peak[index] = max(self.peak[index], self.used(index))
        return True, count


class Lease:
    """Read-only borrowed snapshot; origin 0=fast, 1=slow, 2=loader.

    finish(True) reports admission independently of successful serving: an
    oversized request may serve successfully without being retained. On any
    finish error this lease is closed, protection is released and the cache can
    be retried. The caller must never mutate value in place, including tensors.
    """

    def __init__(self, owner, key, value, origin, priority):
        self.owner = owner
        self.key = key
        self.value = value
        self.origin = origin
        self.priority = priority
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.closed:
            self.finish(False)

    def finish(self, success=True):
        if self.closed:
            raise RuntimeError("lease is already closed")
        if type(success) is not bool:
            raise TypeError("success must be bool")
        self.closed = True
        cache = self.owner
        if self.origin < 2:
            cache._unpin(self.key)
        if not success:
            return dict(admitted=False, origin=self.origin)
        admitted, count = cache._commit(self)
        cache.stats[("fast_hit", "slow_hit", "source")[self.origin]] += 1
        if self.origin == 2:
            # _commit already validated this value's byte count; avoid another
            # fallible callback after the retained state has been published.
            cache.stats["source_bytes"] += count
        return dict(admitted=admitted, origin=self.origin)
