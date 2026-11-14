"""Additional whole-prefix CPU history options using the existing payload contract.

These experimental options retain serialized bytes for one serial consumer, not
native GPU slabs. Request completion, identity/expiry checks, copy ownership and
staging capacity are inherited from :mod:`.placement`; no baseline copy or model
adapter is embedded here.
"""

import math
import time
from collections import Counter

from .placement import Demand, Placement

HISTORY_POLICIES = ("lfu", "slru", "fifo", "filo", "mru")


class HistoryCache(Placement):
    """Select ``lfu``, ``slru``, ``fifo``, ``filo`` or ``mru`` for CPU history.

    Use ``begin(group, request_id)``, ``request.bind_fetcher(source_fetch)``
    and ``request.finish(reuse_success=...)`` just as with Placement. Only a
    complete successful 32-layer reuse commits heat. Duplicate successful IDs
    update recency but do not increment frequency or SLRU promotion hits.

    LFU uses lifetime successful-request frequency (including nonresident keys),
    breaks ties by LRU, and refuses to displace a hotter resident for a colder
    incoming key. SLRU inserts into probation and promotes after promotion_hits
    fresh successful history hits. Protected bytes are limited to the configured
    fraction; demotion preserves last-success time. FIFO/FILO use first insertion
    in the current residency; MRU uses most recent successful use.

    Capacity includes retained and admitted staging payload bytes only, not
    metadata/model working tensors. Evictions reserve space at begin, so a later
    failure does not restore evicted history. The inherited single-active-request
    guard protects in-use bytes. Lifetime dedup/frequency/event metadata is
    unbounded, matching the research adapter's finite-run scope.
    """
    def __init__(self, policy, capacity_bytes, *, window_s=60.0, clock=time.time,
                 protected_fraction=0.5, promotion_hits=1):
        if policy not in HISTORY_POLICIES:
            raise ValueError("unknown policy")
        if not math.isfinite(protected_fraction) or not 0 <= protected_fraction <= 1:
            raise ValueError("invalid protected fraction")
        if type(promotion_hits) is not int or promotion_hits < 1:
            raise ValueError("invalid promotion hit threshold")
        super().__init__("lru", capacity_bytes,
                         window_s=window_s, clock=clock)
        self.policy = policy
        self.frequency = Counter()  # lifetime unique successful requests, including nonresident
        self.inserted = {}        # current residency's first insertion sequence
        self.history_hits = Counter()  # current residency's fresh successful hits
        self.protected = set()
        self.protected_capacity = int(capacity_bytes * protected_fraction)
        self.promotion_hits = promotion_hits

    def _drop(self, key, reason):
        super()._drop(key, reason)
        self.inserted.pop(key, None)
        self.history_hits.pop(key, None)
        self.protected.discard(key)

    def _ranked(self):
        def rank(key):
            recent = self.entries[key][2]
            if self.policy == "lfu":
                return self.frequency[key], recent, key
            if self.policy == "slru":
                return int(key in self.protected), recent, key
            if self.policy == "fifo":
                return self.inserted[key], key
            if self.policy == "filo":
                return -self.inserted[key], key
            return -recent, key  # MRU: most recent successful demand first
        return sorted(self.entries, key=rank)

    def begin(self, group, request_id):
        started = time.perf_counter()
        event_start = len(self.events)
        if self.active is not None:
            raise RuntimeError("one active demand only")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request id required")
        now = self._now()
        for key, (old, _, _) in list(self.entries.items()):
            if old.expiry <= now:
                self._drop(key, "ttl")
        if group.expiry <= now:
            raise ValueError("expired group")
        key = group.key
        hit = key in self.entries
        admit = group.nbytes <= self.capacity
        if not hit and admit:
            available, victims = self.capacity - self.resident_bytes, []
            for victim in self._ranked():
                if available >= group.nbytes:
                    break
                # Same observed-history gate as baseline frequency policies.
                # The pending request's possible success is not counted yet.
                if self.policy == "lfu" and self.frequency[victim] > self.frequency[key]:
                    admit = False
                    break
                victims.append(victim)
                available += self.entries[victim][0].nbytes
            if available < group.nbytes:
                admit = False
            if admit:
                for victim in victims:
                    self._drop(victim, "capacity")
        session = _HistoryDemand(self, group, request_id, hit, admit and not hit)
        session.started, session.event_start = started, event_start
        self.active = session
        return session

    def _after_finish(self, row):
        key = row["key"]
        if row["success_increment"]:
            self.frequency[key] += 1
        if row["reuse_success"] and key in self.entries:
            if not row["hit"]:
                self.inserted[key] = self.sequence
                self.history_hits[key] = 0
            elif row["success_increment"]:
                self.history_hits[key] += 1
                if self.policy == "slru" and self.history_hits[key] >= self.promotion_hits:
                    self.protected.add(key)
            if self.policy == "slru":
                while sum(self.entries[k][0].nbytes for k in self.protected) > self.protected_capacity:
                    demote = min(self.protected, key=lambda k: (self.entries[k][2], k))
                    self.protected.remove(demote)
                    self.events.append(dict(event="demote", key=demote, reason="protected_byte_budget"))


class _HistoryDemand(Demand):
    def finish(self, *, reuse_success):
        row = super().finish(reuse_success=reuse_success)
        self.owner._after_finish(row)
        return row
