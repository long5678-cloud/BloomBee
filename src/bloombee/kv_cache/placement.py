"""Whole-prefix bounded CPU placement. No torch, disk writes, or network policy.

Single consumer; every successful request commits ONE heat event, not 32.
Capacity covers retained raw payload + admitted in-flight staging payload.
Caller/model working tensors and the currently fetched HTTP blob are separate.
"""

import hashlib
import json
import math
import struct
import time
from collections import Counter, deque
from dataclasses import dataclass
from functools import cached_property


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Layer:
    identity_json: str
    sha256: str
    nbytes: int
    expires_at: float
    endpoint: str

    @classmethod
    def from_record(cls, identity, record):
        identity = identity.to_dict() if hasattr(identity, "to_dict") else identity
        record = record.to_dict() if hasattr(record, "to_dict") else record
        if record["identity_digest"] != digest(identity):
            raise ValueError("record/identity mismatch")
        return cls(
            json.dumps(identity, sort_keys=True, separators=(",", ":")),
            record["payload_sha256"],
            record["payload_nbytes"],
            record["expires_at"],
            record["endpoint"],
        )

    @property
    def identity(self):
        return json.loads(self.identity_json)


@dataclass(frozen=True)
class Group:
    layers: tuple

    def __post_init__(self):
        if len(self.layers) != 32:
            raise ValueError("exactly 32 layers required")
        base = None
        for i, layer in enumerate(self.layers):
            ident = layer.identity
            if ident.pop("layer_start") != i or ident.pop("layer_end") != i + 1:
                raise ValueError("incomplete/out-of-order layer coverage")
            if base is None:
                base = ident
            elif base != ident:
                raise ValueError("mixed prefix/model/execution identities")
            if (
                type(layer.nbytes) is not int
                or layer.nbytes <= 0
                or not isinstance(layer.expires_at, (int, float))
                or not math.isfinite(layer.expires_at)
                or len(layer.sha256) != 64
            ):
                raise ValueError("invalid size/expiry/digest")
            int(layer.sha256, 16)
        if len({x.sha256 for x in self.layers}) != 32:
            raise ValueError("duplicate layer objects")

    @cached_property
    def key(self):
        return digest([(x.identity_json, x.sha256, x.nbytes) for x in self.layers])

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self.layers)

    @property
    def expiry(self):
        return min(x.expires_at for x in self.layers)


def verify(layer, blob, now):
    if layer.expires_at <= now:
        raise ValueError("expired layer")
    if type(blob) is not bytes or len(blob) != layer.nbytes:
        raise ValueError("payload size/type mismatch")
    if hashlib.sha256(blob).hexdigest() != layer.sha256:
        raise ValueError("payload SHA mismatch")
    if len(blob) < 12 or blob[:8] != b"KVSLAB01":
        raise ValueError("payload magic")
    (size,) = struct.unpack("!I", blob[8:12])
    if size > 65536 or size + 12 > len(blob):
        raise ValueError("payload header length")
    if json.loads(blob[12 : 12 + size]).get("identity") != layer.identity:
        raise ValueError("payload identity mismatch")
    # The consumer validates the tensor schema when decoding this payload.


class Placement:
    """Retain validated whole-prefix replicas for one serial consumer.

    Select 'hotness' for windowed frequency, 'decay' for decayed frequency,
    or 'lru' for least recently used. 'remote-only' disables retention.
    Use a clock in the directory records' expiry time domain.
    """

    def __init__(self, policy, capacity_bytes, *, window_s=60.0, clock=time.time):
        if policy not in ("remote-only", "lru", "hotness", "decay"):
            raise ValueError("unknown policy")
        if type(capacity_bytes) is not int or capacity_bytes < 0:
            raise ValueError("invalid byte capacity")
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("invalid window")
        if policy == "decay" and window_s != 60.0:
            raise ValueError("decay policy has a fixed 60-second window")
        self.success_sequence = 0
        self.decay_events = deque()
        self.policy, self.capacity, self.window, self.clock = policy, capacity_bytes, window_s, clock
        self.entries, self.recent, self.seen = {}, deque(), set()
        self.sequence, self.active, self.last_now = 0, None, -math.inf
        self.peak_bytes = 0
        self.events = []

    @property
    def resident_bytes(self):
        return sum(g.nbytes for g, _, _ in self.entries.values())

    def _now(self):
        now = self.clock()
        if not math.isfinite(now) or now < self.last_now:
            raise ValueError("clock must be finite/nondecreasing")
        self.last_now = now
        while self.recent and self.recent[0][0] <= now - self.window:
            self.recent.popleft()
        while self.decay_events and self.decay_events[0][0] <= now - self.window:
            self.decay_events.popleft()
        return now

    def _heat_scores(self):
        if self.policy != "decay":
            return Counter(k for _, k in self.recent)
        scores = Counter()
        for _, key, sequence in self.decay_events:
            scores[key] += 2.0 ** (-(self.success_sequence - sequence) / 4.0)
        return scores

    def _drop(self, key, reason):
        group, _, _ = self.entries.pop(key)
        self.events.append(dict(event="evict", key=key, nbytes=group.nbytes, reason=reason))

    def begin(self, group, request_id):
        """Open one request. Finish after the caller confirms successful reuse."""
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
        victims = []
        admit = self.policy != "remote-only" and group.nbytes <= self.capacity
        if not hit and admit:
            heat = self._heat_scores()
            ranked = sorted(
                self.entries,
                key=lambda k: ((heat[k] if self.policy in ("hotness", "decay") else 0), self.entries[k][2], k),
            )
            available = self.capacity - self.resident_bytes
            for victim in ranked:
                if available >= group.nbytes:
                    break
                if self.policy in ("hotness", "decay") and heat[victim] > heat[key]:
                    admit = False
                    break
                victims.append(victim)
                available += self.entries[victim][0].nbytes
            if available < group.nbytes:
                admit = False
            if admit:
                for victim in victims:
                    self._drop(victim, "capacity")
        session = Demand(self, group, request_id, hit, admit and not hit)
        session.started = started
        session.event_start = event_start
        self.active = session
        return session


class Demand:
    def __init__(self, owner, group, request_id, hit, admit):
        self.owner, self.group, self.request_id = owner, group, request_id
        self.hit, self.admit = hit, admit
        self.staged, self.fetched = {}, set()
        self.remote_bytes = self.remote_objects = self.copy_bytes = 0
        self.fetch_seconds = self.verify_seconds = self.copy_seconds = 0.0
        self.peak_working_blob_bytes = 0
        self.started = time.perf_counter()
        self.finished = False

    def fetch(self, endpoint, sha256, **kwargs):
        raise RuntimeError("bind_fetcher(remote_fetch) before use")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.finished:
            self.finish(reuse_success=False)

    def bind_fetcher(self, remote_fetch):
        """Return the fetch callback for this request, preserving endpoint and size checks."""
        by_digest = {x.sha256: (i, x) for i, x in enumerate(self.group.layers)}

        def fetch(endpoint, sha256, **kwargs):
            if self.finished or self.owner.active is not self:
                raise RuntimeError("inactive demand")
            i, layer = by_digest[sha256]
            if endpoint != layer.endpoint or endpoint not in kwargs.get("allowed_endpoints", ()):
                raise ValueError("endpoint mismatch/not allowed")
            if layer.nbytes > kwargs.get("max_bytes", 256 * 1024**2):
                raise ValueError("object exceeds caller bound")
            if self.hit:
                blob = self.owner.entries[self.group.key][1][i]
            elif i in self.staged:
                blob = self.staged[i]
            else:
                tick = time.perf_counter()
                blob = remote_fetch(endpoint, sha256, **kwargs)
                self.fetch_seconds += time.perf_counter() - tick
                self.remote_objects += 1
                self.remote_bytes += len(blob)
                self.peak_working_blob_bytes = max(self.peak_working_blob_bytes, len(blob))
            tick = time.perf_counter()
            now = self.owner._now()
            verify(layer, blob, now)
            if self.hit and self.owner.entries[self.group.key][0].expiry <= now:
                raise ValueError("cached receipt expired")
            self.verify_seconds += time.perf_counter() - tick
            if self.admit and i not in self.staged:
                tick = time.perf_counter()
                # Force owned bytes, rather than retaining remote source aliases.
                self.staged[i] = memoryview(blob).tobytes()
                self.copy_seconds += time.perf_counter() - tick
                self.copy_bytes += len(blob)
                usage = self.owner.resident_bytes + sum(map(len, self.staged.values()))
                if usage > self.owner.capacity:
                    raise AssertionError("resident + staging capacity exceeded")
                self.owner.peak_bytes = max(self.owner.peak_bytes, usage)
            self.fetched.add(i)
            return blob

        fetch.source_kind = "cpu_near" if self.hit else "http"
        self.fetch = fetch
        return fetch

    def finish(self, *, reuse_success):
        """Count a complete successful reuse once and retain admitted payloads."""
        if self.finished:
            raise RuntimeError("already finished")
        try:
            now = self.owner._now()
        except ValueError:
            self.staged.clear()
            self.owner.active = None
            self.finished = True
            raise
        cached_valid = not self.hit or self.owner.entries[self.group.key][0].expiry > now
        complete = len(self.fetched) == 32 and self.group.expiry > now and cached_valid
        success = bool(reuse_success and complete)
        dedup = (self.group.key, self.request_id)
        fresh = success and dedup not in self.owner.seen
        if fresh:
            self.owner.seen.add(dedup)
            self.owner.recent.append((now, self.group.key))
            if self.owner.policy == "decay":
                self.owner.success_sequence += 1
                self.owner.decay_events.append((now, self.group.key, self.owner.success_sequence))
        self.owner.sequence += 1
        if success and self.admit:
            self.owner.entries[self.group.key] = (
                self.group,
                tuple(self.staged[i] for i in range(32)),
                self.owner.sequence,
            )
            self.owner.events.append(dict(event="insert", key=self.group.key, nbytes=self.group.nbytes))
        elif success and self.hit:
            old, blobs, _ = self.owner.entries[self.group.key]
            self.owner.entries[self.group.key] = old, blobs, self.owner.sequence
        staged_bytes = sum(map(len, self.staged.values()))
        self.staged.clear()
        self.owner.active = None
        self.finished = True
        row = dict(
            event="demand",
            key=self.group.key,
            request_id=self.request_id,
            hit=self.hit,
            complete=complete,
            reuse_success=success,
            success_increment=int(fresh),
            remote_bytes=self.remote_bytes,
            remote_objects=self.remote_objects,
            copy_bytes=self.copy_bytes,
            fetch_seconds=self.fetch_seconds,
            verify_seconds=self.verify_seconds,
            copy_seconds=self.copy_seconds,
            admission=self.admit,
            evicted_bytes=sum(x["nbytes"] for x in self.owner.events[self.event_start :] if x["event"] == "evict"),
            rolled_back_bytes=staged_bytes if not success else 0,
            resident_bytes=self.owner.resident_bytes,
            peak_retained_and_staging_bytes=self.owner.peak_bytes,
            peak_http_working_blob_bytes=self.peak_working_blob_bytes,
            total_seconds=time.perf_counter() - self.started,
        )
        if self.owner.policy == "decay":
            row["success_sequence"] = self.owner.success_sequence
            row["decayed_group_score_after"] = self.owner._heat_scores()[self.group.key]
        self.owner.events.append(row)
        return row
