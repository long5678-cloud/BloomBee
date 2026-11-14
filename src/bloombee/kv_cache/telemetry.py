"""Bounded request/group telemetry over the public Hivemind DHT API.

Default event time is UTC epoch seconds, not process-relative monotonic time.
Membership is explicit. A snapshot is advisory and not a global transaction.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import threading
import time
import uuid
from typing import Optional


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _name(value):
    if not isinstance(value, str) or not 0 < len(value) <= 256 or any(ord(c) < 32 for c in value):
        raise ValueError("expected a bounded nonempty identifier")
    return value


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected finite epoch seconds")
    return value


def _positive(value):
    if _number(value) <= 0:
        raise ValueError("expected positive duration")
    return value


def _sha(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA256 identity")
    return value


@dataclass(frozen=True)
class PublisherRef:
    """Logical using node and its process incarnation, independent of transport peer."""
    node_id: str
    session_id: str

    def __post_init__(self):
        _name(self.node_id)
        _name(self.session_id)

    @property
    def key(self):
        return _digest(asdict(self))


@dataclass(frozen=True)
class Event:
    kv_id: Optional[str]
    request_id: str
    kind: str
    occurred_at: float
    evidence: str = "observed"
    nbytes: int = 0
    tokens_match: Optional[bool] = None
    numerical_passed: Optional[bool] = None
    copy_integrity: Optional[bool] = None
    backend_work: Optional[bool] = None
    output_digest: Optional[str] = None
    source_digest: Optional[str] = None

    def __post_init__(self):
        if self.kv_id is not None:
            _sha(self.kv_id)
        elif self.kind == "reuse_success":
            raise ValueError("successful use requires a stable KV group identity")
        _name(self.request_id)
        _number(self.occurred_at)
        if self.kind not in Telemetry.EVENTS or self.evidence not in ("observed", "test", "historical"):
            raise ValueError("unknown stage or evidence")
        if type(self.nbytes) is not int or self.nbytes < 0:
            raise ValueError("invalid bytes")
        for value in (self.tokens_match, self.numerical_passed, self.copy_integrity, self.backend_work):
            if value is not None and type(value) is not bool:
                raise ValueError("correctness must be bool or None")
        for value in (self.output_digest, self.source_digest):
            if value is not None:
                _sha(value)

    @property
    def key(self):
        # One stage per request/group/incarnation; retries cannot multiply a use.
        return self.request_id, self.kv_id, self.kind


class IncompleteTelemetry(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__("telemetry is partial; inspect result['issues'] and coverage")


class Telemetry:
    """One publisher's bounded in-memory ledger. Publish is explicit, no workers.

    Each new process gets a new session_id. Caller request IDs must be unique
    within that session. No durable exactly-once guarantee across process loss.
    """
    EVENTS = frozenset(("lookup", "directory_hit", "fetch_success", "import_success", "reuse_success",
                        "fallback", "prefetch_fetch", "prefetch_used", "prefetch_unused_bytes",
                        "request_failed", "cold_recompute"))

    def __init__(self, dht, *, namespace, node_id, session_id=None, retention_s=3600,
                 max_events=4096, clock=time.time, history_start=None, timeout_s=5):
        self.dht, self.namespace = dht, _name(namespace)
        self.ref = PublisherRef(node_id, uuid.uuid4().hex if session_id is None else session_id)
        self.retention_s, self.timeout_s = _positive(retention_s), _positive(timeout_s)
        if type(max_events) is not int or not 0 < max_events <= 65536:
            raise ValueError("max_events must be in [1,65536]")
        self.max_events, self.clock = max_events, clock
        self._last = _number(clock())
        self._start = self._last if history_start is None else _number(history_start)
        if self._start > self._last:
            raise ValueError("history_start is in the future")
        self._events, self._lock = {}, threading.Lock()
        self._publish_lock = threading.Lock()
        self._last_expiration = -math.inf

    def _now(self):
        now = _number(self.clock())
        if now < self._last:
            raise ValueError("UTC clock moved backwards; correct clock before publishing")
        self._last = now
        cutoff = max(self._start, now - self.retention_s)
        self._events = {k:e for k,e in self._events.items() if e.occurred_at > cutoff}
        return now, cutoff

    def record(self, event):
        if not isinstance(event, Event):
            raise TypeError("record expects Event")
        with self._lock:
            now, cutoff = self._now()
            if not cutoff < event.occurred_at <= now:
                raise ValueError("event outside retained/observed time; stale retries are rejected")
            old = self._events.get(event.key)
            if old is not None:
                if old != event:
                    raise ValueError("conflicting retransmission for request/group/stage")
                return False
            if len(self._events) >= self.max_events:
                raise OverflowError("retained event capacity reached; no silent eviction")
            self._events[event.key] = event
            return True

    def publish(self, *, ttl_s=60):
        # Serialize snapshot capture AND network publication. record() only takes
        # _lock, and can proceed while a prior snapshot is being sent.
        with self._publish_lock:
            return self._publish(ttl_s=ttl_s)

    def _publish(self, *, ttl_s):
        from hivemind.utils import get_dht_time
        _positive(ttl_s)
        with self._lock:
            now, cutoff = self._now()
            events = [asdict(e) for e in self._events.values()]
        generation, pages = _digest(events), []
        expiry = get_dht_time() + ttl_s  # TTL clock is separate from injectable event time.
        if expiry < self._last_expiration:
            raise ValueError("cannot shorten an existing DHT publication TTL; use a new session/namespace")
        self._last_expiration = expiry
        prefix = "kv.telemetry:v1:" + self.namespace
        for i in range(0, len(events), 64):
            chunk = events[i:i+64]
            key = prefix + ":" + self.ref.key + ":" + generation + ":" + str(i//64)
            if not self.dht.store(key, chunk, expiration_time=expiry, return_future=True).result(timeout=self.timeout_s):
                raise RuntimeError("DHT page publication failed")
            pages.append(dict(key=key, digest=_digest(chunk), count=len(chunk)))
        snapshot = dict(schema=1, publisher=asdict(self.ref), retained_since=cutoff, observed_until=now,
                        generation=generation, count=len(events), pages=pages)
        if not self.dht.store(prefix + ":index", snapshot, subkey=self.ref.key,
                              expiration_time=expiry, return_future=True).result(timeout=self.timeout_s):
            raise RuntimeError("DHT index publication failed")
        return snapshot


def query(dht, *, namespace, publishers, start, end, kv_id=None, allow_partial=False,
          clock=time.time, timeout_s=5):
    """Return exact observed counts for (start,end], with explicit coverage.

    Default raises IncompleteTelemetry instead of presenting unknown as zero.
    allow_partial returns observed counts and complete=False with per-publisher issues.
    """
    from hivemind.utils import get_dht_time
    _name(namespace)
    _number(start)
    _number(end)
    now = _number(clock())
    _positive(timeout_s)
    if not start < end or end > now:
        raise ValueError("require start < end <= current UTC clock")
    if kv_id is not None:
        _sha(kv_id)
    refs = list(publishers)
    if not refs or any(not isinstance(r, PublisherRef) for r in refs) or len({r.key for r in refs}) != len(refs):
        raise ValueError("explicit unique PublisherRef membership required")
    prefix = "kv.telemetry:v1:" + namespace
    issues, coverage, counts, stages, details = [], [], defaultdict(Counter), Counter(), []
    nodes = set()
    try:
        found = dht.get(prefix + ":index", latest=True, return_future=True).result(timeout=timeout_s)
    except TimeoutError:
        found = None
    for ref in refs:
        try:
            entry = None if found is None else found.value.get(ref.key)
            if entry is None or entry.expiration_time <= get_dht_time():
                raise LookupError("missing_or_expired")
            s = entry.value
            if s["schema"] != 1 or s["publisher"] != asdict(ref):
                raise ValueError("invalid publisher snapshot")
            retained, observed = _number(s["retained_since"]), _number(s["observed_until"])
            if retained > observed:
                raise ValueError("invalid coverage")
            if observed > now:
                raise ValueError("publisher clock ahead of query clock")
            if type(s["count"]) is not int or not 0 <= s["count"] <= 65536 or len(s["pages"]) > 1024:
                raise ValueError("snapshot exceeds event bound")
            records = []
            for page in s["pages"]:
                got = dht.get(page["key"], latest=True, return_future=True).result(timeout=timeout_s)
                if got is None or got.expiration_time <= get_dht_time():
                    raise LookupError("missing_or_expired_page")
                if len(got.value) != page["count"] or _digest(got.value) != page["digest"]:
                    raise ValueError("invalid page digest/count")
                records.extend(got.value)
            if len(records) != s["count"] or _digest(records) != s["generation"]:
                raise ValueError("incomplete snapshot")
            events = [Event(**record) for record in records]
            seen = {}
            for e in events:
                if not retained < e.occurred_at <= observed:
                    raise ValueError("event outside declared snapshot coverage")
                if e.key in seen and seen[e.key] != e:
                    raise ValueError("conflicting snapshot event")
                seen[e.key] = e
            coverage.append(dict(publisher=asdict(ref), retained_since=retained, observed_until=observed))
            nodes.add(ref.node_id)
            if start < retained or end > observed:
                issues.append(dict(publisher=asdict(ref), reason="window_not_fully_observed_or_retained"))
            for e in seen.values():
                if start < e.occurred_at <= end and (kv_id is None or e.kv_id == kv_id):
                    stages[e.kind] += 1
                    details.append(dict(publisher=asdict(ref), **asdict(e)))
                    if e.kind == "reuse_success":
                        counts[e.kv_id][ref.node_id] += 1
        except (LookupError, ValueError, TypeError, KeyError, AttributeError, TimeoutError) as error:
            issues.append(dict(publisher=asdict(ref), reason=str(error)))
    rows = [dict(kv_id=k, total=sum(c.values()), nodes={n:c.get(n,0) for n in sorted(nodes)}) for k,c in counts.items()]
    rows.sort(key=lambda r:(-r["total"], r["kv_id"]))
    details.sort(key=lambda e:(e["occurred_at"], e["publisher"]["node_id"], e["publisher"]["session_id"], e["request_id"],e["kv_id"] or "",e["kind"]))
    uses = [e for e in details if e["kind"] == "reuse_success"]
    result = dict(window=dict(start_exclusive=start, end_inclusive=end), complete=not issues,
                  coverage=coverage, issues=issues, ranked=rows, stage_counts=dict(stages), events=details,
                  total_uses=len(uses), numerical_passed=sum(e["numerical_passed"] is True for e in uses),
                  numerical_failed=sum(e["numerical_passed"] is False for e in uses),
                  copy_integrity_failed=sum(e["copy_integrity"] is False for e in uses),
                  backend_work_failed=sum(e["backend_work"] is False for e in uses),
                  tokens_matched=sum(e["tokens_match"] is True for e in uses),
                  tokens_mismatched=sum(e["tokens_match"] is False for e in uses))
    if issues and not allow_partial:
        raise IncompleteTelemetry(result)
    return result
