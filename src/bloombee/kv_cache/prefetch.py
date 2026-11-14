"""History-only bounded remote prefix prefetch; explicit backend integration.

No trace lookahead, tensor import, demand fetching, or request scheduling here.
A caller supplies a CURRENT published directory and a trusted remote fetcher.
"""

import hashlib
import json
import math
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .placement import Layer, verify


@dataclass(frozen=True)
class RemotePrefix:
    """A complete backend-compatible snapshot; namespace includes model/revision/schema."""

    namespace: str
    blocks: tuple
    layers: tuple

    def __post_init__(self):
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("namespace required")
        if type(self.blocks) is not tuple or not self.blocks:
            raise ValueError("nonempty immutable block chain required")
        if any(type(x) not in (int, str) for x in self.blocks):
            raise ValueError("block IDs must be integers or strings")
        if type(self.layers) is not tuple or not self.layers:
            raise ValueError("complete ordered layer tuple required")
        base = None
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, Layer):
                raise ValueError("placement.Layer required")
            ident = layer.identity
            if ident.pop("layer_start", None) != i or ident.pop("layer_end", None) != i + 1:
                raise ValueError("incomplete/out-of-order layers")
            if base is None:
                base = ident
            elif ident != base:
                raise ValueError("mixed prefix/model/execution identities")
            if (type(layer.nbytes) is not int or layer.nbytes <= 0
                    or not math.isfinite(layer.expires_at) or len(layer.sha256) != 64
                    or not layer.endpoint):
                raise ValueError("invalid layer record")
            int(layer.sha256, 16)
        if len({x.sha256 for x in self.layers}) != len(self.layers):
            raise ValueError("duplicate layer payload")

    @property
    def key(self):
        value = [self.namespace, self.blocks,
                 [(x.identity_json, x.sha256, x.nbytes) for x in self.layers]]
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self.layers)

    @property
    def expiry(self):
        return min(x.expires_at for x in self.layers)


@dataclass(frozen=True)
class DemandResult:
    request_id: str
    status: str
    prefix: object = None
    payloads: tuple = ()
    waited_s: float = 0.0
    ticket_id: object = None
    arrival_ns: int = 0
    _owner: object = field(default=None, repr=False, compare=False)


@dataclass
class _Ticket:
    id: str
    prefix: RemotePrefix
    event: threading.Event = field(default_factory=threading.Event)
    cancelled: bool = False
    state: str = "pending"
    payloads: tuple = ()
    ready_ns: int = 0
    claimed: bool = False
    reserved: bool = True


class Prefetcher:
    """One finite session, serial demand consumer and bounded background workers.

    capacity_bytes reserves complete payloads before launching, including pending,
    in-flight, cancelled-but-not-returned, and ready data. The fetch callback must
    bound its own network buffer and timeout. Claimed consumer/model buffers and
    event/request audit metadata are outside this prefetch staging budget.
    """

    def __init__(self, fetch, capacity_bytes, *, allowed_endpoints,
                 max_inflight=1, history_size=128, max_candidates=1, max_entries=8,
                 clock=time.time):
        for value in (capacity_bytes, max_inflight, history_size, max_candidates, max_entries):
            if type(value) is not int or value <= 0:
                raise ValueError("positive integer limits required")
        self.fetch = fetch
        self.capacity_bytes = capacity_bytes
        self.allowed_endpoints = frozenset(allowed_endpoints)
        self.max_inflight = max_inflight
        self.max_candidates = max_candidates
        self.max_entries = max_entries
        self.clock = clock
        self._history = deque(maxlen=history_size)
        self._tickets = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_inflight, thread_name_prefix="kv-prefetch")
        self._reserved = 0
        self._seq = 0
        self._closed = False
        self._arrivals = set()
        self._results = {}
        self._finished = set()
        self._events = []

    def _emit(self, kind, ticket=None, **fields):
        self._events.append(dict(event=kind, monotonic_ns=time.monotonic_ns(),
                                 ticket_id=ticket.id if ticket else None,
                                 key=ticket.prefix.key if ticket else None, **fields))

    @property
    def events(self):
        with self._lock:
            return [dict(x) for x in self._events]

    def events_since(self, cursor=0):
        """Return (next_cursor, copied_new_events) without copying prior history."""
        with self._lock:
            if type(cursor) is not int or not 0 <= cursor <= len(self._events):
                raise ValueError("invalid event cursor")
            return len(self._events), [dict(event) for event in self._events[cursor:]]

    @property
    def reserved_bytes(self):
        with self._lock:
            return self._reserved

    def observe(self, namespace, blocks):
        """Record ONLY a request already disclosed at actual arrival (not its future)."""
        blocks = tuple(blocks)
        if not namespace or any(type(x) not in (int, str) for x in blocks):
            raise ValueError("invalid observed chain")
        with self._lock:
            if self._closed:
                raise RuntimeError("closed")
            self._history.append((namespace, blocks))
            self._emit("observe", namespace=namespace, block_count=len(blocks))

    def _release(self, ticket):
        if ticket.reserved:
            self._reserved -= ticket.prefix.nbytes
            ticket.reserved = False

    def _expire(self):
        for ticket_id, ticket in list(self._tickets.items()):
            if ticket.event.is_set() and (ticket.cancelled or ticket.state != "ready"
                                         or ticket.prefix.expiry <= self.clock()):
                self._release(ticket)
                del self._tickets[ticket_id]
                self._emit("discard", ticket, reason="expired_or_terminal")

    def schedule(self, directory, *, local_keys=()):
        """Return launched ticket IDs. Snapshot must not contain unpublished/future records.

        Rank by (#matching observed requests, latest matching history position,
        prefix length, stable key). Never infer a candidate from the next trace row.
        """
        snapshot = tuple(directory)
        with self._lock:
            if self._closed:
                raise RuntimeError("closed")
            self._expire()
            active_keys = {t.prefix.key for t in self._tickets.values()}
            local_keys = set(local_keys)
            ranked = []
            seen = set()
            for prefix in snapshot:
                key = prefix.key
                if key in seen:
                    continue
                seen.add(key)
                matches = [i for i, (ns, chain) in enumerate(self._history)
                           if ns == prefix.namespace and chain[:len(prefix.blocks)] == prefix.blocks]
                reason = None
                if not matches:
                    reason = "no_observed_history"
                elif key in local_keys:
                    reason = "already_local"
                elif key in active_keys:
                    reason = "already_scheduled"
                elif prefix.expiry <= self.clock():
                    reason = "expired"
                elif any(x.endpoint not in self.allowed_endpoints for x in prefix.layers):
                    reason = "untrusted_endpoint"
                if reason:
                    self._emit("skip", key_value=key, reason=reason)
                    continue
                ranked.append((len(matches), matches[-1], len(prefix.blocks), key, prefix))
            ranked.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
            launched = []
            for count, recent, length, key, prefix in ranked:
                active = sum(not t.event.is_set() for t in self._tickets.values())
                if (len(launched) >= self.max_candidates or active >= self.max_inflight
                        or len(self._tickets) >= self.max_entries):
                    self._emit("skip", key_value=key, reason="slot_limit")
                    continue
                if self._reserved + prefix.nbytes > self.capacity_bytes:
                    self._emit("skip", key_value=key, reason="byte_capacity")
                    continue
                self._seq += 1
                ticket = _Ticket(str(self._seq), prefix)
                self._tickets[ticket.id] = ticket
                self._reserved += prefix.nbytes
                self._emit("launch", ticket, reserved_bytes=self._reserved,
                           history_count=count, history_recent=recent, block_count=length)
                self._executor.submit(self._run, ticket)
                launched.append(ticket.id)
            return tuple(launched)

    def _run(self, ticket):
        payloads = []
        try:
            with self._lock:
                if ticket.cancelled:
                    return
                ticket.state = "fetching"
                self._emit("fetch_start", ticket)
            for layer in ticket.prefix.layers:
                with self._lock:
                    if ticket.cancelled:
                        return
                    if ticket.prefix.expiry <= self.clock():
                        raise ValueError("expired prefix")
                blob = self.fetch(layer.endpoint, layer.sha256,
                                  allowed_endpoints=self.allowed_endpoints)
                with self._lock:
                    self._emit("fetch_bytes", ticket, nbytes=len(blob))
                    if ticket.cancelled:
                        return
                verify(layer, blob, self.clock())
                payloads.append(blob)
            with self._lock:
                if ticket.prefix.expiry <= self.clock():
                    raise ValueError("expired prefix")
                if ticket.cancelled:
                    return
                ticket.payloads = tuple(payloads)  # preserve original bytes objects
                ticket.ready_ns = time.monotonic_ns()
                ticket.state = "ready"
                self._emit("ready", ticket, nbytes=ticket.prefix.nbytes,
                           ready_ns=ticket.ready_ns)
        except Exception as error:
            with self._lock:
                ticket.state = "failure"
                # Do not put URLs, payloads or arbitrary callback error text in events.
                self._emit("failure", ticket, error_type=type(error).__name__)
        finally:
            with self._lock:
                if ticket.cancelled:
                    ticket.state = "cancelled"
                    ticket.payloads = ()
                if ticket.state != "ready":
                    self._release(ticket)
                ticket.event.set()
                self._emit("fetch_end", ticket, state=ticket.state,
                           reserved_bytes=self._reserved)

    def arrive(self, request_id, namespace, blocks, *, wait=True, timeout=None):
        """Call immediately at A's actual arrival; wait is demand wait, never lead time.

        Miss/failure/no-wait/timeout returns no payload; caller performs normal demand.
        Successful claim transfers the exact bytes to caller and releases staging.
        """
        arrival_ns = time.monotonic_ns()
        blocks = tuple(blocks)
        with self._lock:
            if self._closed:
                raise RuntimeError("closed")
            if not isinstance(request_id, str) or not request_id or request_id in self._arrivals:
                raise ValueError("unique nonempty request_id required")
            self._arrivals.add(request_id)
            self._emit("arrival", request_id=request_id, arrival_ns=arrival_ns,
                       namespace=namespace, block_count=len(blocks))
            eligible = [t for t in self._tickets.values()
                        if not t.claimed and not t.cancelled and t.prefix.namespace == namespace
                        and blocks[:len(t.prefix.blocks)] == t.prefix.blocks
                        and t.prefix.expiry > self.clock()]
            eligible.sort(key=lambda t: (-len(t.prefix.blocks), t.id))
            ticket = eligible[0] if eligible else None
            if ticket:
                ticket.claimed = True
        waited_s = 0.0
        if ticket and not ticket.event.is_set() and wait:
            start = time.monotonic()
            ticket.event.wait(timeout)
            waited_s = time.monotonic() - start
        with self._lock:
            status, prefix, payloads = "miss", None, ()
            if ticket:
                if not ticket.event.is_set():
                    status = "inflight_timeout" if wait else "inflight_no_wait"
                    ticket.claimed = False
                elif (not ticket.cancelled and ticket.state == "ready"
                      and ticket.prefix.expiry > self.clock()):
                    status = ("ready_before_arrival" if ticket.ready_ns <= arrival_ns
                              else "inflight_wait")
                    prefix, payloads = ticket.prefix, ticket.payloads
                    ticket.payloads = ()
                    self._release(ticket)
                    self._tickets.pop(ticket.id, None)
                else:
                    status = "failure" if ticket.state == "failure" else "unavailable"
                    self._release(ticket)
                    self._tickets.pop(ticket.id, None)
                self._emit("claim", ticket, request_id=request_id, status=status,
                           waited_s=waited_s, nbytes=sum(map(len, payloads)))
            result = DemandResult(request_id, status, prefix, payloads, waited_s,
                                  ticket.id if ticket else None, arrival_ns, self)
            self._results[request_id] = (id(result), bool(payloads))
            return result

    def finish(self, result, *, reuse_success):
        """Confirm actual model import+use, NOT merely retrieval; exactly once."""
        with self._lock:
            record = self._results.get(result.request_id)
            if result._owner is not self or record is None or record[0] != id(result):
                raise ValueError("foreign result")
            if result.request_id in self._finished:
                raise ValueError("result already finished")
            if reuse_success and not record[1]:
                raise ValueError("cannot claim prefetch reuse without payload")
            self._finished.add(result.request_id)
            if reuse_success:
                self._emit("used", request_id=result.request_id, ticket_ref=result.ticket_id,
                           nbytes=sum(map(len, result.payloads)))
            self._emit("result", request_id=result.request_id, ticket_ref=result.ticket_id,
                       reuse_success=bool(reuse_success), status=result.status)

    def cancel(self, ticket_id):
        """Cooperative: never free capacity while a blocking fetch is still running."""
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if ticket is None or ticket.cancelled:
                return False
            ticket.cancelled = True
            if ticket.event.is_set():
                ticket.payloads = ()
                self._release(ticket)
                self._tickets.pop(ticket.id, None)
            self._emit("cancel", ticket, reserved_bytes=self._reserved)
            return True

    def close(self):
        with self._lock:
            self._closed = True
            for ticket_id in list(self._tickets):
                self.cancel(ticket_id)
        self._executor.shutdown(wait=True)
        with self._lock:
            self._expire()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
