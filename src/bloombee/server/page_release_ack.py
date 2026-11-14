"""Receipts for page leases actually returned through an owning runtime's free FIFO."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
import threading
import uuid

import torch


@dataclass(frozen=True)
class ReturnedLease:
    attempt_id: str
    uid: str
    handle: int
    page_count: int
    generations_sha256: str
    completion: object


def capture_release(record, handle, was_failed):
    """Called inside the runtime free callback after release, under the pool lock."""
    if was_failed or not record["session"].closed:
        return None
    info = record["initial_info"]
    config = getattr(info, "dynamo_cache", None)
    attempt = config.get("attempt_id") if isinstance(config, dict) else None
    if not isinstance(attempt, str) or not attempt:
        return None
    session = record["session"]
    # PageSession.release fences every slot on this runtime stream, including
    # all its predecessor events. Capture the last recorded event by value
    # before any other allocation can change that slot's generation or event.
    if not session.slots:
        return None
    pool = session.pool
    if any(pool.pages[i].generation != gen for i, gen in session.slots):
        raise RuntimeError("release generation changed before capture")
    fence = pool.pages[session.slots[-1][0]].completion
    if pool.device.type == "cuda" and fence is None:
        raise RuntimeError("missing physical-page release fence")
    digest = hashlib.sha256(json.dumps(session.slots, separators=(",", ":")).encode()).hexdigest()
    return ReturnedLease(attempt, record["uid"], handle, len(session.slots), digest, fence)


class ReleaseJournal:
    """Immutable free observations, bounded by completed allocations. Eviction fails closed."""
    def __init__(self, limit=512):
        self.instance = uuid.uuid4().hex
        self.pid = os.getpid()
        self.runtime_thread = None
        self.limit = limit
        self.lock = threading.Lock()
        self.freed = OrderedDict()
        self.attempts = OrderedDict()
        self.allocations = OrderedDict()

    def register(self, attempt, handle, uid, allocation):
        if not isinstance(attempt, str) or not attempt or not allocation or handle not in allocation:
            return
        allocation = tuple(allocation)
        with self.lock:
            if attempt not in self.allocations:
                if attempt in self.attempts:
                    self.attempts[attempt] = None
                    return
                self.allocations[attempt] = (allocation, {})
                if len(self.allocations) > self.limit:
                    self.allocations.popitem(last=False)
            registered = self.allocations[attempt]
            if registered is None:
                return
            group, bindings = registered
            if (allocation != group or (handle in bindings and bindings[handle] != uid)
                    or any(u == uid and h != handle for h, u in bindings.items())):
                self.allocations[attempt] = self.attempts[attempt] = None
                return
            bindings[handle] = uid

    def expected(self, attempt):
        # Independent of the free-receipt table. The group originated in a
        # create-FIFO record. Each UID was linked by its actual paged prefill.
        with self.lock:
            registered = self.allocations.get(attempt)
            if registered is None:
                return None
            handles, bindings = registered
            if len(bindings) != len(handles):
                return None
            return dict(runtime_instance=self.instance, allocation_generation=handles[0],
                        handles=list(handles), backend_uids=[bindings[h] for h in handles])

    def observe_free(self, handles, leases):
        # Only the real FIFO consumer calls this, after every handle was removed.
        handles, leases = tuple(handles), tuple(leases)
        if not leases or any(not isinstance(lease, ReturnedLease) for lease in leases):
            return
        if tuple(row.handle for row in leases) != handles or len({row.attempt_id for row in leases}) != 1:
            return
        with self.lock:
            self.freed[handles] = leases
            if len(self.freed) > self.limit:
                self.freed.popitem(last=False)

    def seal(self, instance, attempt, handles, uids):
        handles, uids = tuple(handles), tuple(uids)
        expected = self.expected(attempt)
        if not expected or expected["handles"] != list(handles) or expected["backend_uids"] != list(uids):
            return False
        with self.lock:
            records = self.freed.get(handles)
            if (instance != self.instance or not records or len(set(handles)) != len(handles)
                    or len(set(uids)) != len(uids) or tuple(row.uid for row in records) != uids
                    or any(row.attempt_id != attempt for row in records)):
                return False
            # Repeating the same close is idempotent. Two allocations claiming
            # one attempt are ambiguous even if a client reused that ID.
            if attempt in self.attempts and self.attempts[attempt] != handles:
                self.attempts[attempt] = None
                return False
            self.attempts[attempt] = handles
            if len(self.attempts) > self.limit:
                self.attempts.popitem(last=False)
            return True

    def status(self, attempt):
        with self.lock:
            handles = self.attempts.get(attempt)
            rows = self.freed.get(handles)
            binding = self.allocations.get(attempt)
            if not rows or binding is None or binding[0] != handles:
                return {"status": "unknown"}
        # Immutable events are queried, never synchronized here. No model,
        # cache, or metadata lock is held while inspecting CUDA readiness.
        try:
            ready = all(row.completion is None or row.completion.query() for row in rows)
        except RuntimeError:
            ready = None
        with self.lock:
            # Linearize the response against the same allocation and immutable
            # free record whose completion events were queried outside the lock.
            if (self.attempts.get(attempt) != handles
                    or self.allocations.get(attempt) is not binding
                    or self.freed.get(handles) is not rows):
                return {"status": "unknown"}
            if ready is None:
                return {"status": "failed"}
            return dict(status="released" if ready else "pending", attempt_id=attempt,
                        runtime_pid=self.pid, runtime_instance=self.instance,
                        allocation_generation=handles[0], handles=list(handles),
                        backend_uids=[row.uid for row in rows], page_counts=[row.page_count for row in rows],
                        slot_generation_sha256=[row.generations_sha256 for row in rows])


class RuntimeReleaseTask:
    def __init__(self, memory):
        self.memory = memory

    def __call__(self, instance, attempt, handles, uids):
        memory = self.memory
        if (os.getpid() != memory.runtime_pid or instance != memory.release_journal.instance
                or memory.release_journal.runtime_thread != threading.get_ident()):
            raise ValueError("wrong release runtime")
        # Runtime.process_batch owns this call. Empty handles pump the exact
        # same allocate/free FIFO without requesting any missing tensor.
        with memory.use_cache():
            pass
        sealed = memory.release_journal.seal(instance, attempt, handles, uids)
        # Hivemind obtains batch size from outputs[0].size(0).
        return (torch.tensor([sealed], dtype=torch.bool),)
