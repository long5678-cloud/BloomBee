"""Engine-owned physical KV pages for opt-in ordinary Llama inference.

This is a BloomBee engine adaptation, not Dynamo's Rust managed CPU pool.
All pages share a fixed allocation; reclaiming a page reuses only that slot,
never invalidates other pages sharing its allocation. Matching is by caller's
complete compatible prefix-block keys. Refs protect active readers/writers.
No full-prefix K/V gather exists in the attention interface.
"""
from collections import deque
from dataclasses import dataclass
import threading

import torch


class PageCapacityError(RuntimeError):
    pass


@dataclass
class _Page:
    key: object = None
    priority: int = 0
    refs: int = 0
    tick: int = 0
    valid: int = 0
    generation: int = 0
    completion: object = None


def _uint(value, name, positive=False):
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


class EnginePagePool:
    """One real backend layer, batch one, time-major expanded attention heads.

    Empty slots first, otherwise idle lowest priority then earliest returned.
    This finite policy models the audited GPU free-queue/claim contract only;
    no duration, windows, offload queue, speculative decoding or TP movement.
    CUDA events on release protect in-flight reads before slot reuse. Pool
    capacity remains allocated; free slots are reusable capacity, not CUDA free.
    """
    def __init__(self, capacity_pages, block_size, num_heads, head_dim, *, dtype, device,
                 default_priority=35, event_limit=256):
        self.capacity_pages = _uint(capacity_pages, "capacity_pages", True)
        self.block_size = _uint(block_size, "block_size", True)
        self.num_heads = _uint(num_heads, "num_heads", True)
        self.head_dim = _uint(head_dim, "head_dim", True)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda") or not dtype.is_floating_point:
            raise ValueError("dense floating CPU/CUDA pages required")
        self.dtype = dtype
        if type(default_priority) is not int or not 0 <= default_priority <= 100:
            raise ValueError("invalid GPU engine default priority")
        self.default_priority = default_priority
        shape = (capacity_pages, block_size, num_heads, head_dim)
        self.k = torch.empty(shape, dtype=dtype, device=self.device)
        self.v = torch.empty_like(self.k)
        self.pages = [_Page() for _ in range(capacity_pages)]
        self.lookup = {}
        self.tick = 0
        self.events = deque(maxlen=_uint(event_limit, "event_limit", True))
        self.lock = threading.RLock()
        self.poisoned = False

    def _check(self):
        if self.poisoned:
            raise RuntimeError("page pool invalidated by device-copy failure")

    def _ready(self, page):
        return page.refs == 0 and (page.completion is None or page.completion.query())

    def _wait(self, page):
        if self.device.type == "cuda" and page.completion is not None:
            torch.cuda.current_stream(self.device).wait_event(page.completion)

    def _fence(self, page):
        if self.device.type == "cuda":
            # Consolidate all prior reader/writer streams into one dependency.
            # One event per physical page, never one growing list per hit.
            self._wait(page)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            page.completion = event

    def _wait_slots(self, slots):
        """One wait per distinct outstanding event, including other streams' readers."""
        if self.device.type != "cuda":
            return
        stream = torch.cuda.current_stream(self.device)
        seen = set()
        for slot, _ in slots:
            event = self.pages[slot].completion
            if event is not None and id(event) not in seen:
                stream.wait_event(event)
                seen.add(id(event))

    def _fence_slots(self, slots):
        """One immutable completion event can protect every page in this operation."""
        if self.device.type != "cuda" or not slots:
            return
        self._wait_slots(slots)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        for slot, _ in slots:
            self.pages[slot].completion = event

    def drain_events(self):
        with self.lock:
            result = list(self.events)
            self.events.clear()
            return result

    def begin(self, keys=(), priorities=None):
        """Claim a continuous matching prefix; stop at first missing block."""
        keys = tuple(keys)
        hints = (None,)*len(keys) if priorities is None else tuple(priorities)
        if len(hints)!=len(keys) or any(p is not None and (type(p) is not int or not 0<=p<=100) for p in hints):
            raise ValueError("claim priorities must be optional 0..100, one per key")
        for key in keys:
            hash(key)
            if key is None:
                raise ValueError("None is reserved for an unregistered physical page")
        if len(keys) != len(set(keys)):
            raise ValueError("prefix block keys must be unique and position-compatible")
        with self.lock:
            self._check()
            slots = []
            for key,priority in zip(keys,hints):
                slot = self.lookup.get(key)
                if slot is None:
                    break
                page = self.pages[slot]
                if page.valid != self.block_size:
                    raise RuntimeError("partial block illegally entered reusable lookup")
                page.refs += 1
                if priority is not None:
                    old_priority=page.priority
                    page.priority=priority
                    if old_priority!=priority:
                        self.events.append(dict(event="priority_update",slot=slot,
                            generation=page.generation,old_priority=old_priority,
                            priority=priority,reason="explicit_claim"))
                self._wait(page)
                slots.append((slot, page.generation))
            return PageSession(self, slots, len(slots) * self.block_size)

    def matching_tokens(self, keys):
        with self.lock:
            count = 0
            for key in keys:
                if key not in self.lookup:
                    break
                count += self.block_size
            return count

    def _plan_slots(self, count):
        available = [i for i, page in enumerate(self.pages) if self._ready(page)]
        if len(available) < count and self.device.type == "cuda":
            # Only wait for returned pages' submitted work, never for an active
            # claimant. This is a safe unfused fallback, not a no-sync claim.
            for page in self.pages:
                if page.refs == 0:
                    if page.completion is not None:
                        page.completion.synchronize()
            available = [i for i, page in enumerate(self.pages) if self._ready(page)]
        from bloombee.kv_cache import page_retention
        if page_retention.enabled():
            # Work slots may never silently evict registered history.
            work = [i for i in available if self.pages[i].key is None]
            if len(work) < count:
                raise PageCapacityError("insufficient work pages outside the retained history budget")
            return work[:count]
        from bloombee.kv_cache.retention_completion import order, snapshot
        import time
        now = time.monotonic()
        history = snapshot(now)
        available.sort(key=lambda i: (self.pages[i].key is not None,
            order(self.pages[i].key, (self.pages[i].priority, self.pages[i].tick, i),
                  view=history, now=now) if self.pages[i].key is not None else ()))
        if len(available) < count:
            raise PageCapacityError("insufficient unclaimed/completed physical pages")
        return available[:count]

    def _claim_new(self, slots):
        for slot in slots:
            page = self.pages[slot]
            old_key = page.key
            if old_key is not None:
                self.lookup.pop(old_key)
                self.events.append(dict(event="reclaim", slot=slot, key=repr(old_key),
                                        priority=page.priority, tick=page.tick))
            page.key = None
            page.refs = 1
            page.valid = 0
            page.priority = self.default_priority
            page.generation += 1

    def describe(self):
        with self.lock:
            return dict(capacity_pages=self.capacity_pages,
                        active_pages=sum(page.refs > 0 for page in self.pages),
                        cached_pages=len(self.lookup),
                        payload_bytes=self.k.numel() * self.k.element_size() * 2,
                        slots=[dict(slot=i, refs=p.refs, generation=p.generation,
                                    key=repr(p.key), priority=p.priority, valid=p.valid)
                               for i, p in enumerate(self.pages)])


class PageSession:
    """An active request lease; retain full published pages after release.

    Caller appends only newly computed tokens or a CPU-recovered missing suffix.
    Pinned prefix pages are immutable; writes append at logical length only.
    Publishing receives keys for a leading subset of FULL logical blocks,
    including the reused prefix. A service can deliberately leave its working
    prefill/decode tail unpublished. Those slots become free on release.
    """
    def __init__(self, pool, slots, length):
        self.pool = pool
        self.slots = list(slots)
        self.length = length
        self.closed = False

    def _check(self):
        self.pool._check()
        if self.closed:
            raise RuntimeError("closed page session")
        for slot, generation in self.slots:
            page = self.pool.pages[slot]
            if page.generation != generation or page.refs <= 0:
                raise RuntimeError("stale or unclaimed physical page")

    def append(self, key, value):
        pool = self.pool
        with pool.lock:
            self._check()
            if (not torch.is_tensor(key) or not torch.is_tensor(value)
                    or key.ndim != 3 or key.shape != value.shape
                    or tuple(key.shape[1:]) != (pool.num_heads, pool.head_dim)
                    or key.dtype != pool.dtype or value.dtype != pool.dtype):
                raise ValueError("append requires matching (tokens,heads,dim) K/V schema")
            # CPU recovery copies only supplied missing suffix; attention history
            # never round-trips through CPU. Report copy origin to the caller.
            if key.device != value.device or key.device.type not in ("cpu", "cuda"):
                raise ValueError("incompatible append devices")
            count = key.shape[0]
            if count == 0:
                return dict(tokens=0, copied_bytes=0, source_device=str(key.device))
            target = self.length + count
            needed = (target + pool.block_size - 1) // pool.block_size - len(self.slots)
            selected = pool._plan_slots(needed) if needed else []
            pool._claim_new(selected)
            self.slots.extend((slot, pool.pages[slot].generation) for slot in selected)
            start = self.length
            offset = 0
            # A partial tail can have been read on another stream.
            pool._wait_slots(self.slots[start // pool.block_size:])
            try:
                while offset < count:
                    logical = start + offset
                    slot = self.slots[logical // pool.block_size][0]
                    page_offset = logical % pool.block_size
                    take = min(count - offset, pool.block_size - page_offset)
                    pool.k[slot, page_offset:page_offset + take].copy_(key[offset:offset + take])
                    pool.v[slot, page_offset:page_offset + take].copy_(value[offset:offset + take])
                    pool.pages[slot].valid = page_offset + take
                    offset += take
            except Exception:
                # An actual device error may have overwritten a victim; do not
                # falsely restore its lookup or claim transactional byte repair.
                pool.poisoned = True
                raise
            self.length = target
            pool._fence_slots(self.slots)
            row = dict(tokens=count, logical_start=start, logical_end=target,
                       copied_bytes=key.numel() * key.element_size() * 2,
                       source_device=str(key.device), target_device=str(pool.device),
                       physical_pages=[slot for slot, _ in self.slots])
            pool.events.append(dict(event="append", **row))
            return row

    def publish(self, keys, priorities=None):
        pool = self.pool
        keys = tuple(keys)
        priorities = (None,) * len(keys) if priorities is None else tuple(priorities)
        if len(keys) > self.length // pool.block_size or len(priorities) != len(keys):
            raise ValueError("publish keys must describe a leading complete-block prefix")
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate content keys")
        if any(p is not None and (type(p) is not int or not 0 <= p <= 100) for p in priorities):
            raise ValueError("priority must be 0..100")
        with pool.lock:
            self._check()
            for index, (key, priority) in enumerate(zip(keys, priorities)):
                hash(key)
                if key is None:
                    raise ValueError("None is reserved for unregistered pages")
                slot = self.slots[index][0]
                page = pool.pages[slot]
                if page.key is not None and page.key != key:
                    raise ValueError("cannot rename pinned reused block")
            for index, (key, priority) in enumerate(zip(keys, priorities)):
                slot = self.slots[index][0]
                page = pool.pages[slot]
                existing = pool.lookup.get(key)
                if existing is None or existing == slot:
                    page.key = key
                    if priority is not None:
                        page.priority = priority
                    pool.lookup[key] = slot
                    pool._fence(page)
                # An independently computed duplicate need not overwrite the
                # already registered page; its unregistered slot frees on close.

    def operands(self, length=None):
        self._check()
        length = self.length if length is None else _uint(length, "prefix length")
        if length > self.length:
            raise ValueError("prefix exceeds committed session bytes")
        return (PagedOperand(self, "key", length), PagedOperand(self, "value", length))

    def block(self, index):
        """A leased logical block's real slot/generation and zero-copy K/V views.

        CPU retention may read only newly eligible complete blocks through this
        interface. It must not keep unleased views after release/reclamation.
        The final working tail can be partial and reports its actual length.
        """
        with self.pool.lock:
            self._check()
            _uint(index, "logical block index")
            if index >= (self.length + self.pool.block_size - 1) // self.pool.block_size:
                raise IndexError("logical block outside committed session")
            slot, generation = self.slots[index]
            self.pool._wait(self.pool.pages[slot])
            length = min(self.pool.block_size, self.length - index * self.pool.block_size)
            return dict(index=index, slot=slot, generation=generation, length=length,
                        key=self.pool.k[slot, :length], value=self.pool.v[slot, :length])

    def release(self):
        pool = self.pool
        with pool.lock:
            if self.closed:
                raise RuntimeError("page session already released")
            # Release remains possible after poison so refs do not leak.
            for slot, generation in self.slots:
                page = pool.pages[slot]
                if page.generation != generation or page.refs <= 0:
                    raise RuntimeError("invalid page release identity")
                pool._fence(page)
                page.refs -= 1
                if page.refs == 0:
                    pool.tick += 1
                    page.tick = pool.tick
                    if page.key is None:
                        page.valid = 0
            import os
            from bloombee.kv_cache.retention_completion import cap_idle_history
            from bloombee.kv_cache import page_retention
            if not page_retention.enabled():
                cap_idle_history(pool, int(os.environ.get('BLOOMBEE_RETAINED_GPU_BLOCKS', str(pool.capacity_pages))))
            self.closed = True

    def record_stream_use(self):
        """Called by the actual attention kernel before leaving its CUDA stream."""
        with self.pool.lock:
            self._check()
            self.pool._fence_slots(self.slots)

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *unused):
        self.release()


@dataclass(frozen=True)
class PagedOperand:
    """Tensor-like shape metadata ONLY. No gather, __torch_function__ or data copy."""
    session: PageSession
    kind: str
    length: int
    head_start: int = 0
    head_end: object = None

    def __post_init__(self):
        if self.kind not in ("key", "value"):
            raise ValueError("page operand kind must be key or value")
        _uint(self.length, "operand length")
        end = self.session.pool.num_heads if self.head_end is None else self.head_end
        if not 0 <= self.head_start < end <= self.session.pool.num_heads:
            raise ValueError("invalid page operand head range")
        if self.length > self.session.length:
            raise ValueError("operand exceeds committed page sequence")

    @property
    def heads(self):
        end = self.session.pool.num_heads if self.head_end is None else self.head_end
        return end - self.head_start

    @property
    def shape(self):
        return (1, self.heads, self.length, self.session.pool.head_dim)

    @property
    def device(self):
        return self.session.pool.device

    @property
    def dtype(self):
        return self.session.pool.dtype

    def dim(self):
        return 4

    def head_slice(self, start, end):
        if not 0 <= start < end <= self.heads:
            raise ValueError("invalid paged head slice")
        return PagedOperand(self.session, self.kind, self.length,
                            self.head_start + start, self.head_start + end)

    def segments(self):
        self.session._check()
        pool = self.session.pool
        backing = pool.k if self.kind == "key" else pool.v
        end = pool.num_heads if self.head_end is None else self.head_end
        remaining = self.length
        for slot, generation in self.session.slots:
            if remaining <= 0:
                break
            take = min(remaining, pool.block_size)
            pool._wait(pool.pages[slot])
            yield backing[slot, :take, self.head_start:end, :]
            remaining -= take


def _page_attention_inputs(key):
    """Use only physical-page views. The final projected tokens remain separate."""
    pool = key.session.pool
    key.session._check()
    pages = (key.length + pool.block_size - 1) // pool.block_size
    slots = key.session.slots[:pages]
    pool._wait_slots(slots)
    first = slots[0][0] if slots else 0
    contiguous = all(slot == first + i for i, (slot, _) in enumerate(slots))
    return slots, contiguous


def _attention_from_pages(q, key, k_new, v_new, mask):
    """Fully vectorized attention on real page storage, preserving one global softmax."""
    from bloombee.server.page_attention_kernel import physical_page_attention
    slots, contiguous = _page_attention_inputs(key)
    return physical_page_attention(q, key, k_new, v_new, mask, slots, contiguous)


def paged_attention_value(q, key, value, k_new, v_new, mask, *, batch, heads):
    """Apply attention to physical KV pages and newly projected tokens.

    Read the original projected Q and KV with a global softmax over history
    and newly projected tokens. The same physical pool remains authoritative.
    """
    if (not isinstance(key, PagedOperand) or not isinstance(value, PagedOperand)
            or key.kind != "key" or value.kind != "value"
            or key.session is not value.session or key.length != value.length
            or key.head_start != value.head_start or key.head_end != value.head_end):
        raise ValueError("paired page operands required")
    pool = key.session.pool
    if batch != 1 or heads != key.heads or value.heads != heads:
        raise ValueError("paged ordinary attention currently supports batch one")
    if q.device != pool.device or q.dtype != pool.dtype:
        raise ValueError("paged attention must use the original page device and dtype")
    if (q.ndim != 3 or q.shape[0] != heads or q.shape[2] != pool.head_dim
            or k_new.ndim != 3 or k_new.shape != v_new.shape
            or tuple(k_new.shape[1:]) != (heads, pool.head_dim)
            or k_new.device != q.device or v_new.device != q.device
            or k_new.dtype != q.dtype or v_new.dtype != q.dtype):
        raise ValueError("new projected KV does not match paged attention schema")
    expected = (batch, 1, q.shape[1], key.length + k_new.shape[0])
    if mask.numel() != expected[0] * expected[2] * expected[3]:
        raise ValueError("attention mask does not match paged logical prefix")
    result = _attention_from_pages(q, key, k_new, v_new, mask.reshape(expected).to(q.device))
    page_count = (key.length + pool.block_size - 1) // pool.block_size
    pages = [slot for slot, _ in key.session.slots[:page_count]]
    item_size = pool.k.element_size()
    stride = pool.k.stride(0) * item_size
    k_base = pool.k.data_ptr() + key.head_start * pool.head_dim * item_size
    v_base = pool.v.data_ptr() + key.head_start * pool.head_dim * item_size
    pool.events.append(dict(event="attention_direct_pages", prefix_tokens=key.length,
                            pages=pages,
                            key_ptrs=[k_base + stride * slot for slot in pages],
                            value_ptrs=[v_base + stride * slot for slot in pages],
                            head_start=key.head_start, heads=heads, head_dim=pool.head_dim,
                            element_size=pool.k.element_size(),
                            history_kv_gather_bytes=0))
    key.session.record_stream_use()
    return result.to(q.dtype).view(batch, heads, q.shape[1], pool.head_dim)
