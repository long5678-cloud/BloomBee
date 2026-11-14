"""Native dense-slab history: ownership transfer, never prefix re-materialization.

GPU lifetime unit is the existing engine's allocation, NOT Dynamo's independently
allocated block. Reclaiming a slab co-evicts its other blocks. The victim block
metadata order follows managed PriorityKey; this granularity difference is public.
Only idle allocations are transferable/reclaimable. A MemoryCache free message
ends the session lease, not an inference_step return.
"""
from dataclasses import dataclass
import os

import torch


def raw(value):
    return value if torch.is_tensor(value) else value.data


def storage_id(tensor):
    return {"device": str(tensor.device), "pointer": tensor.untyped_storage().data_ptr(),
            "offset": tensor.storage_offset(), "shape": list(tensor.shape),
            "stride": list(tensor.stride())}


@dataclass(eq=False)
class Slot:
    serial: int
    native: tuple
    ids: tuple
    inputs: torch.Tensor
    outputs: torch.Tensor
    priorities: tuple
    returned_tick: int
    acquired_tick: int
    handle: int
    active: bool = True
    ready_event: object = None

    @property
    def nbytes(self):
        return sum(raw(t).numel() * raw(t).element_size() for t in self.native) + sum(
            t.numel() * t.element_size() for t in (self.inputs, self.outputs))


class NativeHistory:
    def __init__(self, binding, block_size, budget_bytes, max_slots=1):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("GPU history budget must be nonnegative bytes")
        self.binding, self.block_size, self.budget_bytes = binding, block_size, budget_bytes
        if type(max_slots) is not int or max_slots < 0:
            raise ValueError("GPU history slots must be nonnegative")
        self.max_slots = max_slots
        self.pid = os.getpid()
        self.slots = []
        self.tick = self.serial = 0
        self.evictions = []

    @property
    def resident_bytes(self):
        return sum(slot.nbytes for slot in self.slots)

    def _owner(self):
        if os.getpid() != self.pid:
            raise RuntimeError("GPU allocation cannot be inherited by another runtime")

    def find(self, ids, inputs, shape, dtype, device, max_tokens=None):
        """Only compare actual GPU history/input; no CPU history read here."""
        self._owner()
        best = None
        for slot in self.slots:
            key = raw(slot.native[0])
            if (slot.active or tuple(key.shape) != tuple(shape) or key.dtype != dtype or key.device != device):
                continue
            if slot.ready_event is not None:
                torch.cuda.current_stream(key.device).wait_event(slot.ready_event)
            count = 0
            for left, right in zip(slot.ids, ids):
                if left != right:
                    break
                count += 1
            count = count // self.block_size * self.block_size
            if max_tokens is not None:
                count = min(count, max_tokens)
            if count and (best is None or count > best[1]):
                if inputs.dtype == slot.inputs.dtype and torch.equal(
                        inputs[:, :count].to(slot.inputs.device), slot.inputs[:, :count]):
                    best = slot, count
        return best

    def acquire(self, slot, handle):
        self._owner()
        if slot not in self.slots or slot.active:
            raise RuntimeError("native allocation is not idle")
        if slot.ready_event is not None:
            torch.cuda.current_stream(raw(slot.native[0]).device).wait_event(slot.ready_event)
        self.tick += 1
        slot.active, slot.handle, slot.acquired_tick = True, handle, self.tick

    def release(self, slot, handle):
        self._owner()
        if slot not in self.slots or not slot.active or slot.handle != handle:
            raise RuntimeError("native allocation lease mismatch")
        self.tick += 1
        if raw(slot.native[0]).device.type == "cuda":
            slot.ready_event = torch.cuda.Event()
            slot.ready_event.record(torch.cuda.current_stream(raw(slot.native[0]).device))
        slot.returned_tick, slot.active = self.tick, False

    def _make_room(self, required, exclude=None):
        others = lambda: [s for s in self.slots if s is not exclude]
        if (required + sum(s.nbytes for s in others() if s.active) > self.budget_bytes
                or len([s for s in others() if s.active]) >= self.max_slots):
            return False
        while (sum(s.nbytes for s in others()) + required > self.budget_bytes
               or len(others()) >= self.max_slots):
            candidates = [slot for slot in self.slots if not slot.active and slot is not exclude]
            if not candidates:
                return False
            # Lowest individual block wins; the native dense allocation is the
            # physical reclamation unit. Explicitly record collateral eviction.
            victim = min(candidates, key=lambda s: (min(s.priorities, default=0),
                          s.returned_tick, s.acquired_tick, s.ids))
            if victim.ready_event is not None:
                victim.ready_event.synchronize()
            self.evictions.append({"slot_id": victim.serial, "reason": "idle_capacity",
                "unit": "native_dense_slab", "coevicted_blocks": len(victim.priorities),
                "priority": min(victim.priorities, default=0),
                "returned_tick": victim.returned_tick, "acquired_tick": victim.acquired_tick,
                "freed_payload_bytes": victim.nbytes,
                "key_storage": storage_id(raw(victim.native[0]))})
            self.slots.remove(victim)
        return True

    def retain(self, native, ids, inputs, outputs, priorities, handle, existing=None):
        self._owner()
        count = len(ids) // self.block_size * self.block_size
        if not count or not self.budget_bytes or not self.max_slots:
            return None
        hints = tuple(0 if v is None else v for v in (priorities or ()))
        hints = (hints + (0,) * (count // self.block_size))[:count // self.block_size]
        # Auxiliary state remains on the same native execution device. It is
        # counted in addition to FULL slab capacity (including unused tail).
        key = raw(native[0])
        source_input = inputs[:, :count].detach().to(key.device, copy=True)
        source_output = outputs[:, :count].detach().to(key.device, copy=True)
        proposed = Slot(self.serial + 1, native, tuple(ids[:count]), source_input,
                        source_output, hints, 0, self.tick + 1, handle)
        if proposed.nbytes > self.budget_bytes or not self.max_slots:
            return None
        if existing is not None:
            # The current session has exclusive write ownership, so old valid
            # history is replaced atomically; no stale branches survive writes.
            if existing not in self.slots or not existing.active or existing.handle != handle:
                raise RuntimeError("cannot replace another session's history")
            if not self._make_room(proposed.nbytes, exclude=existing):
                self.slots.remove(existing)
                return None
            self.slots.remove(existing)
        elif not self._make_room(proposed.nbytes):
            return None
        self.serial += 1
        self.tick += 1
        proposed.serial, proposed.acquired_tick = self.serial, self.tick
        self.slots.append(proposed)
        return proposed


def register_session(memory, pool, slot, handle):
    memory._dynamo_active_history[handle] = (pool, slot)
    memory._dynamo_session_release[handle] = lambda: (
        pool.release(slot, handle) if slot in pool.slots else None)


def invalidate_session(memory, handle):
    current = memory._dynamo_active_history.pop(handle, None)
    memory._dynamo_session_release.pop(handle, None)
    if current is not None:
        pool, slot = current
        if slot in pool.slots:
            pool.slots.remove(slot)


def protect_history_before_step(memory, info, hypo_ids):
    """Invalidate before reset/reorder/custom writes, including opt-out steps."""
    for handle in info.cache_handles:
        current = memory._dynamo_active_history.get(handle)
        if current is None:
            continue
        _, slot = current
        def enabled(value):
            return bool(value.numel() and value.bool().any().item()) if torch.is_tensor(value) else bool(value)
        explicit = getattr(info, "kv_cache_position_ids", None)
        bad_positions = explicit is not None and (not torch.is_tensor(explicit) or bool(explicit.numel()))
        bad_hypo = hypo_ids is not None and torch.is_tensor(hypo_ids) and hypo_ids.numel() and not torch.equal(
            hypo_ids.cpu(), torch.zeros(1, dtype=hypo_ids.dtype))
        if (info.prefix_length < len(slot.ids) or info.batch_offset
                or info.full_batch_size not in (0, 1) or info.micro_batch_size not in (0, 1)
                or info.active_adapter or enabled(info.is_spec_dec) or enabled(info.need_pruning)
                or bad_positions or bad_hypo):
            invalidate_session(memory, handle)
