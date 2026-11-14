"""Terminal-confirmed retention of one real 16-token page per backend layer."""
import os
import time

from .retention_completion import policy, prefix_key, snapshot


def tier():
    value = os.environ.get("BLOOMBEE_HISTORY_TIER", "both")
    if value not in ("cpu", "gpu", "both"):
        raise ValueError("invalid history tier")
    return value


def enabled():
    return policy() != "dynamo" or tier() != "both"


def _scores(view, now):
    # Accumulate in ledger order, as Placement._heat_scores does.
    name = policy()
    frequency, recency = {}, {}
    use = []
    for row in view["events"]:
        if type(row.get("kv_reuse_success")) is not bool:
            raise ValueError("complete request must declare verified KV reuse")
        if row["kv_reuse_success"] or name == "dynamo":
            use.append(row)
    for seq, row in enumerate(use, 1):
        key = row["prefix_key"]
        recency[key] = seq
        if row["completed_monotonic"] > now - 60.0:
            contribution = 2.0 ** (-(len(use) - seq) / 4.0) if name == "decay" else 1
            frequency[key] = frequency.get(key, 0) + contribution
    return frequency, recency


def _reserve(keys, busy, limit, reserved, incoming, frequency, recency):
    if incoming in keys:
        return "resident", None
    if len(keys) + reserved < limit:
        return "stage", None
    idle = keys - busy
    if not idle or limit == 0:
        return "no_idle_capacity", None
    name = policy()
    def rank(key):
        digest = prefix_key(key[1])
        heat = frequency.get(digest, 0) if name in ("hotness", "decay") else 0
        return heat, recency.get(digest, 0), digest
    victim = min(idle, key=rank)
    if name in ("hotness", "decay") and rank(victim)[0] > rank(incoming)[0]:
        return "incoming_colder", None
    return "stage", victim


def begin_prefill(state, config):
    """Called with the owning state lock before any actual page lookup."""
    attempt = config["attempt_id"]
    ids = tuple(config["token_ids"])
    if state.cache.block_size != 16 or len(ids) < 16:
        raise ValueError("this profile requires one complete 16-token prefix")
    if not hasattr(state, "retention_pending"):
        state.retention_pending, state.retention_done = {}, {}
    if attempt in state.retention_pending or attempt in state.retention_done:
        raise ValueError("repeated physical attempt")
    now = time.monotonic()
    view = snapshot(now)
    freq, last = _scores(view, now)
    key = (state.binding, ids[:16])
    with state.pool.lock:
        pending = state.retention_pending
        receipt = dict(attempt_id=attempt, prefix_key=prefix_key(ids), policy=policy(),
                       tier=tier(), selection_sequence=view["sequence"], ledger=view["sha256"])
        item = dict(key=key, token_ids=ids, receipt=receipt, cpu=None, gpu=None, source=None, staged=False)
        gpu_budget = int(os.environ.get("BLOOMBEE_RETAINED_GPU_BLOCKS", str(state.pool.capacity_pages)))
        if gpu_budget < 0 or gpu_budget > state.pool.capacity_pages:
            raise ValueError("invalid retained GPU budget")
        for side, keys, busy, limit in (
            ("cpu", set(state.cache.entries), set(state.cache._active_keys), state.cache.capacity_blocks),
            ("gpu", set(state.pool.lookup), {k for k,i in state.pool.lookup.items() if state.pool.pages[i].refs}, gpu_budget)):
            if tier() in (side, "both"):
                holds = [p for p in pending.values() if p[side + "_action"] == "stage"]
                # Pending slots cannot be accessed by other requests or evicted.
                if len(keys) + len(holds) > limit:
                    raise RuntimeError("existing history exceeds this group's budget")
                action, victim = _reserve(keys, busy, limit, len(holds), key, freq, last)
            else:
                if keys:
                    raise RuntimeError("disabled history tier must start empty")
                action, victim = "disabled", None
            item[side + "_action"] = action
            receipt[side] = dict(action=action, victim=None if victim is None else prefix_key(victim[1]),
                                 retained_before=len(keys), capacity=limit)
            item[side + "_victim"] = victim
        # Both plans validated. Old Placement evicts at begin, before copy.
        for side in ("cpu", "gpu"):
            victim = item[side + "_victim"]
            if victim is not None:
                if side == "cpu":
                    del state.cache.entries[victim]
                    state.remote_cpu_sources.pop(victim, None)
                else:
                    slot = state.pool.lookup.pop(victim)
                    state.pool.pages[slot].key = None  # Preserve its CUDA completion fence.
        pending[attempt] = item
    return receipt


def stage(state, session, config, read_layers):
    """Capture owned CPU tensors / one physical GPU slot; neither becomes readable here."""
    item = state.retention_pending[config["attempt_id"]]
    if item["staged"]:
        raise RuntimeError("attempt already staged")
    key = item["key"]
    slot, generation = session.slots[0]
    with state.pool.lock:
        page = state.pool.pages[slot]
        if session.closed or page.refs <= 0 or page.generation != generation or page.valid != 16:
            raise RuntimeError("physical first-page lease invalid")
        if item["cpu_action"] == "stage":
            # Use the genuine page pool schema/digest/copy validator. Publish only at terminal.
            if key not in state.cache.entries:
                result = state.cache.put_completed_blocks(key[1], read_layers, priorities=(100,))
                if not result["admitted"] or result["evicted"] or result["new_blocks"] != 1:
                    raise RuntimeError("CPU staging violated its capacity reservation")
                item["cpu"] = state.cache.entries.pop(key)
                provenance = dict(state.remote_gpu_sources.get((slot,generation), {"origin":"local_compute"}))
                digest = item["cpu"].digest
                if provenance.get("source_block_digest") not in (None,digest):
                    provenance = dict(origin="local_transformed", previous_source=provenance.get("sha256"))
                item["source"] = dict(provenance, last_transfer="gpu_to_cpu", cpu_block_digest=digest)
        if item["gpu_action"] == "stage" and key not in state.pool.lookup:
            if page.key is not None:
                raise RuntimeError("cannot stage a differently registered physical page")
            page.refs += 1
            item["gpu"] = slot, generation
        item["staged"] = True
    return dict(item["receipt"], cpu_staged_bytes=item["cpu"].nbytes if item["cpu"] else 0,
                gpu_staged_slot=item["gpu"])


def finish_attempt(state, attempt_id, success, *, expected_tokens=None):
    """CPU-only metadata control. Called for every backend, after success ledger append or failure."""
    if type(success) is not bool:
        raise ValueError("success must be bool")
    with state.lock, state.pool.lock:
        done = getattr(state, "retention_done", {})
        if attempt_id in done:
            if done[attempt_id]["success"] != success:
                raise ValueError("contradictory repeated terminal")
            return done[attempt_id]
        item = getattr(state, "retention_pending", {}).get(attempt_id)
        row = None
        reuse = False
        if success:
            matches = [e for e in snapshot()["events"] if e.get("attempt_id") == attempt_id]
            if len(matches) != 1 or type(matches[0].get("kv_reuse_success")) is not bool:
                raise ValueError("missing uniquely attested complete terminal and reuse flag")
            row = matches[0]
            reuse = row["kv_reuse_success"]
            if expected_tokens is not None and list(expected_tokens) != row["token_ids"]:
                raise ValueError("terminal input mismatch")
        if item is None:
            if reuse:
                raise KeyError("successful reuse did not enter this layer")
            return dict(attempt_id=attempt_id, success=success, kv_reuse_success=False, status="absent")
        key, cache, pool = item["key"], state.cache, state.pool
        if success and (row["prefix_key"] != prefix_key(key[1]) or tuple(row["token_ids"]) != item["token_ids"]):
            raise ValueError("prefill and terminal full input differ")
        if reuse and not item["staged"]:
            raise ValueError("successful reuse missing validated layer stage")
        # Single-tier baselines also need successful computed KV for pre-seeding.
        # R3--R8 have the original Placement success predicate for actual copies.
        publish = item["staged"] and success and (reuse or policy() == "dynamo")
        if item["gpu"] is not None:
            slot, generation = item["gpu"]
            page = pool.pages[slot]
            if page.generation != generation or page.refs <= 0 or page.key is not None:
                raise RuntimeError("lost staged GPU lease")
            if publish and key not in pool.lookup:
                page.key = key
                pool.lookup[key] = slot
            page.refs -= 1  # Previous model stream fence stays valid.
        if publish and item["cpu"] is not None and key not in cache.entries:
            cache.entries[key] = item["cpu"]
            state.remote_cpu_sources[key] = item["source"]
        row = dict(item["receipt"], success=success, kv_reuse_success=reuse,
                   status="committed" if publish else ("cold_discard" if success else "aborted"),
                   cpu_resident=key in cache.entries, gpu_resident=key in pool.lookup,
                   cpu_keys=sorted(prefix_key(k[1]) for k in cache.entries),
                   gpu_keys=sorted(prefix_key(k[1]) for k in pool.lookup),
                   cpu_resident_bytes=cache.resident_bytes(),
                   gpu_slot=pool.lookup.get(key))
        del state.retention_pending[attempt_id]
        done[attempt_id] = row
        return row


def terminal(state, payload):
    if not enabled():
        return dict(policy="dynamo", status="native", attempt_id=payload["attempt_id"])
    status = payload["status"]
    if status not in ("complete_success", "failed", "cancelled"):
        raise ValueError("unknown terminal status")
    return finish_attempt(state, payload["attempt_id"], status == "complete_success",
                          expected_tokens=payload.get("token_ids"))
