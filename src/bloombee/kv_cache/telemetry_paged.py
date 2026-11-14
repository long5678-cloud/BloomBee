"""Completion adapter for ordinary BloomBee paged generation (fixed-length workload)."""
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import threading

from .identity import CacheIdentity, canonical_digest, token_ids_digest
from .telemetry import Event
from .telemetry_adapter import cache_group_id


class EvidenceError(ValueError):
    """The supplied evidence cannot support a terminal event; publish no guess."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


@dataclass(frozen=True)
class BoundRows:
    attempt_id: str
    rows: tuple


class ExclusiveLogWindow:
    """Legacy single-provider log. Every producer must serialize the entire call here.

    Use tagged rows instead for overlapping RPCs. A rotated or truncated log is rejected.
    Call collect() before exiting; it reads only complete lines from this byte range.
    """
    def __init__(self, path, attempt_id):
        self.path, self.attempt_id = Path(path), attempt_id

    def __enter__(self):
        self.file = self.path.open("rb")
        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
            self.start = self.file.seek(0, os.SEEK_END)
            if self.start:
                self.file.seek(-1, os.SEEK_END)
                require(self.file.read(1) == b"\n", "capture begins inside a log line")
        except BaseException:
            self.file.close()
            raise
        return self

    def collect(self):
        stat = os.fstat(self.file.fileno())
        require(os.path.samestat(stat, self.path.stat()) and stat.st_size >= self.start,
                "provider log rotated or truncated during capture")
        self.file.seek(self.start)
        data = self.file.read(stat.st_size - self.start)
        require(not data or data.endswith(b"\n"), "capture ends inside a log line")
        events = []
        for line in data.splitlines():
            _, marker, payload = line.partition(b"DYNAMO_SERVICE ")
            if marker:
                events.append(json.loads(payload))
        return BoundRows(self.attempt_id, tuple(events))

    def __exit__(self, *exc):
        self.file.close()


def _prefill_rows(rows, uids, attempt):
    by_uid = {}
    bound = isinstance(rows, BoundRows)
    if bound:
        require(rows.attempt_id == attempt, "wrong exclusive window")
    for row in rows.rows if bound else rows:
        if row.get("event") != "dynamo_service":
            continue
        tag = row.get("telemetry_attempt_id")
        if tag is None:
            require(bound, "untagged provider row requires an exclusive capture")
        elif tag != attempt:
            require(not bound, "foreign tagged prefill inside exclusive capture")
            continue
        uid = row.get("uid")
        require(uid in uids and uid not in by_uid, "unexpected or duplicate layer in final attempt")
        by_uid[uid] = row
    require(set(by_uid) == set(uids), "final attempt has missing transformer layers")
    return [by_uid[u] for u in uids]


def _count(row, field):
    value = row.get(field)
    require(type(value) is int and value >= 0, "invalid layer field " + field)
    return value


def _layer_skip(row, length):
    skip = _count(row, "skipped_tokens")
    require(_count(row, "requested_tokens") == _count(row, "returned_tokens") == length and
            _count(row, "computed_tokens") + skip == length and skip < length,
            "layer work conflicts with prompt length")
    mode = row.get("mode")
    if mode == "fallback":
        require(skip == 0, "fallback layer claims a skipped prefix")
        return 0
    require(row.get("gpu_engine") == "physical_pages" and mode == ("reuse" if skip else "cold"),
            "unsupported or inconsistent ordinary prefill mode")
    gpu, cpu, remote = (_count(row, key) for key in ("gpu_hit_tokens", "cpu_hit_tokens", "remote_hit_tokens"))
    require(skip == gpu + cpu + remote, "tier hits do not sum to actual skipped tokens")
    if skip:
        require(row.get("native_attention_in_place") is True, "page attention use unconfirmed")
    if cpu or remote:
        require(row.get("native_import_exact") is True, "page import unconfirmed")
    return skip


def _layer_identity(row, depth, prefix_hash, layer):
    pages = [p for p in row.get("active_gpu_pages", ()) if p["logical_start"] + p["tokens"] == depth]
    require(len(pages) == 1, "common prefix endpoint is not an adopted page")
    page = pages[0]
    sources = [s for s in row.get("gpu_block_sources", ()) if s.get("logical_start") == page["logical_start"]]
    require(len(sources) == 1 and (sources[0].get("slot"), sources[0].get("generation")) ==
            (page["slot"], page["generation"]), "source page and adopted page differ")
    records = [sources[0]] + [r for r in row.get("remote_records", ()) if r.get("block_end") == depth]
    identities = []
    for record in records:
        if record.get("cache_identity") is not None:
            ident = CacheIdentity.from_dict(record["cache_identity"])
            require(record.get("identity_digest", ident.digest()) == ident.digest(), "source identity digest mismatch")
            identities.append(ident)
    # Locally computed pages may lack remote provenance. The provider can attach
    # its actual block_identity(..., end) for the adopted prefix boundary.
    supplied = ([row["telemetry_identity"]] if row.get("telemetry_identity") else []) + list(row.get("telemetry_identities", ()))
    for value in supplied:
        if value.get("prefix_length") == depth:
            identities.append(CacheIdentity.from_dict(value))
    require(bool(identities), "provider identity missing for adopted common prefix")
    ref = identities[0]
    require(all(x == ref for x in identities) and ref.prefix_length == depth and ref.prefix_tokens_digest == prefix_hash
            and (ref.layer_start, ref.layer_end) == (layer, layer + 1), "prefix, layer or source identity mismatch")
    return ref


class PagedTelemetry:
    """Use one instance per live Telemetry publisher and stable logical request IDs."""

    def __init__(self, publisher, *, layer_uids):
        self.publisher = publisher
        self.uids = tuple(layer_uids)
        require(len(self.uids) == 32 and len(set(self.uids)) == 32 and
                all(isinstance(uid, str) and uid for uid in self.uids), "expected 32 ordered, unique provider layer UIDs")
        self._finished = {}
        self._lock = threading.Lock()

    def finish(self, *, request_id, attempt_id, terminal, completed_at, input_ids, output_ids,
               max_new_tokens, rows=(), evidence="observed", purpose="measured"):
        require(purpose in ("measured", "warmup", "prefetch"), "unknown invocation purpose")
        if purpose != "measured":
            return dict(kind="excluded", recorded=False, event=None, kv_id=None,
                        common_prefix_tokens=0, terminal=terminal, consumer=self.publisher.ref.node_id)
        require(isinstance(attempt_id, str) and bool(attempt_id), "missing final attempt ID")
        require(terminal in ("completed", "cancelled", "failed"), "expected final generate/session status")
        require(type(max_new_tokens) is int and max_new_tokens > 0, "expected fixed positive decode length")
        prompt = list(input_ids)
        prompt_digest = token_ids_digest(prompt)
        kv_id, depth, out_digest = None, 0, None
        if terminal == "completed":
            output = list(output_ids)
            require(len(output) == len(prompt) + max_new_tokens and output[:len(prompt)] == prompt and
                    all(type(t) is int and t >= 0 for t in output), "terminal output does not match fixed-length generate")
            out_digest = canonical_digest(output)
            prefill = _prefill_rows(rows, self.uids, attempt_id)
            skips = [_layer_skip(row, len(prompt)) for row in prefill]
            depth = min(skips)
            if depth:
                prefix_hash = token_ids_digest(prompt[:depth])
                kv_id = cache_group_id([_layer_identity(row, depth, prefix_hash, i) for i, row in enumerate(prefill)])
                kind = "reuse_success"
            elif any(skips) or any(r.get("mode") == "fallback" or r.get("remote_errors") for r in prefill):
                kind = "fallback"
            else:
                kind = "cold_recompute"
        else:
            kind = "request_failed"

        source = canonical_digest([attempt_id, terminal, prompt_digest, max_new_tokens])
        event = Event(kv_id, request_id, kind, completed_at, evidence=evidence,
                      backend_work=True if kind == "reuse_success" else None,
                      output_digest=out_digest, source_digest=source)
        with self._lock:
            now = self.publisher.clock()
            cutoff = now - self.publisher.retention_s
            self._finished = {k: v for k, v in self._finished.items() if v.occurred_at > cutoff}
            previous = self._finished.get(request_id)
            require(previous is None or previous == event, "conflicting final terminal for logical request")
            recorded = self.publisher.record(event)
            self._finished[request_id] = event
        return dict(kind=kind, kv_id=kv_id, common_prefix_tokens=depth, recorded=recorded,
                    terminal=terminal, consumer=self.publisher.ref.node_id, event=event)

