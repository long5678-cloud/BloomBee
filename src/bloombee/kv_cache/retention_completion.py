"""Retention ordering from disclosed full successful demands only.

This adapter controls the actual page/CPU victims. It does not use prefill,
per-layer reads or speculative fetches as demand events. Each group gets its
own node-local immutable append-only terminal ledger, supplied by the client.
"""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

POLICIES = ('dynamo', 'lru', 'hotness', 'decay')
_lock = threading.RLock()
_cache = {}


def policy():
    name = os.environ.get('BLOOMBEE_RETENTION_POLICY', 'dynamo')
    if name not in POLICIES:
        raise ValueError('unsupported actual retention policy')
    return name


def prefix_key(tokens):
    return hashlib.sha256(json.dumps(list(tokens[:16]), separators=(',', ':')).encode()).hexdigest()


def snapshot(now=None):
    """Read committed newline-terminated entries; reject contradictory history."""
    now = time.monotonic() if now is None else now
    file = os.environ.get('BLOOMBEE_RETENTION_TERMINALS')
    if not file:
        if policy() != 'dynamo':
            raise ValueError('non-Dynamo policy requires an explicit terminal ledger')
        return dict(sequence=0, events=[], sha256=hashlib.sha256(b'').hexdigest())
    path = Path(file)
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        # A concurrent append may be incomplete. Use only the committed prefix.
        raw = raw[:raw.rfind(b'\n')+1]
    digest = hashlib.sha256(raw).hexdigest()
    with _lock:
        if (file, digest) in _cache:
            return _cache[file, digest]
        events = []
        seen = set()
        for line in raw.splitlines():
            row = json.loads(line)
            if row['request_id'] in seen:
                raise ValueError('duplicate terminal demand in retention ledger')
            if row['completion_event_seq'] != len(events)+1:
                raise ValueError('noncontiguous retention demand sequence')
            if row['status'] != 'complete_success' or row['actual_output_tokens'] != row['requested_output_tokens']:
                raise ValueError('nonterminal or incomplete demand in retention ledger')
            if not isinstance(row['terminal_ref'], str) or not row['terminal_ref']:
                raise ValueError('terminal evidence missing')
            if not math.isfinite(row['completed_monotonic']) or row['completed_monotonic'] > now:
                raise ValueError('future retention event')
            if len(row['token_ids']) < 16:
                raise ValueError('short demand must not be silently dropped or bridged')
            if row['prefix_key'] != prefix_key(row['token_ids']):
                raise ValueError('retention prefix identity mismatch')
            if events and row['completed_monotonic'] < events[-1]['completed_monotonic']:
                raise ValueError('unordered retention disclosure')
            seen.add(row['request_id'])
            events.append(row)
        result = dict(sequence=len(events), events=events, sha256=digest)
        _cache.clear()
        _cache[file, digest] = result
        return result


def score(key, view, now):
    """Original Placement semantics: window60, decay half-life4 successes."""
    name = policy()
    events = [e for e in view['events'] if e['prefix_key'] == key]
    last = max((e['completion_event_seq'] for e in events), default=0)
    if name == 'lru':
        return last
    recent = [e for e in events if e['completed_monotonic'] > now-60.0]
    if name == 'hotness':
        return len(recent)
    if name == 'decay':
        return math.fsum(2.0 ** (-(view['sequence']-e['completion_event_seq'])/4.0) for e in recent)
    raise ValueError('Dynamo uses its native priority/release ordering')


def order(native_key, native_order, *, view=None, now=None):
    """One actual compatible block key is (binding, full prefix tokens)."""
    if policy() == 'dynamo':
        return native_order
    now = time.monotonic() if now is None else now
    view = snapshot(now) if view is None else view
    return score(prefix_key(native_key[1]), view, now), native_order


def cap_idle_history(pool, limit):
    """Bound retained keys after lease release, protecting all active pages."""
    if type(limit) is not int or limit < 0:
        raise ValueError('invalid retained page budget')
    idle = [i for i, p in enumerate(pool.pages) if p.key is not None and p.refs == 0]
    now = time.monotonic()
    view = snapshot(now)
    idle.sort(key=lambda i: order(pool.pages[i].key,
                                  (pool.pages[i].priority, pool.pages[i].tick, i),
                                  view=view, now=now))
    for slot in idle[:max(0, len(pool.lookup)-limit)]:
        page = pool.pages[slot]
        pool.lookup.pop(page.key)
        pool.events.append(dict(event='retention_reclaim', policy=policy(), slot=slot,
                                key=repr(page.key), demand_ledger_sha256=view['sha256'],
                                completion_sequence=view['sequence'], reason='idle_history_capacity'))
        page.key = None
        # Existing CUDA completion fences remain intact; allocation still waits
        # for actual stream completion through the original _ready/_wait path.

