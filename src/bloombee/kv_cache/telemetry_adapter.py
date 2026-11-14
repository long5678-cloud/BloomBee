"""Thin completion adapter; no experiment receipt schema or model execution.

Call once AFTER Native/RemotePrefixGenerator.generate returns. The existing
generator's record_reuse callback can collect layer_reuses; never count each
layer as a request. Diagnostic stage events must also be request/group summaries.
"""
from .telemetry import Event, _digest, _sha


def cache_group_id(identities):
    """Stable logical group ID; distinct from placement.Group.key's payload key.

    Accept Group, ordered identity dicts, or objects exposing to_dict(). Hash
    ordered canonical layer identity digests, preserving the old cache_group_id.
    """
    if hasattr(identities, "layers"):
        identities = [layer.identity for layer in identities.layers]
    rows = [x.to_dict() if hasattr(x, "to_dict") else dict(x) for x in identities]
    if not rows:
        raise ValueError("nonempty ordered layer identities required")
    common = None
    for i, row in enumerate(rows):
        if type(row.get("layer_start")) is not int or row["layer_start"] != i or row.get("layer_end") != i+1:
            raise ValueError("incomplete/out-of-order model layers")
        base = {k:v for k,v in row.items() if k not in ("layer_start", "layer_end")}
        if common is not None and common != base:
            raise ValueError("mixed model/prefix/execution identities")
        common = base
    return _digest([_digest(row) for row in rows])


def record_generation(telemetry, *, request_id, kv_id, metrics, layer_reuses,
                      expected_layers, input_length, prefix_tokens, max_new_tokens,
                      output_ids, completed_at=None, evidence="observed",
                      tokens_match=None, numerical_passed=None, source_digest=None):
    """Translate a completed native generation result into ONE group event.

    kv_id must equal cache_group_id(layer_reuses) on successful reuse. For cold
    or failed reuse the caller still supplies the selected/logical group ID.
    Supports the native RemotePrefixGenerator's fixed-length generation contract
    (no early EOS). Correctness/probe results are independent, not runtime gates.
    Nothing is emitted while decoding or on individual layer callbacks.
    """
    if kv_id is not None:
        _sha(kv_id)
    for value in (expected_layers, input_length, max_new_tokens):
        if type(value) is not int or value <= 0:
            raise ValueError("layer/input/generation counts must be positive integers")
    if type(prefix_tokens) is not int or not 0 <= prefix_tokens < input_length:
        raise ValueError("invalid selected prefix")
    if prefix_tokens and kv_id is None:
        raise ValueError("selected prefix requires a stable group identity")
    ids = list(output_ids)
    if any(type(t) is not int or t < 0 for t in ids):
        raise ValueError("invalid native token output")
    generation_complete = len(ids) == input_length + max_new_tokens
    copies = metrics.get("kv_copy_integrity")
    steps = metrics.get("actual_backend_steps")
    expected_work = input_length - prefix_tokens + max_new_tokens - 1
    work = None if steps is None else all(len([e for e in steps if e.get("layer") == i]) == max_new_tokens and
               sum(e.get("computed_tokens", 0) for e in steps if e.get("layer") == i) == expected_work
               for i in range(expected_layers)) and len(steps) == expected_layers * max_new_tokens
    identities = list(layer_reuses)
    identity_match = False
    if len(identities) == expected_layers:
        try:
            identity_match = cache_group_id(identities) == kv_id
        except (ValueError, TypeError, KeyError):
            identity_match = False
    reused = bool(generation_complete and prefix_tokens and metrics.get("hit") is True and
                  metrics.get("fallback") is None and metrics.get("imported_layers") == expected_layers and
                  metrics.get("skipped_tokens_per_layer") == prefix_tokens and identity_match)
    copy_integrity = None if copies is None else len(copies) == expected_layers and all(
        e.get("key_exact") is True and e.get("value_exact") is True for e in copies)
    kind = "reuse_success" if reused else "request_failed" if not generation_complete else "cold_recompute" if not prefix_tokens else "fallback"
    event = Event(kv_id, request_id, kind, telemetry.clock() if completed_at is None else completed_at,
                  evidence=evidence, tokens_match=tokens_match, numerical_passed=numerical_passed,
                  copy_integrity=copy_integrity, backend_work=work,
                  output_digest=_digest(ids), source_digest=source_digest)
    return dict(reuse_completed=reused, kind=kind, recorded=telemetry.record(event), event=event)


def generate_with_telemetry(generator, telemetry, token_ids, *, request_id, prefix_length,
                            max_new_tokens=1, evidence="observed"):
    """Call the existing native generator; preserve its original completion hook.

    Serial use of a generator only. No model construction or public RPC client
    routing is added. A telemetry error does not invalidate completed generation.
    Returns (output_ids, metrics, telemetry_status). Correctness is caller-owned.
    """
    ids = list(token_ids)
    identities = list(generator.identity_factory(ids[:prefix_length])) if prefix_length else []
    kv_id = cache_group_id(identities) if identities else None
    expected_layers = len(generator.fixture.backends)
    original, completed = generator.record_reuse, []
    def on_reuse(identity):
        completed.append(identity)
        if original is not None:
            original(identity)
    generator.record_reuse = on_reuse
    try:
        output, metrics = generator.generate(ids, prefix_length=prefix_length, max_new_tokens=max_new_tokens)
    except Exception:
        try:
            telemetry.record(Event(kv_id, request_id, "request_failed", telemetry.clock(), evidence=evidence))
        except Exception:
            pass
        raise
    finally:
        generator.record_reuse = original
    try:
        status = record_generation(telemetry, request_id=request_id, kv_id=kv_id, metrics=metrics,
            layer_reuses=completed, expected_layers=expected_layers, input_length=len(ids),
            prefix_tokens=prefix_length, max_new_tokens=max_new_tokens, output_ids=output, evidence=evidence)
    except Exception as error:
        status = dict(recorded=False, error=str(error))
    return output, metrics, status
