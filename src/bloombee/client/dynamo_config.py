"""Validate optional first-prefill metadata without changing tensor RPC layout."""


def normalize_dynamo_config(config):
    if not isinstance(config, dict) or set(config) - {"context", "token_ids", "priorities", "attempt_id"}:
        raise ValueError("dynamo_cache expects context, token_ids and optional priorities")
    context = config.get("context")
    if not isinstance(context, str) or not context:
        raise ValueError("dynamo_cache requires a nonempty model/execution context")
    ids = config.get("token_ids")
    if not isinstance(ids, (list, tuple)) or not ids or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("dynamo_cache token_ids must be the actual nonempty first-prefill IDs")
    result = {"context": context, "token_ids": list(ids)}
    if "attempt_id" in config:
        attempt_id = config["attempt_id"]
        if not isinstance(attempt_id, str) or not attempt_id or len(attempt_id)>200:
            raise ValueError("invalid stable attempt identity")
        result["attempt_id"] = attempt_id
    priorities = config.get("priorities")
    if priorities is not None:
        if not isinstance(priorities, (list, tuple)) or any(
            x is not None and (type(x) is not int or not 0 <= x <= 100) for x in priorities
        ):
            raise ValueError("dynamo_cache priorities must contain 0..100 integers or None")
        result["priorities"] = list(priorities)
    return result
