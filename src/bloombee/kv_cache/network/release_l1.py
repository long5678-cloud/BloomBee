"""Release one owned transport chunk after the ordinary cache has taken ownership.

Caller holds the per-arm, per-key lease across observation, retirement and
verification. Use on the target node, against its registered MP endpoint.
This uses only metadata and L1 retirement, never a KV read or a storage reset.
"""
import ipaddress
import time
from urllib.parse import urlsplit


def release_l1_chunk(client, *, layout, target, coordinator_url, source_instance,
                     expected_key, warm_registration, target_incarnation, confirm_seconds=2.0):
    """Return a retirement receipt; any error or unverified status is not a pass.

    expected_key is the complete wire key from the checked warm receipt.
    source_instance is the instance reporting the source's Valkey L2 placement.
    This frozen workload uses one 16-token chunk, TP1 and group zero.
    """
    started = time.monotonic_ns()
    endpoint = urlsplit(target["http_url"])
    ip = ipaddress.ip_address(endpoint.hostname)
    if (endpoint.scheme != "http" or endpoint.path not in ("", "/") or endpoint.query
            or endpoint.username or endpoint.password or endpoint.fragment
            or (not ip.is_loopback and str(ip) != warm_registration["ip"])):
        raise ValueError("expected trusted target-local or registered MP endpoint")
    if (not source_instance or target["instance_id"] == source_instance
            or not layout.cache_salt or len(layout.tokens) != 16
            or expected_key.get("model_name") != layout.model_name
            or expected_key.get("cache_salt") != layout.cache_salt
            or expected_key.get("object_group_id") != 0
            or expected_key.get("kv_rank") != 16777472):
        raise ValueError("unexpected transfer identity")
    if not 0 < confirm_seconds <= 30:
        raise ValueError("invalid confirmation timeout")
    token_query = dict(model_name=layout.model_name, token_ids=list(layout.tokens),
                       world_size=1, cache_salt=layout.cache_salt)

    def locations():
        resp = client.post(coordinator_url.rstrip("/") + "/directory/lookup", json=token_query)
        resp.raise_for_status()
        body = resp.json()
        rows = body["results"]
        if body["chunks"] != 1 or len(rows) != 1 or rows[0]["key"] != expected_key:
            raise ValueError("lookup disagrees with warm receipt key")
        return rows[0]["placements"]

    def source_present(rows):
        return any(p["tier"] == "l2" and p["backend"] == "valkey"
                   and p["instance_id"] == source_instance and p["size_bytes"] > 0 for p in rows)

    def target_l1(rows):
        return [p for p in rows if p["tier"] == "l1" and p["instance_id"] == target["instance_id"]]

    def registration():
        resp = client.get(coordinator_url.rstrip("/") + "/instances")
        resp.raise_for_status()
        matches = [i for i in resp.json()["instances"] if i["instance_id"] == target["instance_id"]]
        if len(matches) != 1 or matches[0]["http_port"] != endpoint.port:
            raise ValueError("target coordinator registration/port mismatch")
        fields = ("instance_id", "ip", "http_port", "registration_time")
        return all(matches[0][f] == warm_registration[f] for f in fields)

    if not registration():
        raise ValueError("target incarnation changed before retirement")
    before = locations()
    local = target_l1(before)
    receipt = dict(key=expected_key, before=before, started_ns=started)
    if (len(local) != 1 or local[0]["incarnation"] != target_incarnation
            or not source_present(before)):
        return dict(receipt, status="preconditions_unverified")
    resp = client.request("DELETE", target["http_url"].rstrip("/") + "/cache/objects",
                          json=dict(keys=[expected_key], tier="l1", force=False))
    resp.raise_for_status()
    ack = resp.json()
    receipt["mp_result"] = ack
    if ack.get("ok") is not True or ack.get("deleted") != 1 or ack.get("skipped") != 0:
        return dict(receipt, status="not_confirmed_by_target")
    deadline = time.monotonic() + confirm_seconds
    while True:
        after = locations()
        if not target_l1(after) and source_present(after) and registration():
            return dict(receipt, status="released", after=after, completed_ns=time.monotonic_ns())
        if time.monotonic() >= deadline:
            return dict(receipt, status="directory_unconfirmed", after=after)
        time.sleep(0.02)
