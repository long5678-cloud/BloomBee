import hashlib
import json
import struct

import pytest

from bloombee.kv_cache import Group, Layer, Placement


def make_prefix(label):
    layers = []
    payloads = {}
    for index in range(32):
        identity = {"prefix": label, "layer_start": index, "layer_end": index + 1}
        header = json.dumps({"identity": identity}).encode()
        blob = b"KVSLAB01" + struct.pack("!I", len(header)) + header + b"test payload"
        sha = hashlib.sha256(blob).hexdigest()
        layers.append(Layer(json.dumps(identity), sha, len(blob), 1000.0, "cache-node"))
        payloads[sha] = blob
    return Group(tuple(layers)), payloads


def consume(store, prefix, request_id):
    group, payloads = prefix
    with store.begin(group, request_id) as request:
        fetch = request.bind_fetcher(lambda endpoint, sha, **kwargs: payloads[sha])
        for layer in group.layers:
            blob = fetch(layer.endpoint, layer.sha256, allowed_endpoints=[layer.endpoint])
            assert hashlib.sha256(blob).hexdigest() == layer.sha256
        return request.finish(reuse_success=True)


@pytest.mark.parametrize("policy", ["hotness", "decay", "lru"])
def test_second_request_uses_retained_prefix(policy):
    group, payloads = make_prefix("A")
    store = Placement(policy, group.nbytes, clock=lambda: 1.0)
    first = consume(store, (group, payloads), "first")
    second = consume(store, (group, {}), "second")
    retry = consume(store, (group, {}), "second")
    assert first["remote_objects"] == 32
    assert second["hit"] and second["remote_objects"] == 0
    assert second["success_increment"] == 1 and retry["success_increment"] == 0
    assert store.resident_bytes == group.nbytes


@pytest.mark.parametrize("policy, next_hit", [("hotness", False), ("decay", True), ("lru", True)])
def test_change_of_hot_prefix(policy, next_hit):
    # Only one prefix fits. A has three early uses, followed by three uses of B.
    # Before the third B arrives its two recent uses outweigh A only with decay.
    a, b = make_prefix("A"), make_prefix("B")
    store = Placement(policy, a[0].nbytes, clock=lambda: 1.0)
    for i, prefix in enumerate([a, a, a, b, b, b]):
        consume(store, prefix, str(i))
    assert consume(store, b, "next")["hit"] is next_hit


def test_failed_request_can_be_retried():
    group, payloads = make_prefix("A")
    store = Placement("lru", group.nbytes, clock=lambda: 1.0)
    with pytest.raises(ValueError, match="size"):
        with store.begin(group, "first") as request:
            fetch = request.bind_fetcher(lambda *args, **kwargs: b"bad")
            layer = group.layers[0]
            fetch(layer.endpoint, layer.sha256, allowed_endpoints=[layer.endpoint])
    assert store.resident_bytes == 0
    assert consume(store, (group, payloads), "first")["success_increment"] == 1


def test_clock_error_releases_request():
    timestamps = iter([2.0, 1.0, 3.0, 4.0])
    group, _ = make_prefix("A")
    store = Placement("lru", group.nbytes, clock=lambda: next(timestamps))
    with pytest.raises(ValueError, match="clock"):
        with store.begin(group, "first"):
            pass
    with store.begin(group, "retry") as request:
        assert not request.hit
