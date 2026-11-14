"""CPU functional demo: two real DHT publishers and an independent query peer.

All events are explicitly test fixtures, not new model-success evidence.
Run from an installed BloomBee environment: python examples/kv_cache_telemetry.py
"""
import hashlib
import json
import time
import uuid

from hivemind import DHT
from bloombee.kv_cache.telemetry import Event, Telemetry, query
from bloombee.kv_cache.telemetry_adapter import generate_with_telemetry


def call_existing_native_generator(generator, telemetry, token_ids, request_id, prefix_length):
    """Actual callable integration; caller owns the already-loaded native backend.

    This helper does not construct a model or pretend the CPU demo is a model.
    The supported native generator returns (output_ids, metrics), emits one
    record_reuse callback per imported layer AFTER fixed-length decoding.
    """
    output_ids, metrics, status = generate_with_telemetry(
        generator, telemetry, token_ids, request_id=request_id,
        prefix_length=prefix_length, max_new_tokens=3)
    if status.get("recorded"):
        status["snapshot"] = telemetry.publish()
    return output_ids, metrics, status


def main():
    nodes, cleanup = [], []
    try:
        first = DHT(start=True, host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False, client_mode=False)
        nodes.append(first)
        for _ in range(2):
            nodes.append(DHT(start=True, initial_peers=first.get_visible_maddrs(),
                             host_maddrs=["/ip4/127.0.0.1/tcp/0"], use_relay=False, client_mode=False))
        epoch = time.time()
        clock = lambda: epoch + 10
        namespace = "demo-" + uuid.uuid4().hex
        pubs = [Telemetry(n, namespace=namespace, node_id="test-node-" + str(i),
                          clock=clock, history_start=epoch) for i,n in enumerate(nodes[:2])]
        kv = hashlib.sha256(b"test-only-cache-group").hexdigest()
        for p in pubs:
            event = Event(kv, "request", "reuse_success", epoch + 5, evidence="test")
            assert p.record(event) and not p.record(event)
        pubs[0].record(Event(kv, "failed-request", "fallback", epoch+6, evidence="test"))
        pubs[1].record(Event(kv, "unused-prefetch", "prefetch_unused_bytes", epoch+6, evidence="test"))
        for p in pubs:p.publish()
        result = query(nodes[2], namespace=namespace, publishers=[p.ref for p in pubs],
                       start=epoch, end=epoch+10, kv_id=kv, clock=clock)
        assert result["total_uses"] == 2 and result["ranked"][0]["nodes"] == {"test-node-0":1,"test-node-1":1}
        report = dict(event_evidence="test", real_model_executed=False, real_DHT_peers=[dict(pid=n.pid, peer_id=str(n.peer_id)) for n in nodes], result=result)
    finally:
        for n in reversed(nodes):
            n.shutdown();n.join(timeout=10)
            cleanup.append(dict(pid=n.pid, alive=n.is_alive(), exitcode=n.exitcode))
    assert all(not p["alive"] and p["exitcode"] == 0 for p in cleanup)
    report["cleanup"] = cleanup
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
