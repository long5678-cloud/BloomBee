import hashlib
import json
import struct
import threading
import time
import unittest

from bloombee.kv_cache.placement import Layer
from bloombee.kv_cache.prefetch import Prefetcher, RemotePrefix


def fixture(blocks=(0,), label="a", count=2):
    layers, blobs = [], {}
    for i in range(count):
        identity = dict(prefix=label, layer_start=i, layer_end=i + 1)
        header = json.dumps(dict(identity=identity)).encode()
        blob = b"KVSLAB01" + struct.pack("!I", len(header)) + header + b"cpu-test-data"
        sha = hashlib.sha256(blob).hexdigest()
        layers.append(Layer(json.dumps(identity), sha, len(blob), time.time() + 600, "remote"))
        blobs[sha] = blob
    return RemotePrefix("model/revision/schema", tuple(blocks), tuple(layers)), blobs


def terminal(pool, ticket):
    # Test-only barrier; never used in trace replay to create lead time.
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if any(e["event"] == "fetch_end" and e["ticket_id"] == ticket for e in pool.events):
            return
        time.sleep(.001)
    raise AssertionError("worker did not complete")


class PrefetchTests(unittest.TestCase):
    def make_pool(self, blobs, **kwargs):
        pool = Prefetcher(lambda endpoint, sha, **kw: blobs[sha],
                          kwargs.pop("capacity_bytes", 100000), allowed_endpoints={"remote"}, **kwargs)
        self.addCleanup(pool.close)
        return pool

    def test_cold_start_and_unobserved_directory_do_not_fetch(self):
        prefix, blobs = fixture()
        pool = self.make_pool(blobs)
        self.assertEqual(pool.schedule([prefix]), ())
        result = pool.arrive("first", prefix.namespace, prefix.blocks)
        self.assertEqual(result.status, "miss")
        pool.finish(result, reuse_success=False)
        self.assertFalse(any(e["event"] == "fetch_start" for e in pool.events))

    def test_ready_exact_bytes_identity_and_once_only_success(self):
        prefix, blobs = fixture()
        pool = self.make_pool(blobs)
        pool.observe(prefix.namespace, (0, 1))
        ticket, = pool.schedule([prefix])
        terminal(pool, ticket)
        result = pool.arrive("A", prefix.namespace, (0, 99))
        self.assertEqual(result.status, "ready_before_arrival")
        for layer, blob in zip(prefix.layers, result.payloads):
            self.assertIs(blob, blobs[layer.sha256])
        self.assertEqual(pool.reserved_bytes, 0)
        pool.finish(result, reuse_success=True)
        with self.assertRaises(ValueError):
            pool.finish(result, reuse_success=True)
        self.assertEqual(sum(e["event"] == "fetch_bytes" for e in pool.events), len(prefix.layers))

    def test_inflight_wait_is_after_actual_arrival(self):
        prefix, blobs = fixture()
        entered, release = threading.Event(), threading.Event()
        def fetch(endpoint, sha, **kwargs):
            entered.set()
            if not release.wait(2):
                raise TimeoutError()
            return blobs[sha]
        pool = Prefetcher(fetch, 100000, allowed_endpoints={"remote"})
        self.addCleanup(pool.close)
        self.addCleanup(release.set)
        pool.observe(prefix.namespace, prefix.blocks)
        pool.schedule([prefix])
        self.assertTrue(entered.wait(1))
        threading.Timer(.02, release.set).start()  # controlled test fixture, not trace
        result = pool.arrive("A", prefix.namespace, prefix.blocks)
        self.assertEqual(result.status, "inflight_wait")
        self.assertGreater(result.waited_s, 0)
        ready = next(e for e in pool.events if e["event"] == "ready")
        self.assertGreater(ready["ready_ns"], result.arrival_ns)

    def test_cancel_does_not_release_running_reservation(self):
        prefix, blobs = fixture()
        entered, release = threading.Event(), threading.Event()
        def fetch(endpoint, sha, **kwargs):
            entered.set()
            if not release.wait(2):
                raise TimeoutError()
            return blobs[sha]
        pool = Prefetcher(fetch, prefix.nbytes, allowed_endpoints={"remote"})
        self.addCleanup(pool.close)
        self.addCleanup(release.set)
        pool.observe(prefix.namespace, prefix.blocks)
        ticket, = pool.schedule([prefix])
        self.assertTrue(entered.wait(1))
        pool.cancel(ticket)
        self.assertEqual(pool.reserved_bytes, prefix.nbytes)
        self.assertEqual(pool.schedule([prefix]), ())
        release.set()
        terminal(pool, ticket)
        self.assertEqual(pool.reserved_bytes, 0)
        self.assertFalse(any(e["event"] == "ready" for e in pool.events))

    def test_timeout_keeps_ticket_for_later_demand(self):
        prefix, blobs = fixture()
        release = threading.Event()
        def fetch(endpoint, sha, **kwargs):
            release.wait(2)
            return blobs[sha]
        pool = Prefetcher(fetch, prefix.nbytes, allowed_endpoints={"remote"})
        self.addCleanup(pool.close)
        self.addCleanup(release.set)
        pool.observe(prefix.namespace, prefix.blocks)
        ticket, = pool.schedule([prefix])
        result = pool.arrive("A", prefix.namespace, prefix.blocks, timeout=0)
        self.assertEqual(result.status, "inflight_timeout")
        self.assertEqual(pool.reserved_bytes, prefix.nbytes)
        pool.finish(result, reuse_success=False)
        release.set()
        terminal(pool, ticket)
        self.assertTrue(pool.arrive("B", prefix.namespace, prefix.blocks).payloads)

    def test_corruption_failure_no_payload_or_success(self):
        prefix, blobs = fixture()
        pool = self.make_pool({sha: b"corrupt" for sha in blobs})
        pool.observe(prefix.namespace, prefix.blocks)
        ticket, = pool.schedule([prefix])
        terminal(pool, ticket)
        result = pool.arrive("A", prefix.namespace, prefix.blocks)
        self.assertEqual(result.status, "failure")
        self.assertFalse(result.payloads)
        self.assertEqual(pool.reserved_bytes, 0)
        with self.assertRaises(ValueError):
            pool.finish(result, reuse_success=True)
        pool.finish(result, reuse_success=False)

    def test_capacity_local_and_trust_skips(self):
        prefix, blobs = fixture()
        pool = self.make_pool(blobs, capacity_bytes=prefix.nbytes - 1)
        pool.observe(prefix.namespace, prefix.blocks)
        self.assertEqual(pool.schedule([prefix]), ())
        self.assertIn("byte_capacity", [e.get("reason") for e in pool.events])
        other = self.make_pool(blobs)
        other.observe(prefix.namespace, prefix.blocks)
        self.assertEqual(other.schedule([prefix], local_keys={prefix.key}), ())
        other.allowed_endpoints = frozenset()
        self.assertEqual(other.schedule([prefix]), ())
        self.assertIn("untrusted_endpoint", [e.get("reason") for e in other.events])

    def test_frequency_recency_and_bounded_history(self):
        a, blobs = fixture((0,), "a")
        b, more = fixture((1,), "b")
        blobs.update(more)
        pool = self.make_pool(blobs, history_size=2)
        pool.observe(a.namespace, a.blocks)
        pool.observe(b.namespace, b.blocks)
        pool.observe(b.namespace, b.blocks)  # a is no longer visible
        ticket, = pool.schedule([a, b])
        terminal(pool, ticket)
        self.assertEqual(pool.arrive("A", a.namespace, a.blocks).status, "miss")
        self.assertEqual(pool.arrive("B", b.namespace, b.blocks).prefix, b)

    def test_wrong_guess_does_not_count_used(self):
        prefix, blobs = fixture()
        pool = self.make_pool(blobs)
        pool.observe(prefix.namespace, prefix.blocks)
        ticket, = pool.schedule([prefix])
        terminal(pool, ticket)
        result = pool.arrive("A", prefix.namespace, (42,))
        self.assertEqual(result.status, "miss")
        pool.finish(result, reuse_success=False)
        self.assertFalse(any(e["event"] == "used" for e in pool.events))
        self.assertEqual(pool.reserved_bytes, prefix.nbytes)
        pool.cancel(ticket)
        self.assertEqual(pool.reserved_bytes, 0)

    def test_expired_ready_is_not_consumed(self):
        prefix, blobs = fixture()
        now = [time.time()]
        pool = self.make_pool(blobs, clock=lambda: now[0])
        pool.observe(prefix.namespace, prefix.blocks)
        ticket, = pool.schedule([prefix])
        terminal(pool, ticket)
        now[0] += 1000
        self.assertEqual(pool.arrive("A", prefix.namespace, prefix.blocks).status, "miss")
        pool.schedule([prefix])
        self.assertEqual(pool.reserved_bytes, 0)

    def test_namespace_and_chain_not_last_block_match(self):
        prefix, blobs = fixture((1, 2))
        pool = self.make_pool(blobs)
        pool.observe(prefix.namespace, (9, 2))
        pool.observe("different-model", (1, 2))
        self.assertEqual(pool.schedule([prefix]), ())

    def test_prefix_record_validates_layer_coverage_and_identity(self):
        prefix, _ = fixture()
        with self.assertRaises(ValueError):
            RemotePrefix(prefix.namespace, prefix.blocks, tuple(reversed(prefix.layers)))
        other, _ = fixture(label="other")
        with self.assertRaises(ValueError):
            RemotePrefix(prefix.namespace, prefix.blocks, (prefix.layers[0], other.layers[1]))


if __name__ == "__main__":
    unittest.main()
