"""Legacy trace/loopback HTTP example; synthetic CPU bytes, not model KV or speed evidence."""

import argparse
import hashlib
import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice
from urllib.request import urlopen

from bloombee.kv_cache.placement import Layer, verify
from bloombee.kv_cache.prefetch import Prefetcher, RemotePrefix


def replay(trace, limit=20):
    objects, directory, reveals, requests = {}, {}, [], []
    calls = []
    namespace = "synthetic-cpu/qwen-block-ids/16/v1"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            blob = objects.get(self.path.removeprefix("/"))
            if blob is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"

    def fetch(endpoint_arg, sha, *, allowed_endpoints):
        if endpoint_arg not in allowed_endpoints:
            raise ValueError("untrusted endpoint")
        with urlopen(f"{endpoint_arg}/{sha}", timeout=3) as response:
            blob = response.read(65537)
            if len(blob) > 65536:
                raise ValueError("payload too large")
        calls.append(dict(sha=sha, nbytes=len(blob), monotonic_ns=time.monotonic_ns()))
        return blob

    def publish(blocks):
        layers = []
        prefix_digest = hashlib.sha256(json.dumps(blocks).encode()).hexdigest()
        for i in range(2):
            identity = dict(namespace=namespace, prefix=prefix_digest,
                            layer_start=i, layer_end=i + 1)
            header = json.dumps(dict(identity=identity), sort_keys=True).encode()
            # Tiny deterministic stand-in; never described as a model tensor.
            data = hashlib.sha256(f"{prefix_digest}/{i}".encode()).digest() * 32
            blob = b"KVSLAB01" + struct.pack("!I", len(header)) + header + data
            sha = hashlib.sha256(blob).hexdigest()
            objects[sha] = blob
            layers.append(Layer(json.dumps(identity, sort_keys=True), sha, len(blob),
                                time.time() + 60, endpoint))
        prefix = RemotePrefix(namespace, tuple(blocks), tuple(layers))
        directory[prefix.key] = prefix

    try:
        with Prefetcher(fetch, 65536, allowed_endpoints={endpoint}) as pool:
            with open(trace) as stream:
                # No list(stream), next-row hint, sort, or precomputed future candidates.
                for line_index, line in enumerate(islice(stream, limit)):
                    row = json.loads(line)
                    blocks = tuple(row["hash_ids"])
                    reveals.append(dict(line=line_index + 1, timestamp=row["timestamp"],
                                        block_count=len(blocks), monotonic_ns=time.monotonic_ns()))
                    before = len(calls)
                    result = pool.arrive(str(line_index + 1), namespace, blocks)
                    pool.observe(namespace, blocks)
                    used, payloads = result.prefix, result.payloads
                    fallback = False
                    if not payloads:
                        candidates = [p for p in directory.values()
                                      if blocks[:len(p.blocks)] == p.blocks]
                        if candidates:
                            used = max(candidates, key=lambda p: len(p.blocks))
                            payloads = tuple(fetch(x.endpoint, x.sha256, allowed_endpoints={endpoint})
                                             for x in used.layers)
                            fallback = True
                    for layer, blob in zip(used.layers if used else (), payloads):
                        verify(layer, blob, time.time())
                    checksum = hashlib.sha256(b"".join(payloads)).hexdigest() if payloads else None
                    pool.finish(result, reuse_success=bool(result.payloads))
                    requests.append(dict(line=line_index + 1, status=result.status,
                                         waited_s=result.waited_s, prefix_blocks=len(used.blocks) if used else 0,
                                         prefetch_reuse=bool(result.payloads), demand_fallback=fallback,
                                         checksum=checksum, payload_bytes=sum(map(len, payloads)),
                                         fetch_callbacks_during_request=len(calls) - before))
                    # Only the disclosed row can create/publish new remote descriptors.
                    for length in (1, 4, 16):
                        if len(blocks) >= length:
                            publish(blocks[:length])
                    pool.schedule(tuple(directory.values()))
            # Context closes/cancels remaining wrong guesses, recording wasted bytes.
        events = pool.events
        return dict(evidence="synthetic CPU consumer + real loopback HTTP, no model",
                    pacing="immediate sequential disclosure; no manufactured request delay",
                    trace=str(trace), reveals=reveals, requests=requests, events=events,
                    http_callbacks=calls, reserved_after_close=pool.reserved_bytes,
                    summary=dict(rows=len(requests), cold_first=requests[0]["status"] if requests else None,
                                 ready_before_arrival=sum(x["status"] == "ready_before_arrival" for x in requests),
                                 inflight_wait=sum(x["status"] == "inflight_wait" for x in requests),
                                 misses=sum(x["status"] == "miss" for x in requests),
                                 fetched_payload_bytes=sum(x["nbytes"] for x in calls),
                                 consumed_prefetch_bytes=sum(x["payload_bytes"] for x in requests if x["prefetch_reuse"])))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(replay(args.trace, args.limit), ensure_ascii=False, indent=2))
