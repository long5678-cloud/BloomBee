"""CPU-only public API examples; fixtures are not model/throughput evidence.

Run from a checkout with its src on PYTHONPATH:
    python -m examples.kv_cache_options
"""
import hashlib
import json
import math
import struct

import torch

from bloombee.kv_cache import (
    CacheKey, Group, Layer, RetentionRange, create_cache, describe_options,
    move_tensor_tree, tensor_nbytes,
)
from bloombee.kv_cache.placement import digest


def pack_example(identity, tensors):
    """Example-owned KVSLAB01 float32 encoding, no pickle/network dependencies.

    The cache owns bytes and checks header identity/SHA; a real application's
    transport/backend owns its encoder, decoder and model-specific schema.
    This intentionally small example codec is not a general transport service.
    """
    parts, specs = [], []
    for name, tensor in sorted(tensors.items()):
        if tensor.dtype != torch.float32:
            raise ValueError("example encoder supports float32 only")
        raw = tensor.detach().cpu().contiguous().numpy().tobytes()
        parts.append(raw)
        specs.append(dict(name=name, dtype="float32", shape=list(tensor.shape), nbytes=len(raw)))
    header = json.dumps(dict(identity=identity, tensors=specs), sort_keys=True,
                        separators=(",", ":")).encode()
    return b"KVSLAB01" + struct.pack("!I", len(header)) + header + b"".join(parts)


def unpack_example(blob, identity):
    if len(blob) < 12 or blob[:8] != b"KVSLAB01":
        raise ValueError("example payload magic/length")
    size, = struct.unpack("!I", blob[8:12])
    header = json.loads(blob[12:12+size])
    if header["identity"] != identity:
        raise ValueError("example payload identity")
    offset, tensors = 12 + size, {}
    for spec in header["tensors"]:
        nbytes = math.prod(spec["shape"]) * 4
        if spec["dtype"] != "float32" or spec["nbytes"] != nbytes or offset+nbytes > len(blob):
            raise ValueError("example tensor descriptor")
        tensors[spec["name"]] = torch.frombuffer(bytearray(blob[offset:offset+nbytes]),
                                                 dtype=torch.float32).reshape(spec["shape"])
        offset += nbytes
    if offset != len(blob):
        raise ValueError("example trailing data")
    return tensors


def example_group(label="P", tensors=None, *, identity_extra=None, expiry=1e6):
    tensors = tensors or [{"key": torch.tensor([[float(i)]]),
                           "value": torch.tensor([[float(i+1)]])} for i in range(32)]
    layers, payloads = [], {}
    for index, values in enumerate(tensors):
        identity = dict(model="public-example-fixture", revision="1", prefix=label,
                        layer_start=index, layer_end=index+1, **(identity_extra or {}))
        blob = pack_example(identity, values)
        sha = hashlib.sha256(blob).hexdigest()
        record = dict(identity_digest=digest(identity), payload_sha256=sha,
                      payload_nbytes=len(blob), expires_at=expiry, endpoint="example-holder")
        layers.append(Layer.from_record(identity, record));payloads[sha] = blob
    return Group(tuple(layers)), payloads


def consume_example(cache, group, payloads, request_id):
    """Decode every actual payload before confirming a successful fixture use."""
    calls = []
    def source(endpoint, sha, **kwargs):
        calls.append(sha)
        return payloads[sha]
    with cache.begin(group, request_id) as request:
        fetch = request.bind_fetcher(source)
        for layer in group.layers:
            unpack_example(fetch(layer.endpoint, layer.sha256,
                                 allowed_endpoints=[layer.endpoint]), layer.identity)
        result = request.finish(reuse_success=True)
    return dict(source_calls=len(calls), **result)


def demo():
    result = []
    for info in describe_options():
        option = info["name"]
        if info["family"] == "dynamo-host-prefix":
            ids = [1,2,3,4]
            layers = {i: {"key":torch.ones(4,2)*i,"value":torch.ones(4,2)*(i+1)} for i in range(2)}
            cache = create_cache(option,capacity_blocks=2,block_size=2,context="example/model/schema",
                                 expected_layers=2,host_offload_min_priority=50)
            put = cache.put_completed(ids,layers,priorities=[50,80])
            retained,n,event = cache.get(ids)
            try:
                assert n==4 and all(torch.equal(retained[i][k],layers[i][k]) for i in layers for k in layers[i])
            finally:
                if n:cache.release()
            result.append(dict(option=option,family=info["family"],reused_tokens=n,source=event["source"],
                               cpu_bytes=cache.resident_bytes(),put=put))
        elif info["family"] == "group-bytes":
            group, payloads = example_group()
            cache = create_cache(option, capacity_bytes=group.nbytes, clock=lambda: 1.0)
            first = consume_example(cache, group, payloads, "first")
            second = consume_example(cache, group, payloads, "second")
            result.append(dict(option=option, family=info["family"], first_hit=first["hit"],
                               second_hit=second["hit"], source_calls=second["source_calls"]))
        elif info["family"] == "tiered-snapshot":
            value = {"key": torch.arange(4, dtype=torch.float32).reshape(2,2),
                     "value": torch.ones(2,2)}
            key = CacheKey("example-model/schema", hashlib.sha256(b"complete-prefix-and-content").hexdigest())
            def validate(actual_key, actual):
                if actual_key != key or any(not torch.equal(actual[k], value[k]) for k in value):
                    raise ValueError("incompatible example snapshot")
            cache = create_cache(option, fast_bytes=tensor_nbytes(value),
                slow_bytes=tensor_nbytes(value), size=tensor_nbytes,
                move=move_tensor_tree, validate=validate, context=key.context)
            origins = []
            for _ in range(2):
                with cache.acquire(key, lambda key:value) as lease:
                    validate(key, lease.value);origins.append(lease.origin);lease.finish(True)
            result.append(dict(option=option, family=info["family"], origins=origins))
        else:
            ids = [1,2,3,4]
            layers = {i: {"key":torch.ones(4,2)*i, "value":torch.ones(4,2)*(i+1)} for i in range(2)}
            cache = create_cache(option, capacity_blocks=2, context="example/model/schema",
                expected_layers=2, block_size=2, slow_capacity_blocks=2 if option=="F" else 0)
            cache.put_completed(ids, layers, now=0.0)
            retained, n, event = cache.get(ids, now=1.0)
            try:
                assert n==4 and all(torch.equal(retained[i][k], layers[i][k]) for i in layers for k in layers[i])
            finally:
                if n:cache.release()
            result.append(dict(option=option, family=info["family"], reused_tokens=n, source=event["source"]))
    return dict(kind="CPU functional tensor/byte fixtures; no language-model generation", options=result)


if __name__ == "__main__":
    print(json.dumps(demo(), ensure_ascii=False, indent=2))
