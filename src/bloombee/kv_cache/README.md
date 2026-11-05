# Local KV prefix cache policies

A BloomBee consumer can keep a complete prefix snapshot in CPU memory and
reuse it on a later request. The caller selects one retention policy per
`Placement` store and shares that store across successive requests.

## Package layout

```text
src/                           Python source root
└── bloombee/                  BloomBee package
    └── kv_cache/              Local prefix cache
        ├── README.md          Usage and policy rules
        ├── __init__.py        Public imports: Group, Layer, Placement
        └── placement.py       Request lifecycle, scoring and eviction
tests/
└── test_kv_cache_placement.py  Public API and policy tests
```

## Follow one request

1. **Look up the prefix.** Pass the 32 layer records and a request ID to
   `begin`. The store removes expired entries and checks for a local copy.
2. **Read the layers.** On a hit, the bound fetcher reads retained bytes.
   On a miss, it calls the caller's fetch function. The caller imports each
   layer and continues inference.
3. **Confirm reuse.** After all 32 layers have been read and used successfully,
   call `finish(reuse_success=True)`. If the snapshot is still valid, the store
   records one access per distinct (prefix, request ID) pair. An admitted miss
   becomes a local entry. A successful hit updates recency.

## Choose a retention policy

| API option | Rule when a new prefix needs space |
| --- | --- |
| `lru` — Least Recently Used | Admit a prefix that fits the total budget. Evict entries in order of their last successful use, oldest first, until enough space is free. |
| `hotness` — Sliding-window access frequency | Score each prefix by its number of successful accesses in the trailing `window_s` seconds (default 60). Consider the lowest scores for eviction first. |
| `decay` — Exponentially decayed access frequency | Within a fixed 60-second window, each successful access contributes a weight of `2 ** (-n / 4)`, where `n` counts later distinct successful accesses across the store. Sum the weights per prefix. Four later accesses halve an event's weight. Consider the lowest scores first. |

Both frequency policies use the scores available at request start. The
incoming prefix must score at least as high as every entry it would displace.
Among residents with equal scores, the least recently used goes first.
A successful request counts even when its prefix is too large or scores too
low to retain. For a comparison with retention disabled, use `remote-only`.

## Call the API

The ordinary inference path uses the cache only when its caller wires in the
following steps. The example assumes the caller already has directory records,
a trusted fetch function and model-specific import and inference functions.

```python
from bloombee.kv_cache import Group, Layer, Placement

cache = Placement(policy="lru", capacity_bytes=256 * 1024**2)

# For each request: records in layer order for one compatible prefix.
group = Group(tuple(Layer.from_record(identity, record) for identity, record in records))
with cache.begin(group, request_id) as request:
    fetch = request.bind_fetcher(fetch_from_holder)
    for layer in group.layers:
        blob = fetch(layer.endpoint, layer.sha256, allowed_endpoints=trusted_endpoints)
        import_layer(layer.identity, blob)
    output = continue_inference()
    request.finish(reuse_success=True)
```

The caller supplies the directory records, authorized endpoints, fetch function,
tensor import, and inference. Call `finish` with success only after actual reuse.
Exiting the context without confirming success discards staged data.
A hit reads retained bytes through the same checked callback. The caller still
decodes and validates the tensor schema. The cache expects complete 32-layer
KVSLAB01 prefix groups and tracks one active request per store. Use a clock in
the same time domain as the expiry fields.

The byte budget covers retained and staged payloads. Working tensors, fetch
buffers, request-ID deduplication state, and the event history are separate.
Request IDs and events are kept for the lifetime of this finite-session store.

Run the example requests and check their eviction behavior from the checkout:

```sh
python -m pytest tests/test_kv_cache_placement.py -q
```
