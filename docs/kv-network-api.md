# KV network API

## Local retention

The four current names select retention configurations, not four NVIDIA runtimes.

| Name | Whole-group Placement rule | Ordinary page service |
|---|---|---|
| `lru` | Evict the least recently successfully used group | Terminal-confirmed retention on physical CPU/GPU infrastructure |
| `hotness` | Successful uses in `(now-window_s,now]`; ties use LRU; reject incoming colder groups | Fixed 60-second comparison window, one 16-token prefix page per layer |
| `decay` | Fixed 60-second window; weight halves every four later successful reuse events; ties use LRU | Same event semantics and one-page comparison profile |
| `dynamo` | Separate CPU block API, not serialized Placement | Priority plus idle return order; GPU-first, CPU fills missing range |

```python
from bloombee.kv_cache import Group, Layer, Placement, create_cache
cache = create_cache("hotness", capacity_bytes=64 << 20, window_s=60.0)
with cache.begin(group, request_id) as demand:
    fetch = demand.bind_fetcher(actual_remote_fetch)
    # Import every verified layer into the real consumer, then confirm completion.
    for layer in group.layers:
        blob = fetch(layer.endpoint, layer.sha256, allowed_endpoints=approved_endpoints)
        consume_layer(blob)
    receipt = demand.finish(reuse_success=True)
```

`Group` requires exactly 32 ordered, compatible layers; capacity is resident plus admitted staging payload bytes. The caller's model working tensors and transient source blob are separate. One demand is active at a time. Failed/context-exited demands do not learn success. Duplicate successful group/request events do not increment frequency again. `decay` accepts only the original 60-second window. Do not pass `policy=` alongside the factory option.

```python
cache = create_cache("dynamo", capacity_blocks=16, block_size=16,
    context=execution_context, expected_layers=32, host_offload_min_priority=50)
write_event = cache.put_completed(token_ids, layers, priorities=priorities,
                                  request_id=request_id, final=True)
values, reused_tokens, read_event = cache.get(token_ids)
if values is not None:
    try:
        consume(values, reused_tokens)
    finally:
        cache.release()
```

Dynamo-derived CPU retention copies completed contiguous blocks. Priority equality passes the threshold; the first lower-priority block stops further offload for that request, including subsequent chunks. CPU and GPU may hold copies simultaneously. Reclaim only idle objects, ordered by priority then return/acquisition order. No incoming-hotter-than-victim admission test is added to this rule. This CPU object does not allocate GPU pages.

The ordinary engine owns `EnginePagePool`, completion fences and auxiliary input/output slots. `BLOOMBEE_HISTORY_TIER=cpu|gpu|both` controls residency independently. Non-Dynamo service retention needs its committed terminal ledger and currently supports one retained 16-token prefix page per real layer. Process-level policies are not arbitrary concurrent per-request switches.

The inherited 18-name `create_cache` registry is retained for compatibility. `A/B/A+B`, `E/F/G/H`, tiered and other historical research variants are separate storage contracts, not current formally selected four-option configurations. `remote-only` disables old Placement retention; it is not a fifth service policy.

## Telemetry

```python
from bloombee.kv_cache.telemetry import Event, Telemetry, query, IncompleteTelemetry
publisher = Telemetry(dht, namespace="application", node_id="consumer-a",
    retention_s=3600, max_events=4096, history_start=observed_start)
publisher.record(Event(kv_group_id, request_id, "reuse_success", completed_at))
publisher.publish(ttl_s=60)
result = query(reader_dht, namespace="application", publishers=[publisher.ref],
               start=start_epoch_seconds, end=end_epoch_seconds)
```

Only verified complete-group successful reuse enters `ranked` and `total_uses`; lookup/fetch/import/warm are not reuse. Count by consuming node. One complete 32-layer group used by one request is one event. Use `cache_group_id` from `telemetry_adapter` for computation identity; `Placement.Group.key` also binds payload identity and is not interchangeable.

Windows are `(start,end]` in UTC epoch seconds. Counts sort descending, ties by KV ID; `kv_id=` restricts one group. Membership is an explicit unique `PublisherRef` list, not automatic whole-network discovery. Missing/expired publications, missing pages, checksum errors and uncovered windows raise `IncompleteTelemetry`. `allow_partial=True` returns `complete=False` and issues; unknown is not zero. Publication TTL and event retention use different clocks; publishing/renewing is explicit. Capacity overflow raises; no silent loss of retained events.

`record` returns False for identical replay and raises on conflicting replay. A new process has a new session; no durable cross-restart exactly-once guarantee. Preserve timestamps on retries. Numeric pass status is independent of an observed reuse fact. `stage_counts` counts events; `nbytes` carries bytes. Test/historical namespaces must be separate from measured namespaces.

`PagedTelemetry.finish` validates a complete fixed-length generation and 32 tagged layer records. Missing/contradictory evidence raises, rather than recording zero. `warmup`/`prefetch` purposes are excluded. Native `generate_with_telemetry` and ordinary `PagedTelemetry` are two adapters for the same statistical implementation.

## Prefetch

```python
from bloombee.kv_cache.network.api import score
result = score(qualified_context, "lrfu")  # Or "transition"
```

Both strategies validate the shared schema and score the same C/H. H contains causally disclosed successful full requests, never future arrivals, queue contents, failed requests or speculative fetches. C is qualified from history, remote complete availability and target absence. Unknown qualification abstains; invalid provenance raises. Disabling prefetch is a separate option.

`lrfu`: sum `2**(-(N-s)/4)` over matching completion sequences. Half-life is four successful request events, not seconds or accesses. `math.fsum`, highest positive score, ascending candidate-key tie break; otherwise abstain.

`transition`: use the current identity-scope contiguous completion suffix. Condition on its last prefix P; all outgoing P transitions, including targets outside C, enter the denominator. Select the C target only if edge support >=2 and exact confidence >=1/2; ties by key. No smoothing or LRFU fallback. Truncated/insufficient history abstains. These constants are frozen, not public tuning parameters. At most one complete prefix is selected per decision.

Pure scoring does no I/O. Native execution requires official LMCache services and valid source manifests. The shared data path is coordinator prefetch → SDK retrieve → complete wire/identity validation → bounded ready bank → reader prepare → acknowledgement → exact target L1 retirement → activate → ordinary model consumption. No simulated SDK success substitutes for this chain.

After a successful non-final request discloses its history, the controller publishes a single execute command for an audited, nonempty selection. The existing worker rechecks eligibility and warms asynchronously. On ingress the controller claims a ready matching object or falls back immediately if it is unavailable. An optional initial history decision still starts at the first actual ingress. Classify a claim as ready before arrival only when its bank-ready timestamp precedes that request's actual ingress in the same clock domain. Model adoption is verified separately. Final decisions start no future warm. Completed-request history remains active with distributed Telemetry disabled.
