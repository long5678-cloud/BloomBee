# Ordinary BloomBee requests and native runtime

## Local-only model request

The existing model/session interface remains the execution path:

```python
with model.inference_session(max_length=36, dynamo_cache={
    "context": execution_context, "token_ids": actual_ids,
    "priorities": [100, 0], "attempt_id": unique_attempt_id}):
    output = model.generate(input_tensor, max_new_tokens=3, min_new_tokens=3,
                            do_sample=False, attention_mask=attention_mask)
```

The service must enable `BLOOMBEE_DYNAMO_SERVICE=1` and `BLOOMBEE_DYNAMO_GPU_ENGINE=pages`. The ordinary integration currently targets batch one, one active paged session, 32-layer Llama, FP16. Other model families/concurrent sessions are not certified. CPU pseudo-layers 0/1 carry K/V and auxiliaries within one real transformer layer. LMCache's 64 storage slots represent 32 real layers.

`configs/server.local.json` is a local-only template. Replace the context with a compatible immutable execution identity and choose capacities for actual resources. GPU page budget includes K/V and auxiliary data; it is not total process VRAM. The source model/revision, tokenizer and numeric execution settings must match the cache identity.

## Request adapter

`OrdinaryClient(model, output_root=NEW_DIRECTORY, context=..., config=..., job=...)` wraps an already loaded normal model. It accepts actual token tensors and never maps future trace hashes. `generate` returns `(output_tensor, terminal_receipt)` and persists the actual model outcome before observer actions. Fixed-length full completion is required; an early EOS is outside this profile. It creates a new append-only `terminal_ledger`; configure the server to read that same node-local path before requests.

For local terminal-confirmed `lru/hotness/decay`, set `cache_evidence=True`, `provider_log`, `consumer_node_id` and `telemetry_namespace`, with `distributed_telemetry=False` if DHT publication is disabled. The adapter validates real layer evidence and writes successful reuse facts to the local retention ledger. Creating a local serialized `Placement` object does not automatically wire it into physical model pages.

For distributed Telemetry set `distributed_telemetry=True`. When enabled, its namespace and explicit consumer ID are mandatory. With B disabled no distributed publisher or namespace is required; strict local layer validation uses a nonpublishing evidence sink. This uses the normal model's DHT, but the adapter publishes only final complete-request observations. Provider logs must be accessible to the consumer through the chosen deployment design; this profile reads actual node-local tagged logs. It is not an arbitrary remote logging service.

For Prefetch set `prefetch_adapter="builtin"` and supply `prefetch_controller` containing:

- `inbox`: new local protocol directory shared with the CPU worker.
- `worker_config_path`: complete immutable worker JSON.
- `request_count`: positive finite number of ordinary requests in this controller lifetime.
- `history_boundary`: explicit consumer/window/clock/initial completion sequence and nontruncation declaration.
- `initial_history`: real completed terminal records only.
- `source_manifests`: disclosed immutable source bundle manifests, never a future trace.
- `score_metadata`: segment/run/source/config provenance, independent of current request contents.
- `independent_audit`: omitted or `"builtin"` uses the delivered independent metadata audit; external `{path,sha256,function}` is supported. Explicit `null` disables authorization and blocks execute. See [native configuration](native-configuration.md) for complete generation, validation and launch commands.
- Optional `control_timeout_s` and `ipc_timeout_s`: explicit error reporting bounds; they do not kill a model.

Set `ingress_clock_domain` to the actual ingress host's domain matching the page-control status for meaningful readiness ordering. Missing/mismatching domains make readiness ordering unknown, not false proof of prearrival. `control_endpoint` identifies the actual page-control service. Closing a controller drains accepted work and releases owned tickets; it does not select/stop GPUs or unrelated processes.

## Native execution dependencies and configuration

The runtime retains the official LMCache SDK and coordinator routes. It requires a correctly built Linux LMCache v0.5.7 source at commit `236f3eec32a2d330662a1795d90a01a1e0782c49`, coordinator, multiprocess CPU cache instance on the consumer, an actual source instance, and Redis/Valkey TCP L2. Historical Redis was 7.2.4. BloomBee/model and native CPU interpreters may be separate, but both must import this package. Use real endpoint values and normal install/PYTHONPATH rather than historical machine-specific bootstrap paths.

Worker configuration includes:

```text
run_id, source_run_id, source_instance
native_endpoint = {instance_id, url, http_url}
lmcache_target = same actual consumer registration
coordinator_url, page_control_url, candidate_consumer_id
page_identity_template_by_layer = 32 actual compatible layer identities
page_model_name = codec's static namespace
release_helper = {path, sha256} for packaged release_l1.py
strategy_attempt_dir = new local attempt directory
prefetch_strategy = lrfu | transition
clock_domain_file = actual node-local domain receipt
```

Optional ready-bank limits are bounded (`<=64 MiB`, `<=4` prefixes); transfer probes must provide actual physical logger evidence. Missing probes return unknown bytes/GETs rather than deriving them from tensor size. The SDK connects lazily; creating a runtime does not by itself retrieve KV.

Start an explicitly configured worker with:

```bash
CUDA_VISIBLE_DEVICES='' python -m bloombee.kv_cache.network worker \
  --config /path/to/worker.json --inbox /path/to/new-inbox
```

The launcher supplies package-local plugin and source pins. It does not launch official LMCache services, prepare source KV, or manufacture identity templates. Start those services using the pinned official CLI and generate source KV with the real source model. A `source_bundle_manifest` must reflect actual store/readback, identity, checksum and 32-layer geometry. The source ordinary service sets `BLOOMBEE_NATIVE_PAGE_CONFIG` to its real publication config; its adapter is now a normal package import.

Normal model consumption uses prepare → ack → exact target L1 retirement → activate. Reading shared cache files is not a network transfer. Local and remote configuration changes are process-level and must be applied to the matching service; this package does not silently switch global policies per request.

## Finite trace-derived harness

`python -m bloombee.kv_cache.network front --run-dir /path/to/new-run` retains the historical ingress/queue/full-generation instrumentation. The directory supplies `job_config.json` and `front_config.json`, including explicit `model_path`, peer, output directory, ledgers and consumer identity. Set `completion_adapter="builtin"` and optionally `prefetch_adapter="builtin"`. This finite harness is separate from the ordinary token-input SDK.

Historical initial-history restoration and trace mapping remain special harness operations, not general APIs. Reproducing the old short batch requires the archived `group_config_a1.py` that reads `short_test_a1.jsonl`; the later working script reads a different file and is not interchangeable.

## Compatibility and acceptance

A historical pack_a6 bootstrap was located and source-pinned, but was absent from the final run's complete manifest. It conditionally avoided an incompatible Gemma client registration while retaining the real block class. This package uses normal registration; the latest-upstream full model import and Linux native extension environment must be checked before GPU acceptance. No stub is provided as a production replacement.

The instance-local `logits_to_keep` forwarding adapter is retained, including explicit zero/index tensors. Server launch preserves FP16 numeric execution settings. Offload/conversion assets and source-read compatibility are the normal upstream installation's responsibility; the historical machine-specific asset attestation wrapper is not silently claimed to run here. Reorganized/new-upstream model numerics remain a separate acceptance item.

Local reuse without any remote identity configuration now leaves its optional
telemetry identity explicitly unavailable instead of forcing Remote startup.
Any explicit partial/invalid remote identity settings still fail validation;
actual Telemetry requires valid identity evidence from the adopted pages/logs.
This does not weaken the explicit cross-module page-probe contract.

On Linux page-control retains the original kernel boot domain. On systems
without that file it uses an explicit `BLOOMBEE_CLOCK_DOMAIN` or a per-process
UUID domain. Status and ready registry use the same domain. Cross-process or
cross-host monotonic comparisons are unknown unless a genuine common domain is
supplied; a UUID is not a fabricated kernel boot identity.

## Failure and retry semantics

All calls advance an independent attempt sequence. Successful completion history
advances only after validated evidence and a durable terminal ledger write;
failed attempts do not enter predictor history. Layer/telemetry artifacts use
hashed attempt IDs with exclusive creation, preserving prior failures.

Absent priorities remain absent. The ordinary API does not invent a first-block
priority; explicit zero and user-supplied values reach normal validation intact.

The persisted model terminal is independent of cache commit. A successful model
whose observation fails remains `complete_success` in its original receipt,
while the cache control receives an explicit abort/cancelled transaction. No
success ledger is fabricated to force commit. `LifecycleError.terminal` retains
the actual model outcome and `.errors` lists stage failures. If inference itself
failed, that original exception is rethrown with `kv_lifecycle_errors` and
`kv_terminal`; close errors are attached without masking it.

Telemetry is strictly opt-in and boolean. Publish failures remain errors, but
cannot skip page/ticket terminal notification, successful-history disclosure
when qualified, or Prefetch close. Cleanup is attempted independently; partial
close can be retried without replacing evidence or resubmitting model work.
Network failures may still make cleanup unconfirmed; those errors are reported,
not suppressed or reclassified as successful release.


Non-measurement purposes (`warmup` and `prefetch`) still require the same strict
32-layer local evidence before cache commit or predictor history disclosure.
They are excluded from distributed telemetry measurements only. Local receipts
retain `evidence_kind` and actual `kv_reuse_success` separately from publication.
A transient prefetch terminal/disclosure failure retains the original attempt,
cache terminal status and causal history snapshot. The adapter retries these
in order before the next claim or close. It does not replay model generation,
remove successful history, or cancel an already successful cache transaction.
Persistent retry failure remains visible and prevents a new claim or clean close.


History is published as complete generations: a new draft containing the old
history plus the new success must pass full-write, flush and file-fsync checks
before atomic replacement of the reader-visible ledger. Drafts and prior ledger
generations are retained. A publication failure leaves the old ledger and memory
history unchanged and aborts only this cache transaction. This covers process I/O
errors; it does not claim filesystem power-loss or multiwriter crash recovery.

Closing always attempts resource shutdown even when a pending terminal or
history disclosure still fails. The original terminal and history snapshots
remain in the close receipt and the error remains visible. Resource shutdown is
not confirmation that a remote cache terminal or history disclosure succeeded.
The controller preserves each original terminal intent, attempts worker drain
independently, and closes local HTTP clients even after cleanup errors.
