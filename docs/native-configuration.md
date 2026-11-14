# Native configuration and independent execution audit

The package now contains `network/independent_audit.py`, derived from the historical `prefetch_joint_a1/staging/audit_ec.py`. It independently reconstructs E from full successful terminal history H, remote state R and local state L, then checks E=C using metadata snapshots. It does not call the candidate filter, SDK, warm, or ticket claim. Missing/stale/conflicting evidence, incomplete history, changed snapshots and evidence-write failure never authorize execute.

Controller uses this builtin when `independent_audit` is omitted or `"builtin"`. Explicit `null` disables the audit and continues to block execute. An external `{path,sha256,function}` remains available; wrong SHA/import errors block construction. Worker consumes the Controller's actual execute file, verifies its independent-audit receipt path, SHA and context authorization, and only then enters its existing single warm executor.

The adopted gate is the fixed 32-layer FP16/32-head/128-dimension, one 16-token page profile. It requires an untruncated completion history with `initial_completion_seq=0`. It checks actual receipt files/SHA, consumer and clock provenance, full model output, source manifest/run/instance/incarnation, coordinator event/heartbeat state, ticket inventory, page `status`/`page_probe`, reader lease inventory and before/after snapshot consistency. A missing publication is not replaced with a guessed value or fresh timestamp.

The current independent frontier check revises the historical audit (SHA-256 `6ce18ecef90884471a5e791d506aa2dab74862e0a4561d11f333af002b7b0dcf`). The worker pins the revised audit and candidate filter separately. In LMCache v0.5.7, `seconds_since_last_event` measures idle time since admission. `candidate_max_delivery_delay_s` instead bounds the last admitted batch's emitter-to-coordinator delay (default 60 seconds, legacy configuration key `candidate_max_event_age_s`). A valid recent heartbeat, continuous observed sequence, no reported stream loss and successful delivery to the directory consumers are required. Absent or negative delay cannot establish timely delivery. HTTP heartbeats carry no emitter sequence. Two stable coordinator snapshots bound the observed state but cannot exclude a final unreported batch. The subsequent manifest, target inventory and model-use checks still apply.

## Generate a complete bundle

Start from `configs/native-deployment.example.json`. Its `REPLACE` fields, empty membership and null incarnation deliberately fail validation; they are not runnable fake identities. Supply real service endpoints/registrations, source incarnation, actual 32-layer identity templates, immutable source manifests, complete disclosed history, model assets and local directory mappings. The three referenced JSON input files are not inferred from future requests. An empty fresh history is valid but cannot predict a cold first request.

Copy `declared_execution_digest` and `context` from the model launch that produced the source manifests. The generator verifies that these values reproduce the execution digest on all 32 published layers. Existing deployment inputs need the source launch declaration to regenerate a bundle.

Set `max_session_tokens` to the largest input plus requested output in a serial replay. Without explicit `gpu_capacity_blocks`, the generator takes the larger of 17 and the number of 16-token pages needed for that session. An explicit capacity must cover the declared session length. Each physical page costs 524,288 bytes **per layer** in this fixed profile. The default `gpu_budget_bytes` covers all configured pages in one layer. A larger explicit byte ceiling is allowed. Check available device memory separately for all 32 layers, model weights and other allocations. Concurrent sessions can require additional pages.

```bash
python -m bloombee.kv_cache.network configure-native \
  --deployment /path/to/real-deployment.json --output /path/to/new-bundle
python -m bloombee.kv_cache.network validate-native --bundle /path/to/new-bundle
```

The bundle contains:

- `worker.json`: native/coordinator/page endpoints, actual source incarnation, 32 identities, strategy/plugin pin, release-helper pin, clock receipt and bounded bank.
- `client.json` and `job.json`: ordinary adapter, complete controller/history/provenance configuration, local provider log, terminal ledger and explicit Telemetry switch.
- `source-native.json`: actual source SDK endpoint and immutable manifest publication path.
- `source-server.json` / `target-server.json`: normal model page configuration and exact Remote identity namespace; target page-control port and source publication hook.
- `source-manifests.json` / `initial-history.json`: caller-supplied real evidence, copied unchanged.
- `launch.json`: explicit per-host Redis/coordinator/source MP/target MP/model/worker command arrays using the official pinned runtime flags. No private bootstrap, model download, SSH or automatic GPU selection.
- `deployment.json`: supplied configuration. Keep any optional ticket-control secret private; commands and normal validation output do not print it.

Paths are actual execution-node paths. Generate using the installed package on the intended target, or deploy the bundle at the explicitly specified config directories and regenerate package-local pins there. Model/state/config paths on the two hosts must match the supplied mappings. Do not treat copying weights as transferring cache data.

## Launch and validate the intended nodes

The package assumes the pinned official Linux LMCache installation and Redis binary exist. Existing services may be reused only if their actual registrations, source namespace and incarnation match the configuration. Before applying this real profile, collect the real source publication/manifests/history; do not relabel synthetic test fixtures as those inputs.

Each launch command starts one explicitly named component on the current host and writes a new local log. Invoke it in the intended node's normal shell; run long-lived services in separate sessions. No remote control is performed by this command.

```bash
# Source-side components (or retain the already matching live services):
python -m bloombee.kv_cache.network launch-native --bundle /path/to/bundle --component redis
python -m bloombee.kv_cache.network launch-native --bundle /path/to/bundle --component coordinator
python -m bloombee.kv_cache.network launch-native --bundle /path/to/bundle --component source_mp
python -m bloombee.kv_cache.network launch-native --bundle /path/to/bundle --component source_model

# Target-side components:
python -m bloombee.kv_cache.network launch-native --bundle /path/to/bundle --component target_mp
python -m bloombee.kv_cache.network launch-native --bundle /path/to/bundle --component target_model
CUDA_VISIBLE_DEVICES='' python -m bloombee.kv_cache.network launch-native \
  --bundle /path/to/bundle --component worker
```

Source and target model peer/swarm routing must identify the intended ordinary provider. The input peer/swarm values are deployment facts; generation does not discover or attest them. Use actual input tokens via `examples/ordinary_request.py --config client.json --context ACTUAL_CONTEXT ...` after the services and source publication are ready. The consumer provider log must be visible at the configured node-local path to this adapter. Its append-only terminal ledger must be the same path configured for physical retention.

On the target, before requests:

```bash
python -m bloombee.kv_cache.network validate-native \
  --bundle /path/to/bundle --runtime-files
```

This additional check verifies actual model/log/clock/terminal references and local history clock. It still does not assert service health, cross-machine clock comparability or physical KV transfer. The real audit consults live coordinator/registrations, bank inventory, page-control status/probe and reader lease registry for every execution decision. The ticket endpoint is obtained from the actual worker-ready receipt; its optional authentication is supplied by the same worker configuration. Clock-domain labels are not fabricated boot-time evidence.

## Validation scope

CPU regressions retain all original 28 metadata cases plus a receipt-SHA mismatch case. Actual Controller file-gate tests cover builtin authorization, explicit-disabled refusal, stale/missing/local/changed state, wrong external SHA and worker consumption/rejection. These deterministic metadata controls are not official SDK/GPU transfers. Real Linux native extensions, 32-layer model adoption and cross-host numerical acceptance require separately versioned runtime receipts.


## Reusing one admitted ordinary provider for finite cases

`max_session_tokens` also supplies the default normal-server `inference_max_length`, `attn_cache_tokens` and task-pool `max_batch_size`. Explicit values cannot be smaller. An optional positive `max_chunk_size_bytes` bounds ordinary prefill chunks identically on the cold and cached providers. The page budget covers K/V and auxiliary buffers per layer; it does not cover weights, the original dense request cache, attention workspace or CUDA reservations. Admission must account for those allocations on the actual device. `provider_log` can name the absolute log of an already running normal provider; it does not fabricate or attest that provider.

Frozen page-control group files may set `retained_gpu_blocks` independently of the physical page count needed by an active sequence. The value must fit every registered layer, and failed switches leave the previous state intact. A C-only case can use `policy=dynamo`, `tier=gpu`, `retained_gpu_blocks=0`: the mandatory model page reader remains enabled, while no idle CPU/GPU prefix is retained after completion. It must still prove an actual official native transfer and subsequent model adoption. No per-request clearing is needed to manufacture remote eligibility.

`examples/cache_case_replay.py` loads the ordinary model once, runs explicit JSONL cases, preserves original offered offsets separately from actual ordinary-client ingress, and records every failure. Its plan names existing configs, immutable frozen group IDs and optional official native sidecars. The driver owns the future workload; the predictor receives only terminal history through `OrdinaryClient`. Each row may specify `purpose=warmup`, `measured` or `prefetch`. Warmup remains a real model request and causal history event but is excluded from measured Telemetry uses.

The first forward-logits timestamp is a compute-side TTFT observation. The ordinary API returns the full fixed-length output at completion, so this timestamp is not a streamed-token delivery claim. Decode rate uses observed forward intervals; total output throughput must additionally report the exact measured wall-time denominator, failed requests and warmup boundary. Cache completion records use a flushed single stdout write, including the newline, on the configured ordinary provider log; human WARNING verbosity must not hide this machine-readable protocol. This does not promise atomic records on every possible external pipe.

Keep source preparation, warmup, measured requests, factory/CPU tests and actual two-host native/GPU receipts separate. A generated bundle, ready receipt or successful cold output is not itself proof of remote adoption.
