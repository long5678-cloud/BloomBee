# Current validation

The latest-upstream source checkout and the separately installed wheel each pass **279 tests**, with **one CUDA-only test skipped**. An additional **205 terminal/page-pool comparisons** pass in an independent CPU process. The five runnable A/B/AB/C/ABC component examples exit successfully. The retained legacy Prefetch pool also passes 12 compatibility tests from the installed wheel.

## Current checks

- Original Placement byte contracts, rollback, expiry, capacity and replay accounting.
- Dynamo CPU pools, actual physical page allocator, priority/idle reclaim, active leases, continuation and cleanup.
- Actual miniature production Llama attention/loader/backend, automatic multi-chunk processing, prefix reuse and exception cleanup. The updated fixture uses a real local HF checkpoint and latest identity-bound conversion, without bypassing the loader or using network weights.
- Latest upstream GLM non-equal-width MLA chunk/prefill/decode comparisons against the actual HF block; recovery-history regression.
- Real local Hivemind peers and independent readers, DHT windows/missing/expired publications, idempotency/capacity and exact counts.
- Real CPU tensor data over local DHT/HTTP in the standalone demand-remote profile. This is separate from the official LMCache path and is not cross-physical-host proof.
- Codec/checksum/identity/frame/byte-bound checks, separate CPU HTTP processes, ready-bank lifecycle, prepare/ack/release rejection and candidate-state controls.
- Actual A/B/AB/C/ABC component construction/use/close, including Placement, DHT publication and native runtime's local ticket lifecycle. The C lifecycle examples do not claim an official SDK transfer or model adoption.
- Both public strategy options, fixed event decay and transition thresholds/denominators, abstention/invalid provenance, d1 warm-before-reserve ordering and worker-done exit race.
- Four archived ordinary 32-layer outcomes replayed against the delivered PagedTelemetry adapter with two new real local DHT peers and strict negative controls; no new model request is claimed for this replay.
- Cold/failed ordinary adapter wiring, explicit consumer identity without experiment GPU-job coupling, configuration/module-off/import checks. Model doubles are confined to explicitly labeled wiring tests.
- Real model/metadata/LoRA class imports. Packaging verifies all **214 installed Python runtime files** against the checkout; the upstream source-only template generator is intentionally not a wheel package.
- A new target installation with `--no-deps`, followed by imports and the same 279-test suite using the installed package with the source directory absent from the import path.

The exact package-file count and wheel checksum are recorded in the accompanying wheel-content receipt; use that machine receipt if a subsequent source revision differs from this document.

## Fixes found by current testing

Local cache hits previously constructed Remote identity settings unconditionally and could fail without a remote deployment. Optional identity logging now stays unavailable for completely unconfigured Local use; explicitly supplied partial/invalid settings still fail strictly. Telemetry does not fabricate missing cache identity.

Linux keeps its original boot clock domain. Non-Linux page-control no longer fails solely because `/proc` is unavailable; it uses an explicit local label or a process UUID. Registry/status remain bound to the same runtime PID/domain, with no invented cross-host monotonic guarantee.

Lazy independent KV imports required a matching lazy utility export to prevent a direct metadata import cycle. Real metadata/model imports are checked in the installed artifact.

Old fixture failures are retained in internal logs. The old profiler's absolute small-concatenation count changed with latest RoPE; the retained assertion verifies exactly two eliminated historical K/V concatenations, unchanged original prefix, suffix writes and the original numerical tolerance. No model numeric threshold was relaxed.

## Explicit gaps

The skipped test requires CUDA completion/fence behavior. No new 32-layer GPU ordinary RPC/native LMCache/coordinator/Redis cross-host experiment was authorized or run. Full numeric output equivalence, actual official SDK/native extension startup and real model adoption on this reorganized latest SHA remain unverified. Prior historical GPU evidence is described separately in evidence.md.

The observed macOS source-review dependencies differ from upstream's declared full model constraints; see installation.md and requirements-review.lock.txt. The new target wheel install verifies contents/imports against that observed CPU profile, not a fresh complete GPU dependency resolver/build. No complete historical transitive model/native environment lock is claimed.

The suite is finite and does not certify arbitrary models, concurrent policy switching, production readiness, universal acceleration or absence of bugs. OrdinaryClient uses the documented fixed-length completion and node-local evidence profile; broader EOS/streaming services need their own integration contract.

## Cross-review repair candidate

The first independent reviews found wrapper/configuration defects beyond the
original 223 tests. Those reports and reproductions remain intact. Current
repair tests cover absent priorities, distinct failed/observed attempts, real
cache abort/references, publication/control errors, real ticket shutdown,
retryable close, explicit B-off/C-only, builtin independent audit/Worker gate,
configuration generation and source/receipt/endpoint rejection.

The source and a freshly installed wheel each pass 279 tests (one CUDA skip),
plus the 205 page comparisons and 12 legacy pool checks. This is the repair
candidate pending the three owners' independent re-review; these counts alone
do not close review findings or extend the native/GPU acceptance scope.


The a4 repair also covers ledger partial-write/flush/fsync/publication faults
followed by successful requests with real CPU page-reference cleanup. Persistent
terminal/disclosure errors still trigger actual ticket service shutdown while
retaining unresolved original handoff evidence. Source and installed-wheel tests
pass 279 cases (one CUDA skip); independent owner review remains a separate gate.


## A5 working-draft status (latest)

The a4 counts and three-owner review above belong to the a4 frozen snapshot,
not the current a5 working changes. Fresh installation acceptance was withdrawn.
The minimal real attempt reached source model readiness but its harness supplied
128 input tokens plus 3 output tokens against a 128-token server limit. No model
completion or remote adoption was accepted. The second consumer had no idle GPU
at admission and was not started. All owned services and GPU processes were
released. A5 has not received final real-environment or three-owner acceptance.
