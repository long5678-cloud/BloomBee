# BloomBee KV network — local review branch

Composable local retention, distributed usage telemetry, and LMCache-backed remote prefetch for the existing BloomBee project. This checkout is based on upstream `0028a5e89ddb634106a2d86fed0197105a377b97`; the earlier fork branch was `9612af3f4002a69f74ec317b7e3e97af61bcadf6`. It is an uncommitted review candidate, not a published release.

```text
src/bloombee/
  kv_cache/
    placement.py                  # Original whole-prefix byte API
    api.py, dynamo.py              # Cache factories and CPU block retention
    telemetry*.py                 # Ledger, DHT query, model-evidence adapters
    page_retention.py             # Terminal-confirmed physical retention
    network/
      api.py                      # Options, Components, validated score()
      ordinary.py                 # Ordinary token-input model request adapter
      controller.py               # Current ordinary-request controller (d1)
      worker.py, worker_protocol.py
      joint_runtime.py, joint_strategy.py
      native_page_bridge.py       # Official LMCache SDK/coordinator bridge
      ready_bank.py, bundle_contract.py
      lrfu_policy.py, transition_policy.py
      __main__.py                 # Explicit check/score/worker/front/server CLI
  client/                         # Normal model/session RPC, opt-in metadata
  server/                         # Physical pages, ready reader, normal backend
examples/kv_network.py             # Real CPU retention example
examples/ordinary_request.py       # Ordinary token-input call
configs/server.local.json          # Explicit local service settings template
docs/                             # Installation, API, runtime, evidence
provenance/                       # Source manifest and change evidence
tests/kv_network/                 # Portable CPU/contract checks and fixtures
```

## Module → options → calls

| Module | Options | Calls and responsibility |
|---|---|---|
| Local retention | `lru`, `hotness`, `decay`, `dynamo` | `create_cache(...)`; storage contracts differ. Normal service selects the same four names with `BLOOMBEE_RETENTION_POLICY`. |
| Telemetry | enabled/disabled; window, retention, TTL, membership | `record(Event(...)) → publish() → query(start,end)`; successful complete-group reuse only. |
| Prefetch | enabled/disabled; `lrfu` or `transition` | `score(context,strategy)` is pure; native runtime qualifies history, warms via LMCache, prepares/acks/retires/activates a bundle, then normal BloomBee uses it. |

A=local retention, B=Telemetry, C=Prefetch in the examples below. These module combinations are unrelated to the historical cache factory options named `A`, `B`, and `A+B`.

## Quick start

Use Python 3.11 for the local checks. See [installation](docs/installation.md) for model and optional native dependencies.

```bash
python -m pip install -r requirements-review.lock.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m bloombee.kv_cache.network check --local lru
python examples/kv_network.py --policy hotness
python -m pytest --noconftest tests/kv_network tests/test_kv_cache_placement.py tests/test_kv_cache_telemetry_api.py -q -p no:cacheprovider
python tests/kv_network/check_page_retention.py /path/to/new-test-output
```

The last command requires a new output directory. The CPU suite includes real loopback DHT peers and HTTP transfer; synthetic tensors and model doubles are explicitly labeled and do not count as GPU/LMCache model validation.

## Independent and combined calls

The following API sketch assumes an application-supplied `dht`, qualified
`context`, and complete `native_config`; it is not a standalone copy/paste script.
Runnable CPU examples (real retention/DHT/ticket lifecycle; no model or LMCache
transfer claim) are:

```bash
python examples/kv_network_combinations.py --modules A
python examples/kv_network_combinations.py --modules B
python examples/kv_network_combinations.py --modules AB
python examples/kv_network_combinations.py --modules C
python examples/kv_network_combinations.py --modules ABC --strategy transition
```

For real C transfer/model consumption, prepare the actual services, manifests,
identity and controller config in [ordinary runtime](docs/ordinary-runtime.md).
The CPU example deliberately does not invent this deployment state.

```python
from bloombee.kv_cache.network.api import Components, Options, score

# A only. Does not load Transformers, Hivemind or LMCache.
a = Components(Options(local="lru"), local_kwargs={"capacity_bytes": 64 << 20})

# B only. Supply an actual DHT object; creating a publisher starts no service.
b = Components(Options(telemetry=True), telemetry_kwargs={
    "dht": dht, "namespace": "application", "node_id": "consumer-a"})

# A+B. A's actual completion facts must be passed to B by the caller.
ab = Components(Options(local="decay", telemetry=True),
    local_kwargs={"capacity_bytes": 64 << 20},
    telemetry_kwargs={"dht": dht, "namespace": "application", "node_id": "consumer-a"})

# Pure C selection: validated completed history and the already qualified C set.
selection = score(context, "transition")

# A+B+C: real native configuration, no implicit model or worker launch.
abc = Components(Options(local="lru", telemetry=True, prefetch="lrfu"),
    local_kwargs={"capacity_bytes": 64 << 20},
    telemetry_kwargs={"dht": dht, "namespace": "application", "node_id": "consumer-a"},
    prefetch_kwargs={"config": native_config})
ready = abc.prefetch.start()  # Explicit loopback ticket service startup.
# The configured worker/controller and normal model use this runtime's APIs.
abc.close()
```

`Components` exposes real implementations; it does not silently convert serialized bytes into GPU pages or make every model call use a cache. Use [the ordinary request adapter](docs/ordinary-runtime.md) for model integration. C does not require distributed B publication, but it always requires successful full-request history. Disabling any component does not launch it.

## Ordinary BloomBee

```bash
# Use an existing compatible model environment; see docs/installation.md.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=0 python -m bloombee.kv_cache.network server \
  --config configs/server.local.json -- /path/to/model \
  --new_swarm --block_indices 0:32 --device cuda:0 --torch_dtype float16
```

Edit the immutable execution context and capacities in the configuration first. Use the normal BloomBee peer address produced by the server. The server command preserves disabled SDPA and disabled FP16 reduced-precision reduction. It does not select a device, download weights, prepare a conversion, or launch LMCache. Normal upstream [server instructions](docs/upstream-readme.md) still apply.

For a minimal token request see `examples/ordinary_request.py`. For remote C, install/run the official pinned coordinator/MP/Redis services and supply their real identities and endpoints as described in [ordinary runtime](docs/ordinary-runtime.md). Cache data must cross the configured network endpoint; shared weights are not shared cache evidence.

## Validation and limits

Historical integrated execution: ten groups, 60/60 short requests, 180 output tokens; eight Remote groups each used a real LMCache cross-node 16 MiB transfer followed by local reuse. R1–R6 used c1; R7/R8 used d1. Those measurements precede this reorganization and the latest-upstream merge.

The short workload used 33 mapped input tokens and three generated tokens, not full original coding requests. It did not prove prediction ready before natural queue-head arrival, elimination of on-demand waiting, a winning predictor, or general acceleration. Prefetch's target is CPU. Numeric correctness and actual cache-use events remain separate facts.

Current local results and exact gaps are in [validation](docs/validation.md). The new latest-upstream GPU/model/native end-to-end path requires a separate real-environment acceptance; CPU checks cannot certify it. No zero-bug, arbitrary-model, concurrency, TP, or production-readiness guarantee is made.

See [native configuration and audit](docs/native-configuration.md), [API](docs/kv-network-api.md), [evidence](docs/evidence.md), [upstream compatibility](docs/upstream-compatibility.md), and [licenses](NOTICE-KV.md).
