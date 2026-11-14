# BloomBee KV Network

BloomBee can retain key/value (KV) tensors on a node, publish successful reuse events to a distributed hash table (DHT), and prefetch remote prefixes through LMCache. An ordinary inference request can then use the prefetched tensors on the consumer GPU.

## Modules and options

```text
BloomBee KV Network
├── Local retention
│   ├── lru         Evict by least recent successful use
│   ├── hotness     Count successful uses in a time window
│   ├── decay       Discount older successful uses
│   └── dynamo      Apply block priorities and GPU/CPU retention rules
├── Distributed telemetry
│   └── Record successful model reuse and query it by time window
└── Remote prefetch
    ├── lrfu        Rank prefixes by recency and frequency
    └── transition  Predict the next prefix from observed transitions
```

## Repository layout

```text
src/bloombee/
├── kv_cache/           Local KV retention and distributed telemetry
│   └── network/        Public API, prefetch policies and LMCache integration
├── client/             Inference sessions and request metadata
└── server/             Model execution, physical KV pages and prefix reuse

configs/                Service and deployment configuration
examples/               Runnable API and inference examples
tests/kv_network/       Tests for the KV Network modules
docs/                   Installation, API, runtime and validation guides
```

See the [API reference](docs/kv-network-api.md) for eviction rules, tie handling, priorities and storage contracts.

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

The examples use A for local retention, B for telemetry and C for prefetch. The following API sketch assumes an application-supplied `dht`, qualified
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

## Validation

On two H20 hosts, three consecutive requests from the public Qwen Coder trace produced all 234 reference output tokens. For the final request, LRFU selected a prefix, LMCache transferred 16 MiB, and the consumer reused 16 tokens across 32 layers. The final request took 9.46 seconds with reuse and 2.40 seconds cold.

See [validation and workload details](docs/validation.md), [native configuration](docs/native-configuration.md), [upstream compatibility](docs/upstream-compatibility.md), and [licenses](NOTICE-KV.md).
