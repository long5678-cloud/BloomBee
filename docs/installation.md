# Installation and dependency boundaries

## Lightweight policies and scoring

The package's top-level model initialization is lazy. Serialized Placement, Telemetry ledger types, validated pure strategy scoring and option validation can be imported without loading model packages or LMCache. DHT publication requires Hivemind. Tensor/Dynamo APIs require torch. HTTP/ticket operations require httpx. Native execution additionally requires official LMCache and pyzmq.

For source-based CPU review:

```bash
python3.11 -m venv .venv-review
. .venv-review/bin/activate
python -m pip install -r requirements-review.lock.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m bloombee.kv_cache.network check --local lru --telemetry
```

This is a CPU test environment, not the historical model runtime lock. The delivered local receipt records its exact resolved distributions. The normal model package still declares upstream dependencies; model/GPU use requires a compatible complete BloomBee installation:

```bash
python -m pip install -e '.[dev]'
```

The upstream setup includes CUDA-oriented packages and is not a promise of full model inference on macOS. This review's Python tests use an existing CPU torch environment with isolated project-local dependency targets; neither those absolute development paths nor installed targets belong in a published repository.

## Official native runtime

Pin LMCache source to v0.5.7 commit `236f3eec32a2d330662a1795d90a01a1e0782c49`. Build/install it in its supported Linux environment according to the pinned source's installation instructions and native extension prerequisites. A conceptual source pin is:

```text
lmcache @ git+https://github.com/LMCache/LMCache.git@236f3eec32a2d330662a1795d90a01a1e0782c49
```

This is a source requirement, not proof that a generic pip command builds all official native components on every platform. Redis 7.2.4 TCP was used in the historical run. The native SDK, coordinator, multiprocess server and official Valkey L2 adapter are required; no vLLM model process is substituted. BloomBee model and native CPU environments can differ, but the matching package must be importable in each.

The historical native CPU import receipt reported torch `2.8.0+cu128`, CUDA uninitialized, and source-based LMCache metadata `unknown`. A complete historical transitive environment lock was not recovered. The source commits, four official loaded-source pins, model revision, Redis version and actual execution evidence are retained; they must not be represented as a complete pip freeze.

## Import and bootstrap

Normal installation replaces historical hardcoded `pack_a6` source-path injection. Historical bootstrap compatibility was scoped to missing Gemma client registration and did not replace real Llama blocks. The latest-main model import is a separate check in validation. The delivery does not silently install a global import overlay or fake model module.

## Reproducibility

Use `provenance/source_manifest.json` to inspect destination hashes and their sources. Historical runtime files have old hashes; reorganized package files have new hashes. Worker protocol pins refer to the new packaged files. Changing one requires regenerating those explicit pins and re-running the relevant checks. Old GPU evidence cannot certify changed source behavior.

## Declared model dependencies versus the observed CPU profile

The CPU source-review environment resolved Transformers 5.20.2, torch 2.14.1,
Hugging Face Hub 2.2.0 and NumPy 2.4.6. Its exact observed distributions are in
`requirements-review.lock.txt`. This profile is distinct from upstream's model
package constraints (`huggingface-hub<2`, `numpy<2`) and the historical GPU
runtime. A normal dependency-resolving model installation under those bounds
has not been certified by these tests. The wheel is separately installed into
a new target with `--no-deps` and tested against the observed CPU profile;
that checks packaged contents/imports, not a fresh complete GPU resolver/build.
The constraints were not silently widened merely to make this environment pass.


## Linux model environment: independent installation

Use Linux x86-64, Python 3.10 or 3.11, an NVIDIA GPU and a driver supporting
CUDA 12.8 for the initial torch 2.8.0 acceptance profile. This is a finite
model profile, not a claim that every GPU or CUDA version is supported. A normal
Python interpreter with the standard `venv` module is required. The script uses
`--without-pip` and either system pip >=22.3 as a bootstrap tool or the official
PyPA HTTPS bootstrap, so Ubuntu's missing
`ensurepip` does not require a system package change. HTTPS access to PyPA,
PyPI/PyTorch dependencies and the pinned public Hivemind Git repository is needed.

From this candidate checkout, choose a new absolute environment directory on
your data disk. For example, if your own data directory is `/data/user`:

```bash
bash scripts/install_linux.sh /data/user/env-model
/data/user/env-model/bin/python -m bloombee.kv_cache.network check --local dynamo
```

Replace `/data/user` with an existing directory you own; it is an example, not a
server address or repository-specific requirement. Do not reuse an existing
environment name. The script rejects it and preserves its contents. It clears
Python import overlays, creates an isolated environment, resolves all declared
model and development dependencies under `constraints/model-linux.txt`, runs
`pip check` and imports the real model classes. It does not use `--no-deps` or
system site packages, select a GPU, launch a model or download model weights.
If the standalone bootstrap is needed, its SHA is retained. The dependency
install report and `resolved.txt` are always retained in
the new environment. Preserve the full terminal log for a failed install.

This profile is under real-environment acceptance; a successful dependency
install alone does not establish model, native LMCache or cross-host correctness.


### Slow shared filesystem

An environment contains many small files. If installation blocks on a shared
filesystem, an isolated node-local memory-backed environment is supported on
Linux: check `df -h /dev/shm` and reserve at least 20 GiB of free memory-backed
space, then use a new name such as `/dev/shm/env_a1` as the script's environment
argument. This stores executable dependencies in RAM, not on the system disk.
It is ephemeral and must be recreated after reboot. Keep the source, full terminal
log, downloaded-asset cache and copies of the install report/resolved versions on
your data disk. The source directory must remain present for this editable install.
Do not change global Python, shared installations, system mounts or other jobs.


## Current real-function acceptance scope

Fresh environment creation/resolution is outside the current acceptance scope.
The optional installer above has not been certified and is not a prerequisite
for the current functional checks. Reuse an existing compatible model/native
environment without modifying it. The source-mode command shown in README
explicitly selects this checkout's `src`; retain the resulting module paths and
file hashes. It is a source selection, not a private compatibility bootstrap or
a claim of a clean dependency install. Missing dependencies are reported as an
environment prerequisite; do not change another user's shared environment.
