#!/usr/bin/env bash
# Independent full model install. Does not alter the system Python environment.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/install_linux.sh /absolute/path/to/new-environment" >&2
  exit 2
fi
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="$1"
case "$ENV_ROOT" in /*) ;; *) echo "Environment path must be absolute" >&2; exit 2;; esac
if [ -e "$ENV_ROOT" ]; then echo "Use a new environment path; existing files are preserved" >&2; exit 2; fi
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
python3 -c 'import sys; assert sys.version_info[:2] in ((3,10),(3,11)), "Use Python 3.10 or 3.11"'
python3 -m venv --without-pip "$ENV_ROOT"
# A modern system pip can provision a new venv without inheriting its packages.
# Otherwise obtain the official standalone bootstrap with bounded networking.
if python3 -m pip --help 2>/dev/null | grep -q -- '--python'; then
  python3 -m pip --python "$ENV_ROOT/bin/python" install 'pip==25.3' 'setuptools==80.9.0' 'wheel==0.45.1'
else
  curl --fail --location --connect-timeout 15 --max-time 180 \
    https://bootstrap.pypa.io/get-pip.py --output "$ENV_ROOT/bootstrap.py"
  sha256sum "$ENV_ROOT/bootstrap.py" > "$ENV_ROOT/bootstrap.sha256"
  "$ENV_ROOT/bin/python" "$ENV_ROOT/bootstrap.py" 'pip==25.3' 'setuptools==80.9.0' 'wheel==0.45.1'
fi
cd "$SOURCE_ROOT"
"$ENV_ROOT/bin/python" -m pip install -c constraints/model-linux.txt --report "$ENV_ROOT/install-report.json" -e '.[dev]'
"$ENV_ROOT/bin/python" -m pip check
"$ENV_ROOT/bin/python" -m pip freeze --all > "$ENV_ROOT/resolved.txt"
"$ENV_ROOT/bin/python" -c 'import torch,transformers,bloombee; from bloombee import AutoDistributedModelForCausalLM; from bloombee.kv_cache.network.ordinary import OrdinaryClient; print("Model import passed", torch.__version__, transformers.__version__, bloombee.__file__)'
