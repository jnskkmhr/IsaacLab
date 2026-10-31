#!/usr/bin/env bash
# Run inside a cloud task after the source bundle is extracted.
set -euo pipefail
if [[ -n "${PARTICLE_OUTPUT:-}" ]]; then
    mkdir -p "$PARTICLE_OUTPUT"
    exec > >(tee -a "$PARTICLE_OUTPUT/runtime.log") 2>&1
fi
bundle_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
export UV_INSTALL_DIR=/tmp/particle-uv UV_NO_MODIFY_PATH=1
export PATH="$UV_INSTALL_DIR:$PATH"
unset PYTHONPATH VIRTUAL_ENV DISPLAY
if ! command -v uv >/dev/null; then
    curl --fail --location --retry 3 https://astral.sh/uv/0.12.12/install.sh | sh
fi
uv venv --python 3.12 /tmp/particle-gnn-venv
runtime_python=/tmp/particle-gnn-venv/bin/python
runtime_site=$($runtime_python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cp -a "$bundle_root/vendor/." "$runtime_site/"
uv pip install --python "$runtime_python" --prerelease allow \
    --index-strategy unsafe-best-match \
    --index https://download.pytorch.org/whl/cu130 \
    --index https://pypi.nvidia.com \
    -r "$bundle_root/requirements.txt"
export PYTHONPATH="$bundle_root/isaaclab/examples"
for package in "$bundle_root"/isaaclab/source/*; do
    if [[ -d "$package" ]]; then export PYTHONPATH="$PYTHONPATH:$package"; fi
done
export PATH="/tmp/particle-gnn-venv/bin:$PATH"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export WARP_CACHE_PATH=/tmp/particle-warp-cache
cd "$bundle_root"
python - <<'PY'
import importlib.metadata as m
import json
import torch
import warp as wp
wp.init()
versions = {n: m.version(n) for n in ("newton", "warp-lang", "torch", "mujoco", "mujoco-warp", "numpy")}
for provider in ("usd-core", "usd-exchange"):
    try:
        versions[provider] = m.version(provider)
    except m.PackageNotFoundError:
        pass
from pxr import Usd
print(json.dumps({"runtime_versions": versions, "usd_version": Usd.GetVersion(), "gpu": torch.cuda.get_device_name(0)}, indent=2))
assert torch.cuda.is_available()
PY
exec "$@"
