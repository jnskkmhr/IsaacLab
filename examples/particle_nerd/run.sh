#!/usr/bin/env bash
# Run this example against its owning IsaacLab worktree.
set -euo pipefail
example_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$example_dir/../.." && pwd)
if [[ -z ${PARTICLE_NERD_PYTHON:-} ]]; then
    common_dir=$(git -C "$repo_root" rev-parse --path-format=absolute --git-common-dir)
    main_checkout=$(dirname -- "$common_dir")
    PARTICLE_NERD_PYTHON="$repo_root/.venv/bin/python"
    [[ -x "$PARTICLE_NERD_PYTHON" ]] || PARTICLE_NERD_PYTHON="$main_checkout/.venv/bin/python"
fi
[[ -x "$PARTICLE_NERD_PYTHON" ]] || { echo 'Set PARTICLE_NERD_PYTHON to the tested IsaacLab Python interpreter.' >&2; exit 1; }
export PYTHONPATH="$repo_root/examples${PYTHONPATH:+:$PYTHONPATH}"
for package in "$repo_root"/source/*; do
    [[ ! -d "$package" ]] || export PYTHONPATH="$package:$PYTHONPATH"
done
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec uv run --no-project --python "$PARTICLE_NERD_PYTHON" \
    --with numpy==2.4.6 --with usd-core --with h5py python "$@"
