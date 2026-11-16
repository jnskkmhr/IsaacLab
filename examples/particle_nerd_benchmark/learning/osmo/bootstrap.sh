#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Run in the job container after verifying and extracting the source archive.
set -euo pipefail
output_dir=${1:?Usage: bootstrap.sh OUTPUT_DIR SHAPE MODE or OUTPUT_DIR train DATA_DIR [TRAIN_OPTIONS]}
operation=${2:?Pass box, sphere, cylinder, or train}
if [[ "$operation" == train ]]; then
    data_dir=${3:?Pass a collection dataset directory}
    shift 3
    log_name=training.log
    run_command=(torchrun --standalone --nproc_per_node=4 --module particle_nerd_benchmark.learning.train
        --data "$data_dir" --output "$output_dir/model" "$@")
else
    mode=${3:?Pass prescribed or dynamic}
    log_name=collection.log
    run_command=(python -m particle_nerd_benchmark.learning.collect
        --shape "$operation" --mode "$mode" --output "$output_dir" --device cuda:0)
fi
repo_root=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../../..")
mkdir -p "$output_dir"
exec > >(tee -a "$output_dir/$log_name") 2>&1
unset PYTHONPATH VIRTUAL_ENV DISPLAY
export PATH="${UV_INSTALL_DIR:-/tmp/uv}:$PATH"
if ! command -v uv >/dev/null; then
    curl --fail --location --retry 3 "https://astral.sh/uv/${UV_VERSION:-0.12.12}/install.sh" | sh
fi
uv --version
git --version
cd "$repo_root"
# Keep the repository's exact Newton revision, Warp nightly, and numerical stack.
uv sync --locked --no-dev
cp uv.lock bundle_manifest.json "$output_dir/"
uv pip freeze --python "$UV_PROJECT_ENVIRONMENT/bin/python" > "$output_dir/installed-packages.txt"
export PYTHONPATH="$repo_root/examples"
if [[ "$operation" == train ]]; then
    uv run --no-sync python -m particle_nerd_benchmark.learning.train --help > /dev/null
    expected_test_count=60
    if [[ "${COLLECT_FREEFALL:-0}" == "1" ]]; then
        if ! find "$data_dir" -name "freefall-0000.h5" -print -quit | grep -q .; then
            uv run --no-sync torchrun --standalone --nproc_per_node=4 \
                --module particle_nerd_benchmark.learning.collect_freefall \
                --output "$data_dir/freefall-dynamic"
            cp -r "$data_dir/freefall-dynamic" "$output_dir/freefall-dynamic"
        fi
        expected_test_count=100
    fi
    uv run --no-sync python -m particle_nerd_benchmark.learning.verify \
        --data "$data_dir" --output "$output_dir/data_audit.json"
fi
uv run --no-sync "${run_command[@]}"
if [[ "$operation" == train ]]; then
    # Independent trajectory shards use the same four GPUs after optimization.
    # Each rank writes different episode directories and its own summary.
    uv run --no-sync --with scikit-image==0.27.0rc1 python -m torch.distributed.run \
        --standalone --nproc_per_node=4 --module particle_nerd_benchmark.learning.evaluate \
        --data "$data_dir" --checkpoint "$output_dir/model/best.pt" --output "$output_dir/test_evaluation"
    uv run --no-sync python -m particle_nerd_benchmark.learning.summarize \
        "$output_dir/test_evaluation" --expected "$expected_test_count" --protocol-test
fi
