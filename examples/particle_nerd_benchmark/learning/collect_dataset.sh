#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Collect the 1,000-trajectory training set: 600 object trajectories (box, sphere, cylinder; prescribed and
# dynamic) and 400 free-fall sand trajectories, then verify the complete set.
#
# Roughly 135 GB is written to the output directory. Collection resumes: finished trajectories are skipped.
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 --output DIR [--num_gpu N]

Options:
  --output DIR    Directory that receives the dataset (required).
  --num_gpu N     Number of GPUs to use (default: 1). The six object jobs run N at a time, one per GPU,
                  and the free-fall trajectories are sharded over all N GPUs.
  -h, --help      Show this message.
EOF
}

output=""
num_gpu=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) output=${2:?--output needs a directory}; shift 2 ;;
        --num_gpu | --num_gpus) num_gpu=${2:?--num_gpu needs a number}; shift 2 ;;
        -h | --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ -z "$output" ]]; then
    echo "--output is required." >&2
    usage >&2
    exit 2
fi
if ! [[ "$num_gpu" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num_gpu must be a positive integer, got: $num_gpu" >&2
    exit 2
fi
available=$(nvidia-smi -L 2>/dev/null | wc -l || true)
if (( available > 0 && num_gpu > available )); then
    echo "--num_gpu $num_gpu requested, but only $available GPU(s) are available." >&2
    exit 2
fi

repo_root=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")
cd "$repo_root"
export PYTHONPATH="$repo_root/examples"
mkdir -p "$output"

jobs_list=()
for shape in box sphere cylinder; do
    for mode in prescribed dynamic; do
        jobs_list+=("$shape $mode")
    done
done

# Object trajectories: run num_gpu jobs at a time, each on its own GPU.
slot=0
pids=()
for job in "${jobs_list[@]}"; do
    read -r shape mode <<< "$job"
    uv run --no-sync python -m particle_nerd_benchmark.learning.collect \
        --shape "$shape" --mode "$mode" --output "$output/$shape-$mode" --device "cuda:$slot" &
    pids+=($!)
    slot=$(( (slot + 1) % num_gpu ))
    if (( ${#pids[@]} == num_gpu )); then
        for pid in "${pids[@]}"; do wait "$pid"; done
        pids=()
    fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

# Free-fall trajectories: sharded over all GPUs.
uv run --no-sync torchrun --standalone --nproc_per_node="$num_gpu" \
    --module particle_nerd_benchmark.learning.collect_freefall --output "$output/freefall"

uv run --no-sync python -m particle_nerd_benchmark.learning.verify --data "$output" --output "$output/data_audit.json"
echo "Collected and verified the 1,000-trajectory set in $output"
