#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Submit the complete 1,000-trajectory collection to OSMO: the six object jobs (600 trajectories) and the
# free-fall shards (400 trajectories). Both write directly to the private output bucket.
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 --output URI [osmo workflow submit options, e.g. --pool POOL]

Options:
  --output URI    Private swift:// or s3:// location that receives the dataset (required).
  -h, --help      Show this message.
Any other option, such as --pool, is passed to each OSMO submission.
EOF
}

output_url=""
extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) output_url=${2:?--output needs a swift:// or s3:// URI}; shift 2 ;;
        -h | --help) usage; exit 0 ;;
        *) extra+=("$1"); shift ;;
    esac
done
if [[ -z "$output_url" ]]; then
    echo "--output is required." >&2
    usage >&2
    exit 2
fi
script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
bash "$script_dir/run_collection.sh" "$output_url" ${extra[@]+"${extra[@]}"}
bash "$script_dir/run_collection.sh" --workflow "$script_dir/collect_freefall.yaml" "$output_url" ${extra[@]+"${extra[@]}"}
