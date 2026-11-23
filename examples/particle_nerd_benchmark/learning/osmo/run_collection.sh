#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Usage: run_collection.sh [--workflow YAML] PRIVATE_OUTPUT_URI [osmo workflow submit options]
# --dry-run builds and validates the source selection without uploading or submitting.
set -euo pipefail
script_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
workflow="$script_dir/collect.yaml"
if [[ "${1:-}" == --workflow ]]; then
    workflow=$(realpath "${2:?Pass a workflow YAML path after --workflow}")
    shift 2
fi
output_url=${1:?Usage: $0 [--workflow YAML] PRIVATE_OUTPUT_URI [osmo workflow submit options]}
shift
case "$output_url" in
    swift://* | s3://*) ;;
    *) echo 'Use an authenticated private swift:// or s3:// output URI.' >&2; exit 2 ;;
esac
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
snapshot=$(mktemp -d)
trap 'rm -rf "$snapshot"' EXIT
mkdir "$snapshot/source"
git -C "$repo_root" ls-files --cached --others --exclude-standard -z > "$snapshot/candidates"
uv run --no-project python - "$repo_root" "$snapshot" <<'PYTHON'
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

root, snapshot = map(Path, sys.argv[1:])
root_files = {"pyproject.toml", "uv.lock", "VERSION", "LICENSE", "LICENSE.txt", "LICENSE.md"}
source_suffixes = {".py", ".pyi", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".typed"}
benchmark_suffixes = {".py", ".toml", ".yaml", ".sh"}
benchmark_excluded_parts = {
    "data", "datasets", "logs", "runs", "outputs", "output", "videos", "video", "recordings",
    "wandb", "test", "tests", "docs", "__pycache__",
}
legacy_force_files = {
    "examples/particle_nerd/__init__.py",
    "examples/particle_nerd/shovel/__init__.py",
    "examples/particle_nerd/shovel/sdf.py",
}
# Preserve untracked runtime modules from the main checkout when submitting a worktree.
# Every file present in the worktree takes precedence over that fallback.
worktrees = subprocess.check_output(["git", "-C", str(root), "worktree", "list", "--porcelain"], text=True)
main_root = Path(worktrees.splitlines()[0].removeprefix("worktree "))
candidates = {}
if main_root != root:
    untracked = subprocess.check_output(
        ["git", "-C", str(main_root), "ls-files", "--others", "--exclude-standard", "-z", "source"]
    )
    candidates.update({value: main_root for value in untracked.split(b"\0") if value})
candidates.update({value: root for value in (snapshot / "candidates").read_bytes().split(b"\0") if value})
files = {}
supplemental_source = []
for value in sorted(candidates):
    if not value:
        continue
    relative = Path(value.decode())
    if any(p.startswith(".") for p in relative.parts):
        continue
    in_g1_asset = (
        relative.is_relative_to("source/isaaclab_assets/data/unitree")
        and len(relative.parts) > 4
        and relative.parts[4].startswith("g1_29dof_rev_1_0")
    )
    if relative.parts[0] == "source":
        # These are Python package trees: data/datasets/logs can be runtime modules. The small G1 USD asset is
        # bundled because the dataset's G1 episodes spawn the robot from it.
        if not in_g1_asset and (
            relative.is_relative_to("source/isaaclab_assets/data") or {"test", "tests", "__pycache__"}.intersection(relative.parts)
        ):
            continue
    elif benchmark_excluded_parts.intersection(relative.parts):
        continue
    selected = (
        relative.as_posix() in root_files
        or relative.as_posix() in legacy_force_files
        or (relative.is_relative_to("examples/particle_nerd/force_estimation") and relative.suffix == ".py")
        or in_g1_asset
        or (relative.parts[0] == "source" and relative.suffix in source_suffixes)
        or (relative.parts[0] == "apps" and relative.suffix in {".kit", ".toml"})
        or (relative.is_relative_to("examples/particle_nerd_benchmark") and relative.suffix in benchmark_suffixes)
    )
    if not selected:
        continue
    source_root = root if (root / relative).exists() else candidates[value]
    original = source_root / relative
    if not original.is_file() or original.is_symlink():
        raise SystemExit(f"Selected source is missing or a symbolic link: {relative}")
    destination = snapshot / "source" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, destination)
    files[relative.as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()
    if source_root != root:
        supplemental_source.append(relative.as_posix())
# Independently enumerate the runtime Python inventory to catch broad selection
# filters removing valid packages before an archive reaches a worker.
expected_python = set()
for value in candidates:
    relative = Path(value.decode())
    if relative.parts[0] != "source" or relative.suffix != ".py":
        continue
    if any(part.startswith(".") for part in relative.parts):
        continue
    if {"test", "tests", "__pycache__"}.intersection(relative.parts):
        continue
    if relative.is_relative_to("source/isaaclab_assets/data"):
        continue
    expected_python.add(relative.as_posix())
actual_python = {name for name in files if name.startswith("source/") and name.endswith(".py")}
if actual_python != expected_python:
    raise SystemExit(
        f"Runtime Python inventory mismatch: missing={sorted(expected_python - actual_python)}, "
        f"unexpected={sorted(actual_python - expected_python)}"
    )
required = {
    "pyproject.toml", "uv.lock",
    "source/isaaclab/isaaclab/utils/datasets/__init__.py",
    "examples/particle_nerd_benchmark/learning/collect.py",
    "examples/particle_nerd_benchmark/learning/protocol.py",
    "examples/particle_nerd_benchmark/learning/osmo/bootstrap.sh",
    "source/isaaclab_tasks/isaaclab_tasks/contrib/legacy/particle_nerd/network/features.py",
    "source/isaaclab_tasks/isaaclab_tasks/contrib/particle_nerd/network/nerd_network.py",
    "source/isaaclab_tasks/isaaclab_tasks/contrib/particle_nerd/network/feature_builder.py",
    "source/isaaclab_tasks/isaaclab_tasks/contrib/particle_nerd/modules/attention.py",
    "source/isaaclab_tasks/isaaclab_tasks/contrib/particle_nerd/nerd_kernels.py",
}
if missing := required.difference(files):
    raise SystemExit(f"Bundle is missing required files: {sorted(missing)}")
lock = tomllib.loads((snapshot / "source/uv.lock").read_text())
packages = {p["name"]: {"version": p["version"], "source": p["source"]} for p in lock["package"]}
if "registry" not in packages["warp-lang"]["source"]:
    raise SystemExit("This launcher requires the locked binary Warp registry wheel, not a Git build.")
manifest = {
    "git_revision": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "files_sha256": files,
    "supplemental_source": supplemental_source,
    "locked_runtime": {name: packages[name] for name in ("newton", "warp-lang", "torch", "numpy", "h5py")},
}
(snapshot / "source/bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Bundled {len(files)} source/configuration files, including {len(supplemental_source)} fallback modules.")
print(f"Verified complete runtime inventory: {len(expected_python)} Python source files.")
print("Excluded asset binaries, raw trajectories, checkpoints, and videos.")
print(json.dumps(manifest["locked_runtime"], indent=2))
PYTHON
tar -czf "$snapshot/source.tar.gz" -C "$snapshot/source" .
source_sha256=$(sha256sum "$snapshot/source.tar.gz" | cut -d ' ' -f 1)
snapshot_url="${output_url%/}/snapshots/$source_sha256"
submit_args=()
has_set_string=false
dry_run=false
for arg in "$@"; do
    submit_args+=("$arg")
    case "$arg" in
        --dry-run) dry_run=true ;;
        --set-string)
            submit_args+=("output_url=$output_url" "source_url=$snapshot_url/" "source_sha256=$source_sha256")
            has_set_string=true
            ;;
    esac
done
if ! "$has_set_string"; then
    submit_args+=(--set-string "output_url=$output_url" "source_url=$snapshot_url/" "source_sha256=$source_sha256")
fi
echo "Source archive: $(du -h "$snapshot/source.tar.gz" | cut -f1), SHA256 $source_sha256"
if ! "$dry_run"; then
    osmo data upload "$snapshot_url/" "$snapshot/source.tar.gz"
fi
osmo workflow submit "$workflow" "${submit_args[@]}" --format-type json
