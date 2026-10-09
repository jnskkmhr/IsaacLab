#!/usr/bin/env bash
# Copyright (c) 2022-2026, Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Usage: bash osmo/run_multinode.sh WORKFLOW [osmo workflow submit options]
set -euo pipefail
workflow=$(realpath "${1:?Usage: $0 WORKFLOW [osmo workflow submit options]}")
shift
output_url=""
dry_run=false
for arg in "$@"; do
    case "$arg" in
        output_url=*) output_url="${arg#output_url=}" ;;
        --dry-run) dry_run=true ;;
    esac
done
: "${output_url:?Pass --set-string output_url=STORAGE_URI}"
repo_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
snapshot=$(mktemp -d)
trap 'rm -rf "$snapshot"' EXIT
mkdir "$snapshot/source"
# Let Git select tracked files and non-ignored new files, honoring ! exceptions.
git -C "$repo_root" ls-files --cached --others --exclude-standard -z > "$snapshot/files"
# Apply deployment exclusions to both tracked and untracked files.
git -C "$repo_root" ls-files --cached --others --ignored -z \
    --exclude-from="$repo_root/.dockerignore" --exclude='*.git*' \
    --exclude='/data/' --exclude='.env*' > "$snapshot/excluded"
uv run --no-project python - "$snapshot" <<'PYTHON'
import sys
from pathlib import Path

snapshot = Path(sys.argv[1])
files = set((snapshot / "files").read_bytes().split(b"\0"))
excluded = set((snapshot / "excluded").read_bytes().split(b"\0"))
(snapshot / "files").write_bytes(b"".join(path + b"\0" for path in sorted(files - excluded) if path))
PYTHON
rsync -a --from0 --files-from="$snapshot/files" --ignore-missing-args \
    "$repo_root/" "$snapshot/source/"

# Keep large assets in shared storage, keyed by their contents rather than the run.
mkdir "$snapshot/assets"
mv "$snapshot/source/source/isaaclab_assets/data" "$snapshot/assets/data"
asset_version=$(uv run --no-project python - "$snapshot/assets" <<'PYTHON'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        value = ("link", str(path.readlink()))
    elif path.is_file():
        with path.open("rb") as stream:
            value = ("file", hashlib.file_digest(stream, "sha256").hexdigest())
    else:
        value = ("directory", "")
    digest.update(json.dumps((path.relative_to(root).as_posix(), value)).encode())
print(digest.hexdigest())
PYTHON
)
# Git installs of Warp require native libraries built before packaging. Reuse
# uv's wheel for the locked commit, rather than building on every trainer.
warp_wheel_version=$(uv run --no-project python - "$snapshot" "$(uv cache dir)" <<'PYTHON'
import hashlib
import shutil
import sys
import tomllib
import zipfile
from pathlib import Path

snapshot, cache = map(Path, sys.argv[1:])
with (snapshot / "source/uv.lock").open("rb") as stream:
    package = next(p for p in tomllib.load(stream)["package"] if p["name"] == "warp-lang")
if "registry" in package["source"]:
    # Registry releases include native libraries and are installed by uv sync on each trainer.
    sys.exit(0)
revision = package["source"]["git"].rsplit("#", 1)[1]
wheels = list(cache.glob(f"sdists-v*/git/*/{revision[:16]}/warp_lang-{package['version']}-*.whl"))
if len(wheels) != 1:
    raise SystemExit(
        f"Expected one built Warp wheel for locked commit {revision} in {cache}; found {len(wheels)}. "
        "Build Warp's native libraries with build_lib.py and populate the local uv environment first."
    )
wheel = wheels[0]
with zipfile.ZipFile(wheel) as archive:
    for library in ("warp/bin/warp.so", "warp/bin/warp-clang.so"):
        if library not in archive.namelist():
            raise SystemExit(f"{wheel} is missing {library}.")
(snapshot / "wheels").mkdir()
shutil.copy2(wheel, snapshot / "wheels" / wheel.name)
with wheel.open("rb") as stream:
    print(hashlib.file_digest(stream, "sha256").hexdigest())
PYTHON
)
warp_wheel_url=""
if [[ -n "$warp_wheel_version" ]]; then
    warp_wheel_url="${output_url%/}/wheels/$warp_wheel_version"
fi
warp_wheel_input="${warp_wheel_url:+$warp_wheel_url/wheels/}"
asset_url="${output_url%/}/assets/$asset_version"
snapshot_url="${output_url%/}/snapshots/$(date -u +%Y%m%dT%H%M%SZ)-${snapshot##*.}"
# OSMO replaces repeated --set-string groups; insert into the existing group.
submit_args=()
has_set_string=false
for arg in "$@"; do
    submit_args+=("$arg")
    if [[ "$arg" == --set-string ]]; then
        submit_args+=("assets_url=$asset_url/data/" "source_url=$snapshot_url/source/" "warp_wheel_url=$warp_wheel_input")
        has_set_string=true
    fi
done
if ! "$has_set_string"; then
    submit_args+=(--set-string "assets_url=$asset_url/data/" "source_url=$snapshot_url/source/" "warp_wheel_url=$warp_wheel_input")
fi
if "$dry_run"; then
    osmo workflow submit "$workflow" "${submit_args[@]}"
    exit
fi

# Keep client diagnostics even if the terminal is cleared or the upload fails.
mkdir -p "$repo_root/logs/osmo"
log_file=$(mktemp "$repo_root/logs/osmo/submit-$(date -u +%Y%m%dT%H%M%SZ)-XXXX.log")
exec > >(tee -a "$log_file") 2>&1
echo "Submission log: $log_file"

# Publish the completion marker last so interrupted uploads are never reused.
upload_cached() {
    local directory="$1" url="$2"
    rm -f "$snapshot/cached-files"
    osmo data list "$url/" "$snapshot/cached-files" --regex '(^|/)complete$'
    if [[ ! -s "$snapshot/cached-files" ]]; then
        echo "Uploading $(basename "$directory") to $url (once per content version)..."
        osmo data upload "$url/" "$directory"
        touch "$snapshot/complete"
        osmo data upload "$url/" "$snapshot/complete"
    else
        echo "Reusing $(basename "$directory") from $url"
    fi
}
upload_cached "$snapshot/assets/data" "$asset_url"
if [[ -n "$warp_wheel_url" ]]; then
    upload_cached "$snapshot/wheels" "$warp_wheel_url"
fi
echo "Uploading code snapshot ($(du -sh --apparent-size "$snapshot/source" | cut -f1)) to $snapshot_url..."
osmo data upload "$snapshot_url/" "$snapshot/source"
osmo workflow submit "$workflow" "${submit_args[@]}" --format-type json
