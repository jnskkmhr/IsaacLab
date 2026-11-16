# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the complete 600- or 1,000-trajectory campaign before training and hash its files."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py

from .protocol import DT, DURATION, episodes


def verify(root):
    """Reject missing, duplicate, incomplete, or incorrectly phased recordings."""
    expected_obj = {record["id"]: record for record in episodes(include_freefall=False)}
    expected_all = {record["id"]: record for record in episodes(include_freefall=True)}
    files, splits, groups = {}, Counter(), Counter()
    for path in sorted(root.rglob("manifest.json")):
        encoded = path.read_bytes()
        manifest = json.loads(encoded)
        if manifest.get("schema") != "particle-nerd-object-trajectories-v1":
            continue
        if manifest.get("evaluation_reference") is not None:
            raise ValueError(f"Evaluation reference mixed into training data: {path}")
        manifest_sha = hashlib.sha256(encoded).hexdigest()
        for record in manifest["episodes"]:
            name = record["id"]
            if name in files or record != expected_all.get(name):
                raise ValueError(f"Duplicate or changed campaign episode: {name}")
            file = path.parent / (name + ".h5")
            with h5py.File(file, "r") as handle:
                if not handle.attrs.get("complete") or (
                    record["shape"] != "freefall" and handle.attrs["manifest_sha256"] != manifest_sha
                ):
                    raise ValueError(f"Incomplete recording or manifest mismatch: {file}")
                steps = round(DURATION / DT)
                pos_shape = handle["particle_position"].shape
                valid_particles = (
                    (pos_shape[0] == steps + 1 and pos_shape[2] == 3 and pos_shape[1] >= 200)
                    if record["shape"] == "freefall"
                    else (pos_shape == (steps + 1, 20646, 3))
                )
                if (
                    not valid_particles
                    or handle["body_state"].shape != (steps + 1, 13)
                    or handle["collider_state"].shape != (steps, 13)
                ):
                    raise ValueError(f"Unexpected native-time state dimensions: {file}")
                if "material_force" in handle:
                    raise ValueError(f"Force targets found in state-only training data: {file}")
            with file.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            files[name] = {"path": str(file.relative_to(root)), "sha256": sha, "bytes": file.stat().st_size}
            splits[record["split"]] += 1
            groups[record["shape"] + "/" + record["mode"]] += 1
            if len(files) % 20 == 0:
                print(f"Verified {len(files)} complete trajectories", flush=True)
    target_keys = expected_all.keys() if any(k.startswith("freefall-") for k in files) else expected_obj.keys()
    if missing := target_keys - files.keys():
        raise ValueError(f"Missing {len(missing)} trajectories: {sorted(missing)}")
    return {"trajectories": len(files), "splits": dict(splits), "groups": dict(groups), "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "files"}), flush=True)


if __name__ == "__main__":
    main()
