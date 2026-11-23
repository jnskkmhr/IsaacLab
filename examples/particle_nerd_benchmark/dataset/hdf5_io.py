# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HDF5 reader and writer for one dataset episode (no compression; `particles/pos` is chunked along time)."""

import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

SCHEMA = "particle-nerd-dataset-v1"


def code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_episode(
    path: Path, record: dict, positions: np.ndarray, body: dict | None, attrs: dict, extra: dict | None = None
) -> None:
    """Write one episode atomically.

    Args:
        positions: Particle positions [T, N, 3] (float32).
        body: None, or arrays `pose` [T, 7], `vel` [T, 6], `ext_wrench` [T-1, 6], `contact_wrench` [T-1, 6]
            (one rigid body), or the same with a body axis [T, B, ...] for articulated robots.
        extra: Optional groups of arrays, for example `joints` and `control` for robots.
        attrs: Extra attributes (material, spacing, size, ...).
    """
    steps = positions.shape[0]
    if not np.isfinite(positions).all():
        raise RuntimeError(f"Nonfinite particle positions in {record['id']}")
    partial = path.with_suffix(".partial.h5")
    with h5py.File(partial, "w") as f:
        f.attrs["schema_version"] = SCHEMA
        f.attrs["code_commit"] = code_commit()
        f.attrs["solver"] = "mpm"
        f.attrs["num_steps"] = steps - 1
        f.attrs["num_particles"] = positions.shape[1]
        f.attrs["num_bodies"] = 0 if body is None else (1 if body["pose"].ndim == 2 else body["pose"].shape[1])
        for key in ("id", "terrain", "terrain_variant", "category", "motion", "motion_type", "shape", "split"):
            f.attrs[key if key != "id" else "episode_id"] = str(record.get(key))
        for key in ("seed", "particle_seed"):
            f.attrs[key] = int(record[key])
        f.attrs["dt"] = float(record["dt_s"])
        f.attrs["duration_s"] = float(record["duration_s"])
        f.attrs["settle_s"] = float(record["settle_s"])
        f.attrs["params"] = json.dumps(record)
        for key, value in attrs.items():
            f.attrs[key] = json.dumps(value) if value is None or isinstance(value, (dict, list, tuple)) else value
        f.create_dataset("time", data=np.arange(steps, dtype=np.float32) * np.float32(record["dt_s"]))
        particles = f.create_group("particles")
        particles.create_dataset("pos", data=positions, chunks=(min(16, steps), positions.shape[1], 3))
        if body is not None:
            group = f.create_group("body")
            for name, array in body.items():
                group.create_dataset(name, data=np.asarray(array, dtype=np.float32))
        for group_name, arrays in (extra or {}).items():
            group = f.create_group(group_name)
            for name, array in arrays.items():
                group.create_dataset(name, data=np.asarray(array, dtype=np.float32))
        f.attrs["complete"] = True
    partial.rename(path)


def read_episode(path: Path, particles: bool = True) -> dict:
    """Return `attrs`, `time`, optional `pos` and `body` arrays of one episode."""
    with h5py.File(path, "r") as f:
        if not f.attrs.get("complete", False):
            raise RuntimeError(f"Incomplete episode: {path}")
        out = {"attrs": {k: f.attrs[k] for k in f.attrs}, "time": f["time"][:]}
        if particles:
            out["pos"] = f["particles/pos"][:]
        if "body" in f:
            out["body"] = {k: f["body"][k][:] for k in f["body"]}
    return out
