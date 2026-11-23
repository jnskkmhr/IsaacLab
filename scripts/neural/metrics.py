# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measures of a predicted rollout against the MPM reference, and the check that a recorded bed had settled."""

import json
from pathlib import Path

import h5py
import numpy as np


def position_rmse(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Root-mean-square distance [mm] between predicted and reference particle positions [N, 3] of one frame."""
    return float(np.sqrt(((predicted - reference) ** 2).sum(-1).mean()) * 1e3)


def position_errors(predicted: np.ndarray, reference: np.ndarray, dt: float) -> dict:
    """Position error [mm] of a rollout after 1 s, after 5 s and at its end.

    Args:
        predicted: Predicted positions [T, N, 3], frame 0 is the start state.
        reference: Reference positions [T, N, 3] of the same frames.
        dt: Time between two frames [s].

    Returns:
        ``rmse_mm@1s``, ``rmse_mm@5s`` and ``rmse_mm@end``. A rollout shorter than 1 s or 5 s reports its last
        frame under that key.
    """
    last = min(len(predicted), len(reference)) - 1
    errors = {}
    for key, seconds in (("1s", 1.0), ("5s", 5.0)):
        index = min(int(round(seconds / dt)), last)
        errors[f"rmse_mm@{key}"] = position_rmse(predicted[index], reference[index])
    errors["rmse_mm@end"] = position_rmse(predicted[last], reference[last])
    return errors


def layer_drop(positions: np.ndarray, start: np.ndarray, mask: np.ndarray, index: int) -> dict:
    """Mean vertical displacement [mm] of the top, mid and floor layers of the bed at frame ``index``.

    Args:
        positions: Particle positions [T, N, 3] of a rollout.
        start: Positions [N, 3] at the start; they define the layers and the reference height.
        mask: Particles [N] that belong to the bed.
        index: Frame to measure.
    """
    top = np.percentile(start[mask, 2], 99.5)
    dz = (positions[index][:, 2] - start[:, 2]) * 1e3
    layers = {
        "top3": mask & (start[:, 2] > top - 0.03),
        "mid": mask & (start[:, 2] < top - 0.10) & (start[:, 2] > top - 0.18),
        "floor": mask & (start[:, 2] < top - 0.20),
    }
    return {name: float(dz[m].mean()) if m.any() else float("nan") for name, m in layers.items()}


BARE_BED_LIMIT_MM_PER_10S = 12.0
"""A bed that nothing touches may not move more than this.

Measured in the coupled scene: 3 to 5 mm per 10 s on the flat bed and 5 to 10 mm on the slope and wave beds. Beds
that were recorded before they had settled moved 13 mm (flat) to 28 mm (slope).
"""
BODY_MARGIN_M = 0.40
BLOCK_MARGIN_M = 0.60


def bare_bed_drift(f: h5py.File, first: np.ndarray, last: np.ndarray) -> float | None:
    """Vertical drift [mm per 10 s] of the bed surface where nothing touched it during the episode.

    The surface is the top 3 cm of every 0.1 m column of the bed at the first frame. Particles that come within
    ``BODY_MARGIN_M`` of a rigid body or ``BLOCK_MARGIN_M`` of the released sand block are left out. Returns None
    if fewer than 200 particles remain.
    """
    num_block = int(f.attrs["num_block_particles"]) if "num_block_particles" in f.attrs else 0
    num_bed = first.shape[0] - num_block
    bed = first[:num_bed]
    cell = np.floor(bed[:, :2] / 0.1).astype(np.int64)
    _, column = np.unique(cell[:, 0] * 100003 + cell[:, 1], return_inverse=True)
    top = np.full(column.max() + 1, -np.inf)
    np.maximum.at(top, column, bed[:, 2])
    keep = bed[:, 2] > top[column] - 0.03
    tracks = []  # (points [M, 2] visited by something that disturbs the bed, margin)
    if "body" in f:
        pose = f["body/pose"][::20]
        tracks.append((pose[..., :2].reshape(-1, 2), BODY_MARGIN_M))
    if num_block:
        block = f["particles/pos"][::40, num_bed:, :2]
        tracks.append((block.mean(axis=1), BLOCK_MARGIN_M))
    for points, margin in tracks:
        points = np.unique(np.round(points / 0.05).astype(np.int64), axis=0) * 0.05
        for start in range(0, len(points), 256):
            chunk = points[start : start + 256]
            near = (np.linalg.norm(bed[:, None, :2] - chunk[None], axis=-1) < margin + 0.05).any(axis=1)
            keep &= ~near
    if keep.sum() < 200:
        return None
    seconds = float(f.attrs["duration_s"])
    return float((last[:num_bed][keep, 2] - bed[keep, 2]).mean() * 1e3 * 10.0 / seconds)


def file_stats(path: Path) -> dict:
    """Key statistics of one episode file, including the settled-bed check."""
    with h5py.File(path, "r") as f:
        a = f.attrs
        pos = f["particles/pos"]
        n = int(a["num_steps"])
        first, last = pos[0], pos[n]
        drift = bare_bed_drift(f, first, last)
        row = {
            "file": None,
            "id": str(a["episode_id"]),
            "split": str(a["split"]),
            "category": str(a["category"]),
            "motion": str(a["motion"]),
            "shape": str(a["shape"]),
            "variant": str(a["terrain_variant"]),
            "particles": int(a["num_particles"]),
            "MB": round(path.stat().st_size / 1e6, 1),
            "complete": bool(a.get("complete", False)),
            "finite": bool(np.isfinite(first).all() and np.isfinite(last).all()),
            "p99_z_t0": round(float(np.percentile(first[:, 2], 99)), 3),
            "p99_z_end": round(float(np.percentile(last[:, 2], 99)), 3),
            "max_z_end": round(float(last[:, 2].max()), 3),
            "bare_bed_dz_mm_per_10s": None if drift is None else round(drift, 2),
            "bed_settled": drift is None or abs(drift) <= BARE_BED_LIMIT_MM_PER_10S,
        }
        if "body" in f:
            pose = f["body/pose"][:]
            wrench = f["body/contact_wrench"][:]
            row["finite"] = row["finite"] and bool(np.isfinite(pose).all() and np.isfinite(wrench).all())
            row.update(
                obj_z_start=round(float(pose[0, 2]), 3),
                obj_z_end=round(float(pose[-1, 2]), 3),
                obj_xy_travel=round(float(np.linalg.norm(pose[-1, :2] - pose[0, :2])), 3),
                max_contact_f=round(float(np.linalg.norm(wrench[:, :3], axis=-1).max()), 1),
            )
        return row


def summarize_files(root: Path, out_path: Path, terrain: str | None = None) -> list[dict]:
    """Write and print one line per episode file under ``root`` (restricted to one terrain if given).

    The last printed line counts the files that are broken and the files whose bed had not settled.
    """
    rows = []
    for path in sorted(root.rglob("*.hdf5")):
        if terrain and terrain not in path.parts:
            continue
        try:
            row = file_stats(path)
        except Exception as error:  # a broken file must show up in the log, not stop the summary
            row = {"id": path.stem, "error": repr(error)}
        row["file"] = str(path.relative_to(root))
        rows.append(row)
        print("FILE " + json.dumps(row), flush=True)
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    bad = [r["file"] for r in rows if r.get("error") or not r.get("complete") or not r.get("finite")]
    drifting = [r["file"] for r in rows if r.get("bed_settled") is False]
    total = sum(r.get("MB", 0.0) for r in rows) / 1000.0
    print(
        f"SUMMARY files={len(rows)} size={total:.1f} GB bad={len(bad)} {bad[:5]} "
        f"bed_not_settled={len(drifting)} {drifting[:5]}",
        flush=True,
    )
    return rows
