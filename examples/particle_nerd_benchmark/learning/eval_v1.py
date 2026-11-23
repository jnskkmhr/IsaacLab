# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop evaluation of a Particle NeRD Local checkpoint on the test split of the terrain / category dataset.

Each test episode is rolled out from its first frame with the recorded body motion as input (no MPM state is used by
the model). The prediction is compared with the MPM reference: position error over time and the vertical change of
three bed layers (top 3 cm, mid, floor), which measures sinking directly.
"""

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from . import evaluate as ev
from .data_v1 import DEFAULT_CATEGORIES, EpisodesV1
from .network import ParticleNeRDNetwork


def layer_dz(P, Q0, mask, index):
    """Mean vertical displacement [mm] of the top, mid and floor layers of the bed at frame ``index``."""
    top = np.percentile(Q0[mask, 2], 99.5)
    dz = (P[index][:, 2] - Q0[:, 2]) * 1e3
    layers = {
        "top3": mask & (Q0[:, 2] > top - 0.03),
        "mid": mask & (Q0[:, 2] < top - 0.10) & (Q0[:, 2] > top - 0.18),
        "floor": mask & (Q0[:, 2] < top - 0.20),
    }
    return {name: float(dz[m].mean()) if m.any() else float("nan") for name, m in layers.items()}


def evaluate_file(model, dataset, index, device, sleep_speed=0.0, active=None, max_steps=None):
    record = dataset.records[index][0]
    frames = dataset.num_frames(index)
    horizon = frames - 1 if max_steps is None else min(frames - 1, max_steps)
    sample = dataset.sequence(np.random.default_rng(0), horizon, device, index=index, start=0)
    torch.cuda.reset_peak_memory_stats()
    step_seconds = []
    P, failed = ev.predict(model, sample, sleep_speed=sleep_speed, active=active, step_seconds=step_seconds)
    P = np.asarray(P)
    Q0 = sample["history"][:, -1].cpu().numpy()
    R = np.concatenate((Q0[None], sample["targets"].cpu().numpy()))
    n = min(len(P), len(R))
    dt = record["dt_s"]
    bed = np.ones(len(Q0), dtype=bool)
    if record.get("num_block_particles"):
        bed[len(Q0) - record["num_block_particles"] :] = False
    result = {
        "id": record["id"],
        "category": record["category"],
        "motion": record["motion"],
        "terrain": record["terrain"],
        "shape": record["shape"],
        "failed": bool(failed),
        "finite": bool(np.isfinite(P[:n]).all()),
        "steps": n - 1,
        "particles": int(len(Q0)),
        # model step only (device synchronized, first ten steps and the copy to the host left out)
        "ms_per_step": float(1e3 * np.median(step_seconds)) if step_seconds else float("nan"),
        "gpu_peak_gb": float(torch.cuda.max_memory_allocated() / 1e9),
    }
    for seconds in (1.0, 5.0, (n - 1) * dt):
        i = min(int(round(seconds / dt)), n - 1)
        key = "end" if abs(seconds - (n - 1) * dt) < 1e-6 else f"{seconds:g}s"
        result[f"rmse_mm@{key}"] = float(np.sqrt(((P[i] - R[i]) ** 2).sum(-1).mean()) * 1e3)
    end = n - 1
    model_layers, mpm_layers = layer_dz(P, Q0, bed, end), layer_dz(R, Q0, bed, end)
    for name in model_layers:
        result[f"{name}_dz_mm"] = model_layers[name]
        result[f"{name}_dz_mpm_mm"] = mpm_layers[name]
    if record["category"] == "rest":  # sinking right under the resting object
        centre, half = sample["poses"][0, 0, :2].cpu().numpy(), sample["sizes"][0, :2].cpu().numpy()
        under = bed & (np.abs(Q0[:, 0] - centre[0]) < half[0] + 0.03) & (np.abs(Q0[:, 1] - centre[1]) < half[1] + 0.03)
        under_model, under_mpm = layer_dz(P, Q0, under, end), layer_dz(R, Q0, under, end)
        result["under_object_top3_dz_mm"] = under_model["top3"]
        result["under_object_top3_dz_mpm_mm"] = under_mpm["top3"]
    return result


def summarize(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["category"], row["terrain"])].append(row)
        groups[(row["category"], "all")].append(row)
    keys = ("rmse_mm@1s", "rmse_mm@end", "top3_dz_mm", "top3_dz_mpm_mm", "mid_dz_mm", "floor_dz_mm")
    out = {}
    for (category, terrain), items in sorted(groups.items()):
        entry = {"files": len(items), "failed": sum(i["failed"] or not i["finite"] for i in items)}
        for key in keys:
            values = [i[key] for i in items if np.isfinite(i[key])]
            entry[key] = round(float(np.mean(values)), 2) if values else None
        out[f"{category}/{terrain}"] = entry
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--per-cell", type=int, default=2, help="Test files per (terrain, category, motion)")
    parser.add_argument("--rest-per-cell", type=int, default=3, help="Test files per terrain for rest and rest_bed")
    parser.add_argument(
        "--sleep-speed",
        type=float,
        default=0.0,
        help="Particle sleeping: a particle predicted to move slower than this [m/s] in a step stays still",
    )
    parser.add_argument("--active-k", type=int, help="Run only this many active particles through the network")
    parser.add_argument("--anchors", type=int, default=256, help="Background anchor tokens added to the active set")
    parser.add_argument("--selection-interval", type=int, default=4, help="Steps between reselections")
    parser.add_argument("--max-steps", type=int, help="Limit every rollout to this many steps (smoke check only)")
    parser.add_argument("--max-files", type=int, help="Limit the number of files (smoke check only)")
    args = parser.parse_args()
    active = None
    if args.active_k:
        active = {"k": args.active_k, "anchors": args.anchors, "interval": args.selection_interval}

    wp.init()
    device = "cuda:0"
    model, checkpoint = ParticleNeRDNetwork.load_checkpoint(args.checkpoint, device)
    dataset = EpisodesV1(args.data, "test", categories=tuple(args.categories))
    taken = collections.Counter()
    selected = []
    for index, (record, _, _) in enumerate(dataset.records):
        key = (record["terrain"], record["category"], record["motion"])
        limit = args.rest_per_cell if record["category"] in ("rest", "rest_bed") else args.per_cell
        if taken[key] < limit:
            taken[key] += 1
            selected.append(index)
    selected = selected[: args.max_files] if args.max_files else selected
    print(f"{args.label}: evaluating {len(selected)} of {len(dataset.records)} test files", flush=True)
    rows = []
    start = time.perf_counter()
    for index in selected:
        row = evaluate_file(model, dataset, index, device, args.sleep_speed, active, args.max_steps)
        rows.append(row)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    result = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "step": int(checkpoint.get("step", -1)),
        "sleep_speed_m_s": args.sleep_speed,
        "active": active,
        "ms_per_step": round(float(np.mean([r["ms_per_step"] for r in rows])), 2),
        "gpu_peak_gb": round(float(np.max([r["gpu_peak_gb"] for r in rows])), 2),
        "seconds": round(time.perf_counter() - start, 1),
        "summary": summarize(rows),
        "files": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print("SUMMARY " + json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
