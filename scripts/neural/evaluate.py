# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark a trained Particle NeRD network against MPM on the test episodes of the dataset.

Every selected test episode is rolled out from its first frame over its whole length, without any correction from
the reference. Per episode the script prints one JSON line and stores:

- the position error against MPM after 1 s, after 5 s and at the end [mm];
- the vertical displacement of the top, mid and floor layers of the bed, for the network and for MPM [mm]. A bed
  that sinks more than the MPM bed shows up here;
- the time per network step [ms] and the peak GPU memory [GB].

The file ``<label>.json`` holds these rows and their means per category and terrain. It is written to
``logs/neural/<run name>/``, where the run name is the folder of the checkpoint unless ``--run-name`` is given.

Run from the repository root:

    uv run python scripts/neural/evaluate.py --dataset /data/particle_nerd \\
        --checkpoint logs/neural/my_run/best.pt

Options that change how the network is run: ``--sleep-speed 0.01`` puts particles to sleep that move slower than
1 cm/s, and ``--active-k 2048`` steps only the 2048 particles nearest to the object or moving, as the Isaac Lab
solver does with ``active_k``.
"""

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from dataset import DEFAULT_CATEGORIES, EpisodeWindows
from metrics import layer_drop, position_errors
from rollout import predict

from isaaclab_tasks.contrib.particle_nerd.network.nerd_network import ParticleNeRDNetwork

SUMMARY_KEYS = ("rmse_mm@1s", "rmse_mm@end", "top3_dz_mm", "top3_dz_mpm_mm", "mid_dz_mm", "floor_dz_mm")


def evaluate_file(model, dataset, index, device, sleep_speed=0.0, active=None, max_steps=None) -> dict:
    """Roll the network out over one test episode and compare it with the MPM reference."""
    record = dataset.records[index][0]
    frames = dataset.num_frames(index)
    horizon = frames - 1 if max_steps is None else min(frames - 1, max_steps)
    sample = dataset.sequence(np.random.default_rng(0), horizon, device, index=index, start=0)
    torch.cuda.reset_peak_memory_stats()
    step_seconds = []
    predicted, failed = predict(model, sample, sleep_speed=sleep_speed, active=active, step_seconds=step_seconds)
    start = sample["history"][:, -1].cpu().numpy()
    reference = np.concatenate((start[None], sample["targets"].cpu().numpy()))
    n = min(len(predicted), len(reference))
    bed = np.ones(len(start), dtype=bool)
    if record.get("num_block_particles"):
        bed[len(start) - record["num_block_particles"] :] = False
    result = {
        "id": record["id"],
        "category": record["category"],
        "motion": record["motion"],
        "terrain": record["terrain"],
        "shape": record["shape"],
        "failed": bool(failed),
        "finite": bool(np.isfinite(predicted[:n]).all()),
        "steps": n - 1,
        "particles": int(len(start)),
        # model step only (device synchronized, first ten steps and the copy to the host left out)
        "ms_per_step": float(1e3 * np.median(step_seconds)) if step_seconds else float("nan"),
        "gpu_peak_gb": float(torch.cuda.max_memory_allocated() / 1e9),
    }
    result.update(position_errors(predicted[:n], reference[:n], record["dt_s"]))
    end = n - 1
    model_layers, mpm_layers = layer_drop(predicted, start, bed, end), layer_drop(reference, start, bed, end)
    for name in model_layers:
        result[f"{name}_dz_mm"] = model_layers[name]
        result[f"{name}_dz_mpm_mm"] = mpm_layers[name]
    if record["category"] == "rest":  # sinking right under the resting object
        centre, half = sample["poses"][0, 0, :2].cpu().numpy(), sample["sizes"][0, :2].cpu().numpy()
        under = (
            bed
            & (np.abs(start[:, 0] - centre[0]) < half[0] + 0.03)
            & (np.abs(start[:, 1] - centre[1]) < half[1] + 0.03)
        )
        under_model, under_mpm = layer_drop(predicted, start, under, end), layer_drop(reference, start, under, end)
        result["under_object_top3_dz_mm"] = under_model["top3"]
        result["under_object_top3_dz_mpm_mm"] = under_mpm["top3"]
    return result


def summarize(rows: list[dict]) -> dict:
    """Means of :data:`SUMMARY_KEYS` per category and terrain, and per category over all terrains."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["category"], row["terrain"])].append(row)
        groups[(row["category"], "all")].append(row)
    out = {}
    for (category, terrain), items in sorted(groups.items()):
        entry = {"files": len(items), "failed": sum(i["failed"] or not i["finite"] for i in items)}
        for key in SUMMARY_KEYS:
            values = [i[key] for i in items if np.isfinite(i[key])]
            entry[key] = round(float(np.mean(values)), 2) if values else None
        out[f"{category}/{terrain}"] = entry
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Folder of the dataset that record.py wrote.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint of train.py, for example best.pt.")
    parser.add_argument("--label", default="evaluation", help="Name of this evaluation and of its result file.")
    parser.add_argument("--run-name", default=None, help="Results go to logs/neural/<run name>/.")
    parser.add_argument("--output", type=Path, default=None, help="Result file, instead of the default place.")
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES), help="Categories to evaluate.")
    parser.add_argument("--per-cell", type=int, default=2, help="Test files per (terrain, category, motion).")
    parser.add_argument("--rest-per-cell", type=int, default=3, help="Test files per terrain for rest and rest_bed.")
    parser.add_argument(
        "--sleep-speed",
        type=float,
        default=0.0,
        help="Particle sleeping: a particle predicted to move slower than this [m/s] in a step stays still.",
    )
    parser.add_argument("--active-k", type=int, help="Run only this many active particles through the network.")
    parser.add_argument("--anchors", type=int, default=256, help="Background anchor tokens added to the active set.")
    parser.add_argument("--selection-interval", type=int, default=4, help="Steps between two selections.")
    parser.add_argument("--max-steps", type=int, help="Limit every rollout to this many steps (checks only).")
    parser.add_argument("--max-files", type=int, help="Limit the number of files (checks only).")
    parser.add_argument("--device", default="cuda:0", help="Device to run on.")
    args = parser.parse_args(argv)

    if not args.dataset.is_dir():
        parser.error(f"{args.dataset} is not a folder. Name the dataset that scripts/neural/record.py wrote")
    if not args.checkpoint.is_file():
        parser.error(f"{args.checkpoint} is not a file. Name a checkpoint of scripts/neural/train.py")
    if args.sleep_speed < 0.0:
        parser.error("--sleep-speed is a speed in m/s and cannot be negative")
    if args.active_k is not None and args.active_k < 1:
        parser.error("--active-k is a number of particles and must be 1 or more")
    if args.anchors < 0 or args.selection_interval < 1 or args.per_cell < 1 or args.rest_per_cell < 1:
        parser.error("--anchors cannot be negative; --selection-interval, --per-cell and --rest-per-cell start at 1")
    for name in ("max_steps", "max_files"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be 1 or more")
    run_name = args.checkpoint.resolve().parent.name if args.run_name is None else args.run_name
    output = Path("logs/neural") / run_name / f"{args.label}.json" if args.output is None else args.output
    active = None
    if args.active_k:
        active = {"k": args.active_k, "anchors": args.anchors, "interval": args.selection_interval}

    wp.init()
    try:
        model, checkpoint = ParticleNeRDNetwork.load_checkpoint(args.checkpoint, args.device)
        dataset = EpisodeWindows(args.dataset, "test", categories=tuple(args.categories))
    except ValueError as error:
        parser.error(str(error))
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
        row = evaluate_file(model, dataset, index, args.device, args.sleep_speed, active, args.max_steps)
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1))
    print("SUMMARY " + json.dumps(result["summary"]), flush=True)
    print(f"The results are in {output.resolve()}.")


if __name__ == "__main__":
    main()
