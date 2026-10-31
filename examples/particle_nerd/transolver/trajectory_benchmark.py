# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect six unseen force-driven paths and evaluate two frozen Transolver checkpoints."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from particle_nerd.data_collection.campaign import validate
from particle_nerd.data_collection.trajectories import PATHS


def motion_metrics(references, predictions, records):
    """Consecutive-step errors, independent of accumulated position offsets."""
    results = []
    for record in records:
        index = record["index"]
        with (
            h5py.File(references / record["file"], "r") as ref,
            h5py.File(predictions / f"rollout-{index:05d}.hdf5", "r") as pred,
        ):
            start = int(pred.attrs["start_frame"])
            valid = int(pred.attrs["valid_frames"])
            truth = ref["particle_position"][start : start + valid]
            estimate = pred["particle_position"][:valid]
        true_dx, pred_dx = np.diff(truth, axis=0), np.diff(estimate, axis=0)
        sq_error = np.sum((true_dx - pred_dx) ** 2, axis=-1)
        true_sq = np.sum(true_dx**2, axis=-1)
        pred_sq = np.sum(pred_dx**2, axis=-1)
        moving = true_sq > 0.0001**2
        row = dict(
            index=index,
            case=record["case"],
            direction=record["direction"],
            dt=0.02,
            all_step_rmse_m=float(np.sqrt(sq_error.mean())),
            moving_step_rmse_m=float(np.sqrt(sq_error[moving].mean())) if moving.any() else None,
            stationary_step_rmse_m=float(np.sqrt(sq_error[~moving].mean())) if (~moving).any() else None,
            moving_sample_count=int(moving.sum()),
            stationary_sample_count=int((~moving).sum()),
            moving_motion_rms_ratio=float(np.sqrt(pred_sq[moving].sum() / true_sq[moving].sum()))
            if moving.any()
            else None,
            moving_missed_fraction=float((pred_sq[moving] <= 0.0001**2).mean()) if moving.any() else None,
            per_step_rmse_m=np.sqrt(sq_error.mean(axis=1)).tolist(),
        )
        results.append(row)
    (predictions / "motion_metrics.json").write_text(
        json.dumps(
            dict(moving_definition="Reference displacement >0.1 mm in this 20 ms step", recurrent=True, rows=results),
            indent=2,
        )
        + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--baseline-checkpoint", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    references = args.output / "reference"
    references.mkdir(exist_ok=True)
    records = []
    for path in PATHS:
        for direction in (1, -1):
            index = 10000 + len(records)
            name = f"episode-{index:05d}"
            command = [
                sys.executable,
                "-m",
                "particle_nerd.data_collection.collect",
                "--case",
                "continuous-drag",
                "--drag-path",
                path,
                "--path-direction",
                str(direction),
                "--duration",
                "5",
                "--body-mass",
                "7.5",
                "--seed",
                "620000",
                "--name",
                name,
                "--output",
                str(references),
            ]
            # Every path uses the same material seed and box mass; pose evolves dynamically.
            print(json.dumps(dict(event="collect", index=index, path=path, direction=direction)), flush=True)
            subprocess.run(command, check=True)
            quality = validate(references / f"{name}.hdf5")
            record = dict(
                index=index,
                file=f"{name}.hdf5",
                split="test",
                case=path,
                direction=direction,
                seed=620000,
                mass_kg=7.5,
                quality=quality,
            )
            (references / f"{name}.campaign.json").write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            (args.output / "collection.json").write_text(json.dumps(records, indent=2) + "\n")
    for label, checkpoint in (("long", args.checkpoint), ("baseline", args.baseline_checkpoint)):
        print(json.dumps(dict(event="evaluate", model=label)), flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "particle_nerd.transolver.evaluate",
                "--data",
                str(references),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(args.output / label),
                "--ids",
                *[str(r["index"]) for r in records],
                "--steps",
                "250",
                "--mode",
                "all",
            ],
            check=True,
        )
        evaluation_path = args.output / label / "evaluation.json"
        evaluation = json.loads(evaluation_path.read_text())
        evaluation["caveat"] = (
            "New path tests; frozen checkpoint, no retraining; recorded dynamic box motion, no force estimator"
        )
        evaluation_path.write_text(json.dumps(evaluation, indent=2) + "\n")
        motion_metrics(references, args.output / label, records)
    provenance = dict(
        trained=False,
        cases=records,
        checkpoint_sha256={
            label: hashlib.sha256(path.read_bytes()).hexdigest()
            for label, path in (("long", args.checkpoint), ("baseline", args.baseline_checkpoint))
        },
        scene="Unchanged open-ground material bed and box geometry; new external-force target paths",
        conditioning="Recorded dynamic MPM-MJWarp box motion; recurrent predicted particles, no state resets",
    )
    (args.output / "COMPLETE.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
