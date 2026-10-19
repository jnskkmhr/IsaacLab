# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train activity selectors and evaluate frozen motion weights on disjoint trajectories."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from particle_nerd.gnn.data import Episodes
from particle_nerd.gnn.evaluate_selection import PREVIOUS_TEST_IDS


def balanced_ids(records, per_case):
    rng = np.random.default_rng(41763)
    chosen = []
    for case in sorted({r["case"] for r in records}):
        candidates = sorted(r["index"] for r in records if r["case"] == case)
        if len(candidates) < per_case:
            raise ValueError(f"Insufficient {case} trajectories")
        chosen.extend(int(x) for x in rng.choice(candidates, per_case, replace=False))
    return sorted(chosen)


def batch(commands, output):
    processes = []
    try:
        for gpu, name, command in commands:
            log = (output / f"{name}.log").open("w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((name, process, log))
            print(
                json.dumps(dict(event="launch", name=name, gpu=gpu, command=command)),
                flush=True,
            )
        failures = []
        for name, process, log in processes:
            code = process.wait()
            log.close()
            print(
                json.dumps(dict(event="finished", name=name, exit_code=code)),
                flush=True,
            )
            if code:
                failures.append((name, code))
        if failures:
            raise RuntimeError(f"Failed children: {failures}; inspect their logs")
    finally:
        for _, process, log in processes:
            if process.poll() is None:
                process.wait()
            if not log.closed:
                log.close()


def validation_score(path):
    report = json.loads(path.read_text())
    rows = report["rollouts"]
    if any(r["failure"] or r["frames"] != 250 for r in rows):
        return float("inf")
    # Equal weight on all-particle and genuinely displaced-particle error;
    # the latter prevents the much larger stationary population dominating.
    return float(
        np.mean(
            [
                r["final_position_rmse_m"] + r["moving_final_rmse_m"]
                for r in rows
                if r["moving_final_rmse_m"] is not None
            ]
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--training-steps", type=int, default=6000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    validation = balanced_ids(Episodes(args.data, "validation").records, 2)
    test_records = [r for r in Episodes(args.data, "test").records if r["index"] not in PREVIOUS_TEST_IDS]
    test = balanced_ids(test_records, 4)
    receipt = dict(
        validation_ids=validation,
        test_ids=test,
        previous_test_ids_excluded=PREVIOUS_TEST_IDS,
        label_distances_m=[0.0001, 0.0005],
        label_horizon_frames=5,
        training_steps=args.training_steps,
        selection=(
            "Minimum mean validation (all-particle RMSE + displaced-particle RMSE); "
            "threshold calibrated on validation classification with 95% recall target"
        ),
        motion_model="Fixed original checkpoint",
        rigid_motion="Recorded MPM box motion; no force estimator",
        computation="Complete particle graph; no compute reduction claimed",
    )
    (args.output / "protocol.json").write_text(json.dumps(receipt, indent=2) + "\n")
    variants = [("0001", 0.0001), ("0005", 0.0005)]
    batch(
        [
            (
                gpu,
                f"train_{name}",
                [
                    sys.executable,
                    "-m",
                    "particle_nerd.gnn.train_selector",
                    "--data",
                    str(args.data),
                    "--output",
                    str(args.output / f"selector_{name}"),
                    "--steps",
                    str(args.training_steps),
                    "--evaluate-every",
                    "1000",
                    "--label-distance",
                    str(distance),
                ],
            )
            for gpu, (name, distance) in enumerate(variants)
        ],
        args.output,
    )

    def evaluate(name, gpu, split, ids, modes, distance, selector=None):
        command = [
            sys.executable,
            "-m",
            "particle_nerd.gnn.evaluate_selection",
            "--data",
            str(args.data),
            "--checkpoint",
            str(args.checkpoint),
            "--output",
            str(args.output / name),
            "--split",
            split,
            "--episodes",
            str(len(ids)),
            "--ids",
            *map(str, ids),
            "--modes",
            *modes,
            "--label-distance",
            str(distance),
            "--compile",
        ]
        if selector:
            command += ["--selector", str(selector)]
        return gpu, name, command

    batch(
        [
            evaluate(
                f"validation_{name}",
                gpu,
                "validation",
                validation,
                ["learned"],
                distance,
                args.output / f"selector_{name}" / "best.pt",
            )
            for gpu, (name, distance) in enumerate(variants)
        ],
        args.output,
    )
    scores = {name: validation_score(args.output / f"validation_{name}" / "evaluation.json") for name, _ in variants}
    best_name, best_distance = min(variants, key=lambda item: scores[item[0]])
    if not np.isfinite(scores[best_name]):
        raise RuntimeError("Both selector variants failed validation rollouts")
    selected = args.output / f"selector_{best_name}" / "best.pt"
    receipt.update(
        validation_scores_m=scores,
        selected_variant=best_name,
        selected_label_distance_m=best_distance,
        selected_checkpoint=str(selected),
    )
    (args.output / "selection.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(event="validation_selection", **receipt)), flush=True)
    batch(
        [
            evaluate(
                f"test_{mode}",
                gpu,
                "test",
                test,
                [mode],
                best_distance,
                selected if mode == "learned" else None,
            )
            for gpu, mode in enumerate(["all", "oracle", "velocity", "learned"])
        ],
        args.output,
    )
    (args.output / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
