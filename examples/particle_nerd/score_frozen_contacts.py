# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate frozen contact parameters on new datasets without parameter selection."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from evaluate_body_contact import evaluate, load


def summarize_episodes(rows, dt):
    summaries = {}
    for name in sorted({r["episode"] for r in rows}):
        selected = [r for r in rows if r["episode"] == name]
        predicted = np.array([r["estimate"] for r in selected])
        reference = np.array([r["reference"] for r in selected])
        error = predicted - reference
        peak = int(np.argmax(reference[:, 2]))
        summaries[name] = {
            "samples": len(selected),
            "force_squared_vector_error_sum": float(np.sum(error[:, :3] ** 2)),
            "torque_squared_vector_error_sum": float(np.sum(error[:, 3:] ** 2)),
            "force_vector_rmse_N": float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
            "torque_vector_rmse_Nm": float(np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1)))),
            "force_bias_N": np.mean(error[:, :3], axis=0).tolist(),
            "reference_peak_upward_force_N": float(reference[peak, 2]),
            "estimated_force_at_reference_peak_N": float(predicted[peak, 2]),
            "estimated_peak_upward_force_N": float(predicted[:, 2].max()),
            "estimated_peak_time_error_s": float((np.argmax(predicted[:, 2]) - peak) * dt),
            "last_100_mean_reference_force_N": np.mean(reference[-100:, :3], axis=0).tolist(),
            "last_100_mean_estimated_force_N": np.mean(predicted[-100:, :3], axis=0).tolist(),
        }
    rng = np.random.default_rng(30031)
    values = list(summaries.values())
    draws = rng.integers(len(values), size=(2000, len(values)))
    counts = np.array([v["samples"] for v in values])
    confidence = {}
    for key in ("force", "torque"):
        sums = np.array([v[key + "_squared_vector_error_sum"] for v in values])
        boot = np.sqrt(sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
        confidence[key + "_vector_rmse_95_percent_episode_bootstrap"] = np.quantile(boot, [0.025, 0.975]).tolist()
    return summaries, confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a frozen evaluation.")
    methods = {}
    for path in args.experiments:
        experiment = json.loads(path.read_text())
        if not experiment["complete"]:
            raise ValueError("Incomplete fitting experiment.")
        methods[path.stem] = experiment["selected_parameters"]
    report = {
        "complete": False,
        "methods": methods,
        "selection": "Parameters frozen before opening these datasets; no fitting in this script.",
        "limitations": "Reference particle motion; one-step box force evaluation, not a learned rollout.",
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "body_contact_reconstruction.py",
                "contact_iterations.py",
                "evaluate_body_contact.py",
                Path(__file__).name,
            )
        },
        "datasets": {},
    }
    for path in args.datasets:
        manifest, episodes = load(path)
        result = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": manifest["seed"],
            "body_mass_kg": manifest["body_mass_kg"],
            "mpm_iterations": manifest["max_iterations"],
            "methods": {},
        }
        report["datasets"][path.stem] = result
        for method, parameters in methods.items():
            metrics, rows = evaluate(
                episodes,
                sorted(episodes),
                parameters,
                body=manifest["moving_object_id"],
                gravity=manifest["gravity"],
                dense=True,
            )
            per_episode, confidence = summarize_episodes(rows, manifest["dt"])
            with h5py.File(path) as recording:
                direct = {
                    name: np.linalg.norm(
                        e["reference/rigid_contact_impulse"][:, manifest["moving_object_id"], :3], axis=1
                    )
                    > 1e-8
                    for name, e in recording["data"].items()
                }
            strata = {}
            for active in (False, True):
                selected = [r for r in rows if bool(direct[r["episode"]][r["step"]]) == active]
                if not selected:
                    continue
                estimate = np.array([r["estimate"] for r in selected])
                target = np.array([r["reference"] for r in selected])
                error = estimate - target
                strata["direct_rigid_contact" if active else "no_direct_rigid_contact"] = {
                    "samples": len(selected),
                    "force_vector_rmse_N": float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
                    "relative_force_L2": float(np.linalg.norm(error[:, :3]) / np.linalg.norm(target[:, :3])),
                }
            result["methods"][method] = {
                "metrics": metrics,
                "confidence": confidence,
                "direct_contact_strata": strata,
                "episodes": per_episode,
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({"dataset": path.stem, "method": method, "metrics": metrics, "confidence": confidence}),
                flush=True,
            )
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
