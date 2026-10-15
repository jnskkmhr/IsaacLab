# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproducible DEM-prior experiment; calibration and evaluation use different episodes.

All losses are against MPM interval-average reactions, not learned motion.
Reference forces enter scoring and calibration only, never force reconstruction.
"""

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import scipy
from dem_reconstruction import ContactHistory, DemParameters, assemble_dem_frame, fit_dem_frame
from reconstruction import reconstruct_reactions

CALIBRATION_STEPS = [99, 159, 169, 199, 249, 299]
# Fixed time grid, augmented by one target-defined peak diagnostic per episode.
EVALUATION_STEPS = sorted(set(range(19, 400, 40)) | set(range(140, 191, 5)) | set(CALIBRATION_STEPS))


def load_episodes(path):
    with h5py.File(path) as f:
        manifest = json.loads(f.attrs["particle_nerd_manifest"])
        if not manifest.get("complete"):
            raise ValueError("Dataset is incomplete.")
        episodes = {}
        for name, e in f["data"].items():
            episodes[name] = {
                key: e["states/" + key][:].astype(float)
                for key in ("particle_position", "particle_velocity", "body_pose", "body_velocity")
            }
            episodes[name].update(
                {
                    key: e["constants/" + key][0].astype(float)
                    for key in ("particle_mass", "particle_radius", "body_size", "body_friction")
                }
            )
            episodes[name]["dt"] = float(e["constants/dt"][0])
            episodes[name]["reference"] = e["reference/body_impulse"][:].astype(float) / episodes[name]["dt"]
    if len(episodes) < 4 or any(len(e["reference"]) != 400 for e in episodes.values()):
        raise ValueError("This protocol requires at least four complete 400-step episodes.")
    return manifest, episodes


def iterate_frames(e, parameters, steps, *, gravity, noise_std=0, seed=0):
    history = ContactHistory()
    # Persistent offsets isolate geometry sensitivity; velocities/accelerations remain exact.
    offsets = np.random.default_rng(seed).normal(0, noise_std, e["particle_position"].shape[1:])
    stop = max(steps) + 1
    for t in range(stop):
        if parameters.friction_model == "velocity" and t not in steps:
            continue
        frame = assemble_dem_frame(
            e["particle_position"][t + 1] + offsets,
            e["particle_velocity"][t + 1],
            e["particle_mass"],
            e["particle_radius"],
            e["body_pose"][t + 1],
            e["body_velocity"][t + 1],
            e["body_size"],
            e["body_friction"],
            (e["particle_velocity"][t + 1] - e["particle_velocity"][t]) / e["dt"],
            dt=e["dt"],
            parameters=parameters,
            history=history,
            gravity=gravity,
        )
        if t in steps:
            yield t, frame


def record(name, t, estimate, reference, **diagnostics):
    return {
        "episode": name,
        "step": t,
        "estimated_wrench": np.asarray(estimate).tolist(),
        "reference_wrench": np.asarray(reference).tolist(),
        **diagnostics,
    }


def metrics(records, moving_body):
    pred = np.asarray([r["estimated_wrench"] for r in records])
    ref = np.asarray([r["reference_wrench"] for r in records])
    diff = pred - ref
    return {
        "samples": len(records),
        "force_mse_N2": float(np.mean(diff[:, :, :3] ** 2)),
        "force_rmse_N_per_component": float(np.sqrt(np.mean(diff[:, :, :3] ** 2))),
        "relative_force_L2": float(np.linalg.norm(diff[:, :, :3]) / max(np.linalg.norm(ref[:, :, :3]), 1e-12)),
        "body_force_vector_rmse_N": np.sqrt(np.mean(np.sum(diff[:, :, :3] ** 2, axis=2), axis=0)).tolist(),
        "box_force_mse_N2": float(np.mean(diff[:, moving_body, :3] ** 2)),
        "box_force_vector_rmse_N": float(np.sqrt(np.mean(np.sum(diff[:, moving_body, :3] ** 2, axis=1)))),
        "body_torque_vector_rmse_Nm": np.sqrt(np.mean(np.sum(diff[:, :, 3:] ** 2, axis=2), axis=0)).tolist(),
        "mean_balance_relative_residual": float(np.mean([r["relative_residual"] for r in records]))
        if "relative_residual" in records[0]
        else None,
        "solver_failures": sum(not r.get("converged", True) for r in records),
        "median_seconds": float(np.median([r.get("solve_seconds", 0) + r.get("assembly_seconds", 0) for r in records])),
    }


def save(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    manifest, episodes = load_episodes(args.dataset)
    names = sorted(episodes)
    calibration, evaluation = names[:2], names[2:4]
    gravity, moving = manifest["gravity"], manifest["moving_object_id"]
    steps_by_episode = {}
    peak_steps = {}
    for name in evaluation:
        peak = int(np.argmax(np.linalg.norm(episodes[name]["reference"][:, moving, :3], axis=1)))
        steps_by_episode[name] = sorted(set(EVALUATION_STEPS) | {peak})
        peak_steps[name] = peak
    report = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "calibration_episodes": calibration,
        "evaluation_episodes": evaluation,
        "moving_body": moving,
        "calibration_steps": CALIBRATION_STEPS,
        "evaluation_steps": steps_by_episode,
        "peak_steps": peak_steps,
        "loss_definition": "Mean squared error across samples, six bodies, three force components; units N^2.",
        "limitations": [
            "Reference MPM motion, not NeRD inference; only two evaluation episodes from one scene.",
            "The previously viewed validation file is reused; this is not a pristine final test set.",
            "DEM parameters are effective approximations, not a conversion of MPM Young modulus.",
            "Frictional pair forces omit grain spin and APIC angular-momentum reconstruction.",
            "End-of-step force estimates are compared to interval-average MPM forces.",
            "The reference-peak frame is included explicitly for peak diagnostics; other frames use a fixed schedule.",
        ],
        "methods": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.baseline_only:
        records = []
        for name in evaluation:
            e = episodes[name]
            for t in steps_by_episode[name]:
                r = reconstruct_reactions(
                    e["particle_position"][t + 1],
                    e["particle_velocity"][t],
                    e["particle_velocity"][t + 1],
                    e["particle_mass"],
                    e["body_pose"][t + 1],
                    e["body_size"],
                    e["body_friction"],
                    dt=e["dt"],
                    gravity=gravity,
                )
                records.append(
                    record(
                        name,
                        t,
                        r.body_wrench,
                        e["reference"][t],
                        relative_residual=r.relative_residual,
                        converged=r.converged,
                        solve_seconds=r.solve_seconds,
                    )
                )
                print(f"original {name} {t}: {r.solve_seconds:.2f}s, converged={r.converged}", flush=True)
                report["methods"]["original"] = {"metrics": metrics(records, moving), "records": records}
                save(args.output, report)
        report["complete"] = True
        save(args.output, report)
        return
    # Select effective stiffness and prior weight on calibration episodes only.
    trials, selected = [], {}
    for mode in ("velocity", "history"):
        for stiffness in (100.0, 1000.0, 10000.0):
            params = DemParameters(
                normal_stiffness=stiffness, friction_model=mode, internal_friction=manifest["material"]["friction"]
            )
            # Advance history once per trajectory and reuse selected frames across weights.
            frames = [
                (name, t, frame)
                for name in calibration
                for t, frame in iterate_frames(episodes[name], params, CALIBRATION_STEPS, gravity=gravity)
            ]
            for weight in (0.001, 0.1):
                records = []
                for name, t, frame in frames:
                    r = fit_dem_frame(frame, law_weight=weight)
                    records.append(record(name, t, r.pop("body_wrench"), episodes[name]["reference"][t], **r))
                trial = {"parameters": asdict(params), "law_weight": weight, "metrics": metrics(records, moving)}
                trials.append(trial)
                print("calibration", mode, stiffness, weight, json.dumps(trial["metrics"]), flush=True)
                report["calibration_trials"] = trials
                save(args.output, report)
        candidates = [t for t in trials if t["parameters"]["friction_model"] == mode]
        admissible = [
            t
            for t in candidates
            if t["metrics"]["solver_failures"] == 0 and t["metrics"]["mean_balance_relative_residual"] <= 0.05
        ]
        pool = admissible or [t for t in candidates if t["metrics"]["solver_failures"] == 0] or candidates
        selected[mode] = min(pool, key=lambda t: t["metrics"]["force_mse_N2"])
        selected[mode]["passed_calibration_balance_gate"] = bool(admissible)
    report["selected"] = selected
    report["selection_rule"] = (
        "Lowest force MSE per friction model among converged trials with mean balance residual <=5%; "
        "if none, report failure of gate and use converged minimum MSE."
    )
    chosen_mode = min(selected, key=lambda mode: selected[mode]["metrics"]["force_mse_N2"])
    report["noise_test_mode_selected_on_calibration"] = chosen_mode
    for mode in ("velocity", "history"):
        params = DemParameters(**selected[mode]["parameters"])
        weight = selected[mode]["law_weight"]
        fitted, direct, zero = [], [], []
        for name in evaluation:
            e = episodes[name]
            for t, frame in iterate_frames(e, params, range(400), gravity=gravity):
                direct.append(
                    record(
                        name, t, frame.direct_body_wrench, e["reference"][t], assembly_seconds=frame.assembly_seconds
                    )
                )
                if t in steps_by_episode[name]:
                    r = fit_dem_frame(frame, law_weight=weight)
                    fitted.append(record(name, t, r.pop("body_wrench"), e["reference"][t], **r))
                    if mode == chosen_mode:
                        r = fit_dem_frame(frame, law_weight=weight, zero_prior=True)
                        zero.append(record(name, t, r.pop("body_wrench"), e["reference"][t], **r))
            print("evaluation", name, mode, json.dumps(metrics(fitted, moving)), flush=True)
        report["methods"][mode + "_fit"] = {"metrics": metrics(fitted, moving), "records": fitted}
        matched = [r for r in direct if r["step"] in steps_by_episode[r["episode"]]]
        report["methods"][mode + "_direct"] = {
            "metrics": metrics(matched, moving),
            "full_trajectory_metrics": metrics(direct, moving),
            "records": direct,
        }
        if zero:
            report["methods"]["zero_prior_same_basis"] = {"metrics": metrics(zero, moving), "records": zero}
        save(args.output, report)
    # Persistent, isotropic per-particle position offsets; no artificial acceleration noise.
    report["geometry_sensitivity"] = []
    params = DemParameters(**selected[chosen_mode]["parameters"])
    for noise in (0.0001, 0.001):
        records, changes = [], []
        clean = {
            (r["episode"], r["step"]): np.array(r["estimated_wrench"])
            for r in report["methods"][chosen_mode + "_fit"]["records"]
        }
        for index, name in enumerate(evaluation):
            for t, frame in iterate_frames(
                episodes[name], params, CALIBRATION_STEPS, gravity=gravity, noise_std=noise, seed=2026 + index
            ):
                r = fit_dem_frame(frame, law_weight=selected[chosen_mode]["law_weight"])
                estimate = r.pop("body_wrench")
                changes.append(estimate[:, :3] - clean[(name, t)][:, :3])
                records.append(record(name, t, estimate, episodes[name]["reference"][t], **r))
        report["geometry_sensitivity"].append(
            {
                "position_noise_std_m_per_axis": noise,
                "metrics": metrics(records, moving),
                "force_change_rmse_N_per_component": float(np.sqrt(np.mean(np.asarray(changes) ** 2))),
                "records": records,
            }
        )
        print("geometry sensitivity", noise, json.dumps(metrics(records, moving)), flush=True)
        save(args.output, report)
    report["complete"] = True
    save(args.output, report)
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
