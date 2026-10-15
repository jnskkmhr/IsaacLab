# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a joint particle/body contact projection and its particle-motion changes."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import h5py
import numpy as np
from evaluate_body_contact import STEPS, load
from network_contact_reconstruction import project_contact_network


def evaluate(episodes, names, manifest, parameters, dense=False):
    errors, corrections, times, iterations, residuals, failures = [], [], [], [], [], 0
    body = manifest["moving_object_id"]
    rows = []
    for name in names:
        e = episodes[name]
        for t in range(len(e["reference"])) if dense else STEPS:
            result = project_contact_network(
                e["particle_position"][t],
                e["particle_velocity"][t + 1],
                e["particle_mass"],
                e["particle_radius"],
                e["body_pose"][t],
                e["body_velocity"][t],
                e["body_mass"],
                e["body_inertia"],
                e["body_size"],
                body,
                dt=e["dt"],
                gravity=manifest["gravity"],
                **parameters,
            )
            error = result.body_wrench[body] - e["reference"][t, body]
            correction = result.particle_velocity - e["particle_velocity"][t + 1]
            errors.append(error)
            corrections.append(float(np.mean(np.sum(correction**2, axis=1))))
            times.append(result.solve_seconds)
            iterations.append(result.iterations)
            residuals.append(result.momentum_correction_residual)
            failures += not result.converged
            rows.append(
                {
                    "episode": name,
                    "step": t,
                    "estimate": result.body_wrench[body].tolist(),
                    "reference": e["reference"][t, body].tolist(),
                    "particle_velocity_change_rmse_mps": float(np.sqrt(corrections[-1])),
                    "converged": result.converged,
                    "status": result.status,
                }
            )
        print(
            json.dumps(
                {"episode": name, "parameters": parameters, "samples_so_far": len(errors), "failures": failures}
            ),
            flush=True,
        )
    errors = np.array(errors)
    return {
        "samples": len(errors),
        "force_vector_rmse_N": float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
        "torque_vector_rmse_Nm": float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
        "particle_velocity_change_rmse_mps": float(np.sqrt(np.mean(corrections))),
        "mean_momentum_correction_residual_Ns": float(np.mean(residuals)),
        "median_seconds": float(np.median(times)),
        "median_iterations": float(np.median(iterations)),
        "solver_failures": int(failures),
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--radius-scales", default=".75,1,1.25")
    parser.add_argument("--stabilizations", default="0,.01,.05")
    parser.add_argument("--dense-validation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an experiment.")
    manifest, episodes = load(args.calibration)
    with h5py.File(args.calibration) as f:
        for name, e in episodes.items():
            e["particle_radius"] = f["data"][name]["constants/particle_radius"][0].astype(float)
    names = sorted(episodes)
    fitting, validation = names[:4], names[4:]
    report = {
        "complete": False,
        "dataset": str(args.calibration),
        "fitting_episodes": fitting,
        "validation_episodes": validation,
        "friction": "none; normal contacts only",
        "limitations": (
            "Particle velocities are corrected, not held equal to the reference. "
            "Material points are approximated by noncohesive spheres. "
            "This is a postprojection experiment, not an inverse constitutive law."
        ),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("network_contact_reconstruction.py", Path(__file__).name)
        },
        "trials": [],
    }
    for radius, stabilization in itertools.product(
        map(float, args.radius_scales.split(",")), map(float, args.stabilizations.split(","))
    ):
        parameters = {"radius_scale": radius, "stabilization": stabilization}
        metrics, _ = evaluate(episodes, fitting, manifest, parameters)
        report["trials"].append({"parameters": parameters, **metrics})
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["trials"][-1]), flush=True)
    candidates = [t for t in report["trials"] if t["solver_failures"] == 0]
    if not candidates:
        raise RuntimeError("No fitting candidate converged.")
    chosen = min(candidates, key=lambda t: t["force_vector_rmse_N"])
    report["selected_parameters"] = chosen["parameters"]
    report["validation_metrics"], report["validation_records"] = evaluate(
        episodes, validation, manifest, chosen["parameters"], dense=args.dense_validation
    )
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": chosen, "validation": report["validation_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
