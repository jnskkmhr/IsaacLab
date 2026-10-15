# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure frozen contact-model sensitivity to particle errors and iteration count.

Noise is independent Gaussian error per particle coordinate. Body state remains
exact. These are controlled input perturbations, not a learned-error distribution.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from body_contact_reconstruction import reconstruct_body_contact
from evaluate_body_contact import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a sensitivity report.")
    parameters = json.loads(args.experiment.read_text())["selected_parameters"]
    manifest, episodes = load(args.dataset)
    names = sorted(episodes)[4:]
    body = manifest["moving_object_id"]
    settings = [
        (0, 0, 2000),
        (0.0001, 0, 2000),
        (0.001, 0, 2000),
        (0.005, 0, 2000),
        (0, 0.01, 2000),
        (0, 0.05, 2000),
        (0, 0.1, 2000),
        (0.001, 0.05, 2000),
        (0, 0, 25),
        (0, 0, 50),
        (0, 0, 100),
    ]
    report = {
        "complete": False,
        "dataset": str(args.dataset),
        "episodes": names,
        "parameters": parameters,
        "sampling": "every fifth transition",
        "noise_seed": 90471,
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("body_contact_reconstruction.py", "contact_iterations.py", Path(__file__).name)
        },
        "trials": [],
    }
    reference_estimates = None
    for position_sigma, velocity_sigma, maximum_iterations in settings:
        rng = np.random.default_rng(90471)
        estimates, targets, iterations, residuals, times, converged = [], [], [], [], [], []
        for name in names:
            e = episodes[name]
            previous_impulse = np.zeros_like(e["particle_position"][0])
            for t in range(len(e["reference"])):
                position = e["particle_position"][t + int(parameters["geometry"] == "end")].copy()
                velocity = e["particle_velocity"][t + 1].copy()
                position += position_sigma * rng.standard_normal(position.shape)
                velocity += velocity_sigma * rng.standard_normal(velocity.shape)
                result = reconstruct_body_contact(
                    position,
                    velocity,
                    e["particle_mass"],
                    e["body_pose"][t, body],
                    e["body_velocity"][t, body],
                    e["body_mass"][body],
                    e["body_inertia"][body],
                    e["body_size"][body],
                    dt=e["dt"],
                    gravity=manifest["gravity"],
                    contact_offset=parameters["contact_offset"],
                    effective_mass_scale=parameters["effective_mass_scale"],
                    extrapolate_velocity=parameters.get("affine", False),
                    stabilization=parameters.get("stabilization", 0),
                    previous_particle_impulse=previous_impulse,
                    history_weight=parameters.get("history_weight", 0.0),
                    friction=float(e["body_friction"][body]) if parameters.get("frictional", False) else None,
                    max_iterations=maximum_iterations,
                )
                previous_impulse = result.particle_impulse
                if t % 5:
                    continue
                estimates.append(result.body_wrench)
                targets.append(e["reference"][t, body])
                iterations.append(result.iterations)
                residuals.append(result.impulse_residual)
                times.append(result.solve_seconds)
                converged.append(result.converged)
        estimates, targets = np.array(estimates), np.array(targets)
        if reference_estimates is None:
            reference_estimates = estimates.copy()
        error = estimates - targets
        report["trials"].append(
            {
                "position_sigma_m": position_sigma,
                "velocity_sigma_mps": velocity_sigma,
                "max_iterations": maximum_iterations,
                "samples": len(targets),
                "force_vector_rmse_N": float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
                "torque_vector_rmse_Nm": float(np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1)))),
                "force_vector_change_rmse_N": float(
                    np.sqrt(np.mean(np.sum((estimates[:, :3] - reference_estimates[:, :3]) ** 2, axis=1)))
                ),
                "median_iterations": float(np.median(iterations)),
                "maximum_iterations_used": int(max(iterations)),
                "maximum_impulse_residual_Ns": float(max(residuals)),
                "solver_failures": sum(not c for c in converged),
                "median_seconds": float(np.median(times)),
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["trials"][-1]), flush=True)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
