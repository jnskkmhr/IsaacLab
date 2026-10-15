# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare reconstructed body forces with recorded MPM impulses on held-out episodes."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from dataset import validate_dataset
from reconstruction import reconstruct_reactions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--optimizer", choices=("trf", "osqp"), default="trf")
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--neighbor_radius", type=float, default=0.065)
    parser.add_argument("--contact_distance", type=float, default=0.04)
    parser.add_argument(
        "--max_iterations", type=int, default=100, help="Inverse least-squares limit; unrelated to MPM iterations."
    )
    args = parser.parse_args()
    if args.stride < 1 or args.start_step < 0:
        parser.error("Stride must be positive and start_step nonnegative.")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    validation = validate_dataset(args.dataset)
    estimates, references, residuals, timings, records = [], [], [], [], []
    with h5py.File(args.dataset, "r") as dataset:
        manifest = json.loads(dataset.attrs["particle_nerd_manifest"])
        for name, episode in dataset["data"].items():
            dt = float(episode["constants/dt"][0])
            for t in range(args.start_step, len(episode["actions"]), args.stride):
                result = reconstruct_reactions(
                    episode["states/particle_position"][t + 1],
                    episode["states/particle_velocity"][t],
                    episode["states/particle_velocity"][t + 1],
                    episode["constants/particle_mass"][0],
                    episode["states/body_pose"][t + 1],
                    episode["constants/body_size"][0],
                    episode["constants/body_friction"][0],
                    dt=dt,
                    gravity=manifest["gravity"],
                    neighbor_radius=args.neighbor_radius,
                    contact_distance=args.contact_distance,
                    regularization=args.regularization,
                    max_iterations=args.max_iterations,
                    optimizer=args.optimizer,
                )
                estimates.append(result.body_wrench)
                references.append(episode["reference/body_impulse"][t] / dt)
                residuals.append(result.relative_residual)
                timings.append(result.solve_seconds)
                records.append(
                    {
                        "episode": name,
                        "step": t,
                        "converged": result.converged,
                        "optimizer_status": result.optimizer_status,
                        "objective": result.objective,
                        "projected_gradient_relative": result.projected_gradient_relative,
                        "relative_residual": result.relative_residual,
                        "solve_seconds": result.solve_seconds,
                        "internal_edges": result.internal_edges,
                        "contacts": result.contacts,
                    }
                )
                print(
                    f"{name} step {t}: residual={result.relative_residual:.3g}, {result.solve_seconds:.2f} s",
                    flush=True,
                )
    if not estimates:
        raise ValueError("No transitions selected for evaluation.")
    estimates, references = np.asarray(estimates), np.asarray(references)
    error = estimates - references
    report = {
        "dataset": str(args.dataset.resolve()),
        "dataset_validation": validation,
        "split": manifest["split"],
        "samples": len(estimates),
        "closure": "compressive central particle forces; four-ray Coulomb contacts; quadratic force penalty",
        "settings": {
            key: getattr(args, key)
            for key in ("neighbor_radius", "contact_distance", "regularization", "max_iterations", "optimizer")
        },
        "force_mse_N2": float(np.mean(error[..., :3] ** 2)),
        "force_rmse_N_per_body": np.sqrt(np.mean(np.sum(error[..., :3] ** 2, axis=-1), axis=0)).tolist(),
        "torque_rmse_Nm_per_body": np.sqrt(np.mean(np.sum(error[..., 3:] ** 2, axis=-1), axis=0)).tolist(),
        "relative_force_error": float(np.linalg.norm(error[..., :3]) / max(np.linalg.norm(references[..., :3]), 1e-12)),
        "mean_particle_balance_relative_residual": float(np.mean(residuals)),
        "median_reconstruction_seconds": float(np.median(timings)),
        "solver_failures": sum(not item["converged"] for item in records),
        "records": records,
        "estimated_body_wrench": estimates.tolist(),
        "reference_body_wrench": references.tolist(),
        "interpretation": (
            "Small balance residual does not establish unique or accurate per-body reactions. "
            "This closure is not MPM's material law."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("records", "estimated_body_wrench", "reference_body_wrench")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
