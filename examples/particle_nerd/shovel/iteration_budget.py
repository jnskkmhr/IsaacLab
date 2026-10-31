# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare contact iteration limits on identical recorded learned-shovel states.

This measures numerical error against the converged estimator, not MPM accuracy.
Host loading and diagnostic readback are outside the synchronized estimator timing.
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import warp as wp

from particle_nerd.force_estimation.estimator import ContactEstimator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limits", nargs="+", type=int, default=[25, 50, 100, 200, 400, 800])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--relaxation", type=float, default=0.025)
    args = parser.parse_args()
    if args.stride < 1 or any(n < 1 for n in args.limits):
        parser.error("Stride and iteration limits must be positive")
    wp.init()
    result = dict(
        gpu=torch.cuda.get_device_name(),
        relaxation=args.relaxation,
        reference_relaxation=0.025,
        effective_mass_scale=180,
        tolerance=1e-9,
        reference_limit=10000,
        stride=args.stride,
        paths={},
    )
    for path in ("wide-circle", "figure-eight"):
        with h5py.File(args.reference_dir / f"{path}.hdf5") as f:
            mass = f["particle_mass"][:]
        with h5py.File(args.rollouts / path / "rollout.hdf5") as f:
            m = json.loads(f.attrs["manifest"])
            positions = f["particle_position"][:]
            poses, velocities = f["sampled_body_pose"][:], f["body_velocity"][:]
            external = f["external_wrench"][::4]
        estimator = ContactEstimator(
            mass,
            m["body_mass_kg"],
            m["body_inertia"],
            m["body_size_m"],
            0.02,
            shape="shovel",
            friction=0.5,
            effective_mass_scale=180,
            max_iterations=10000,
            jacobi_relaxation=0.025,
        )
        x = wp.empty(len(mass), dtype=wp.vec3, device="cuda:0")
        v = wp.empty_like(x)
        pose = wp.empty(1, dtype=wp.transform, device="cuda:0")
        velocity = wp.empty(1, dtype=wp.spatial_vector, device="cuda:0")
        force = wp.empty_like(velocity)
        records = {n: [] for n in [10000, *args.limits]}
        for step in range(0, len(positions) - 1, args.stride):
            x.assign(positions[step])
            v.assign((positions[step + 1] - positions[step]) / 0.02)
            pose.assign(poses[step : step + 1])
            velocity.assign(velocities[step : step + 1])
            force.assign(external[step : step + 1])
            if step == 0:
                for _ in range(3):
                    estimator.estimate(x, v, pose, velocity, force)
                wp.synchronize()
            for limit in records:
                estimator.max_iterations = limit
                estimator.jacobi_relaxation = 0.025 if limit == 10000 else args.relaxation
                wp.synchronize()
                start = time.perf_counter()
                estimator.estimate(x, v, pose, velocity, force)
                wp.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                records[limit].append(dict(step=step, wall_ms=elapsed, **estimator.diagnostics()))
        reference = np.array([r["wrench"] for r in records[10000]])
        assert all(r["finite"] and r["converged"] for r in records[10000])
        summaries = {}
        for limit, rows in records.items():
            errors = np.array([r["wrench"] for r in rows]) - reference
            summaries[limit] = dict(
                samples=len(rows),
                median_estimator_ms=float(np.median([r["wall_ms"] for r in rows])),
                force_rmse_N=float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=1)))),
                force_max_N=float(np.linalg.norm(errors[:, :3], axis=1).max()),
                torque_rmse_Nm=float(np.sqrt(np.mean(np.sum(errors[:, 3:] ** 2, axis=1)))),
                torque_max_Nm=float(np.linalg.norm(errors[:, 3:], axis=1).max()),
                converged_fraction=float(np.mean([r["converged"] for r in rows])),
                max_residual=float(max(r["residual"] for r in rows)),
            )
        result["paths"][path] = dict(summary=summaries, measurements=records)
        print(json.dumps({path: summaries}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
