# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare reconstructed forces with MPM interval-average forces in current recordings."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import warp as wp

from particle_nerd.force_estimation.estimator import ContactEstimator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--max-iterations", type=int, default=2000)
    args = p.parse_args()
    wp.init()
    predicted, reference, diagnostics = [], [], []
    with h5py.File(args.episode, "r") as f:
        m = json.loads(f.attrs["manifest"])
        frames = f["frame_steps"][:]
        start = int(np.searchsorted(frames, m["settled_step"]))
        size, mass = np.array(m["body_size_m"]), m["body_mass_kg"]
        inertia = np.diag(mass / 12 * (np.sum(size**2) - size**2))
        dt = 1 / m["record_hz"]
        estimator = ContactEstimator(
            f["particle_mass"][:],
            mass,
            inertia,
            size,
            dt,
            max_iterations=args.max_iterations,
            friction=float(m.get("body_contact_friction", 0.6)),
        )
        for frame in range(start, min(start + args.steps, len(frames) - 1)):
            a, b = frames[frame : frame + 2]
            if not np.isclose((b - a) * m["dt"], dt):
                raise ValueError("Irregular recorded interval")
            external = f["external_wrench"][a:b].mean(axis=0)
            arrays = [
                wp.array(f["particle_position"][frame], dtype=wp.vec3, device="cuda:0"),
                wp.array(f["particle_velocity"][frame + 1], dtype=wp.vec3, device="cuda:0"),
                wp.array(f["body_pose"][a : a + 1], dtype=wp.transform, device="cuda:0"),
                wp.array(f["body_velocity"][a : a + 1], dtype=wp.spatial_vector, device="cuda:0"),
                wp.array(external[None], dtype=wp.spatial_vector, device="cuda:0"),
            ]
            predicted.append(estimator.estimate(*arrays).numpy()[0])
            reference.append(f["material_on_body_wrench"][a:b].mean(axis=0))
            diagnostics.append(estimator.diagnostics())
    if not predicted:
        raise ValueError("No transitions available")
    predicted, reference = np.array(predicted), np.array(reference)
    report = dict(
        episode=str(args.episode),
        particle_motion="recorded MPM; not learned",
        reference="MPM material-on-body wrench averaged over each particle recording interval",
        force_rmse_N=float(np.sqrt(np.mean((predicted[:, :3] - reference[:, :3]) ** 2))),
        torque_rmse_Nm=float(np.sqrt(np.mean((predicted[:, 3:] - reference[:, 3:]) ** 2))),
        predicted_wrench=predicted.tolist(),
        reference_wrench=reference.tolist(),
        diagnostics=diagnostics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["predicted_wrench", "reference_wrench", "diagnostics"]}
        )
    )


if __name__ == "__main__":
    main()
