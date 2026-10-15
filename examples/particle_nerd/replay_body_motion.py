# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a free box against recorded material motion, without running MPM.

This is an offline rigid-body diagnostic, not a coupled NeRD rollout: material
motion remains fixed to its reference recording. A reference-force replay checks
the integration approximation. Predicted rigid-ground/wall intersections are
reported as invalid; this diagnostic does not silently resolve rigid collisions.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from body_contact_reconstruction import reconstruct_body_contact
from evaluate_body_contact import load
from scipy.spatial.transform import Rotation


def integrate_body(pose, velocity, wrench, mass, inertia, gravity, dt):
    """Semi-implicit translation and a first-order gyroscopic rotation update."""
    rotation = Rotation.from_quat(pose[3:])
    matrix = rotation.as_matrix()
    world_inertia = matrix @ inertia @ matrix.T
    updated = velocity.copy()
    updated[:3] += dt * (np.asarray(gravity) + wrench[:3] / mass)
    updated[3:] += dt * np.linalg.solve(
        world_inertia, wrench[3:] - np.cross(velocity[3:], world_inertia @ velocity[3:])
    )
    orientation = Rotation.from_rotvec(updated[3:] * dt) * rotation
    return np.r_[pose[:3] + dt * updated[:3], orientation.as_quat()], updated


def replay(episode, parameters, body, gravity, *, reference=False):
    pose = episode["body_pose"][0, body].copy()
    velocity = episode["body_velocity"][0, body].copy()
    dt = episode["dt"]
    mass = episode["body_mass"][body]
    inertia = episode["body_inertia"][body]
    poses = [pose.copy()]
    velocities = [velocity.copy()]
    forces = []
    previous = np.zeros(6)
    previous_particle_impulse = np.zeros_like(episode["particle_position"][0])
    failures = 0
    minimum_ground_clearance = np.inf
    minimum_wall_clearance = np.inf
    half = episode["body_size"][body] / 2
    corners = np.array(
        [[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])]
    )
    ground_top = episode["body_pose"][0, 0, 2] + episode["body_size"][0, 2] / 2
    wall_limits = np.array([0.16, 0.16])  # Inner faces of the fixed collection tray.
    for t in range(len(episode["reference"])):
        if reference:
            current = episode["reference"][t, body]
        else:
            result = reconstruct_body_contact(
                episode["particle_position"][t + int(parameters["geometry"] == "end")],
                episode["particle_velocity"][t + 1],
                episode["particle_mass"],
                pose,
                velocity,
                mass,
                inertia,
                episode["body_size"][body],
                dt=dt,
                gravity=gravity,
                contact_offset=parameters["contact_offset"],
                effective_mass_scale=parameters["effective_mass_scale"],
                friction=float(episode["body_friction"][body]) if parameters.get("frictional", False) else None,
                extrapolate_velocity=parameters.get("affine", False),
                stabilization=parameters.get("stabilization", 0),
                previous_particle_impulse=previous_particle_impulse,
                history_weight=parameters.get("history_weight", 0.0),
                max_iterations=2000 if parameters.get("frictional", False) else 10000,
            )
            current = result.body_wrench
            previous_particle_impulse = result.particle_impulse
            failures += not result.converged
        # Match the collector's one-step delay in applying material reactions.
        pose, velocity = integrate_body(pose, velocity, previous, mass, inertia, gravity, dt)
        previous = current
        poses.append(pose.copy())
        velocities.append(velocity.copy())
        forces.append(current.tolist())
        vertices = pose[:3] + corners @ Rotation.from_quat(pose[3:]).as_matrix().T
        minimum_ground_clearance = min(minimum_ground_clearance, float(vertices[:, 2].min() - ground_top))
        minimum_wall_clearance = min(
            minimum_wall_clearance, float(np.min(wall_limits - np.max(np.abs(vertices[:, :2]), axis=0)))
        )
        if not np.isfinite(pose).all() or np.linalg.norm(velocity) > 100 or np.linalg.norm(pose[:3]) > 5:
            break
    poses = np.array(poses)
    velocities = np.array(velocities)
    count = len(poses)
    target = episode["body_pose"][:count, body]
    position_error = np.linalg.norm(poses[:, :3] - target[:, :3], axis=1)
    angle_error = (Rotation.from_quat(poses[:, 3:]) * Rotation.from_quat(target[:, 3:]).inv()).magnitude()
    return {
        "transitions": len(forces),
        "completed": len(forces) == len(episode["reference"]),
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "final_position_error_m": float(position_error[-1]),
        "orientation_rmse_deg": float(np.sqrt(np.mean(angle_error**2)) * 180 / np.pi),
        "final_orientation_error_deg": float(angle_error[-1] * 180 / np.pi),
        "minimum_ground_clearance_m": minimum_ground_clearance,
        "minimum_wall_clearance_m": minimum_wall_clearance,
        "valid_without_rigid_contact": minimum_ground_clearance >= 0 and minimum_wall_clearance >= 0,
        "solver_failures": int(failures),
        "poses": poses.tolist(),
        "velocities": velocities.tolist(),
        "forces": forces,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("experiments", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a replay report.")
    manifest, episodes = load(args.dataset)
    with h5py.File(args.dataset) as f:
        if any(np.max(np.abs(e["reference/rigid_contact_impulse"][:])) > 1e-9 for e in f["data"].values()):
            raise ValueError("This free-body diagnostic requires reference episodes without direct rigid contacts.")
    methods = {"reference_force": None}
    for path in args.experiments:
        experiment = json.loads(path.read_text())
        if not experiment["complete"]:
            raise ValueError(f"Incomplete experiment: {path}")
        methods[path.stem] = experiment["selected_parameters"]
    output = {
        "dataset": str(args.dataset),
        "integration": "semi-implicit translation; first-order gyroscopic rotation",
        "limitation": "Recorded material motion is held fixed. No particle motion is predicted or coupled back.",
        "methods": {},
    }
    for method, parameters in methods.items():
        output["methods"][method] = {}
        for name, episode in episodes.items():
            result = replay(
                episode, parameters, manifest["moving_object_id"], manifest["gravity"], reference=parameters is None
            )
            output["methods"][method][name] = result
            print(
                json.dumps(
                    {
                        "method": method,
                        "episode": name,
                        **{k: v for k, v in result.items() if k not in ("poses", "velocities", "forces")},
                    }
                ),
                flush=True,
            )
            args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
