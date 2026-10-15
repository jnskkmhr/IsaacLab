# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read and validate the particle NeRD episode contract without simulation imports."""

import json

import h5py
import numpy as np


def validate_dataset(path):
    """Check step alignment, finite values, and particle/body identity in every episode.

    Also report global linear momentum balance. This is a diagnostic, not proof
    that individual body reactions or torque measurements are accurate.
    """
    residuals, force_scales = [], []
    box_residuals, box_force_scales = [], []
    with h5py.File(path, "r") as dataset:
        if "particle_nerd_manifest" not in dataset.attrs:
            raise ValueError("Dataset is incomplete: collection manifest is missing.")
        manifest = json.loads(dataset.attrs["particle_nerd_manifest"])
        if not manifest["complete"]:
            raise ValueError("Collection was not completed.")
        episodes = dataset["data"]
        if not len(episodes):
            raise ValueError("Dataset contains no episodes.")
        particle_counts, total_steps = set(), 0
        for name, episode in episodes.items():
            x = episode["states/particle_position"][:]
            v = episode["states/particle_velocity"][:]
            pose = episode["states/body_pose"][:]
            body_velocity = episode["states/body_velocity"][:]
            action = episode["actions"][:]
            impulse = episode["reference/body_impulse"][:]
            mass = episode["constants/particle_mass"][0]
            dt = float(episode["constants/dt"][0])
            steps, bodies = impulse.shape[:2]
            if x.shape != v.shape or x.shape[0] != steps + 1 or len(action) != steps or len(pose) != steps + 1:
                raise ValueError(f"{name}: states and transitions are not aligned.")
            if (
                pose.shape[1:] != (bodies, 7)
                or body_velocity.shape != (steps + 1, bodies, 6)
                or impulse.shape[-1] != 6
                or action.shape != (steps, 0 if manifest["schema_version"] >= 2 else 3)
            ):
                raise ValueError(f"{name}: invalid body geometry or impulse shape.")
            if not np.array_equal(episode["constants/body_object_id"][0], np.arange(bodies)):
                raise ValueError(f"{name}: inconsistent object IDs.")
            if len(mass) != x.shape[1] or not np.array_equal(episode["constants/particle_id"][0], np.arange(len(mass))):
                raise ValueError(f"{name}: inconsistent particle IDs or masses.")
            if manifest["schema_version"] == 1 and not np.allclose(
                pose[1:, manifest["moving_object_id"], :3], action, atol=1e-6
            ):
                raise ValueError(f"{name}: prescribed body positions do not match recorded actions.")
            for array in (x, v, pose, body_velocity, action, impulse, mass):
                if not np.isfinite(array).all():
                    raise ValueError(f"{name}: nonfinite data.")
            if np.any(mass <= 0) or dt <= 0:
                raise ValueError(f"{name}: mass and timestep must be positive.")
            if not np.allclose(np.linalg.norm(pose[..., 3:], axis=-1), 1, atol=1e-4):
                raise ValueError(f"{name}: invalid body quaternion.")
            if manifest["schema_version"] >= 2:
                applied = episode["reference/applied_body_impulse"][:]
                following = episode["reference/next_body_impulse"][:]
                if applied.shape != impulse.shape or not np.isfinite(applied).all():
                    raise ValueError(f"{name}: invalid applied impulse.")
                if not np.allclose(applied[0], 0, atol=1e-7):
                    raise ValueError(f"{name}: coupling feedback was not cleared at reset.")
                if not np.allclose(applied[1:], following[:-1], atol=1e-6):
                    raise ValueError(f"{name}: lagged coupling impulses are not aligned.")
                box = manifest["moving_object_id"]
                body_mass = episode["constants/body_mass"][0, box]
                box_force = body_mass * (
                    np.diff(body_velocity[:, box, :3], axis=0) / dt - np.asarray(manifest["gravity"])
                )
                box_residuals.extend(
                    np.linalg.norm(
                        box_force - (applied[:, box, :3] + episode["reference/rigid_contact_impulse"][:, box, :3]) / dt,
                        axis=1,
                    ).tolist()
                )
                box_force_scales.extend(
                    np.maximum(
                        np.linalg.norm(box_force, axis=1), body_mass * np.linalg.norm(manifest["gravity"])
                    ).tolist()
                )
            particle_counts.add(len(mass))
            total_steps += steps
            material_force = np.sum(mass[None, :, None] * np.diff(v, axis=0) / dt, axis=1)
            material_force -= mass.sum() * np.asarray(manifest["gravity"])
            body_force = impulse[..., :3].sum(axis=1) / dt
            residuals.extend(np.linalg.norm(material_force + body_force, axis=1).tolist())
            weight = mass.sum() * np.linalg.norm(manifest["gravity"])
            force_scales.extend(np.maximum(np.linalg.norm(body_force, axis=1), weight).tolist())
        return {
            "episodes": len(episodes),
            "transitions": total_steps,
            "particles_per_episode": sorted(particle_counts),
            "box_mean_force_balance_residual_N": float(np.mean(box_residuals)) if box_residuals else None,
            "box_relative_force_balance_residual": float(
                np.linalg.norm(box_residuals) / max(np.linalg.norm(box_force_scales), 1e-12)
            )
            if box_residuals
            else None,
            "mean_linear_momentum_residual_N": float(np.mean(residuals)),
            "relative_linear_momentum_residual": float(
                np.linalg.norm(residuals) / max(np.linalg.norm(force_scales), 1e-12)
            ),
        }
