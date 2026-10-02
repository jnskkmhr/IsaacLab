# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validated, named trajectory data for WBC reference commands."""

import numpy as np
import torch


class WholeBodyDataset:
    """Load static poses or trajectories, preserving boundaries and resolving columns by name.

    All positions are metres, velocities are per second, and quaternions are XYZW.
    No pickle objects or implicit body/joint order conventions are accepted.
    """

    def __init__(self, path: str, joint_names: list[str], body_names: list[str], device: str):
        with np.load(path, allow_pickle=False) as data:
            if data["schema_version"].item() != 1 or data["quaternion_order"].item() != "xyzw":
                raise ValueError("WBC requires schema_version=1 and XYZW quaternions")
            dataset_kind = data["dataset_kind"].item() if "dataset_kind" in data else "trajectories"
            if dataset_kind not in ("static_poses", "trajectories"):
                raise ValueError(f"Unsupported WBC dataset kind: {dataset_kind}")
            self.is_static = dataset_kind == "static_poses"
            self.fps = float(data["fps"].item())
            if not np.isfinite(self.fps) or self.fps <= 0:
                raise ValueError("Dataset fps must be finite and positive")
            stored_joints = data["joint_names"].tolist()
            stored_bodies = data["body_names"].tolist()
            if len(set(stored_joints)) != len(stored_joints) or set(stored_joints) != set(joint_names):
                raise ValueError("Dataset and robot must contain the same unique joint names")
            if len(set(stored_bodies)) != len(stored_bodies) or not set(body_names) <= set(stored_bodies):
                raise ValueError("Dataset must contain every tracked body exactly once")
            joint_indices = [stored_joints.index(name) for name in joint_names]
            body_indices = [stored_bodies.index(name) for name in body_names]
            frame_count = len(data["joint_pos"])
            starts, lengths = data["clip_starts"], data["clip_lengths"]
            if (
                starts.ndim != 1
                or len(starts) == 0
                or lengths.shape != starts.shape
                or not np.issubdtype(starts.dtype, np.integer)
                or not np.issubdtype(lengths.dtype, np.integer)
                or np.any(lengths < (1 if self.is_static else 2))
                or (self.is_static and np.any(lengths != 1))
                or not np.array_equal(starts, np.r_[0, lengths.cumsum()[:-1]])
                or lengths.sum() != frame_count
            ):
                raise ValueError(
                    "Clip boundaries must partition all frames: one frame per static pose, at least two per trajectory"
                )
            self.clip_starts = torch.tensor(starts, dtype=torch.long, device=device)
            self.clip_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            arrays = {}
            for name, width in (
                ("joint_pos", None),
                ("joint_vel", None),
                ("body_pos_w", 3),
                ("body_quat_w", 4),
                ("body_lin_vel_w", 3),
                ("body_ang_vel_w", 3),
                ("foot_contact", 2),
            ):
                value = data[name]
                expected = (
                    (frame_count, len(stored_joints))
                    if width is None
                    else (frame_count, 2)
                    if name == "foot_contact"
                    else (frame_count, len(stored_bodies), width)
                )
                if value.shape != expected or not np.isfinite(value).all():
                    raise ValueError(f"{name} must have shape {expected} and finite values")
                if width is None:
                    value = value[:, joint_indices]
                elif name != "foot_contact":
                    value = value[:, body_indices]
                arrays[name] = torch.tensor(value, dtype=torch.float32, device=device)
            if not torch.allclose(
                arrays["body_quat_w"].norm(dim=-1), torch.ones((frame_count, len(body_names)), device=device), atol=1e-4
            ):
                raise ValueError("Dataset quaternions must be normalized")
            if not np.isin(data["foot_contact"], (0, 1)).all():
                raise ValueError("Foot contact labels must be zero or one")
            if self.is_static:
                if any(np.any(data[name] != 0) for name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")):
                    raise ValueError("Static pose references must have zero velocities")
                if not np.all(data["foot_contact"] == 1):
                    raise ValueError("Static WBC poses must keep both feet in contact")
        self.joint_pos = arrays["joint_pos"]
        self.joint_vel = arrays["joint_vel"]
        self.body_pos_w = arrays["body_pos_w"]
        self.body_quat_w = arrays["body_quat_w"]
        self.body_lin_vel_w = arrays["body_lin_vel_w"]
        self.body_ang_vel_w = arrays["body_ang_vel_w"]
        self.foot_contact = arrays["foot_contact"]
        self.stance_ids = None
        with np.load(path, allow_pickle=False) as data:
            if "stance_ids" in data:
                stance_ids = data["stance_ids"]
                positions, orientations = data["stance_foot_pos_w"], data["stance_foot_quat_w"]
                if (
                    not self.is_static
                    or stance_ids.shape != (frame_count,)
                    or not np.issubdtype(stance_ids.dtype, np.integer)
                    or positions.ndim != 3
                    or positions.shape[1:] != (2, 3)
                    or orientations.shape != (len(positions), 2, 4)
                    or np.any(stance_ids < 0)
                    or np.any(stance_ids >= len(positions))
                    or not np.isfinite(positions).all()
                    or not np.isfinite(orientations).all()
                    or not np.allclose(np.linalg.norm(orientations, axis=-1), 1.0, atol=1e-4)
                ):
                    raise ValueError("Invalid static stance IDs or foot anchors")
                feet = [stored_bodies.index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
                foot_error = np.linalg.norm(data["body_pos_w"][:, feet] - positions[stance_ids], axis=-1)
                dot = np.abs(np.sum(data["body_quat_w"][:, feet] * orientations[stance_ids], axis=-1))
                angle = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
                if np.any(foot_error > 0.0011) or np.any(angle > 0.0051):
                    raise ValueError("Poses in a stance group must share fixed foot anchors")
                self.stance_ids = torch.tensor(stance_ids, dtype=torch.long, device=device)
                self.stance_foot_pos_w = torch.tensor(positions, dtype=torch.float32, device=device)
                self.stance_foot_quat_w = torch.tensor(orientations, dtype=torch.float32, device=device)
