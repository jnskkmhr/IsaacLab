# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sagittal reflection for the WBC observation contract and named G1 action order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from ..config.g1_29dof.robot_constants import JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

JOINT_PARTNERS = [
    JOINT_NAMES.index(
        name.replace("left_", "right_", 1) if name.startswith("left_") else name.replace("right_", "left_", 1)
    )
    for name in JOINT_NAMES
]
JOINT_SIGNS = [-1 if name.endswith(("_roll_joint", "_yaw_joint")) else 1 for name in JOINT_NAMES]


def mirror_joints(values: torch.Tensor) -> torch.Tensor:
    """Exchange left/right joints and reflect rotations about X and Z."""
    return values[..., JOINT_PARTNERS] * values.new_tensor(JOINT_SIGNS)


def mirror_observation(values: torch.Tensor, group: str) -> torch.Tensor:
    """Reflect a complete student, teacher, or critic observation in its declared ordering."""
    expected = {"student": 138, "teacher": 167, "critic": 201}[group]
    if values.shape[-1] != expected:
        raise ValueError(f"{group} expects {expected} observation values, got {values.shape[-1]}")
    result = values.clone()
    result[..., :3] *= values.new_tensor([-1, 1, -1])
    result[..., 3:6] *= values.new_tensor([1, -1, 1])
    for start in (6, 35, 64):
        result[..., start : start + 29] = mirror_joints(values[..., start : start + 29])
    poses = values[..., 93:138].reshape(*values.shape[:-1], 5, 9)[..., [0, 2, 1, 4, 3], :].clone()
    poses *= values.new_tensor([1, -1, 1, 1, -1, -1, 1, 1, -1])
    result[..., 93:138] = poses.flatten(-2)
    if group != "student":
        result[..., 138:167] = mirror_joints(values[..., 138:167])
    if group == "critic":
        result[..., 167:170] *= values.new_tensor([1, -1, 1])
        result[..., 170:199] = mirror_joints(values[..., 170:199])
        result[..., 199:201] = values[..., [200, 199]]
    return result


def augment_symmetry(
    env: ManagerBasedRLEnv, obs: TensorDict | None = None, actions: torch.Tensor | None = None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append reflected samples for RSL-RL PPO; dataset mirroring also covers distillation."""
    augmented = None
    if obs is not None:
        reflected = TensorDict(
            {key: mirror_observation(value, key) for key, value in obs.items()}, batch_size=obs.batch_size
        )
        augmented = torch.cat((obs, reflected), dim=0)
    return augmented, None if actions is None else torch.cat((actions, mirror_joints(actions)), dim=0)
