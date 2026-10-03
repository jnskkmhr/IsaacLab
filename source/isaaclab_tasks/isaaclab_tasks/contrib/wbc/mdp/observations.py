# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""WBC target observations expressed in the measured pelvis heading frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_conjugate, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_body_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Target body displacements from the pelvis, in its yaw-aligned frame, in metres."""
    reference = env.command_manager.get_term("whole_body")
    robot = reference.robot
    heading = yaw_quat(robot.data.root_quat_w.torch)[:, None, :].expand(-1, len(reference.body_ids), -1)
    return quat_apply_inverse(heading, reference.target_body_pos_w - robot.data.root_pos_w.torch[:, None]).flatten(1)


def target_body_orientations(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Target rotations in the pelvis heading frame as row-flattened first two matrix columns."""
    reference = env.command_manager.get_term("whole_body")
    heading = yaw_quat(reference.robot.data.root_quat_w.torch)[:, None, :].expand(-1, len(reference.body_ids), -1)
    orientation = quat_mul(quat_conjugate(heading), reference.target_body_quat_w)
    return matrix_from_quat(orientation)[..., :2].flatten(1)


def target_joint_velocities(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reference joint velocities in the command's configured joint order, in radians per second."""
    reference = env.command_manager.get_term("whole_body")
    return reference.dataset.joint_vel[reference.frame]


def target_foot_contacts(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reference contact flags ordered as left foot, right foot."""
    reference = env.command_manager.get_term("whole_body")
    return reference.dataset.foot_contact[reference.frame]
