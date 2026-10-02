# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_body_position(env: ManagerBasedRLEnv, standard_deviation: float) -> torch.Tensor:
    """Reward world-frame tracking of pelvis, ankles, and wrists."""
    reference = env.command_manager.get_term("whole_body")
    difference = reference.robot.data.body_pos_w.torch[:, reference.body_ids] - reference.target_body_pos_w
    return torch.exp(-difference.square().sum(-1) / standard_deviation**2).mean(-1)


def track_body_orientation(env: ManagerBasedRLEnv, standard_deviation: float) -> torch.Tensor:
    """Reward orientation agreement using a sign-invariant quaternion angle."""
    reference = env.command_manager.get_term("whole_body")
    current = reference.robot.data.body_quat_w.torch[:, reference.body_ids]
    cosine = (current * reference.target_body_quat_w).sum(-1).abs().clamp(max=1.0)
    angle = 2 * torch.acos(cosine)
    return torch.exp(-angle.square() / standard_deviation**2).mean(-1)


def track_joint_position(env: ManagerBasedRLEnv, standard_deviation: float) -> torch.Tensor:
    reference = env.command_manager.get_term("whole_body")
    difference = reference.robot.data.joint_pos.torch[:, reference.joint_ids] - reference.command
    return torch.exp(-difference.square().mean(-1) / standard_deviation**2)


def action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (env.action_manager.action - env.action_manager.prev_action).square().sum(-1)


def joint_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["robot"].data.joint_vel.torch.square().sum(-1)


def stance_foot_sliding(env: ManagerBasedRLEnv) -> torch.Tensor:
    reference = env.command_manager.get_term("whole_body")
    velocity = reference.robot.data.body_lin_vel_w.torch[:, reference.body_ids[1:3], :2]
    return (velocity.square().sum(-1) * reference.dataset.foot_contact[reference.frame]).sum(-1)
