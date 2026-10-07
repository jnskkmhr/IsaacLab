# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Current relative link state and fixed goal observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pose_goals(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the command's 29-component goal encoding."""
    return env.command_manager.get_term("pose_goal").command


def torso_orientation_p(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Measured torso rotation relative to full pelvis rotation, in 6D."""
    term = env.command_manager.get_term("pose_goal")
    return matrix_from_quat(term.torso_quat_p)[..., :2].flatten(1)


def hand_positions_t(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Measured left/right wrist positions in the full torso frame [m]."""
    return env.command_manager.get_term("pose_goal").hand_pos_t.flatten(1)


def hand_orientations_t(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Measured left/right wrist rotations in the full torso frame, in 6D."""
    return matrix_from_quat(env.command_manager.get_term("pose_goal").hand_quat_t)[..., :2].flatten(1)


def foot_contacts(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Measured left/right contact flags; deploy with contact sensing or estimation."""
    return env.command_manager.get_term("pose_goal").foot_contacts.float()


def foot_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Measured left/right minimum sole height above the flat plane [m]."""
    return env.command_manager.get_term("pose_goal").sole_clearance


def foot_forces(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged foot normal forces in pelvis-heading coordinates [N]."""
    term = env.command_manager.get_term("pose_goal")
    heading = yaw_quat(term.pelvis_quat_w)[:, None].expand(-1, 2, -1)
    return quat_apply_inverse(heading, term.foot_forces_w).flatten(1)
