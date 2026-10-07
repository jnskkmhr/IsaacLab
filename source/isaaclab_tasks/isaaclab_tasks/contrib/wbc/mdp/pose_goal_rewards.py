# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Terminal-pose reaching and measured-state step-quality rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pelvis_distance(env: ManagerBasedRLEnv, distance_scale: float = 1.0) -> torch.Tensor:
    """Broad horizontal goal proximity, retaining signal outside fine tracking tolerance."""
    term = env.command_manager.get_term("pose_goal")
    return 1.0 / (1.0 + term.pelvis_position_error_w[:, :2].norm(dim=-1) / distance_scale)


def pelvis_position(env: ManagerBasedRLEnv, standard_deviation: float = 0.1) -> torch.Tensor:
    """Fine horizontal pelvis position accuracy."""
    error = env.command_manager.get_term("pose_goal").pelvis_position_error_w[:, :2]
    return torch.exp(-error.square().sum(dim=-1) / standard_deviation**2)


def pelvis_height(env: ManagerBasedRLEnv, standard_deviation: float = 0.08) -> torch.Tensor:
    """Final pelvis height, with reduced priority during navigation."""
    term = env.command_manager.get_term("pose_goal")
    return torch.exp(-term.pelvis_position_error_w[:, 2].square() / standard_deviation**2) * (
        1 - 0.7 * term.navigation_gate
    )


def pelvis_yaw(env: ManagerBasedRLEnv, standard_deviation: float = 0.5) -> torch.Tensor:
    """Wrapped world heading accuracy, independent of torso-relative twist."""
    return torch.exp(-env.command_manager.get_term("pose_goal").pelvis_yaw_error.square() / standard_deviation**2)


def torso_orientation(env: ManagerBasedRLEnv, standard_deviation: float = 0.3) -> torch.Tensor:
    """Relative torso rotation accuracy, permitting transit posture deviations."""
    term = env.command_manager.get_term("pose_goal")
    error = quat_error_magnitude(term.torso_quat_p, term.goal_torso_quat_p)
    return torch.exp(-error.square() / standard_deviation**2) * (1 - 0.5 * term.navigation_gate)


def hand_position(env: ManagerBasedRLEnv, standard_deviation: float = 0.08) -> torch.Tensor:
    """Both hands' torso-relative position accuracy, equally weighted."""
    term = env.command_manager.get_term("pose_goal")
    error = (term.hand_pos_t - term.goal_hand_pos_t).square().sum(dim=-1)
    return torch.exp(-error / standard_deviation**2).mean(dim=-1)


def hand_orientation(env: ManagerBasedRLEnv, standard_deviation: float = 0.4) -> torch.Tensor:
    """Both hands' full torso-relative rotation accuracy."""
    term = env.command_manager.get_term("pose_goal")
    error = quat_error_magnitude(term.hand_quat_t, term.goal_hand_quat_t)
    return torch.exp(-error.square() / standard_deviation**2).mean(dim=-1)


def settled_hold(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward current settling and sustained holding, without ending the episode."""
    term = env.command_manager.get_term("pose_goal")
    return term.settled.float() * (1 + (term.hold_time / term.cfg.hold_duration).clamp(max=1))


def arrival_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize pelvis motion near arrival; do not discourage transit velocity."""
    term = env.command_manager.get_term("pose_goal")
    linear = term.robot.data.body_lin_vel_w.torch[:, term.pelvis_id]
    angular = term.robot.data.body_ang_vel_w.torch[:, term.pelvis_id]
    return (1 - term.navigation_gate) * (linear.square().sum(dim=-1) + 0.2 * angular.square().sum(dim=-1))


def pelvis_tilt(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Soft pelvis stability; no world-upright constraint on the torso."""
    term = env.command_manager.get_term("pose_goal")
    gravity = torch.zeros(env.num_envs, 3, device=env.device)
    gravity[:, 2] = -1
    return quat_apply_inverse(term.pelvis_quat_w, gravity)[:, :2].square().sum(dim=-1)


def foot_clearance(
    env: ManagerBasedRLEnv, clearance_height: float = 0.05, standard_deviation: float = 0.03, speed_scale: float = 0.5
) -> torch.Tensor:
    """Bounded sole-clearance reward for moving airborne feet while another foot supports.

    All heights/speeds are measured, not trajectory references. Navigation includes
    in-place yaw turns; this reward is zero inside position and heading tolerances.
    """
    if min(clearance_height, standard_deviation, speed_scale) <= 0:
        raise ValueError("Clearance height, deviation, and speed scale must be positive")
    term = env.command_manager.get_term("pose_goal")
    contacts = term.foot_contacts
    velocity = term.robot.data.body_lin_vel_w.torch[:, term.foot_ids, :2].norm(dim=-1)
    speed_gate = (velocity / speed_scale).clamp(max=1)
    height_score = torch.exp(-(term.sole_clearance - clearance_height).square() / standard_deviation**2)
    return term.navigation_gate * contacts.any(dim=-1) * ((~contacts) * speed_gate * height_score).mean(dim=-1)


def foot_sliding(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize tangential motion only for feet with measured contact."""
    term = env.command_manager.get_term("pose_goal")
    speed = term.robot.data.body_lin_vel_w.torch[:, term.foot_ids, :2].square().sum(dim=-1)
    return (speed * term.foot_contacts).sum(dim=-1)


def foot_scuffing(env: ManagerBasedRLEnv, minimum_clearance: float = 0.015) -> torch.Tensor:
    """Penalize moving feet too close to the ground, with a soft bounded deficit."""
    term = env.command_manager.get_term("pose_goal")
    speed = term.robot.data.body_lin_vel_w.torch[:, term.foot_ids, :2].norm(dim=-1).clamp(max=1)
    deficit = ((minimum_clearance - term.sole_clearance) / minimum_clearance).clamp(0, 1)
    return term.navigation_gate * (speed * deficit).mean(dim=-1)


def no_support(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize simultaneous measured loss of both foot contacts, permitting single support."""
    return (~env.command_manager.get_term("pose_goal").foot_contacts.any(dim=-1)).float()


def excessive_foot_force(env: ManagerBasedRLEnv, threshold: float = 400.0) -> torch.Tensor:
    """Penalize current foot force above threshold [N], including landing impacts."""
    forces = env.command_manager.get_term("pose_goal").foot_forces_w.norm(dim=-1)
    return ((forces - threshold).clamp(min=0) / threshold).square().sum(dim=-1)
