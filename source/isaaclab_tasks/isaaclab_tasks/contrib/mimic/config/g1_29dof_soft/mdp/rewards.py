# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grounded balance and landing-settling rewards for soft-contact jumps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.mdp.observations import foot_contact_hybrid

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def grounded_com_foot_center_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    foot_center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tolerance: float = 0.03,
) -> torch.Tensor:
    """Penalize horizontal whole-body CoM displacement from grounded foot centers.

    The support center is the mean of the contacting foot centers, not a force-weighted
    center of pressure. This term is active during grounded takeoff preparation and
    landing recovery, and zero when all feet are airborne.

    Args:
        env: Environment to evaluate.
        sensor_cfg: Contact feet in soft-solver order.
        asset_cfg: Robot with foot bodies in the same order as the contact feet.
        foot_center_offset: Foot-center offset in each selected link frame [m].
        tolerance: Unpenalized horizontal distance from the support center [m].

    Returns:
        Squared excess horizontal distance [m^2], shape [N].
    """
    if tolerance < 0.0:
        raise ValueError("CoM tolerance must be nonnegative.")
    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.data.body_mass.torch
    com = (masses.unsqueeze(-1) * asset.data.body_com_pos_w.torch).sum(dim=1) / masses.sum(dim=1, keepdim=True)
    positions = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids]
    orientations = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    offset = positions.new_tensor(foot_center_offset).expand_as(positions)
    centers = positions + quat_apply(orientations, offset)
    contacts = foot_contact_hybrid(env, rigid_contact_sensor_cfg=sensor_cfg).bool()
    count = contacts.sum(dim=-1)
    center = (centers * contacts.unsqueeze(-1)).sum(dim=1) / count.clamp(min=1).unsqueeze(-1)
    distance = torch.linalg.vector_norm(com[:, :2] - center[:, :2], dim=-1)
    return (distance - tolerance).clamp(min=0.0).square() * (count > 0)


def grounded_torso_tilt_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "motion",
    tolerance: float = 0.17453292519943295,
) -> torch.Tensor:
    """Penalize torso tilt from upright beyond a tolerance while grounded.

    This uses the motion command's torso anchor and ignores yaw. Unlike reference
    orientation tracking, forward lean in the reference does not shift the target.

    Args:
        env: Environment to evaluate.
        sensor_cfg: Contact feet in soft-solver order.
        command_name: Motion command whose anchor is the torso.
        tolerance: Unpenalized tilt from world vertical [rad], default 10 degrees.

    Returns:
        Squared excess tilt [rad^2], shape [N].
    """
    if not 0.0 <= tolerance <= torch.pi:
        raise ValueError("Tilt tolerance must be between zero and pi.")
    command = env.command_manager.get_term(command_name)
    quat = command.robot_anchor_quat_w
    up = torch.zeros_like(quat[:, :3])
    up[:, 2] = 1.0
    up_b = quat_apply_inverse(quat, up)
    tilt = torch.atan2(torch.linalg.vector_norm(up_b[:, :2], dim=-1), up_b[:, 2])
    grounded = foot_contact_hybrid(env, rigid_contact_sensor_cfg=sensor_cfg).bool().any(dim=-1)
    return (tilt - tolerance).clamp(min=0.0).square() * grounded


def post_jump_torso_ang_vel_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "motion",
    tolerance: float = 0.3,
) -> torch.Tensor:
    """Penalize excessive torso angular speed during grounded final reference stance.

    The final stance must end at phase one and start after phase zero. Its existing
    stance weight scales the penalty. Earlier stance and airborne motion are excluded.

    Args:
        env: Environment to evaluate.
        sensor_cfg: Contact feet in soft-solver order.
        command_name: Motion command defining the torso and final stance interval.
        tolerance: Unpenalized torso angular speed [rad/s].

    Returns:
        Weighted squared excess angular speed [(rad/s)^2], shape [N].
    """
    if tolerance < 0.0:
        raise ValueError("Angular speed tolerance must be nonnegative.")
    command = env.command_manager.get_term(command_name)
    speed = torch.linalg.vector_norm(command.robot_anchor_ang_vel_w, dim=-1)
    final_starts = [start for start, end in command.cfg.stance_phase_ranges if start > 0.0 and end == 1.0]
    if not final_starts:
        return torch.zeros_like(speed)
    grounded = foot_contact_hybrid(env, rigid_contact_sensor_cfg=sensor_cfg).bool().any(dim=-1)
    weight = command.standing_weight * (command.phase >= max(final_starts)) * grounded
    return (speed - tolerance).clamp(min=0.0).square() * weight
