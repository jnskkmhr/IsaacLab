# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Use coupled sand/rigid foot reactions rather than rigid-only contact sensors."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.contrib.push.config.g1_29dof_object_push.mdp.commands import standing_command_mask
from isaaclab_tasks.contrib.push.mdp.rewards import box_velocity_tracking


def feet_slide(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sum lateral foot speeds [m/s] while feet contact the fixed bed."""
    robot = env.scene[asset_cfg.name]
    velocity = robot.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return (velocity.norm(dim=-1) * env.foot_contact).sum(dim=-1)


def stance_foot_angle_penalty(env, asset_cfg: SceneEntityCfg, angle_tolerance: float) -> torch.Tensor:
    """Penalize foot pitch relative to the initial horizontal sand surface [rad squared]."""
    robot = env.scene[asset_cfg.name]
    quaternion = robot.data.body_quat_w.torch[:, asset_cfg.body_ids]
    forward = torch.zeros_like(quaternion[..., :3])
    forward[..., 0] = 1.0
    forward = quat_apply(quaternion, forward)
    pitch = torch.atan2(forward[..., 2], forward[..., :2].norm(dim=-1))
    return ((pitch.abs() - angle_tolerance).clamp_min(0).square() * env.foot_contact).sum(dim=-1)


def _object_contact_magnitudes(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return per-body contact force magnitudes [N], filtered against the box only."""
    return env.object_contact_force[:, asset_cfg.body_ids].norm(dim=-1)


def hand_object_contact_fraction(
    env,
    asset_cfg: SceneEntityCfg | tuple[SceneEntityCfg, ...],
    force_threshold: float = 1.0,
    command_name: str | None = None,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward the fraction of selected hands touching the box, bounded between zero and one.

    A tuple selects one group of palm/finger bodies per hand; any body touching
    counts once for that hand. A single selection treats each body as one hand.
    Contact requires box-filtered force magnitude above ``force_threshold`` [N].
    Increasing force beyond the threshold earns no additional reward. Knee,
    ground, and self contacts do not count when only hand bodies are selected.
    When a command name is provided, stop commands receive no contact credit.
    """
    if isinstance(asset_cfg, tuple):
        # Each selection represents one hand, including its palm and finger links.
        contacts = [(_object_contact_magnitudes(env, group) > force_threshold).any(dim=-1) for group in asset_cfg]
        fraction = torch.stack(contacts, dim=-1).float().mean(dim=-1)
    else:
        fraction = (_object_contact_magnitudes(env, asset_cfg) > force_threshold).float().mean(dim=-1)
    if command_name is not None:
        fraction *= ~standing_command_mask(env.command_manager.get_command(command_name), command_threshold)
    return fraction


def object_velocity_tracking_with_hand_contact(
    env,
    std: float,
    hand_distance: float,
    command_name: str,
    command_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg | tuple[SceneEntityCfg, ...],
    force_threshold: float = 1.0,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Weight box linear-velocity tracking and hand proximity by actual hand contact fraction.

    With two selected wrist bodies, neither/one/both touching the box earns
    zero/half/full tracking credit. Ground and self contacts are excluded by box-only contact filtering.
    Stop commands receive no tracking credit, so approaching a stationary box cannot help.
    """
    tracking = box_velocity_tracking(
        env,
        std,
        hand_distance,
        command_name,
        command_names,
        object_name=env.command_manager.get_term(command_name).cfg.object_name,
    )
    return tracking * hand_object_contact_fraction(env, asset_cfg, force_threshold, command_name, command_threshold)


def object_yaw_velocity_tracking(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg | tuple[SceneEntityCfg, ...],
    force_threshold: float = 1.0,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Track box world-z angular velocity during active commands with hand contact."""
    error = (
        env.scene[env.command_manager.get_term(command_name).cfg.object_name].data.root_ang_vel_w.torch[:, 2]
        - env.command_manager.get_command(command_name)[:, 2]
    )
    return torch.exp(-error.square() / std**2) * hand_object_contact_fraction(
        env, asset_cfg, force_threshold, command_name, command_threshold
    )
