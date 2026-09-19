# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic replacements for the contact-sensor terms of the rigid G1 task.

Coupled Newton solvers keep contact forces in per-entry buffers, which Newton contact sensors do
not support (``isaaclab_contrib.coupling.coupler``), so this task carries no contact sensor. Every
term here reconstructs what the rigid task reads off the sensor from the articulation state
instead: a foot is in contact when its sole is at ground level. That equivalence only holds on flat
ground at ``z = 0``, which is the terrain this task runs on.

Foot *forces* have no kinematic counterpart. The one term that reported them keeps its layout and
returns zeros, and the landing penalty is expressed through the touchdown speed instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

FOOT_SOLE_OFFSET = (0.0, 0.0, -0.03539)
"""Sole of the ankle roll link in the link frame [m]. Same value the rigid task reports heights against."""

FOOT_CONTACT_HEIGHT = 0.01
"""Sole height below which a foot counts as being in contact with the flat ground [m]."""


@dataclass
class FootContactState:
    """Per-foot contact bookkeeping derived from the sole height."""

    contact: torch.Tensor
    """Contact flags, shape ``(num_envs, num_feet)``."""

    first_contact: torch.Tensor
    """Flags of feet that touched down on this step, shape ``(num_envs, num_feet)``."""

    air_time: torch.Tensor
    """Time since each foot last left the ground [s], shape ``(num_envs, num_feet)``."""

    contact_time: torch.Tensor
    """Time since each foot last touched down [s], shape ``(num_envs, num_feet)``."""

    touchdown_speed: torch.Tensor
    """Downward sole speed at touchdown [m/s], zero away from a touchdown, shape ``(num_envs, num_feet)``."""

    step: int
    """Step counter the state was last refreshed at."""


def foot_contact_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    height_threshold: float = FOOT_CONTACT_HEIGHT,
) -> FootContactState:
    """Return the contact state of the selected feet, refreshing it once per environment step.

    Args:
        env: Environment instance.
        asset_cfg: Articulation and bodies to treat as feet.
        height_threshold: Sole height below which a foot is in contact [m].

    Returns:
        The contact state of the selected feet.
    """
    cache: dict = env.__dict__.setdefault("_tree_foot_contact_states", {})
    key = (asset_cfg.name, tuple(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset_cfg.body_ids)

    asset: Articulation = env.scene[asset_cfg.name]
    pose = asset.data.body_pose_w.torch[:, asset_cfg.body_ids, :7]
    offset = torch.tensor(FOOT_SOLE_OFFSET, device=env.device).expand(pose.shape[:-1] + (3,))
    sole_height = pose[..., 2] + math_utils.quat_apply(pose[..., 3:7], offset)[..., 2]
    contact = sole_height < height_threshold

    state = cache.get(key)
    if state is None:
        zeros = torch.zeros_like(sole_height)
        state = FootContactState(
            contact=torch.zeros_like(contact),
            first_contact=torch.zeros_like(contact),
            air_time=zeros.clone(),
            contact_time=zeros.clone(),
            touchdown_speed=zeros.clone(),
            step=-1,
        )
        cache[key] = state
    elif state.step == env.common_step_counter:
        return state

    first_contact = contact & ~state.contact
    air_time = torch.where(contact, torch.zeros_like(state.air_time), state.air_time + env.step_dt)
    contact_time = torch.where(contact, state.contact_time + env.step_dt, torch.zeros_like(state.contact_time))
    # The sole velocity is the link velocity plus the rotational term; the offset is small enough
    # that the link velocity alone is the touchdown speed to well within the contact threshold.
    sole_speed = -asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, 2]

    # A freshly reset environment has no history: start it from the pose it was reset into.
    reset = (env.episode_length_buf == 0).unsqueeze(-1)
    state.contact = contact
    state.first_contact = first_contact & ~reset
    state.air_time = torch.where(reset, torch.zeros_like(air_time), air_time)
    state.contact_time = torch.where(reset, torch.zeros_like(contact_time), contact_time)
    state.touchdown_speed = torch.where(state.first_contact, sole_speed.clamp(min=0.0), torch.zeros_like(sole_speed))
    state.step = env.common_step_counter
    return state


"""
Observations.
"""


def foot_contact(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Contact flag of each foot.

    Args:
        env: Environment instance.
        asset_cfg: Articulation and bodies to treat as feet.

    Returns:
        The flags as ``float``, shape ``(num_envs, num_feet)``.
    """
    return foot_contact_state(env, asset_cfg).contact.float()


def foot_contact_forces(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Placeholder for the foot contact forces of the rigid task.

    The coupled solver keeps its contact forces per entry and no sensor can read them, so the term
    is kept at the rigid-task shape to preserve the critic input layout and filled with zeros.

    Args:
        env: Environment instance.
        asset_cfg: Articulation and bodies to treat as feet.

    Returns:
        Zeros, shape ``(num_envs, 3 * num_feet)``.
    """
    num_feet = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else env.scene[asset_cfg.name].num_bodies
    return torch.zeros(env.num_envs, 3 * num_feet, device=env.device)


def foot_air_time(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    filter_time: float = 0.5,
) -> torch.Tensor:
    """Time each foot has spent off the ground, saturated so that standing stays bounded.

    Args:
        env: Environment instance.
        asset_cfg: Articulation and bodies to treat as feet.
        filter_time: Air time above which the value is reported as zero [s].

    Returns:
        The air times [s], shape ``(num_envs, num_feet)``.
    """
    air_time = foot_contact_state(env, asset_cfg).air_time
    return torch.where(air_time > filter_time, 0.0, air_time)


"""
Rewards.
"""


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward long steps while a single foot is on the ground.

    Kinematic counterpart of ``g1_29dof_rigid.mdp.feet_air_time_positive_biped``.

    Args:
        env: Environment instance.
        command_name: Name of the velocity command term.
        threshold: Air or contact time the reward saturates at [s].
        asset_cfg: Articulation and bodies to treat as feet.
        velocity_threshold: Command norm below which the reward is zero [m/s or rad/s].

    Returns:
        The reward, shape ``(num_envs,)``.
    """
    state = foot_contact_state(env, asset_cfg)
    in_mode_time = torch.where(state.contact, state.contact_time, state.air_time)
    single_stance = torch.sum(state.contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)

    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return reward * (command_norm > velocity_threshold)


def feet_slide(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the horizontal speed of the feet that are on the ground.

    Args:
        env: Environment instance.
        asset_cfg: Articulation and bodies to treat as feet.

    Returns:
        The penalty [m/s], shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact = foot_contact_state(env, asset_cfg).contact
    body_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contact.float(), dim=1)


def feet_pitch_contact(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize a pitched foot at touchdown, so that the robot lands flat.

    Args:
        env: Environment instance.
        asset_cfg: Articulation and bodies to treat as feet.

    Returns:
        The penalty [rad^2], shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    first_contact = foot_contact_state(env, asset_cfg).first_contact
    quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids, :]
    _, pitch, _ = math_utils.euler_xyz_from_quat(quat.reshape(-1, 4))
    pitch = ((pitch + torch.pi) % (2 * torch.pi) - torch.pi).reshape(quat.shape[0], -1)
    return torch.sum(torch.square(pitch) * first_contact.float(), dim=-1)


def soft_landing(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize hard footfalls through the touchdown speed.

    The rigid task penalizes the contact force at touchdown, which the coupled solver does not
    report. Touchdown speed is its kinematic cause and carries the same ordering of footfalls, so
    the term keeps its role but is expressed in [m/s] and needs its own weight.

    Args:
        env: Environment instance.
        command_name: Name of the velocity command term.
        asset_cfg: Articulation and bodies to treat as feet.
        command_threshold: Command norm below which the penalty is zero [m/s or rad/s].

    Returns:
        The penalty [m/s], shape ``(num_envs,)``.
    """
    cost = torch.sum(foot_contact_state(env, asset_cfg).touchdown_speed, dim=1)

    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return cost * (command_norm > command_threshold)


def undesired_ground_proximity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize links other than the feet that come down to the ground.

    Kinematic counterpart of ``isaaclab.envs.mdp.undesired_contacts``: on flat ground a link can
    only touch the terrain by reaching it, so height stands in for the missing contact force. Contacts
    with the logs are not covered, which is deliberate: the robot is meant to push them aside.

    Args:
        env: Environment instance.
        asset_cfg: Articulation and bodies that should stay clear of the ground.
        height_threshold: Link height below which the link is counted [m].

    Returns:
        The number of offending links, shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    return torch.sum((height < height_threshold).float(), dim=1)
