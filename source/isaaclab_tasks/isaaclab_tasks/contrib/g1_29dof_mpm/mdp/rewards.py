# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gait rewards driven by the foot contact against the granular bed.

Contact comes from the reaction wrench that the coupler feeds back from the MPM entry to the
proxy feet, so these are the granular counterparts of the hybrid soft-contact rewards of
``g1_29dof_soft``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from ..g1_mpm_env import G1MPMEnv


def feet_air_time_positive_biped(
    env: G1MPMEnv,
    command_name: str,
    threshold: float,
    velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward long single-stance phases while a motion command is active.

    Args:
        env: Environment instance.
        command_name: Name of the velocity command term.
        threshold: Upper bound on the rewarded stance or swing duration [s].
        velocity_threshold: Command magnitude above which the reward applies [m/s or rad/s].

    Returns:
        The reward, shape ``(num_envs,)``.
    """
    air_time = env.foot_air_time()
    contact_time = env.foot_contact_time()
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no gait is asked of a standing robot
    command = env.command_manager.get_command(command_name)
    commanded = (torch.norm(command[:, :2], dim=1) > velocity_threshold) | (
        torch.abs(command[:, 2]) > velocity_threshold
    )
    return reward * commanded


def no_fly(
    env: G1MPMEnv,
    command_name: str,
    velocity_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize flight phases below a command speed at which running is acceptable.

    Args:
        env: Environment instance.
        command_name: Name of the velocity command term.
        velocity_threshold: Command speed above which flight is no longer penalized [m/s].

    Returns:
        The penalty, shape ``(num_envs,)``.
    """
    airborne = torch.sum(env.foot_contact(), dim=-1) < 0.5
    command_speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return airborne.float() * (command_speed < velocity_threshold).float()


def feet_pitch_contact(
    env: G1MPMEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot pitch at touchdown only, to encourage flat landings.

    Args:
        env: Environment instance.
        asset_cfg: Configuration selecting the foot bodies, in the environment's foot order.

    Returns:
        The penalty, shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids, :]
    _, pitch, _ = euler_xyz_from_quat(body_quat.reshape(-1, 4))
    pitch = pitch.reshape(body_quat.shape[0], body_quat.shape[1])
    return torch.sum(torch.square(pitch) * env.foot_first_contact(), dim=-1)
