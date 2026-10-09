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
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply

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
    air_time = env.foot_air_time
    contact_time = env.foot_contact_time
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
    airborne = torch.sum(env.foot_contact, dim=-1) < 0.5
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
    return torch.sum(torch.square(pitch) * env.foot_first_contact, dim=-1)


def reward_soft_landing(
    env: G1MPMEnv,
    command_name: str | None = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize foot force magnitude at touchdown using the combined terrain contact forces.

    The cost is in newtons, not impulse. Persistent contacts incur no cost.
    A zero motion command disables the penalty, matching the soft-contact task.
    """
    first_contact = env.foot_first_contact
    landing_force = torch.linalg.vector_norm(env.foot_contact_force, dim=-1) * first_contact
    cost = landing_force.sum(dim=-1)
    env.extras["log"]["Metrics/landing_force_mean"] = landing_force.sum() / first_contact.sum().clamp(min=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        active = torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs() > command_threshold
        cost = cost * active
    return cost


class foot_touch_down_angle_penalty(ManagerTermBase):
    """Penalize foot pitch at first contact relative to the flat initial sand/approach surface."""

    def __init__(self, cfg: RewardTermCfg, env: G1MPMEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        tracked_ids, _ = self._asset.find_bodies(env.cfg.foot_body_expr, preserve_order=True)
        selected_ids = cfg.params["asset_cfg"].body_ids
        if isinstance(selected_ids, slice):
            selected_ids = list(range(self._asset.num_bodies))[selected_ids]
        self._contact_ids = torch.tensor([tracked_ids.index(i) for i in selected_ids], device=env.device)
        self._forward = torch.tensor([1.0, 0.0, 0.0], device=env.device)
        if cfg.params["angle_tolerance"] < 0.0:
            raise ValueError("angle_tolerance must be non-negative.")

    def __call__(self, env: G1MPMEnv, asset_cfg: SceneEntityCfg, angle_tolerance: float) -> torch.Tensor:
        foot_quat = self._asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
        forward = quat_apply(foot_quat, self._forward.expand_as(foot_quat[..., :3]))
        angle = torch.atan2(forward[..., 2], torch.linalg.vector_norm(forward[..., :2], dim=-1))
        touchdown = env.foot_first_contact[:, self._contact_ids] > 0.0
        penalty = (angle.abs() - angle_tolerance).clamp_min(0.0).square()
        return torch.where(torch.isfinite(angle) & touchdown, penalty, 0.0).sum(dim=-1)
