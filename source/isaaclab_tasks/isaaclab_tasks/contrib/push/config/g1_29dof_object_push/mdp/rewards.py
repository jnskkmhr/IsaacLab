# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Box velocity tracking and stationary leg posture rewards."""

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from isaaclab_tasks.contrib.push.mdp.rewards import box_velocity_tracking, hand_position_tracking

from .commands import standing_command_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _box_contact_magnitudes(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return per-body contact force magnitudes [N], filtered against the box only."""
    matrix = env.scene[sensor_cfg.name].data.force_matrix_w
    if matrix is None:
        raise ValueError("Box-contact rewards require a contact sensor filtered against the box.")
    forces = matrix.torch[:, sensor_cfg.body_ids]
    return torch.linalg.vector_norm(forces, dim=-1).sum(dim=-1)


def hand_box_contact_fraction(
    env,
    sensor_cfg: SceneEntityCfg | tuple[SceneEntityCfg, ...],
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
    if isinstance(sensor_cfg, tuple):
        # Each selection represents one hand, including its palm and finger links.
        contacts = [(_box_contact_magnitudes(env, group) > force_threshold).any(dim=-1) for group in sensor_cfg]
        fraction = torch.stack(contacts, dim=-1).float().mean(dim=-1)
    else:
        fraction = (_box_contact_magnitudes(env, sensor_cfg) > force_threshold).float().mean(dim=-1)
    if command_name is not None:
        fraction *= ~standing_command_mask(env.command_manager.get_command(command_name), command_threshold)
    return fraction


def box_velocity_tracking_with_hand_contact(
    env,
    std: float,
    hand_distance: float,
    command_name: str,
    command_names: tuple[str, ...],
    sensor_cfg: SceneEntityCfg | tuple[SceneEntityCfg, ...],
    force_threshold: float = 1.0,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Weight box linear-velocity tracking and hand proximity by actual hand contact fraction.

    With two selected wrist bodies, neither/one/both touching the box earns
    zero/half/full tracking credit. Ground and self contacts are excluded by the sensor filter.
    Stop commands receive no tracking credit, so approaching a stationary box cannot help.
    """
    tracking = box_velocity_tracking(env, std, hand_distance, command_name, command_names)
    return tracking * hand_box_contact_fraction(env, sensor_cfg, force_threshold, command_name, command_threshold)


def box_yaw_velocity_tracking(
    env,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg | tuple[SceneEntityCfg, ...],
    force_threshold: float = 1.0,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Track box world-z angular velocity during active commands with hand contact."""
    error = env.scene["box"].data.root_ang_vel_w.torch[:, 2] - env.command_manager.get_command(command_name)[:, 2]
    return torch.exp(-error.square() / std**2) * hand_box_contact_fraction(
        env, sensor_cfg, force_threshold, command_name, command_threshold
    )


def non_hand_box_contact_force_l1(env, sensor_cfg: SceneEntityCfg, force_threshold: float = 1.0) -> torch.Tensor:
    """Sum selected non-hand bodies' box contact force magnitudes above the threshold [N].

    Uses filtered total forces (normal plus friction), never net forces that also
    include the ground. Allowed hand bodies must be excluded in ``sensor_cfg``.
    """
    return (_box_contact_magnitudes(env, sensor_cfg) - force_threshold).clamp(min=0.0).sum(dim=-1)


def leg_joint_deviation_l1(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize leg deviations from the default pose at near-zero commanded box velocity.

    Both planar speed and absolute yaw rate must be below ``command_threshold``.
    Root motion reduces the cost by ``exp(-(linear_speed + angular_speed)**2)``
    so recovery steps remain possible. Speeds use root body-frame velocities;
    selected joint deviations are measured in radians.
    """
    command = env.command_manager.get_command(command_name)
    standing = (torch.linalg.vector_norm(command[:, :2], dim=-1) < command_threshold) & (
        command[:, 2].abs() < command_threshold
    )
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    speed = torch.linalg.vector_norm(asset.data.root_lin_vel_b.torch, dim=-1)
    speed = speed + torch.linalg.vector_norm(asset.data.root_ang_vel_b.torch, dim=-1)
    return deviation.abs().sum(dim=-1) * standing * torch.exp(-speed.square())


def hand_position_tracking_when_moving(
    env, command_name: str, command_names: tuple[str, ...], std: float, command_threshold: float = 0.05
) -> torch.Tensor:
    """Reward reaching only when a nonzero box velocity is requested."""
    moving = ~standing_command_mask(env.command_manager.get_command(command_name), command_threshold)
    return hand_position_tracking(env, std, command_names) * moving


def stand_still_velocity_l2(
    env, command_name: str, asset_cfg: SceneEntityCfg, command_threshold: float = 0.05
) -> torch.Tensor:
    """Penalize root linear and angular motion during stop commands without recovery attenuation."""
    asset = env.scene[asset_cfg.name]
    standing = standing_command_mask(env.command_manager.get_command(command_name), command_threshold)
    return (
        asset.data.root_lin_vel_b.torch.square().sum(dim=-1) + asset.data.root_ang_vel_b.torch.square().sum(dim=-1)
    ) * standing


def stand_still_velocity_exp(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    linear_std: float,
    angular_std: float,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward zero root velocity during stop commands with Gaussian widths in m/s and rad/s."""
    asset = env.scene[asset_cfg.name]
    standing = standing_command_mask(env.command_manager.get_command(command_name), command_threshold)
    error = asset.data.root_lin_vel_b.torch.square().sum(dim=-1) / linear_std**2
    error += asset.data.root_ang_vel_b.torch.square().sum(dim=-1) / angular_std**2
    return torch.exp(-error) * standing


class stand_still_position_l2(ManagerTermBase):
    """Penalize root XY drift from reset position or the start of a later stop interval."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.reference_xy = self.asset.data.root_pos_w.torch[:, :2].clone()
        self.was_standing = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.reference_xy[env_ids] = self.asset.data.root_pos_w.torch[env_ids, :2]
        self.was_standing[env_ids] = True

    def __call__(self, env, command_name: str, asset_cfg: SceneEntityCfg, command_threshold: float = 0.05):
        standing = standing_command_mask(env.command_manager.get_command(command_name), command_threshold)
        position = self.asset.data.root_pos_w.torch[:, :2]
        entering_stop = standing & ~self.was_standing
        self.reference_xy[entering_stop] = position[entering_stop]
        self.was_standing.copy_(standing)
        return (position - self.reference_xy).square().sum(dim=-1) * standing


class stand_still_position_exp(stand_still_position_l2):
    """Reward staying at reset XY or the start of a later stop interval with a Gaussian kernel."""

    def __call__(
        self, env, command_name: str, asset_cfg: SceneEntityCfg, std: float, command_threshold: float = 0.05
    ) -> torch.Tensor:
        """Return a stop-only reward with position width ``std`` in meters."""
        error = super().__call__(env, command_name, asset_cfg, command_threshold)
        standing = standing_command_mask(env.command_manager.get_command(command_name), command_threshold)
        return torch.exp(-error / std**2) * standing
