# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-relative rewards for velocity tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def compute_standing_contact_penalty(
    foot_contact: torch.Tensor,
    velocity_command: torch.Tensor,
    base_lin_vel_b: torch.Tensor,
    base_ang_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    command_threshold: float = 0.05,
    recovery_linear_velocity: float = 0.2,
    recovery_angular_velocity: float = 0.5,
    recovery_tilt: float = 0.2,
) -> torch.Tensor:
    """Compute a standing contact penalty from tensors, without accessing the environment.

    This is a helper for reward terms, not a reward-manager term itself.
    ``velocity_command`` contains at least ``(vx, vy, wz)`` per environment.
    Base linear/angular velocities and projected gravity have shape ``(num_envs, 3)``
    and are expressed in the robot base frame.

    ``foot_contact`` has shape ``(num_envs, num_feet)`` and uses true/nonzero for contact.
    Linear and yaw commands must each be below ``command_threshold`` to enable the term.
    The penalty fades exponentially with measured base linear speed [m/s], angular
    speed [rad/s], and tilt from upright [rad], using the positive recovery scales.
    This permits recovery steps at a reduced cost; it does not guarantee push recovery.
    """
    if min(recovery_linear_velocity, recovery_angular_velocity, recovery_tilt) <= 0.0:
        raise ValueError("Standing contact penalty recovery scales must be positive.")
    standing = (torch.linalg.vector_norm(velocity_command[:, :2], dim=-1) < command_threshold) & (
        velocity_command[:, 2].abs() < command_threshold
    )
    base_tilt = torch.atan2(torch.linalg.vector_norm(projected_gravity_b[:, :2], dim=-1), -projected_gravity_b[:, 2])
    recovery = (
        torch.sum(base_lin_vel_b.square(), dim=-1) / recovery_linear_velocity**2
        + torch.sum(base_ang_vel_b.square(), dim=-1) / recovery_angular_velocity**2
        + (base_tilt / recovery_tilt).square()
    )
    unsupported_feet = (~foot_contact.bool()).sum(dim=-1)
    return unsupported_feet * standing * torch.exp(-recovery)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg,
    standing_position_foot_z: float = 0.039,
    height_sensor_cfg: SceneEntityCfg | None = None,
    ground_height_offset: float = 0.0,
) -> torch.Tensor:
    """Reward foot clearance relative to the mean scanned ground height.

    Args:
        env: Environment instance.
        target_height: Desired clearance above the standing foot height [m].
        std: Exponential denominator [m²], preserving the existing clearance reward.
        tanh_mult: Horizontal foot-speed multiplier [s/m].
        asset_cfg: Robot and foot bodies to measure.
        standing_position_foot_z: Standing foot body-origin height above the ground [m].
        height_sensor_cfg: Ray caster providing world-space ground hits. Without a sensor,
            the ground reference is world Z = 0, matching the existing reward.
        ground_height_offset: Optional offset to the ground height [m]. This is useful for accounting for
            additional layers such as sand or other terrain features above the base ground level.

    Returns:
        Velocity-weighted clearance reward, shape ``(num_envs,)``.
    """
    asset = env.scene[asset_cfg.name]
    ground_height = 0.0
    if height_sensor_cfg is not None:
        sensor = env.scene[height_sensor_cfg.name]
        ground_height = sensor.data.ray_hits_w.torch[..., 2].mean(dim=1, keepdim=True) + ground_height_offset
    height_error = torch.square(
        asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
        - ground_height
        - (target_height + standing_position_foot_z)
    )
    velocity_weight = torch.tanh(
        tanh_mult * torch.linalg.vector_norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=-1)
    )
    return torch.exp(-torch.sum(height_error * velocity_weight, dim=1) / std)


class feet_edge_penalty(ManagerTermBase):
    """Penalize stance-foot centers near terrain height discontinuities.

    Both yaw-aligned scanners must use the same grid, centered on the respective
    sole. Adjacent ray hits locate edges at their XY midpoints. Subtracting the
    median height difference along each grid axis excludes uniform slopes.
    Missing hits are ignored. This measures center clearance, not sole support area.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.scanners = [env.scene[cfg.params[name].name] for name in ("left_sensor_cfg", "right_sensor_cfg")]
        pattern = self.scanners[0].cfg.pattern_cfg
        starts, _ = pattern.func(pattern, env.device)
        other_pattern = self.scanners[1].cfg.pattern_cfg
        other_starts, _ = other_pattern.func(other_pattern, env.device)
        if not torch.equal(starts, other_starts):
            raise ValueError("Foot edge scanners must use identical grid patterns.")
        nx, ny = starts[:, 0].unique().numel(), starts[:, 1].unique().numel()
        indices = torch.arange(starts.shape[0], device=env.device)
        grid = indices.reshape(nx, ny) if pattern.ordering == "yx" else indices.reshape(ny, nx).T
        self.neighbors = []
        for first, second in ((grid[:-1, :], grid[1:, :]), (grid[:, :-1], grid[:, 1:])):
            first, second = first.flatten(), second.flatten()
            midpoint = 0.5 * (starts[first, :2] + starts[second, :2])
            distance = torch.linalg.vector_norm(midpoint, dim=-1)
            self.neighbors.append((first, second, distance))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        left_sensor_cfg: SceneEntityCfg,
        right_sensor_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        height_threshold: float,
        margin: float,
        contact_threshold: float,
    ) -> torch.Tensor:
        heights = torch.stack([scanner.data.ray_hits_w.torch[..., 2] for scanner in self.scanners], dim=1)
        nearest_edge = torch.full(heights.shape[:2], float("inf"), device=env.device)
        for first, second, distance in self.neighbors:
            difference = heights[..., second] - heights[..., first]
            valid = torch.isfinite(difference)
            slope = torch.nanmedian(torch.where(valid, difference, float("nan")), dim=-1, keepdim=True).values
            edge = valid & ((difference - slope).abs() > height_threshold)
            nearest_edge = torch.minimum(nearest_edge, torch.where(edge, distance, float("inf")).amin(dim=-1))
        contact_sensor = env.scene.sensors[sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
        in_contact = torch.linalg.vector_norm(forces, dim=-1) > contact_threshold
        return (in_contact * (1.0 - nearest_edge / margin).clamp(min=0.0).square()).sum(dim=-1)


def capped_foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    reference_foot_height: float,
    sole_offset: tuple[float, float, float],
    tanh_mult: float,
    command_name: str | None = None,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward moving swing feet for local terrain clearance, saturating at the reference height.

    Measure sole-center world Z above the nearest valid terrain ray hit beneath
    each foot. ``sole_offset`` is in the foot body's frame. Asset and contact
    selections must contain the left and right feet in that order.
    Only single-support swings earn reward, weighted by bounded horizontal foot
    speed. Missing terrain hits earn zero. Heights above the reference earn no
    additional reward and incur no clearance penalty.

    When ``command_name`` is provided, require commanded XY speed plus absolute
    yaw rate above ``command_threshold``, matching the velocity air-time reward.
    With None, callers can apply their own task-specific gate.
    """
    if reference_foot_height <= 0.0:
        raise ValueError("reference_foot_height must be positive.")
    asset = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    foot_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
    offset = foot_pos.new_tensor(sole_offset).expand_as(foot_pos)
    sole_pos = foot_pos + quat_apply(foot_quat, offset)
    hits = torch.stack(
        [env.scene[cfg.name].data.ray_hits_w.torch for cfg in (left_sensor_cfg, right_sensor_cfg)], dim=1
    )
    valid = torch.isfinite(hits).all(dim=-1)
    distance = (hits[..., :2] - sole_pos[:, :, None, :2]).square().sum(dim=-1)
    nearest = torch.where(valid, distance, float("inf")).argmin(dim=-1, keepdim=True)
    ground_height = hits[..., 2].gather(-1, nearest).squeeze(-1)
    clearance = torch.where(valid.any(dim=-1), sole_pos[..., 2] - ground_height, 0.0)
    contact = env.scene.sensors[sensor_cfg.name].data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0
    swing = ~contact & (contact.sum(dim=-1, keepdim=True) == 1)
    foot_speed = torch.linalg.vector_norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=-1)
    reward = (swing * torch.tanh(tanh_mult * foot_speed) * (clearance / reference_foot_height).clamp(0.0, 1.0)).sum(
        dim=-1
    )
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        command_speed = torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs()
        reward *= command_speed > command_threshold
    return reward
