# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-relative rewards for velocity tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
        ground_height_offset: Optional offset to the ground height [m]. This is useful for accounting for additional layers
            such as sand or other terrain features above the base ground level.

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
