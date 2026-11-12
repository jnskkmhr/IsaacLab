# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-relative foot measurements for rigid and RFT rough locomotion."""

import torch

from isaaclab.utils.math import quat_apply


def foot_ground_height(env, left_sensor_cfg, right_sensor_cfg):
    """World Z of the surface below each ankle, in left/right order [m]."""
    return torch.stack(
        [env.scene[cfg.name].data.ray_hits_w.torch[..., 2].mean(dim=1) for cfg in (left_sensor_cfg, right_sensor_cfg)],
        dim=-1,
    )


def foot_height(env, asset_cfg, left_sensor_cfg, right_sensor_cfg, offset=(0.0, 0.0, -0.03539)):
    """Foot sole offset above the surface below its ankle [m]."""
    asset = env.scene[asset_cfg.name]
    position = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    rotation = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
    sole = position + quat_apply(rotation, torch.tensor(offset, device=env.device).expand_as(position))
    return sole[..., 2] - foot_ground_height(env, left_sensor_cfg, right_sensor_cfg)


def foot_clearance_reward(
    env, target_height, std, tanh_mult, asset_cfg, left_sensor_cfg, right_sensor_cfg, standing_position_foot_z=0.03539
):
    """Reference clearance reward, measuring each foot against its local rough surface."""
    asset = env.scene[asset_cfg.name]
    ground = foot_ground_height(env, left_sensor_cfg, right_sensor_cfg)
    error = (
        asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2] - ground - target_height - standing_position_foot_z
    ).square()
    weight = torch.tanh(tanh_mult * asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2].norm(dim=-1))
    return torch.exp(-(error * weight).sum(dim=1) / std)


def elevated_grid_pattern(cfg, device):
    """Raise ray starts in the yaw-aligned pattern, not the tilting body frame."""
    from isaaclab.sensors.ray_caster.patterns import grid_pattern

    starts, directions = grid_pattern(cfg, device)
    starts[:, 2] = cfg.height
    return starts, directions
