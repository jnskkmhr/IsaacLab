# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Alternative terrain references and clearance incentives for stair experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _local_ground_height(points_w: torch.Tensor, hits_w: torch.Tensor) -> torch.Tensor:
    """Return the nearest finite terrain sample in XY for each queried body origin.

    Queries have shape (B, bodies, 3); hits have shape (B, rays, 3). All-miss scans
    fall back to world Z=0. A query near a stair edge is limited by grid resolution.
    """
    valid = torch.isfinite(hits_w).all(dim=-1)
    safe_hits = torch.where(valid[..., None], hits_w, 0.0)
    distances = (points_w[:, :, None, :2] - safe_hits[:, None, :, :2]).square().sum(dim=-1)
    nearest = distances.masked_fill(~valid[:, None, :], torch.inf).argmin(dim=-1)
    return safe_hits[..., 2].gather(1, nearest)


def local_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize root height relative to the terrain sample directly below the root."""
    root = env.scene[asset_cfg.name].data.root_pos_w.torch
    ground = _local_ground_height(root[:, None], env.scene[sensor_cfg.name].data.ray_hits_w.torch)[:, 0]
    return (root[:, 2] - ground - target_height).square()


def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    horizontal_vertical_ratio: float = 4.0,
    min_horizontal_force: float = 1.0,
) -> torch.Tensor:
    """Penalize a foot contact dominated by horizontal force, as in AME_Locomotion.

    Compare world XY and world Z components of the normal contact force [N].
    The force floor suppresses near-zero sensor noise. Stair-edge contacts produce
    horizontal surface-normal forces; ordinary ground support predominantly acts
    along world Z. The result is one penalty per environment if either foot trips.
    """
    forces = env.scene.sensors[sensor_cfg.name].data.net_normal_forces_w.torch[:, sensor_cfg.body_ids]
    horizontal = torch.linalg.vector_norm(forces[..., :2], dim=-1)
    tripped = (horizontal > horizontal_vertical_ratio * forces[..., 2].abs()) & (horizontal > min_horizontal_force)
    return tripped.any(dim=-1).float()


def stair_foot_clearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    standing_position_foot_z: float = 0.039,
    ground_height_offset: float = 0.0,
    command_name: str = "base_velocity",
    local_ground: bool = True,
    motion_gate: bool = True,
) -> torch.Tensor:
    """Ablate clearance incentives independently of terrain reference selection.

    ``std`` retains the original reward's squared-meter denominator. Motion gating
    makes motionless feet earn zero instead of maximum reward. The local reference
    uses the terrain sample nearest each foot body in world XY, not a plane fit
    through multiple treads. These flags allow controlled comparisons with the
    existing reward before changing task defaults.
    """
    asset = env.scene[asset_cfg.name]
    feet = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    hits = env.scene[height_sensor_cfg.name].data.ray_hits_w.torch
    if local_ground:
        ground = _local_ground_height(feet, hits)
    else:
        valid = torch.isfinite(hits).all(dim=-1)
        ground = torch.where(valid, hits[..., 2], 0.0).sum(dim=1, keepdim=True)
        ground = ground / valid.sum(dim=1, keepdim=True).clamp_min(1)
    error = (feet[..., 2] - ground - ground_height_offset - target_height - standing_position_foot_z).square()
    speed = torch.linalg.vector_norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=-1)
    speed_weight = torch.tanh(tanh_mult * speed)
    reward = torch.exp(-(error * speed_weight).sum(dim=1) / std)
    if motion_gate:
        command_speed = torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :2], dim=-1)
        reward = reward * speed_weight.mean(dim=1) * (command_speed > 0.05)
    return reward
