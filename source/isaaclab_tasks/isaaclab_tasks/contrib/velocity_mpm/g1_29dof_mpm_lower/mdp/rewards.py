# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint regularization for policy-controlled joints only."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.contrib.velocity.config.vel_mdp.rewards import foot_clearance_reward as clearance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the L2 norm of mechanical power [W] for the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.linalg.vector_norm(torque * velocity, dim=-1)


def action_rate_l2(env: ManagerBasedRLEnv, action_indices: list[int]) -> torch.Tensor:
    """Penalize changes in selected policy action columns, independent of articulation ordering."""
    delta = env.action_manager.action[:, action_indices] - env.action_manager.prev_action[:, action_indices]
    return delta.square().sum(dim=-1)


def commanded_base_height_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward commanded root height relative to the environment origin with exp(-error² / std²); std is in meters."""
    height = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    error = height - env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-error.square() / std**2)


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
    """Apply the lower-body clearance reward to the platform or initial sand surface.

    The scanner queries support geometry, excluding the robot. Add the configured
    sand depth only over the bed. For the tilting tray, convert its normal depth
    to a vertical offset. Like the reference task, this does not estimate settling.
    """
    hits = env.scene[height_sensor_cfg.name].data.ray_hits_w.torch
    spawn = env.cfg.scene.sand.spawn
    if env.cfg.scene.testbed is None:
        local = hits[..., :2] - env.scene.env_origins[:, None, :2]
        lower = local.new_tensor(spawn.lower[:2])
        upper = local.new_tensor(spawn.upper[:2])
        over_bed = ((local >= lower) & (local <= upper)).all(dim=-1)
        offset = over_bed.float().mean(dim=1, keepdim=True) * ground_height_offset
    else:
        tray = env.scene["testbed"]
        ids, _ = tray.find_bodies("tray")
        quat = tray.data.body_quat_w.torch[:, ids[0]]
        normal = quat_apply(quat, quat.new_tensor([0.0, 0.0, 1.0]).expand(env.num_envs, 3))
        offset = ground_height_offset / normal[:, 2:3].clamp_min(0.1)
    return clearance(
        env,
        target_height=target_height,
        std=std,
        tanh_mult=tanh_mult,
        asset_cfg=asset_cfg,
        standing_position_foot_z=standing_position_foot_z,
        height_sensor_cfg=height_sensor_cfg,
        ground_height_offset=offset,
    )
