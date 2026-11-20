# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def foot_height(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset: list[float] = [0.0, 0.0, -0.03539],
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pose = asset.data.body_pose_w.torch[:, asset_cfg.body_ids, :7]
    pose[..., :2] = pose[..., :2] - env.scene.env_origins.unsqueeze(1)[:, :, :2]
    position = pose[..., :3]
    quat = pose[..., 3:7]
    offset_position = torch.zeros_like(position)
    offset_position[..., 0] += offset[0]
    offset_position[..., 1] += offset[1]
    offset_position[..., 2] += offset[2]
    body_position = math_utils.quat_apply(quat, offset_position) + position
    return body_position[..., 2].reshape(env.num_envs, -1)


def foot_air_time(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"), filter_time: float = 0.5
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    air_time = torch.where(air_time > filter_time, 0.0, air_time)
    return air_time


def foot_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"), threshold: float = 1.0
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, :]
    contact = (torch.norm(contact_forces, dim=-1) > threshold).float()
    return contact


def foot_contact_forces(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, :]
    forces_flat = contact_forces.reshape(env.num_envs, -1)
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def root_quat_wxyz(env, asset_cfg=SceneEntityCfg("robot"), make_quat_unique=False):
    """Preserve the reference critic quaternion order at the observation boundary."""
    from isaaclab.envs.mdp import root_quat_w

    return root_quat_w(env, asset_cfg=asset_cfg, make_quat_unique=make_quat_unique)[..., [3, 0, 1, 2]]


def velocity_commands_wxyz(env, command_name):
    """Preserve reference velocity/heading observation layout (vx, vy, wz, w, x, y, z)."""
    command = env.command_manager.get_command(command_name)
    return torch.cat((command[:, :3], command[:, [6, 3, 4, 5]]), dim=-1)
