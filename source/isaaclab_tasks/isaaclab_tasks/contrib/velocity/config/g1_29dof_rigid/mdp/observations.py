# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

"""
gait
"""


def foot_height(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset: list[float] = [0.0, 0.0, -0.03539],
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
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
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    filter_time: float = 0.5,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    air_time = torch.where(air_time > filter_time, 0.0, air_time)  # filter out the air time larger than filter_time
    return air_time


def foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]  # (num_envs, num_body_ids, 3)
    contact = (torch.norm(contact_forces, dim=-1) > threshold).float()
    return contact


def foot_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]  # (num_envs, num_body_ids, 3)
    forces_flat = contact_forces.reshape(env.num_envs, -1)
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))

"""
terrain
"""


def terrain_material_parameters(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terrain material parameters on rigid ground.

    Rigid terrain has no compliant material, so the parameters are reported as constants to keep
    the observation layout identical to the soft-terrain task.

    Returns:
        Friction coefficient, material density [kg/m^3] and internal friction, shape [N, 3].
    """
    friction_coef = torch.ones(env.num_envs, device=env.device)
    rho_c = torch.ones(env.num_envs, device=env.device)
    mu_int = torch.ones(env.num_envs, device=env.device)
    return torch.stack([friction_coef, rho_c, mu_int], dim=-1)
