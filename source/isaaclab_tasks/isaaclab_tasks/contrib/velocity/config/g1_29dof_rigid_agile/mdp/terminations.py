# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        # we have infinite terrain because it is a plane
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w.torch[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w.torch[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def root_height_below_minimum_adaptive(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    min_foot_height = (asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]).min(dim=1).values

    return asset.data.root_pos_w.torch[:, 2] - min_foot_height < minimum_height


def illegal_ground_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    min_height: float,
) -> torch.Tensor:
    """Terminate when the contact force exceeds the force threshold and the asset is below the min_height."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history.torch
    # check if any contact force exceeds the threshold

    asset_height = torch.min(env.scene[asset_cfg.name].data.body_pos_w.torch[:, asset_cfg.body_ids, 2], dim=1)[0]
    on_ground = asset_height < min_height

    in_contact = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
        dim=1,
    )

    return in_contact & on_ground


def illegal_base_height(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_measurement_sensor"),
) -> torch.Tensor:
    """Terminate if the base height is below the threshold."""
    robot = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    base_height = robot.data.root_pos_w.torch[:, 2] - torch.mean(sensor.data.ray_hits_w.torch[..., 2], dim=1)
    return base_height < height_threshold
