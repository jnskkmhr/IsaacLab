# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations relative to the initial sand surface.

Ray casts hit the supporting rigid mesh. Adding the bed depth reconstructs the
initial sand surface; this does not measure deformation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch_agile.mpm_env import G1MovingPatchAgileEnv


def _surface_height(env: G1MovingPatchAgileEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    height = env.scene[sensor_cfg.name].data.ray_hits_w.torch[..., 2]
    depth = env.cfg.scene.terrain.moving_patch_terrain.particle_depth
    if hasattr(env.scene.terrain, "mpm_env_mask"):
        return height + depth * env.scene.terrain.mpm_env_mask[:, None]
    return height + depth


def height_scan(env: G1MovingPatchAgileEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Scanner height above the initial sand surface, minus the observation offset [m]."""
    return env.scene[sensor_cfg.name].data.pos_w.torch[:, 2, None] - _surface_height(env, sensor_cfg) - offset
