# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms using coupled forces and the initial sand surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch_agile.mdp.observations import _surface_height

if TYPE_CHECKING:
    from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch_agile.mpm_env import G1MovingPatchAgileEnv


def illegal_base_height(
    env: G1MovingPatchAgileEnv, height_threshold: float, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate below Agile's root-height threshold above the initial sand surface."""
    height = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - _surface_height(env, sensor_cfg).mean(dim=1)
    return height < height_threshold


def illegal_ground_contact(
    env: G1MovingPatchAgileEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg,
    min_height: float,
    height_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Agile's low-body contact condition, using coupled force history and terrain height."""
    forces = env.body_contact_force_history[:, :3, asset_cfg.body_ids]
    contact = (forces.norm(dim=-1).amax(dim=1) > threshold).any(dim=1)
    height = env.scene[asset_cfg.name].data.body_pos_w.torch[:, asset_cfg.body_ids, 2].amin(dim=1)
    return contact & (height - _surface_height(env, height_sensor_cfg).mean(dim=1) < min_height)
