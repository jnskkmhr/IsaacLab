# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agile rewards evaluated from coupled forces and the initial sand surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp as agile_mdp
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch_agile.mdp.observations import _surface_height

if TYPE_CHECKING:
    from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch_agile.mpm_env import G1MovingPatchAgileEnv


def base_height_exp(
    env: G1MovingPatchAgileEnv, target_height: float, std: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Agile's exponential root-height reward relative to the initial sand surface."""
    height = env.scene["robot"].data.root_pos_w.torch[:, 2] - _surface_height(env, sensor_cfg).mean(dim=1)
    return torch.exp(-(height - target_height).square() / std**2)


def feet_slip(env: G1MovingPatchAgileEnv, contact_threshold: float, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Agile's horizontal ankle speed penalty with three physics samples of contact history."""
    contact = env.body_contact_force_history[:, :3, robot_cfg.body_ids].norm(dim=-1).amax(dim=1) > contact_threshold
    velocity = env.scene[robot_cfg.name].data.body_vel_w.torch[:, robot_cfg.body_ids, :2]
    return (velocity.norm(dim=-1) * contact).sum(dim=-1)


def jumping(env: G1MovingPatchAgileEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Agile's flight penalty using the latest rigid plus MPM force on each ankle."""
    return (env.body_contact_force_history[:, 0, asset_cfg.body_ids].norm(dim=-1) < threshold).all(dim=1).float()


class SmoothContactObserver(agile_mdp.SmoothContactObserver):
    """Keep Agile's loading cost and reset handling; read the coupled physics-step history."""

    def __call__(self, env: G1MovingPatchAgileEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        step = int(env.common_step_counter)
        if self.last_step == step:
            raise ValueError("Loading observer was evaluated twice for one transition")
        history = env.body_contact_force_history[:, :, asset_cfg.body_ids]
        self.costs["interval_loading"] = agile_mdp.interval_loading_cost(history, self.valid)
        self.valid.fill_(True)
        self.last_step = step
        return self.zero
