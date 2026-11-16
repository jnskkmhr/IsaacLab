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
    """Reward root height above the command's initial terrain surface; std is in meters."""
    command = env.command_manager.get_term(command_name)
    height = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - command.terrain_height
    error = height - command.command[:, 0]
    return torch.exp(-error.square() / std**2)


def commanded_torso_orientation_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward torso roll=0 and commanded pitch with Gaussian tolerance std [rad]."""
    command = env.command_manager.get_term(command_name)
    return torch.exp(-command.orientation_error.square().sum(dim=-1) / std**2)
