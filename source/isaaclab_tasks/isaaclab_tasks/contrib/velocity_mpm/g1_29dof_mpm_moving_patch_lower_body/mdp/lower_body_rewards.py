# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Height tracking and joint penalties restricted to policy-controlled joints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lower_body_energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Preserve the original power-norm penalty, excluding randomized arm joints."""
    data = env.scene[asset_cfg.name].data
    power = data.applied_torque.torch[:, asset_cfg.joint_ids] * data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.linalg.vector_norm(power, dim=-1)


def lower_body_action_rate_l2(env: ManagerBasedRLEnv, action_indices: list[int]) -> torch.Tensor:
    """Penalize policy action changes using action columns rather than articulation indices."""
    delta = env.action_manager.action[:, action_indices] - env.action_manager.prev_action[:, action_indices]
    return delta.square().sum(dim=-1)


def commanded_base_height_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward root height tracking with exp(-error² / std²); error and std are in meters."""
    command = env.command_manager.get_term(command_name)
    error = command.measured_height - command.command[:, 0]
    return torch.exp(-error.square() / std**2)
