# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Height tracking for flat-ground locomotion."""

import torch

from isaaclab.managers import SceneEntityCfg


def commanded_base_height_l2(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize squared commanded root-height error above flat ground, in square meters."""
    height = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    error = height - env.command_manager.get_command(command_name)[:, 0]
    return error.square()


def commanded_base_height_exp(env, command_name: str, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward commanded root height above flat ground with exp(-error² / std²); std is in meters."""
    height = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    error = height - env.command_manager.get_command(command_name)[:, 0]
    return torch.exp(-error.square() / std**2)
