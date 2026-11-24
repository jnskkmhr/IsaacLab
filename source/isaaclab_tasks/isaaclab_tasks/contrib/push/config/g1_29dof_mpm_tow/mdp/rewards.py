# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms of the tow task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def lean_orientation_l2(
    env: ManagerBasedRLEnv, forward_lean_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize a tilted base, except for leaning forward up to a limit.

    A robot that tows has to lean into the rope: the rope pulls back at the pelvis, and only a centre of mass
    ahead of the feet balances that. Roll and backward lean are penalized as by ``flat_orientation_l2``; forward
    lean is free up to ``forward_lean_limit`` and penalized beyond it.

    Args:
        env: Environment instance.
        forward_lean_limit: Forward pitch that carries no penalty [rad].
        asset_cfg: The robot.

    Returns:
        Squared tilt outside the allowed forward lean, shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    gravity = asset.data.projected_gravity_b.torch
    # Leaning forward turns the body x-axis towards the ground, so gravity gains a positive x component.
    pitch_excess = torch.relu(gravity[:, 0] - math.sin(forward_lean_limit)) + torch.relu(-gravity[:, 0])
    return torch.square(pitch_excess) + torch.square(gravity[:, 1])
