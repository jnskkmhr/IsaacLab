# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward configuration for the `sphere` two-way coupling task."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from ..mdp.reward import (
    sphere_action_l2,
    sphere_alive_bonus,
    sphere_height_support_reward,
    sphere_target_position_reward,
)


@configclass
class SphereRewardsCfg:
    """Reward terms for stabilizing and supporting the rigid sphere on the granular sand bed."""

    target_position = RewardTermCfg(
        func=sphere_target_position_reward,
        weight=2.0,
        params={"target_pos": (0.0, 0.0, 0.25), "std": 0.25},
    )
    height_support = RewardTermCfg(
        func=sphere_height_support_reward,
        weight=1.0,
        params={"min_height": -0.05},
    )
    alive = RewardTermCfg(func=sphere_alive_bonus, weight=0.5)
    action_l2 = RewardTermCfg(func=sphere_action_l2, weight=-0.01)
