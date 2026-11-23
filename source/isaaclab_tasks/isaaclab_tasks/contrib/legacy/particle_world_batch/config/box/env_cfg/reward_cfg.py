# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward configuration for the `box` two-way coupling task."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from .. import mdp


@configclass
class BoxRewardsCfg:
    """Reward terms for stabilizing and supporting the rigid box on the sand bed."""

    target_pos = RewardTermCfg(
        func=mdp.box_target_position_reward,
        weight=2.0,
        params={"target_pos": (0.0, 0.0, 0.08), "std": 0.25},
    )
    height_support = RewardTermCfg(
        func=mdp.box_height_support_reward,
        weight=1.0,
        params={"min_height": -0.05},
    )
    alive = RewardTermCfg(func=mdp.box_alive_bonus, weight=0.5)
    action_l2 = RewardTermCfg(func=mdp.box_action_l2, weight=-1.0e-3)
