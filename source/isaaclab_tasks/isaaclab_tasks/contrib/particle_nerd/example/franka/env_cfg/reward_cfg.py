# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward configuration for the `franka` sand-pushing task."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from ..mdp.reward import franka_action_l2, franka_alive_bonus, franka_ee_sand_immersion_reward


@configclass
class FrankaRewardsCfg:
    """Reward terms for the `franka` sand-pushing task."""

    ee_immersion = RewardTermCfg(
        func=franka_ee_sand_immersion_reward,
        weight=2.0,
        params={"target_z": 0.08, "std": 0.10},
    )
    alive = RewardTermCfg(func=franka_alive_bonus, weight=0.5)
    action_l2 = RewardTermCfg(func=franka_action_l2, weight=-0.01)
