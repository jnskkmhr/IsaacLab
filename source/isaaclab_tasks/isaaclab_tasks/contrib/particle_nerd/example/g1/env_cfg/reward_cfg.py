# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward configuration for the `g1` humanoid granular locomotion task."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from ..mdp.reward import g1_action_l2, g1_alive_bonus, g1_base_height_reward, g1_forward_vel_reward


@configclass
class G1RewardsCfg:
    """Reward terms for the `g1` humanoid granular locomotion task."""

    forward_vel = RewardTermCfg(
        func=g1_forward_vel_reward,
        weight=2.0,
        params={"target_vx": 0.25, "std": 0.25},
    )
    base_height = RewardTermCfg(
        func=g1_base_height_reward,
        weight=1.0,
        params={"target_z": 0.74, "std": 0.10},
    )
    alive = RewardTermCfg(func=g1_alive_bonus, weight=0.5)
    action_l2 = RewardTermCfg(func=g1_action_l2, weight=-0.01)
