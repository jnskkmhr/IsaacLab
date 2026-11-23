# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward configuration for the `go2` quadruped granular locomotion task."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from ..mdp.reward import go2_action_l2, go2_alive_bonus, go2_base_height_reward, go2_forward_vel_reward


@configclass
class Go2RewardsCfg:
    """Reward terms for the `go2` quadruped granular locomotion task."""

    forward_vel = RewardTermCfg(
        func=go2_forward_vel_reward,
        weight=2.0,
        params={"target_vx": 0.20, "std": 0.25},
    )
    base_height = RewardTermCfg(
        func=go2_base_height_reward,
        weight=1.0,
        params={"target_z": 0.34, "std": 0.08},
    )
    alive = RewardTermCfg(func=go2_alive_bonus, weight=0.5)
    action_l2 = RewardTermCfg(func=go2_action_l2, weight=-0.01)
