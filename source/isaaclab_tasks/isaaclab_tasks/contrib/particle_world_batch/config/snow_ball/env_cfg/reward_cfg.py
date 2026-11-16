# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward configuration for the `snow_ball` slope rolling task."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from .. import mdp


@configclass
class SnowBallRewardsCfg:
    """Reward terms for rolling the snow-ball smoothly down the slope centerline."""

    downslope_velocity = RewardTermCfg(
        func=mdp.snow_ball_downslope_velocity_reward,
        weight=2.0,
        params={"target_speed": 0.35, "std": 0.25},
    )
    centerline = RewardTermCfg(
        func=mdp.snow_ball_track_centerline_reward,
        weight=1.0,
        params={"std": 0.20},
    )
    alive = RewardTermCfg(func=mdp.snow_ball_alive_bonus, weight=0.5)
    action_l2 = RewardTermCfg(func=mdp.snow_ball_action_l2, weight=-1.0e-3)
