# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.wbc.mdp import rewards


@configclass
class G1RewardsCfg:
    body_position = RewardTermCfg(func=rewards.track_body_position, weight=4.0, params={"standard_deviation": 0.12})
    body_orientation = RewardTermCfg(
        func=rewards.track_body_orientation, weight=1.0, params={"standard_deviation": 0.5}
    )
    joint_position = RewardTermCfg(func=rewards.track_joint_position, weight=1.0, params={"standard_deviation": 0.4})
    action_rate = RewardTermCfg(func=rewards.action_rate, weight=-0.01)
    joint_velocity = RewardTermCfg(func=rewards.joint_velocity, weight=-0.0001)
    stance_foot_sliding = RewardTermCfg(func=rewards.stance_foot_sliding, weight=-0.2)
