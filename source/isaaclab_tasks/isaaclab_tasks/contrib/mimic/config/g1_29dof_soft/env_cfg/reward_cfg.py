# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.mimic.config.g1_29dof.env_cfg.reward_cfg import G1RewardsCfg as RigidRewardsCfg
from isaaclab_tasks.contrib.mimic.mdp.rewards import motion_global_anchor_tilt_error_l2


@configclass
class G1RewardsCfg(RigidRewardsCfg):
    """Motion rewards with continuous reference-relative torso tilt regularization."""

    motion_anchor_tilt = RewardTermCfg(
        func=motion_global_anchor_tilt_error_l2,
        weight=-0.5,
        params={"command_name": "motion"},
    )
