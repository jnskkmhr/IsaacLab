# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.mimic.config.g1_29dof.env_cfg.reward_cfg import G1RewardsCfg as RigidRewardsCfg
from isaaclab_tasks.contrib.mimic.mdp.rewards import motion_global_anchor_tilt_error_l2

from ..mdp import rewards as soft_rewards


@configclass
class G1RewardsCfg(RigidRewardsCfg):
    """Motion rewards with continuous reference-relative torso tilt regularization."""

    motion_anchor_tilt = RewardTermCfg(
        func=motion_global_anchor_tilt_error_l2,
        weight=-0.5,
        params={"command_name": "motion"},
    )


@configclass
class G1RewardsFinetuneCfg(G1RewardsCfg):
    """Motion rewards with continuous reference-relative torso tilt regularization."""

    # finetune
    # knee_dof_deviation = RewardTermCfg(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.05,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*knee.*"),
    #     },
    # )

    # waist_dof_deviation = RewardTermCfg(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.5,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*waist_pitch.*"),
    #     },
    # )
    # flat_orientation_l2 = RewardTermCfg(
    #     func=mdp.flat_orientation_l2,
    #     # weight=-20.0,
    #     weight=-1.0,
    # )

    # feet_pitch = RewardTermCfg(
    #     func=g1_mdp.reward_feet_pitch,
    #     # weight=-4.0,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=[".*ankle_pitch.*"],
    #             preserve_order=True,
    #         ),
    #     },
    # )

    com_foot_center = RewardTermCfg(
        func=soft_rewards.grounded_com_foot_center_l2,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            # Center of the box contact patch configured in action_cfg.py.
            "foot_center_offset": (0.038, 0.0, -0.03539),
            "tolerance": 0.03,
        },
    )
    grounded_torso_tilt = RewardTermCfg(
        func=soft_rewards.grounded_torso_tilt_l2,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "tolerance": math.radians(10.0),
        },
    )
    post_jump_torso_ang_vel = RewardTermCfg(
        func=soft_rewards.post_jump_torso_ang_vel_l2,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "tolerance": 0.3,
        },
    )
