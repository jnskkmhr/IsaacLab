# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Equal left/right position and orientation tracking rewards."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..mdp.rewards import elbow_inward_l2, wrist_tracking
from .action_cfg import UPPER_JOINTS


@configclass
class RewardsCfg:
    left_position = RewTerm(func=wrist_tracking, weight=5.0, params={"command_name": "left_ee_pose", "std": 0.2})
    right_position = RewTerm(func=wrist_tracking, weight=5.0, params={"command_name": "right_ee_pose", "std": 0.2})
    left_orientation = RewTerm(
        func=wrist_tracking, weight=1.0, params={"command_name": "left_ee_pose", "std": 0.5, "orientation": True}
    )
    right_orientation = RewTerm(
        func=wrist_tracking, weight=1.0, params={"command_name": "right_ee_pose", "std": 0.5, "orientation": True}
    )
    elbow_inward = RewTerm(
        func=elbow_inward_l2,
        weight=-50.0,
        params={
            "wrist_cfg": SceneEntityCfg(
                "robot", body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True
            ),
            "elbow_cfg": SceneEntityCfg(
                "robot", body_names=["left_elbow_link", "right_elbow_link"], preserve_order=True
            ),
            "lateral_signs": (1.0, -1.0),
            "reference_frame_link": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )
    # end_effector_velocity = RewTerm(
    #     func=end_effector_velocity_l2,
    #     weight=-0.5,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True
    #         ),
    #         "reference_frame_link": SceneEntityCfg("robot", body_names="torso_link"),
    #     },
    # )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-5e-3, params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_JOINTS)}
    )
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_JOINTS)}
    )
