# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.wbc.mdp as mdp


@configclass
class G1RewardsCfg:
    """Tracking rewards and regularization penalties for whole-body control."""

    # -- tracking
    body_position = RewTerm(func=mdp.track_body_position, weight=4.0, params={"standard_deviation": 0.12})
    body_orientation = RewTerm(func=mdp.track_body_orientation, weight=1.0, params={"standard_deviation": 0.5})
    joint_position = RewTerm(func=mdp.track_joint_position, weight=1.0, params={"standard_deviation": 0.4})

    # -- penalties
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    stance_foot_sliding = RewTerm(func=mdp.stance_foot_sliding, weight=-0.2)
    no_fly = RewTerm(
        func=mdp.desired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 5.0,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )
