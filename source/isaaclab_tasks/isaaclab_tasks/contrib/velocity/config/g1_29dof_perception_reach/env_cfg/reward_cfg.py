# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_reach.mdp as reach_mdp
import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp


@configclass
class G1RewardsCfg:
    """Approach, align, and settle at the base-pose target."""

    goal_progress = RewTerm(func=reach_mdp.goal_progress, weight=4.0, params={"command_name": "base_pose"})
    goal_position = RewTerm(
        func=reach_mdp.goal_position_exp, weight=4.0, params={"command_name": "base_pose", "std": 1.0}
    )
    goal_position_fine = RewTerm(
        func=reach_mdp.goal_position_exp, weight=4.0, params={"command_name": "base_pose", "std": 0.25}
    )
    goal_heading = RewTerm(
        func=reach_mdp.goal_heading_exp,
        weight=4.0,
        params={"command_name": "base_pose", "std": 0.5, "position_std": 0.75},
    )
    goal_motion = RewTerm(
        func=reach_mdp.goal_motion_l2, weight=-2.0, params={"command_name": "base_pose", "position_std": 0.35}
    )
    goal_hold = RewTerm(func=reach_mdp.goal_hold, weight=3.0, params={"command_name": "base_pose"})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    action_rate_l2_lower_body = RewTerm(func=g1_mdp.action_rate_l2, weight=-0.01, params={"joint_idx": list(range(12))})
    action_rate_l2_upper_body = RewTerm(
        func=g1_mdp.action_rate_l2, weight=-0.05, params={"joint_idx": list(range(12, 29))}
    )
    energy = RewTerm(func=g1_mdp.energy, weight=-0.001)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    upper_body_posture = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*(waist|shoulder|elbow|wrist).*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    feet_air_time = RewTerm(
        func=reach_mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "base_pose",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 0.5,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link", preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link", preserve_order=True),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
