# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class HECTORRewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_toe"),
        },
    )
    # -- penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0) # type: ignore
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0) # type: ignore
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05) # type: ignore
    
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, # type: ignore
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_toe_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1, # type: ignore
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint"])},
    )
    # joint_deviation_torso = RewTerm(
    #     func=mdp.joint_deviation_l1, # type: ignore
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names="base")},
    # )
    
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5) # type: ignore
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7) # type: ignore
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01) # type: ignore
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, # type: ignore
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base"]), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0) # type: ignore