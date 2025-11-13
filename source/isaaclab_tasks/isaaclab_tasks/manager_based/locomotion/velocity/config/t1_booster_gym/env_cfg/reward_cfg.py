# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.t1_booster_gym.mdp as t1_mdp

@configclass
class T1RewardsCfg:
    """Reward terms for the MDP."""

    # --------
    # tracking
    # --------
    track_lin_vel_xy_exp = RewTerm(
        func=vel_mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.15},
    )

    track_ang_vel_z_exp = RewTerm(
        func=vel_mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.15}
    )

    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-40.0,
        params={
            "target_height": 0.68,
        }
    )

    orientation = RewTerm(
        func=t1_mdp.xy_orientation,
        weight=-7.0
    )

    # --------
    # regularization
    # --------
    # -- general style penalties
    alive = RewTerm(func=mdp.is_alive, weight=1.0) 
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-0.0) 
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.0005) 

    # -- root penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0) 
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2) 
    base_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2, 
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk")},
    )

    # -- joint penalties
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4) # type: ignore
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7) # type: ignore
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4) # type: ignore
    torque_tiredness = RewTerm(
        func=t1_mdp.joint_torque_tiredness,
        weight=0.0
    )
    power = RewTerm(
        func=t1_mdp.power,
        weight=-2e-4
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    dof_posture = RewTerm(
        func=t1_mdp.reward_penalty_pose,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_Hip_Pitch",
                    ".*_Hip_Roll",
                    ".*_Hip_Yaw",
                    # ".*_Knee_Pitch", # don't penalize knee to allow lift off
                    ".*_Ankle_Pitch",
                    ".*_Ankle_Roll",
                ],
                preserve_order=True,
            ),
        },
    )

    # # Penalize deviation from default of the joints that are not essential for locomotion
    # joint_deviation_hip = RewTerm(
    #     func=mdp.joint_deviation_l1, # type: ignore
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*Hip_.*"])},
    # )

    # joint_deviation_torso = RewTerm(
    #     func=mdp.joint_deviation_l1, # type: ignore
    #     # weight=-0.1,
    #     weight=-0.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names="Waist")},
    # )

    # -- foot orientation penalities
    feet_yaw_diff = RewTerm(
        func=t1_mdp.feet_yaw_diff, # type: ignore
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        }
    )

    feet_yaw_mean = RewTerm(
        func=t1_mdp.feet_yaw_mean, # type: ignore
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        }
    )

    feet_roll = RewTerm(
        func=t1_mdp.feet_roll, # type: ignore
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )

    # feet_pitch = RewTerm(
    #     func=t1_mdp.reward_feet_pitch, # type: ignore
    #     weight=-5.0,
    #     params={
    #         "std": 0.1,
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names=["left_foot_link", "right_foot_link"],
    #             preserve_order=True,
    #         ),
    #     },
    # )

    # -- swing foot penalties

    feet_slide = RewTerm(
        func=vel_mdp.feet_slide,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

    foot_distance = RewTerm(
        func = t1_mdp.reward_foot_distance,
        weight=-1.0,
        params={
            "ref_dist": 0.2,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_foot_link", "right_foot_link"],
                preserve_order=True,
            ),
        },
    )

    # Rewards the agent for having feet in the air (e.g., walking, running, not standing still).
    feet_air_time = RewTerm(
        func=vel_mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            # "threshold": 0.34,
            "threshold": 1.0 / 1.2 * 0.2, 
        },
    )

    feet_stepfunc_ref = RewTerm(
        func=t1_mdp.reward_stepfunc_reference,
        weight=3.0,
        params={
            "swing_period_ratio": 0.20,
            "lf_asset_cfg": SceneEntityCfg("robot", body_names="left_foot_link"),
            "rf_asset_cfg": SceneEntityCfg("robot", body_names="right_foot_link"),
            "foot_max_height": 0.1,
            "tracking_sigma": 0.02
        }
    )