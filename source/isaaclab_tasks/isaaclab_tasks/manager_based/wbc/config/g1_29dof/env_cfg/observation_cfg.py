# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.wbc.mdp as wbc_mdp

@configclass
class G1ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"}) # ref joint pos/vel
        # motion_anchor_pos_b = ObsTerm(
        #     func=wbc_mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        # ) # exclude
        motion_anchor_ori_b = ObsTerm(
            func=wbc_mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )

        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5)) # exclude
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            noise=Unoise(n_min=-0.01, n_max=0.01), 
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint", 
                        "left_hip_roll_joint", 
                        "left_hip_yaw_joint", 
                        "left_knee_joint", 
                        "left_ankle_pitch_joint", 
                        "left_ankle_roll_joint", 
                        
                        "right_hip_pitch_joint", 
                        "right_hip_roll_joint", 
                        "right_hip_yaw_joint", 
                        "right_knee_joint", 
                        "right_ankle_pitch_joint", 
                        "right_ankle_roll_joint", 
                        
                        "waist_yaw_joint", 
                        "waist_roll_joint", 
                        "waist_pitch_joint", 
                        
                        "left_shoulder_pitch_joint", 
                        "left_shoulder_roll_joint", 
                        "left_shoulder_yaw_joint", 
                        "left_elbow_joint", 
                        "left_wrist_roll_joint", 
                        "left_wrist_pitch_joint", 
                        "left_wrist_yaw_joint", 

                        "right_shoulder_pitch_joint", 
                        "right_shoulder_roll_joint", 
                        "right_shoulder_yaw_joint", 
                        "right_elbow_joint", 
                        "right_wrist_roll_joint", 
                        "right_wrist_pitch_joint", 
                        "right_wrist_yaw_joint", 
                    ],
                        preserve_order=True,
                    ), 
                    },
            ) 
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            noise=Unoise(n_min=-1.5, n_max=1.5), 
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint", 
                        "left_hip_roll_joint", 
                        "left_hip_yaw_joint", 
                        "left_knee_joint", 
                        "left_ankle_pitch_joint", 
                        "left_ankle_roll_joint", 
                        
                        "right_hip_pitch_joint", 
                        "right_hip_roll_joint", 
                        "right_hip_yaw_joint", 
                        "right_knee_joint", 
                        "right_ankle_pitch_joint", 
                        "right_ankle_roll_joint", 
                        
                        "waist_yaw_joint", 
                        "waist_roll_joint", 
                        "waist_pitch_joint", 
                        
                        "left_shoulder_pitch_joint", 
                        "left_shoulder_roll_joint", 
                        "left_shoulder_yaw_joint", 
                        "left_elbow_joint", 
                        "left_wrist_roll_joint", 
                        "left_wrist_pitch_joint", 
                        "left_wrist_yaw_joint", 

                        "right_shoulder_pitch_joint", 
                        "right_shoulder_roll_joint", 
                        "right_shoulder_yaw_joint", 
                        "right_elbow_joint", 
                        "right_wrist_roll_joint", 
                        "right_wrist_pitch_joint", 
                        "right_wrist_yaw_joint", 
                    ],
                        preserve_order=True,
                    ), 
                    },
            scale=0.05,
            )
        actions = ObsTerm(func=mdp.last_action)

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=wbc_mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=wbc_mdp.motion_anchor_ori_b, params={"command_name": "motion"})

        # privileged observations (for critic only)
        body_pos = ObsTerm(func=wbc_mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=wbc_mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ##

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint", 
                        "left_hip_roll_joint", 
                        "left_hip_yaw_joint", 
                        "left_knee_joint", 
                        "left_ankle_pitch_joint", 
                        "left_ankle_roll_joint", 
                        
                        "right_hip_pitch_joint", 
                        "right_hip_roll_joint", 
                        "right_hip_yaw_joint", 
                        "right_knee_joint", 
                        "right_ankle_pitch_joint", 
                        "right_ankle_roll_joint", 
                        
                        "waist_yaw_joint", 
                        "waist_roll_joint", 
                        "waist_pitch_joint", 
                        
                        "left_shoulder_pitch_joint", 
                        "left_shoulder_roll_joint", 
                        "left_shoulder_yaw_joint", 
                        "left_elbow_joint", 
                        "left_wrist_roll_joint", 
                        "left_wrist_pitch_joint", 
                        "left_wrist_yaw_joint", 

                        "right_shoulder_pitch_joint", 
                        "right_shoulder_roll_joint", 
                        "right_shoulder_yaw_joint", 
                        "right_elbow_joint", 
                        "right_wrist_roll_joint", 
                        "right_wrist_pitch_joint", 
                        "right_wrist_yaw_joint", 
                    ],
                        preserve_order=True,
                    ), 
                    },
            ) 
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint", 
                        "left_hip_roll_joint", 
                        "left_hip_yaw_joint", 
                        "left_knee_joint", 
                        "left_ankle_pitch_joint", 
                        "left_ankle_roll_joint", 
                        
                        "right_hip_pitch_joint", 
                        "right_hip_roll_joint", 
                        "right_hip_yaw_joint", 
                        "right_knee_joint", 
                        "right_ankle_pitch_joint", 
                        "right_ankle_roll_joint", 
                        
                        "waist_yaw_joint", 
                        "waist_roll_joint", 
                        "waist_pitch_joint", 
                        
                        "left_shoulder_pitch_joint", 
                        "left_shoulder_roll_joint", 
                        "left_shoulder_yaw_joint", 
                        "left_elbow_joint", 
                        "left_wrist_roll_joint", 
                        "left_wrist_pitch_joint", 
                        "left_wrist_yaw_joint", 

                        "right_shoulder_pitch_joint", 
                        "right_shoulder_roll_joint", 
                        "right_shoulder_yaw_joint", 
                        "right_elbow_joint", 
                        "right_wrist_roll_joint", 
                        "right_wrist_pitch_joint", 
                        "right_wrist_yaw_joint", 
                    ],
                        preserve_order=True,
                    ), 
                    },
            scale=0.05, 
            )
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PolicyCfg = PolicyCfg(enable_corruption=True, concatenate_terms=True)
    critic: CriticCfg = CriticCfg(enable_corruption=False, concatenate_terms=True)