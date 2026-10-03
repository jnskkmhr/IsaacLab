# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.wbc.mdp as mdp
from isaaclab_tasks.contrib.velocity.config.vel_mdp import MirrorObservationTermCfg as ObsTerm

JOINT_NAME = [
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
]

BODY_NAMES = [
    "pelvis",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


@configclass
class G1ObservationsCfg:
    """Named teacher, student, and critic terms with explicit sagittal mirror rules."""

    @configclass
    class StudentCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, mirror=mdp.mirror_vec3, mirror_params={"axial": True})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, mirror=mdp.mirror_vec3)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAME, preserve_order=True)},
            mirror=mdp.mirror_g1_joints,
            mirror_params={"joint_names": JOINT_NAME},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAME, preserve_order=True)},
            scale=0.05,
            mirror=mdp.mirror_g1_joints,
            mirror_params={"joint_names": JOINT_NAME},
        )
        last_action = ObsTerm(
            func=mdp.last_action,
            mirror=mdp.mirror_g1_joints,
            mirror_params={"joint_names": JOINT_NAME},
        )

        # task space command
        target_body_pos = ObsTerm(
            func=mdp.target_body_positions,
            mirror=mdp.mirror_body_positions,
            mirror_params={"body_names": BODY_NAMES},
        )
        target_body_ori = ObsTerm(
            func=mdp.target_body_orientations,
            mirror=mdp.mirror_body_orientations,
            mirror_params={"body_names": BODY_NAMES},
        )

    @configclass
    class TeacherCfg(StudentCfg):
        target_joint_pos = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "whole_body"},
            mirror=mdp.mirror_g1_joints,
            mirror_params={"joint_names": JOINT_NAME},
        )

    @configclass
    class CriticCfg(TeacherCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, mirror=mdp.mirror_vec3)
        target_joint_vel = ObsTerm(
            func=mdp.target_joint_velocities,
            scale=0.05,
            mirror=mdp.mirror_g1_joints,
            mirror_params={"joint_names": JOINT_NAME},
        )
        target_foot_contact = ObsTerm(func=mdp.target_foot_contacts, mirror=mdp.mirror_foot_scalars)

    teacher: TeacherCfg = TeacherCfg(enable_corruption=True, concatenate_terms=True)
    student: StudentCfg = StudentCfg(enable_corruption=True, concatenate_terms=True)
    critic: CriticCfg = CriticCfg(enable_corruption=False, concatenate_terms=True)
