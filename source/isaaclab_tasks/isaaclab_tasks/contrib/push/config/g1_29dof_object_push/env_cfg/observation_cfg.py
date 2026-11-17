# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local student inputs and the exact observation contracts of both source teachers."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_contrib.mdp import MirrorObservationTermCfg as ObsTerm
from isaaclab_contrib.mdp import mirror_identity, mirror_vec3

from isaaclab_tasks.contrib.push.config.g1_29dof_manipulation.mdp.symmetry import mirror_hand_poses
from isaaclab_tasks.contrib.push.mdp.observations import box_velocity_b, object_pose_b
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.symmetry import mirror_g1_joints
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp.symmetry import mirror_velocity_commands

from ..mdp.observations import action_subset, joint_pos_reference
from ..mdp.symmetry import mirror_pose, mirror_velocity
from .action_cfg import ALL_JOINTS, LOWER_JOINTS, UPPER_JOINTS
from .scene_cfg import LOCOMOTION_REFERENCE_JOINT_POS, MANIPULATION_REFERENCE_JOINT_POS

MANIPULATION_OBS_JOINTS = UPPER_JOINTS + ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]


@configclass
class PolicyCfg(ObservationGroupCfg):
    """104 inputs: joints, previous actions, IMU state, relative box pose and velocity command."""

    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": ALL_JOINTS},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        scale=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": ALL_JOINTS},
    )
    actions = ObsTerm(func=mdp.last_action, mirror=mirror_g1_joints, mirror_params={"joint_names": ALL_JOINTS})
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, mirror=mirror_vec3, mirror_params={"axial": True})
    projected_gravity = ObsTerm(func=mdp.projected_gravity, mirror=mirror_vec3)
    box_pose = ObsTerm(func=object_pose_b, mirror=mirror_pose)
    command = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "box_velocity"}, mirror=mirror_velocity_commands
    )

    height_command = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "base_height"}, mirror=mirror_identity
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class CriticCfg(PolicyCfg):
    """Privileged robot and box velocities assist value estimation."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, mirror=mirror_vec3)
    box_velocity = ObsTerm(func=box_velocity_b, mirror=mirror_velocity)


@configclass
class LocomotionTeacherCfg(ObservationGroupCfg):
    """83 inputs, including pelvis height and only the 15 lower-body actions."""

    joint_pos = ObsTerm(
        func=joint_pos_reference,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True),
            "reference_positions": LOCOMOTION_REFERENCE_JOINT_POS,
        },
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": ALL_JOINTS},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        scale=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": ALL_JOINTS},
    )
    actions = ObsTerm(
        func=action_subset,
        params={"indices": list(range(len(LOWER_JOINTS)))},
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": LOWER_JOINTS},
    )
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, mirror=mirror_vec3, mirror_params={"axial": True})
    projected_gravity = ObsTerm(func=mdp.projected_gravity, mirror=mirror_vec3)
    command = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "box_velocity"}, mirror=mirror_velocity_commands
    )

    height_command = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "base_height"}, mirror=mirror_identity
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class ManipulationTeacherCfg(ObservationGroupCfg):
    """76 inputs; waist observation order is yaw, pitch, roll, matching the reaching policy."""

    joint_pos = ObsTerm(
        func=joint_pos_reference,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MANIPULATION_OBS_JOINTS, preserve_order=True),
            "reference_positions": MANIPULATION_REFERENCE_JOINT_POS,
        },
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": MANIPULATION_OBS_JOINTS},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        scale=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=MANIPULATION_OBS_JOINTS, preserve_order=True)},
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": MANIPULATION_OBS_JOINTS},
    )
    actions = ObsTerm(
        func=action_subset,
        params={"indices": list(range(len(LOWER_JOINTS), len(ALL_JOINTS)))},
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": UPPER_JOINTS},
    )
    hand_targets = ObsTerm(func=mdp.generated_commands, params={"command_name": "hand_pose"}, mirror=mirror_hand_poses)
    hand_pose = ObsTerm(
        func=mdp.body_pose_local,
        params={
            "target_frame": SceneEntityCfg(
                "robot", body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True
            ),
            "parent_frame": SceneEntityCfg("robot", body_names="torso_link"),
        },
        mirror=mirror_hand_poses,
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    policy = PolicyCfg()
    critic = CriticCfg()
    locomotion_teacher_obs = LocomotionTeacherCfg()
    manipulation_teacher_obs = ManipulationTeacherCfg()
