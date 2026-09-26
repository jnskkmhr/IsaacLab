# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp
from isaaclab_tasks.contrib.mimic.mdp.symmetry import (
    mirror_body_orientations,
    mirror_body_positions,
    mirror_rotation_6d,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.symmetry import mirror_g1_joints
from isaaclab_tasks.contrib.velocity.config.vel_mdp import MirrorObservationTermCfg as ObsTerm
from isaaclab_tasks.contrib.velocity.config.vel_mdp import mirror_identity, mirror_vec3

from .action_cfg import G1ActionsCfg
from .commands_cfg import BODY_NAMES, JOINT_NAMES

CONTROLLED_JOINTS = [
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


@configclass
class G1ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "motion"},
            mirror=mirror_g1_joints,
            mirror_params={"joint_names": JOINT_NAMES},
        )  # ref joint pos
        motion_phase = ObsTerm(func=mimic_mdp.motion_phase, params={"command_name": "motion"}, mirror=mirror_identity)

        base_ang_vel = ObsTerm(
            mirror=mirror_vec3,
            mirror_params={"axial": True},
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            mirror=mirror_vec3,
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            mirror=mirror_g1_joints,
            mirror_params={"joint_names": CONTROLLED_JOINTS},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=CONTROLLED_JOINTS,
                    preserve_order=True,
                ),
            },
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            mirror=mirror_g1_joints,
            mirror_params={"joint_names": CONTROLLED_JOINTS},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=CONTROLLED_JOINTS,
                    preserve_order=True,
                ),
            },
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            mirror=mirror_g1_joints,
            mirror_params={"joint_names": G1ActionsCfg().joint_pos.joint_names},
        )

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "motion"},
            mirror=mirror_g1_joints,
            mirror_params={"joint_names": JOINT_NAMES},
        )
        motion_phase = ObsTerm(func=mimic_mdp.motion_phase, params={"command_name": "motion"}, mirror=mirror_identity)

        base_ang_vel = ObsTerm(mirror=mirror_vec3, mirror_params={"axial": True}, func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(mirror=mirror_vec3, func=mdp.projected_gravity)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            mirror=mirror_g1_joints,
            mirror_params={"joint_names": CONTROLLED_JOINTS},
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=CONTROLLED_JOINTS,
                    preserve_order=True,
                ),
            },
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            mirror=mirror_g1_joints,
            mirror_params={"joint_names": CONTROLLED_JOINTS},
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=CONTROLLED_JOINTS,
                    preserve_order=True,
                ),
            },
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            mirror=mirror_g1_joints,
            mirror_params={"joint_names": G1ActionsCfg().joint_pos.joint_names},
        )

        # privileged observations (for critic only)
        motion_anchor_pos_b = ObsTerm(
            func=mimic_mdp.motion_anchor_pos_b,
            params={"command_name": "motion"},
            mirror=mirror_vec3,
        )
        motion_anchor_ori_b = ObsTerm(
            func=mimic_mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            mirror=mirror_rotation_6d,
        )
        body_pos = ObsTerm(
            func=mimic_mdp.robot_body_pos_b,
            params={"command_name": "motion"},
            mirror=mirror_body_positions,
            mirror_params={"body_names": BODY_NAMES},
        )
        body_ori = ObsTerm(
            func=mimic_mdp.robot_body_ori_b,
            params={"command_name": "motion"},
            mirror=mirror_body_orientations,
            mirror_params={"body_names": BODY_NAMES},
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, mirror=mirror_vec3)

    # observation groups
    policy: PolicyCfg = PolicyCfg(enable_corruption=True, concatenate_terms=True)
    critic: CriticCfg = CriticCfg(enable_corruption=False, concatenate_terms=True)
