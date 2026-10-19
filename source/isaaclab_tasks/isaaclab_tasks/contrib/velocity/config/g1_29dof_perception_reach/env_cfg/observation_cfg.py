# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_contrib.mdp import MirrorObservationTermCfg as MirrorObsTerm
from isaaclab_contrib.mdp import mirror_quat, mirror_vec3

import isaaclab_tasks.contrib.velocity.config.g1_29dof_perception_reach.mdp as reach_mdp
import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp import symmetry

from .action_cfg import G1ActionsCfg

ACTIVE_JOINTS = [
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
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    base_ang_vel = MirrorObsTerm(
        func=mdp.base_ang_vel,
        mirror=mirror_vec3,
        mirror_params={"axial": True},
        noise=Unoise(n_min=-0.2, n_max=0.2),
        scale=0.25,
    )
    projected_gravity = MirrorObsTerm(
        func=mdp.projected_gravity, mirror=mirror_vec3, noise=Unoise(n_min=-0.05, n_max=0.05)
    )
    joint_pos = MirrorObsTerm(
        func=mdp.joint_pos_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINTS},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS, preserve_order=True)},
    )
    joint_vel = MirrorObsTerm(
        func=mdp.joint_vel_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINTS},
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS, preserve_order=True)},
        scale=0.05,
    )
    actions = MirrorObsTerm(
        func=mdp.last_action,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": G1ActionsCfg().joint_pos.joint_names},
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 10


@configclass
class CriticCfg(ObsGroup):
    """Observations for policy group."""

    base_ang_vel = MirrorObsTerm(func=mdp.base_ang_vel, mirror=mirror_vec3, mirror_params={"axial": True}, scale=0.25)
    base_quat = MirrorObsTerm(func=mdp.root_quat_w, mirror=mirror_quat, params={"make_quat_unique": True})
    joint_pos = MirrorObsTerm(
        func=mdp.joint_pos_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINTS},
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS, preserve_order=True)},
    )
    joint_vel = MirrorObsTerm(
        func=mdp.joint_vel_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINTS},
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINTS, preserve_order=True)},
        scale=0.05,
    )
    actions = MirrorObsTerm(
        func=mdp.last_action,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": G1ActionsCfg().joint_pos.joint_names},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 10


@configclass
class CommandCfg(ObsGroup):
    """Observations for command group."""

    pose_commands = MirrorObsTerm(
        func=mdp.generated_commands, mirror=reach_mdp.mirror_pose_command, params={"command_name": "base_pose"}
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PrivilegedObsCfg(ObsGroup):
    """Observations for policy group."""

    base_lin_vel = MirrorObsTerm(func=mdp.base_lin_vel, mirror=mirror_vec3)
    foot_height = MirrorObsTerm(
        func=g1_mdp.foot_height,
        mirror=symmetry.mirror_foot_scalars,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )
    foot_contact = MirrorObsTerm(
        func=g1_mdp.foot_contact,
        mirror=symmetry.mirror_foot_scalars,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), "threshold": 5.0},
    )
    foot_contact_force = MirrorObsTerm(
        func=g1_mdp.foot_contact_forces,
        mirror=symmetry.mirror_foot_forces,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    foot_air_time = MirrorObsTerm(
        func=g1_mdp.foot_air_time,
        mirror=symmetry.mirror_foot_scalars,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 10


@configclass
class ElevationMapCfg(ObsGroup):
    """Current XYZ map in the torso yaw frame; retain both grid dimensions."""

    map = MirrorObsTerm(
        func=reach_mdp.elevation_map,
        mirror=reach_mdp.mirror_elevation_map,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "missing_height": -2.0},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.concatenate_dim = -1
        self.history_length = 0


@configclass
class G1PerceptionObservationsCfg:
    """Task-owned observation groups for the perception actor and privileged critic."""

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    command: CommandCfg = CommandCfg()
    privileged: PrivilegedObsCfg = PrivilegedObsCfg()
    elevation_map: ElevationMapCfg = ElevationMapCfg()
