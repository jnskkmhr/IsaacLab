# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.mdp as g1_soft_mdp
import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp import symmetry
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.env_cfg.scene_cfg import G1SceneCfg
from isaaclab_tasks.contrib.velocity.config.vel_mdp import MirrorObservationTermCfg as ObsTerm
from isaaclab_tasks.contrib.velocity.config.vel_mdp import mirror_identity, mirror_quat, mirror_vec3

SOFT_CONTACT_THRESHOLD = 40.0
ACTIVE_JOINT = [
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
LEG_JOINT = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    # "left_ankle_pitch_joint",
    # "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    # "right_ankle_pitch_joint",
    # "right_ankle_roll_joint",
]


@configclass
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        mirror=mirror_vec3,
        mirror_params={"axial": True},
        noise=Unoise(n_min=-0.2, n_max=0.2),
        scale=0.25,
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        mirror=mirror_vec3,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
        scale=0.05,
    )
    actions = ObsTerm(
        func=mdp.last_action, mirror=symmetry.mirror_g1_joints, mirror_params={"joint_names": ACTIVE_JOINT}
    )
    height_scan = ObsTerm(
        func=mdp.height_scan,
        mirror=symmetry.mirror_height_scan,
        mirror_params={"pattern_cfg": G1SceneCfg().height_scanner.pattern_cfg},
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class CriticCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        mirror=mirror_vec3,
        mirror_params={"axial": True},
        scale=0.25,
    )
    base_quat = ObsTerm(
        func=mdp.root_quat_w,
        mirror=mirror_quat,
        params={"make_quat_unique": True},
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
        scale=0.05,
    )
    actions = ObsTerm(
        func=mdp.last_action, mirror=symmetry.mirror_g1_joints, mirror_params={"joint_names": ACTIVE_JOINT}
    )
    height_scan = ObsTerm(
        func=mdp.height_scan,
        mirror=symmetry.mirror_height_scan,
        mirror_params={"pattern_cfg": G1SceneCfg().height_scanner.pattern_cfg},
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class PolicyHistoryCfg(PolicyCfg):
    def __post_init__(self):
        self.history_length = 10

@configclass
class CriticHistoryCfg(CriticCfg):
    def __post_init__(self):
        self.history_length = 10

@configclass
class CommandObsCfg(ObsGroup):
    """Observations for command group."""
    velocity_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "base_velocity"}, mirror=symmetry.mirror_velocity_heading
    )

@configclass
class PrivilegedObsCfg(ObsGroup):
    """Observations for policy group."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, mirror=mirror_vec3)
    foot_height = ObsTerm(
        func=g1_mdp.foot_height,
        mirror=symmetry.mirror_foot_scalars,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )

    foot_contact = ObsTerm(
        func=g1_soft_mdp.foot_contact_hybrid,
        mirror=symmetry.mirror_foot_scalars,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
            "rigid_force_threshold": 5.0,
            "soft_force_threshold": SOFT_CONTACT_THRESHOLD,
        },
    )
    foot_contact_force = ObsTerm(
        func=g1_soft_mdp.foot_contact_forces_hybrid,
        mirror=symmetry.mirror_foot_forces,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
            "rigid_force_filter_threshold": 5.0,
            "soft_force_filter_threshold": SOFT_CONTACT_THRESHOLD,
        },
    )
    foot_air_time = ObsTerm(
        func=g1_soft_mdp.foot_air_time_hybrid,
        mirror=symmetry.mirror_foot_scalars,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
        },
    )

    terrain_material_parameters = ObsTerm(
        # func=mdp.terrain_material_parameters_all_hybrid,
        func=g1_soft_mdp.terrain_material_parameters_hybrid,
        mirror=mirror_identity,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class PrivilegedHistoryCfg(PrivilegedObsCfg):
    def __post_init__(self):
        self.history_length = 10


"""
obs for logging
"""


@configclass
class LoggingObsCfg(ObsGroup):
    """Observations for policy group."""

    base_pos = ObsTerm(func=mdp.root_pos_w, mirror=mirror_vec3)
    base_quat = ObsTerm(func=mdp.root_quat_w, mirror=mirror_quat)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, mirror=mirror_vec3)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, mirror=mirror_vec3, mirror_params={"axial": True})
    commands = ObsTerm(
        func=mdp.generated_commands,
        mirror=symmetry.mirror_velocity_heading,
        params={"command_name": "base_velocity"},
    )
    contact_forces = ObsTerm(
        func=g1_soft_mdp.foot_contact_forces_raw_hybrid,
        mirror=symmetry.mirror_foot_forces,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "soft_contact_sensor_name": "physics_callback",
            "rigid_force_filter_threshold": 5.0,
            "soft_force_filter_threshold": 40.0,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class G1ObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups
    policy: PolicyHistoryCfg = PolicyHistoryCfg()
    critic: CriticHistoryCfg = CriticHistoryCfg()
    command: CommandObsCfg = CommandObsCfg()
    privileged: PrivilegedHistoryCfg = PrivilegedHistoryCfg()

    # logging: LoggingObsCfg = LoggingObsCfg()
