# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPM observation layout with rigid/RFT contact sources and matching mirror rules."""

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_contrib.mdp import MirrorObservationTermCfg as ObsTerm
from isaaclab_contrib.mdp import mirror_identity, mirror_quat, mirror_vec3

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp import symmetry
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft_contact import mdp as soft_mdp
from isaaclab_tasks.contrib.velocity.config.vel_mdp import rough_height

from ..mdp import observations as local_mdp
from .action_cfg import ACTIVE_JOINT


@configclass
class PolicyCfg(ObsGroup):
    """Observations for the actor."""

    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        mirror=mirror_vec3,
        mirror_params={"axial": True},
        noise=Unoise(n_min=-0.2, n_max=0.2),
        scale=0.25,
    )
    projected_gravity = ObsTerm(
        func=mdp.bias_projected_gravity,
        mirror=mirror_vec3,
        params={
            "roll_bias_range": (-math.radians(3.0), math.radians(3.0)),
            "pitch_bias_range": (-math.radians(3.0), math.radians(3.0)),
        },
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINT, preserve_order=True)},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINT, preserve_order=True)},
        scale=0.05,
    )
    actions = ObsTerm(
        func=mdp.last_action, mirror=symmetry.mirror_g1_joints, mirror_params={"joint_names": ACTIVE_JOINT}
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class CriticCfg(ObsGroup):
    """Noise-free proprioception for the critic."""

    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, mirror=mirror_vec3, mirror_params={"axial": True}, scale=0.25)
    base_quat = ObsTerm(func=mdp.root_quat_w, mirror=mirror_quat, params={"make_quat_unique": True})
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINT, preserve_order=True)},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": ACTIVE_JOINT},
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINT, preserve_order=True)},
        scale=0.05,
    )
    actions = ObsTerm(
        func=mdp.last_action, mirror=symmetry.mirror_g1_joints, mirror_params={"joint_names": ACTIVE_JOINT}
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
class CommandCfg(ObsGroup):
    """Command observations for the actor and critic."""

    velocity_commands = ObsTerm(
        func=mdp.generated_commands, mirror=symmetry.mirror_velocity_heading, params={"command_name": "base_velocity"}
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class PrivilegedObsCfg(ObsGroup):
    """Granular-terrain information available to the critic only."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, mirror=mirror_vec3)
    foot_height = ObsTerm(
        func=g1_mdp.foot_height,
        mirror=symmetry.mirror_foot_scalars,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )
    foot_contact = ObsTerm(
        func=soft_mdp.foot_contact_hybrid,
        mirror=symmetry.mirror_foot_scalars,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link", preserve_order=True
            ),
            "rigid_force_threshold": 5.0,
            "soft_force_threshold": 5.0,
        },
    )
    foot_contact_force = ObsTerm(
        func=soft_mdp.foot_contact_forces_hybrid,
        mirror=symmetry.mirror_foot_forces,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link", preserve_order=True
            ),
            "rigid_force_filter_threshold": 5.0,
            "soft_force_filter_threshold": 5.0,
            "retain_negative_forces": True,
        },
    )
    foot_air_time = ObsTerm(
        func=soft_mdp.foot_air_time_hybrid,
        mirror=symmetry.mirror_foot_scalars,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link", preserve_order=True
            ),
            "filter_time": 0.5,
        },
    )
    terrain_material_parameters = ObsTerm(func=local_mdp.terrain_material_parameters, mirror=mirror_identity)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class PrivilegedHistoryCfg(PrivilegedObsCfg):
    def __post_init__(self):
        self.history_length = 10


@configclass
class G1ObservationsCfg:
    """Observation specifications for the MDP."""

    policy: PolicyHistoryCfg = PolicyHistoryCfg()
    critic: CriticHistoryCfg = CriticHistoryCfg()
    command: CommandCfg = CommandCfg()
    privileged: PrivilegedHistoryCfg = PrivilegedHistoryCfg()


@configclass
class RoughPrivilegedHistoryCfg(PrivilegedHistoryCfg):
    """Same history layout, with foot sole height relative to each foot's local surface."""

    foot_height = PrivilegedHistoryCfg().foot_height.replace(
        func=rough_height.foot_height,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "left_sensor_cfg": SceneEntityCfg("left_foot_scanner"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_scanner"),
        },
    )


@configclass
class G1RoughObservationsCfg(G1ObservationsCfg):
    privileged = RoughPrivilegedHistoryCfg()
