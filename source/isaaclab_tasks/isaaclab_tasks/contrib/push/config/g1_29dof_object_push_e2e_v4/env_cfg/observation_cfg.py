# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local observations only: proprioception, the box pose in the base frame, and a robot-frame command."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_contrib.mdp import MirrorObservationTermCfg as ObsTerm
from isaaclab_contrib.mdp import mirror_quat, mirror_vec3

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
from isaaclab_tasks.contrib.push.config.g1_29dof_object_push_e2e import mdp as push_mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp import symmetry

from .action_cfg import ACTIVE_JOINT
from .commands_cfg import NON_HAND_BODIES, WRIST_LINKS

HISTORY_LENGTH = 10
"""Frames of each history-stacked term given to the actor and critic."""


@configclass
class PolicyCfg(ObsGroup):
    """Actor observations: noisy proprioception and the box pose, all robot-relative, with history."""

    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        mirror=mirror_vec3,
        mirror_params={"axial": True},
        noise=Unoise(n_min=-0.2, n_max=0.2),
        scale=0.25,
    )
    projected_gravity = ObsTerm(func=mdp.projected_gravity, mirror=mirror_vec3)
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
    # Box pose relative to the pelvis: a stand-in for onboard perception, with measurement noise.
    object_pose = ObsTerm(
        func=push_mdp.object_pose_b, mirror=push_mdp.mirror_pose, noise=Unoise(n_min=-0.02, n_max=0.02)
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = HISTORY_LENGTH


@configclass
class CommandCfg(ObsGroup):
    """Actor command: forward, lateral, and yaw velocity in the robot frame."""

    velocity_commands = ObsTerm(
        func=push_mdp.velocity_command,
        mirror=push_mdp.mirror_velocity_command,
        params={"command_name": "base_velocity"},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class CriticCfg(ObsGroup):
    """Noise-free actor inputs plus the world base orientation, with history."""

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
    object_pose = ObsTerm(func=push_mdp.object_pose_b, mirror=push_mdp.mirror_pose)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = HISTORY_LENGTH


@configclass
class CriticCommandCfg(ObsGroup):
    """Critic command: the velocity command plus its world-frame heading target."""

    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        mirror=symmetry.mirror_velocity_heading,
        params={"command_name": "base_velocity"},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class PrivilegedCfg(ObsGroup):
    """Base velocity and rigid foot-contact information for the critic only, with history."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, mirror=mirror_vec3)
    foot_height = ObsTerm(
        func=g1_mdp.foot_height,
        mirror=symmetry.mirror_foot_scalars,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )
    foot_contact = ObsTerm(
        func=g1_mdp.foot_contact,
        mirror=symmetry.mirror_foot_scalars,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), "threshold": 5.0},
    )
    foot_contact_force = ObsTerm(
        func=g1_mdp.foot_contact_forces,
        mirror=symmetry.mirror_foot_forces,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    foot_air_time = ObsTerm(
        func=g1_mdp.foot_air_time,
        mirror=symmetry.mirror_foot_scalars,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = HISTORY_LENGTH


@configclass
class ObjectPrivilegedCfg(ObsGroup):
    """Box velocity and measured robot-box contact forces, for the critic only."""

    velocity = ObsTerm(func=push_mdp.box_velocity_b, mirror=push_mdp.mirror_twist)
    contact = ObsTerm(
        func=push_mdp.box_contact_state,
        mirror=push_mdp.mirror_box_contact_state,
        params={
            "left_sensor_cfg": SceneEntityCfg("box_contacts", body_names=WRIST_LINKS[0]),
            "right_sensor_cfg": SceneEntityCfg("box_contacts", body_names=WRIST_LINKS[1]),
            "other_sensor_cfg": SceneEntityCfg("box_contacts", body_names=NON_HAND_BODIES),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class ObservationsCfg:
    """Actor groups are ``policy`` and ``command``; every other group is critic-only."""

    policy: PolicyCfg = PolicyCfg()
    command: CommandCfg = CommandCfg()
    critic: CriticCfg = CriticCfg()
    critic_command: CriticCommandCfg = CriticCommandCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
    object_privileged: ObjectPrivilegedCfg = ObjectPrivilegedCfg()
