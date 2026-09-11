# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations for the G1 29-DoF granular locomotion task.

The proprioceptive groups match the soft-contact task, and so does the privileged group: the
contact terms are served by the reaction wrench that the coupler feeds back from the MPM entry
to the proxy feet, and the terrain-material term keeps the soft-contact layout as a placeholder.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp
import isaaclab_tasks.core.velocity.mdp as mdp

from .. import mdp as mpm_mdp
from .action_cfg import ACTIVE_JOINT


@configclass
class PolicyCfg(ObsGroup):
    """Observations for the actor."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        noise=Unoise(n_min=-0.2, n_max=0.2),
        scale=0.25,
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
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
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class CriticCfg(ObsGroup):
    """Noise-free proprioception for the critic."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        scale=0.25,
    )
    base_quat = ObsTerm(
        func=mdp.root_quat_w,
        params={"make_quat_unique": True},
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
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
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ACTIVE_JOINT,
                preserve_order=True,
            ),
        },
        scale=0.05,
    )
    actions = ObsTerm(func=mdp.last_action)

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
class PrivilegedObsCfg(ObsGroup):
    """Granular-terrain information available to the critic only."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    foot_height = ObsTerm(
        func=g1_mdp.foot_height,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )
    foot_contact = ObsTerm(func=mpm_mdp.foot_contact)
    foot_contact_force = ObsTerm(func=mpm_mdp.foot_contact_force, params={"force_filter_threshold": 5.0})
    foot_air_time = ObsTerm(func=mpm_mdp.foot_air_time)
    terrain_material_parameters = ObsTerm(func=mpm_mdp.terrain_material_parameters)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class PrivilegedHistoryCfg(PrivilegedObsCfg):
    def __post_init__(self):
        self.history_length = 10


@configclass
class LoggingObsCfg(ObsGroup):
    """Quantities recorded for analysis; not consumed by the learning algorithm."""

    base_pos = ObsTerm(func=mdp.root_pos_w)
    base_quat = ObsTerm(func=mdp.root_quat_w)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
    )
    contact_forces = ObsTerm(func=mpm_mdp.foot_contact_force_raw, params={"force_filter_threshold": 5.0})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class G1ObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups
    policy: PolicyHistoryCfg = PolicyHistoryCfg()
    critic: CriticHistoryCfg = CriticHistoryCfg()
    privileged: PrivilegedHistoryCfg = PrivilegedHistoryCfg()

    logging: LoggingObsCfg = LoggingObsCfg()
