# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_contrib.mdp import MirrorObservationTermCfg as ObsTerm
from isaaclab_contrib.mdp import mirror_vec3

import isaaclab_tasks.core.velocity.mdp as mdp

from ..mdp import symmetry
from .action_cfg import CONTROLLED_JOINT_NAMES


@configclass
class PolicyCfg(ObsGroup):
    """Five frames, flattened term by term in Agile collector order (255 values)."""

    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        scale=0.2,
        noise=Unoise(n_min=-0.2, n_max=0.2),
        mirror=mirror_vec3,
        mirror_params={"axial": True},
    )
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), mirror=mirror_vec3)
    velocity_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "base_velocity"}, mirror=symmetry.mirror_velocity_commands
    )
    controlled_joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES, preserve_order=True)},
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": CONTROLLED_JOINT_NAMES},
    )
    controlled_joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        scale=0.05,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES, preserve_order=True)},
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": CONTROLLED_JOINT_NAMES},
    )
    actions = ObsTerm(
        func=mdp.last_action, mirror=symmetry.mirror_g1_joints, mirror_params={"joint_names": CONTROLLED_JOINT_NAMES}
    )

    def __post_init__(self):
        self.history_length = 5
        self.flatten_history_dim = True
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CriticCfg(ObsGroup):
    """Current privileged state (84 values); the network also receives policy history."""

    velocity_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "base_velocity"}, mirror=symmetry.mirror_velocity_commands
    )
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, mirror=mirror_vec3)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, mirror=mirror_vec3, mirror_params={"axial": True})
    projected_gravity = ObsTerm(func=mdp.projected_gravity, mirror=mirror_vec3)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, mirror=symmetry.mirror_g1_joints)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1, mirror=symmetry.mirror_g1_joints)
    actions = ObsTerm(
        func=mdp.last_action, mirror=symmetry.mirror_g1_joints, mirror_params={"joint_names": CONTROLLED_JOINT_NAMES}
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class G1ObservationsCfg:
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
