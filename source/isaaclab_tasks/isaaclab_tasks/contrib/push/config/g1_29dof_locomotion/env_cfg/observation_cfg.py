# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured joint state includes the uncontrolled arms during locomotion."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_contrib.mdp import MirrorObservationTermCfg as ObsTerm
from isaaclab_contrib.mdp import mirror_identity, mirror_vec3

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.symmetry import mirror_g1_joints
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp.symmetry import mirror_velocity_commands

from .action_cfg import ALL_JOINTS, LOWER_JOINTS


@configclass
class PolicyCfg(ObservationGroupCfg):
    """Actor observations, with explicit joint order and mirror rules."""

    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": ALL_JOINTS},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS, preserve_order=True)},
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": ALL_JOINTS},
        scale=0.05,
        noise=Unoise(n_min=-0.5, n_max=0.5),
    )
    actions = ObsTerm(func=mdp.last_action, mirror=mirror_g1_joints, mirror_params={"joint_names": LOWER_JOINTS})
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, mirror=mirror_vec3, mirror_params={"axial": True}, scale=0.25)
    projected_gravity = ObsTerm(
        func=mdp.bias_projected_gravity,
        mirror=mirror_vec3,
        params={"roll_bias_range": (0.0, 0.0), "pitch_bias_range": (0.0, 0.0)},
    )
    command = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "base_velocity"}, mirror=mirror_velocity_commands
    )

    height_command = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "base_height"}, mirror=mirror_identity
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CriticCfg(PolicyCfg):
    """Noise-free measurements for value estimation."""

    projected_gravity = ObsTerm(func=mdp.projected_gravity, mirror=mirror_vec3)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, mirror=mirror_vec3)

    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    policy = PolicyCfg()
    critic = CriticCfg()
