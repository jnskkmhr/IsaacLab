# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg, SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as Term
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid import mdp as rigid_mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft_contact import mdp as soft_mdp

from .. import mdp as local_mdp
from .action_cfg import JOINT_NAMES


@configclass
class PolicyCfg(ObservationGroupCfg):
    enable_corruption = False
    concatenate_terms = True
    history_length = 10
    flatten_history_dim = True
    base_ang_vel = Term(
        func=mdp.base_ang_vel, scale=0.25, noise=UniformNoiseCfg(n_max=0.2, n_min=-0.2, operation="add")
    )
    projected_gravity = Term(
        func=mdp.projected_gravity, noise=UniformNoiseCfg(n_max=0.05, n_min=-0.05, operation="add")
    )
    velocity_commands = Term(func=local_mdp.velocity_commands_wxyz, params={"command_name": "base_velocity"})
    joint_pos = Term(
        func=mdp.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            )
        },
        noise=UniformNoiseCfg(n_max=0.01, n_min=-0.01, operation="add"),
    )
    joint_vel = Term(
        func=mdp.joint_vel_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            )
        },
        scale=0.05,
        noise=UniformNoiseCfg(n_max=1.5, n_min=-1.5, operation="add"),
    )
    actions = Term(func=mdp.last_action)


@configclass
class CriticCfg(ObservationGroupCfg):
    enable_corruption = False
    concatenate_terms = True
    history_length = 10
    flatten_history_dim = True
    base_ang_vel = Term(func=mdp.base_ang_vel, scale=0.25)
    base_quat = Term(func=local_mdp.root_quat_wxyz, params={"make_quat_unique": True})
    velocity_commands = Term(func=local_mdp.velocity_commands_wxyz, params={"command_name": "base_velocity"})
    joint_pos = Term(
        func=mdp.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    joint_vel = Term(
        func=mdp.joint_vel_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=JOINT_NAMES,
                preserve_order=True,
            )
        },
        scale=0.05,
    )
    actions = Term(func=mdp.last_action)


@configclass
class PrivilegedCfg(ObservationGroupCfg):
    enable_corruption = False
    concatenate_terms = True
    history_length = 10
    flatten_history_dim = True
    base_lin_vel = Term(func=mdp.base_lin_vel)
    foot_height = Term(
        func=rigid_mdp.foot_height,
        params={"asset_cfg": SceneEntityCfg(name="robot", body_names=".*_ankle_roll_link", preserve_order=False)},
    )
    foot_contact = Term(
        func=soft_mdp.foot_contact_hybrid,
        params={
            "soft_force_threshold": 40,
            "rigid_force_threshold": 5,
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                name="contact_forces", body_names=".*_ankle_roll_link", preserve_order=False
            ),
            "soft_contact_sensor_name": "physics_callback",
        },
    )
    foot_contact_force = Term(
        func=soft_mdp.foot_contact_forces_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                name="contact_forces", body_names=".*_ankle_roll_link", preserve_order=False
            ),
            "soft_contact_sensor_name": "physics_callback",
            "soft_force_filter_threshold": 40,
            "rigid_force_filter_threshold": 5,
        },
    )
    foot_air_time = Term(
        func=soft_mdp.foot_air_time_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                name="contact_forces", body_names=".*_ankle_roll_link", preserve_order=False
            ),
            "soft_contact_sensor_name": "physics_callback",
        },
    )
    terrain_material_parameters = Term(
        func=local_mdp.terrain_material_parameters_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                name="contact_forces", body_names=".*_ankle_roll_link", preserve_order=False
            ),
            "soft_contact_sensor_name": "physics_callback",
        },
    )


@configclass
class LoggingCfg(ObservationGroupCfg):
    enable_corruption = False
    concatenate_terms = True
    history_length = None
    flatten_history_dim = True
    base_pos = Term(func=mdp.root_pos_w)
    base_quat = Term(func=local_mdp.root_quat_wxyz)
    base_lin_vel = Term(func=mdp.base_lin_vel)
    base_ang_vel = Term(func=mdp.base_ang_vel)
    commands = Term(func=local_mdp.velocity_commands_wxyz, params={"command_name": "base_velocity"})
    contact_forces = Term(
        func=soft_mdp.foot_contact_forces_raw_hybrid,
        params={
            "rigid_contact_sensor_cfg": SceneEntityCfg(
                name="contact_forces", body_names=".*_ankle_roll_link", preserve_order=False
            ),
            "soft_contact_sensor_name": "physics_callback",
            "soft_force_filter_threshold": 40,
            "rigid_force_filter_threshold": 5,
        },
    )


@configclass
class G1ObservationsCfg:
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
    logging: LoggingCfg = LoggingCfg()
