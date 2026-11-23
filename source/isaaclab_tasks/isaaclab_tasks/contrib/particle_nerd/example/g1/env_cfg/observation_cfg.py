# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation configuration for the `g1` humanoid granular locomotion task."""

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.g1_29dof_mpm.mdp.observations import terrain_material_parameters
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.observations import foot_height

from ..mdp.contacts import foot_air_time, foot_contact, foot_contact_force
from ..mdp.observation import g1_velocity_command
from .action_cfg import ACTIVE_JOINT


@configclass
class G1ObservationsCfg:
    """Observation layout of the reference MPM policy (W&B run ``5f1xyc21``).

    The actor consumes 930 history values followed by the seven velocity/heading command values.
    The critic consumes 940 history values, the same command, and 160 privileged history values.
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINT, preserve_order=True)},
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            scale=0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINT, preserve_order=True)},
        )
        actions = ObservationTermCfg(func=mdp.last_action)
        enable_corruption = False
        history_length = 10
        history_order = "term"

    @configclass
    class CriticCfg(ObservationGroupCfg):
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, scale=0.25)
        base_quat = ObservationTermCfg(func=mdp.root_quat_w, params={"make_quat_unique": True})
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINT, preserve_order=True)},
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            scale=0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ACTIVE_JOINT, preserve_order=True)},
        )
        actions = ObservationTermCfg(func=mdp.last_action)
        enable_corruption = False
        history_length = 10
        history_order = "term"

    @configclass
    class CommandCfg(ObservationGroupCfg):
        velocity_commands = ObservationTermCfg(func=g1_velocity_command)
        enable_corruption = False

    @configclass
    class PrivilegedCfg(ObservationGroupCfg):
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        foot_height = ObservationTermCfg(
            func=foot_height,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
        )
        foot_contact = ObservationTermCfg(func=foot_contact)
        foot_contact_force = ObservationTermCfg(func=foot_contact_force, params={"force_filter_threshold": 5.0})
        foot_air_time = ObservationTermCfg(func=foot_air_time)
        terrain_material_parameters = ObservationTermCfg(func=terrain_material_parameters)
        enable_corruption = False
        history_length = 10
        history_order = "term"

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    command: CommandCfg = CommandCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
