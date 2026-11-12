# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Current-command observations and mirror rules for the Newton ablation."""

from collections.abc import Callable

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from isaaclab_contrib.mdp import MirrorObservationTermCfg, mirror_identity, mirror_vec3

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.symmetry import (
    mirror_foot_forces,
    mirror_foot_scalars,
    mirror_g1_joints,
)

from ..mdp.symmetry import mirror_quat_wxyz, mirror_velocity_heading_wxyz
from .action_cfg import JOINT_NAMES
from .observation_cfg import CriticCfg, LoggingCfg, PolicyCfg, PrivilegedCfg


def _mirrored(term: ObservationTermCfg, mirror: Callable, **mirror_params) -> MirrorObservationTermCfg:
    """Copy a reference term's settings and attach its reflection rule."""
    return MirrorObservationTermCfg(**vars(term.copy()), mirror=mirror, mirror_params=mirror_params)


@configclass
class PolicyCurrentCommandCfg(PolicyCfg):
    velocity_commands = None
    base_ang_vel = _mirrored(PolicyCfg().base_ang_vel, mirror_vec3, axial=True)
    projected_gravity = _mirrored(PolicyCfg().projected_gravity, mirror_vec3)
    joint_pos = _mirrored(PolicyCfg().joint_pos, mirror_g1_joints, joint_names=JOINT_NAMES)
    joint_vel = _mirrored(PolicyCfg().joint_vel, mirror_g1_joints, joint_names=JOINT_NAMES)
    actions = _mirrored(PolicyCfg().actions, mirror_g1_joints, joint_names=JOINT_NAMES)


@configclass
class CriticCurrentCommandCfg(CriticCfg):
    velocity_commands = None
    base_ang_vel = _mirrored(CriticCfg().base_ang_vel, mirror_vec3, axial=True)
    base_quat = _mirrored(CriticCfg().base_quat, mirror_quat_wxyz)
    joint_pos = _mirrored(CriticCfg().joint_pos, mirror_g1_joints, joint_names=JOINT_NAMES)
    joint_vel = _mirrored(CriticCfg().joint_vel, mirror_g1_joints, joint_names=JOINT_NAMES)
    actions = _mirrored(CriticCfg().actions, mirror_g1_joints, joint_names=JOINT_NAMES)


@configclass
class CommandCfg(ObservationGroupCfg):
    enable_corruption = False
    concatenate_terms = True
    history_length = 0
    velocity_commands = _mirrored(PolicyCfg().velocity_commands, mirror_velocity_heading_wxyz)


@configclass
class MirroredPrivilegedCfg(PrivilegedCfg):
    base_lin_vel = _mirrored(PrivilegedCfg().base_lin_vel, mirror_vec3)
    foot_height = _mirrored(PrivilegedCfg().foot_height, mirror_foot_scalars)
    foot_contact = _mirrored(PrivilegedCfg().foot_contact, mirror_foot_scalars)
    foot_contact_force = _mirrored(PrivilegedCfg().foot_contact_force, mirror_foot_forces)
    foot_air_time = _mirrored(PrivilegedCfg().foot_air_time, mirror_foot_scalars)
    terrain_material_parameters = _mirrored(PrivilegedCfg().terrain_material_parameters, mirror_identity)


@configclass
class MirroredLoggingCfg(LoggingCfg):
    base_pos = _mirrored(LoggingCfg().base_pos, mirror_vec3)
    base_quat = _mirrored(LoggingCfg().base_quat, mirror_quat_wxyz)
    base_lin_vel = _mirrored(LoggingCfg().base_lin_vel, mirror_vec3)
    base_ang_vel = _mirrored(LoggingCfg().base_ang_vel, mirror_vec3, axial=True)
    commands = _mirrored(LoggingCfg().commands, mirror_velocity_heading_wxyz)
    contact_forces = _mirrored(LoggingCfg().contact_forces, mirror_foot_forces)


@configclass
class G1SymmetricObservationsCfg:
    policy = PolicyCurrentCommandCfg()
    critic = CriticCurrentCommandCfg()
    command = CommandCfg()
    privileged = MirroredPrivilegedCfg()
    logging = MirroredLoggingCfg()


from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.velocity.config.vel_mdp import rough_height


@configclass
class RoughPrivilegedCfg(MirroredPrivilegedCfg):
    foot_height = MirroredPrivilegedCfg().foot_height.replace(
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
class G1RoughObservationsCfg(G1SymmetricObservationsCfg):
    privileged = RoughPrivilegedCfg()
