# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# Adapted from Agile. Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import math

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mdp as mpm_mdp


@configclass
class G1EventCfg:
    """Configuration for events."""

    randomize_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 2.0),
            "operation": "scale",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_distribution_params": (0.0, 0.005),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    randomize_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "armature_distribution_params": (0.4, 1.6),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_bodies_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-1.0, 5.0),
            "operation": "add",
        },
    )

    randomize_bodies_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )

    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "com_range": {"x": (-0.15, 0.25), "y": (-0.05, 0.05), "z": (-0.15, 0.15)},
        },
    )

    apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-5.0, 5.0),
        },
    )

    apply_external_force_torque_extremities = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*wrist_yaw_link.*", ".*ankle_roll_link.*"]),
            "force_range": (-5.0, 5.0),
            "torque_range": (-0.5, 0.5),
        },
    )

    reset_base = EventTerm(
        func=mpm_mdp.reset_root_state_on_terrain,
        mode="reset",
        params={
            "pose_range": {
                "x": (-2.5, 2.5),
                "y": (-2.5, 2.5),
                "z": (-0.0, 0.0),
                "yaw": (-3.14, 3.14),
                "roll": (-math.radians(10), math.radians(10)),
                "pitch": (-math.radians(10), math.radians(10)),
            },
            "velocity_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (-0.0, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll_link.*"]),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (-1.0, 1.0),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.25, 0.25),
            }
        },
    )

    mpm_material = EventTerm(
        func=mpm_mdp.randomize_mpm_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sand"),
            "parameter_ranges": {
                "friction": (0.3, 0.9),
                "density": (1000.0, 3000.0),
                "poisson_ratio": (0.3, 0.3),
                # "hardening": (0.0, 0.0),
                # "dilatancy": (0.0, 0.0),
                # "viscosity": (0.0, 0.0),
            },
            "distribution": "uniform",
        },
    )

    mpm_material_log = EventTerm(
        func=mpm_mdp.randomize_mpm_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("sand"),
            "parameter_ranges": {
                "young_modulus": (1.0e7, 1.0e9),
                "yield_pressure": (1.0e8, 1.0e9),
                # "tensile_yield_ratio": (1.0e-9, 1.0), # TODO: investigate instability in this parameters
                # "yield_stress": (1.0, 1.0e9), # TODO: ablate
            },
            "distribution": "log_uniform",
        },
    )
