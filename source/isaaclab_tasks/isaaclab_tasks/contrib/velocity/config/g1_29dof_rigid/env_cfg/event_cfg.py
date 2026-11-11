# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.velocity.config.vel_mdp import contact_terrain as contact_mdp


@configclass
class G1EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    """
    reset
    """

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,  # type: ignore
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,  # type: ignore
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "distribution": "uniform",
            # "stiffness_distribution_params": (0.9, 1.1),
            # "damping_distribution_params": (0.9, 1.1),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
        },
    )

    """
    interval
    """
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class G1RoughEventCfg:
    """Task-local rough-terrain configuration."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=".*ankle_roll.*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.2, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
        mode="startup",
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names="torso_link"),
            "mass_distribution_params": (0.0, 5.0),
            "operation": "add",
        },
        mode="startup",
    )
    reset_base = EventTerm(
        func=contact_mdp.reset_root_state_on_terrain,
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.141592653589793, 3.141592653589793)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
        mode="reset",
    )
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=".*"),
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
        },
        mode="reset",
    )
    mpm_material = None
    mpm_material_log = None
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        mode="interval",
        interval_range_s=(10.0, 15.0),
    )


@configclass
class G1RoughEventCfg_PLAY(G1RoughEventCfg):
    """Playback configuration."""

    push_robot = None
