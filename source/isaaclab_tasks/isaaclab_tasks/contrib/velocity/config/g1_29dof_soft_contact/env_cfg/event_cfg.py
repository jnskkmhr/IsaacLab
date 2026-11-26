# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPM robot randomization with flat-surface resets and RFT material sampling."""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid import mdp as g1_mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft_contact import mdp as soft_mdp
from isaaclab_tasks.contrib.velocity.config.vel_mdp.contact_terrain import reset_root_state_on_terrain


@configclass
class G1EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.2, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (0.0, 5.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=g1_mdp.reset_root_state_uniform_on_ground,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
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
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )
    randomize_friction = EventTerm(
        func=soft_mdp.randomize_terrain_friction,
        mode="reset",
        params={"friction_range": [0.1, 1], "contact_solver_name": "physics_callback"},
        is_global_time=False,
        interval_range_s=None,
        min_step_count_between_reset=0,
    )
    randomize_stiffness = EventTerm(
        func=soft_mdp.randomize_terrain_stiffness,
        mode="reset",
        params={"stiffness_range": [0.1, 0.9], "contact_solver_name": "physics_callback"},
        is_global_time=False,
        interval_range_s=None,
        min_step_count_between_reset=0,
    )
    randomize_material_density = EventTerm(
        func=soft_mdp.randomize_material_density,
        mode="reset",
        params={
            "bulk_density_range": [1000, 3000],
            "contact_solver_name": "physics_callback",
            "packing_ratio_range": [0.5, 1],
        },
        is_global_time=False,
        interval_range_s=None,
        min_step_count_between_reset=0,
    )


@configclass
class G1RoughEventCfg(G1EventCfg):
    """Sample root clearance relative to the rough surface, including on RFT tiles."""

    reset_base = G1EventCfg().reset_base.replace(func=reset_root_state_on_terrain)
