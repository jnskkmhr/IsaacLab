# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPM robot randomization with flat-surface resets. The analytical sand is one fixed material, as in the sand task."""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid import mdp as g1_mdp

from .. import mdp as tow_mdp
from .scene_cfg import SLED_ASSET, SLED_RESET_DISTANCE


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
            # Every environment of a ground shares one column origin; spread the starts so they do not overlap
            # in the training video. Worlds do not collide with each other either way.
            "pose_range": {"x": (-5.0, 5.0), "y": (-5.0, 5.0), "yaw": (-math.pi, math.pi)},
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
    # After reset_base: the sled goes behind wherever the robot was put.
    reset_sled = EventTerm(
        func=tow_mdp.reset_object_behind_robot_on_ground,
        mode="reset",
        params={
            "distance": SLED_RESET_DISTANCE,
            "clearance": 0.02,
            "asset_cfg": SceneEntityCfg(SLED_ASSET),
            "robot_cfg": SceneEntityCfg("robot"),
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
