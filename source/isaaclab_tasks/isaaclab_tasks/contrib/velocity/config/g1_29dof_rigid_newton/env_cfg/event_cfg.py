# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as Term
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid import mdp as rigid_mdp


@configclass
class G1EventCfg:
    physics_material = Term(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=".*ankle_roll.*", preserve_order=False),
            "num_buckets": 64,
            "restitution_range": [0, 0],
            "static_friction_range": [0.6, 1],
            "dynamic_friction_range": [0.2, 0.6],
        },
        is_global_time=False,
        interval_range_s=None,
        min_step_count_between_reset=0,
    )

    add_base_mass = Term(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names="torso_link", preserve_order=False),
            "operation": "add",
            "mass_distribution_params": [-1, 3],
        },
        is_global_time=False,
        interval_range_s=None,
        min_step_count_between_reset=0,
    )

    reset_base = Term(
        func=rigid_mdp.reset_root_state_uniform_on_ground,
        mode="reset",
        params={
            "pose_range": {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "yaw": [-3.141592653589793, 3.141592653589793]},
            "velocity_range": {"x": [0, 0], "y": [0, 0], "z": [0, 0], "yaw": [0, 0], "roll": [0, 0], "pitch": [0, 0]},
        },
        is_global_time=False,
        interval_range_s=None,
        min_step_count_between_reset=0,
    )

    reset_robot_joints = Term(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": [1, 1], "velocity_range": [0, 0]},
        is_global_time=False,
        interval_range_s=None,
        min_step_count_between_reset=0,
    )

    scale_actuator_gains = Term(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=".*", preserve_order=False),
            "operation": "scale",
            "distribution": "uniform",
            "damping_distribution_params": [0.9, 1.1],
            "stiffness_distribution_params": [0.9, 1.1],
        },
        is_global_time=False,
        interval_range_s=None,
        min_step_count_between_reset=0,
    )

    push_robot = Term(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        params={"velocity_range": {"x": [-1, 1], "y": [-1, 1]}},
        is_global_time=False,
        interval_range_s=[10, 15],
        min_step_count_between_reset=0,
    )


from isaaclab_tasks.contrib.velocity.config.vel_mdp.contact_terrain import reset_root_state_on_terrain


@configclass
class G1RoughEventCfg(G1EventCfg):
    reset_base = G1EventCfg().reset_base.replace(func=reset_root_state_on_terrain)
