# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.g1_soft.mdp as g1_mdp


@configclass
class G1EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
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
        func=mdp.randomize_rigid_body_mass, 
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com, 
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=".*"
            ),
            "operation": "scale",
            "distribution": "uniform",
            # "stiffness_distribution_params": (0.95, 1.05),
            # "damping_distribution_params": (0.95, 1.05),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, 
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

    """
    when fixing upper body
    """
    # reset_unactuated_joints = EventTerm(
    #     func=vel_mdp.reset_joints_target_by_offset, # reset target position of unactuated joints to default plus offset
    #     mode="reset",
    #     params={
    #         "position_range": (0.0, 0.0),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 # waist
    #                 "waist_yaw_joint", 
    #                 "waist_roll_joint", 
    #                 "waist_pitch_joint", 

    #                 # arms
    #                 "left_shoulder_pitch_joint", 
    #                 "left_shoulder_roll_joint", 
    #                 "left_shoulder_yaw_joint", 
    #                 "left_elbow_joint", 

    #                 "left_wrist_roll_joint", 
    #                 "left_wrist_pitch_joint", 
    #                 "left_wrist_yaw_joint", 

    #                 "right_shoulder_pitch_joint", 
    #                 "right_shoulder_roll_joint", 
    #                 "right_shoulder_yaw_joint", 
    #                 "right_elbow_joint", 

    #                 "right_wrist_roll_joint", 
    #                 "right_wrist_pitch_joint", 
    #                 "right_wrist_yaw_joint", 
    #             ],
    #         ),
    #     },
    # )

    # randomize terrain friction
    randomize_friction = EventTerm(
        func=g1_mdp.randomize_terrain_friction, 
        mode="reset",
        params={
            "friction_range": (0.1, 1.0),
            "contact_solver_name": "physics_callback",
        },
    )

    # randomize terrain stiffness
    randomize_stiffness = EventTerm(
        func=g1_mdp.randomize_terrain_stiffness,
        mode="reset",
        params={
            "stiffness_range": (0.2, 0.9),
            "contact_solver_name": "physics_callback",
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )