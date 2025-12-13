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
import isaaclab_tasks.manager_based.locomotion.velocity.config.t1_soft.mdp as t1_mdp


@configclass
class T1EventsCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    add_end_effector_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_link", "right_hand_link"]),
            "mass_distribution_params": (0.0, 2.0),
            "operation": "add",
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Waist",
                    "Left_Hip_Pitch",
                    "Left_Hip_Roll",
                    "Left_Hip_Yaw",
                    "Left_Knee_Pitch",
                    "Left_Ankle_Pitch",
                    "Left_Ankle_Roll",
                    "Right_Hip_Pitch",
                    "Right_Hip_Roll",
                    "Right_Hip_Yaw",
                    "Right_Knee_Pitch",
                    "Right_Ankle_Pitch",
                    "Right_Ankle_Roll",
                ],
            ),
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque, # type: ignore
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, # type: ignore
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,  # BoosterGym uses gaussian but we use uniform.
        mode="reset",
        params={
            "position_range": (0., 0.05),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
    reset_unactuated_joints = EventTerm(
        func=vel_mdp.reset_joints_target_by_offset, # reset target position of unactuated joints to default plus offset
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "AAHead_yaw",
                    "Head_pitch",
                    "Left_Shoulder_Pitch",
                    "Left_Shoulder_Roll",
                    "Left_Elbow_Pitch",
                    "Left_Elbow_Yaw",

                    # "Left_Wrist_Pitch",
                    # "Left_Wrist_Yaw",
                    # "Left_Hand_Roll",

                    "Right_Shoulder_Pitch",
                    "Right_Shoulder_Roll",
                    "Right_Elbow_Pitch",
                    "Right_Elbow_Yaw",

                    # "Right_Wrist_Pitch",
                    # "Right_Wrist_Yaw",
                    # "Right_Hand_Roll",
                ],
            ),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity, # type: ignore
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    # randomize terrain friction
    randomize_friction = EventTerm(
        func=t1_mdp.randomize_terrain_friction, # type: ignore
        mode="reset",
        params={
            "friction_range": (0.1, 1.0),
            "contact_solver_name": "physics_callback",
        },
    )

    # randomize terrain stiffness
    randomize_stiffness = EventTerm(
        func=t1_mdp.randomize_terrain_stiffness, # type: ignore
        mode="reset",
        params={
            "stiffness_range": (0.2, 0.9),
            "contact_solver_name": "physics_callback",
        },
    )