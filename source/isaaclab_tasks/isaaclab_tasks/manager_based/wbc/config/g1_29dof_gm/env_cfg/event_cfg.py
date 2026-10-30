# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
import isaaclab_tasks.manager_based.wbc.mdp as wbc_mdp
# TODO: bundle this mdp into vel_dmp or g1_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.g1_29dof.mdp as g1_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.g1_29dof_soft.mdp as g1_soft_mdp

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

@configclass
class G1EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=wbc_mdp.randomize_rigid_body_material, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    # NOTE: robust policy randomization
    # add_joint_default_pos = EventTerm(
    #     func=wbc_mdp.randomize_joint_default_pos,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    #         "pos_distribution_params": (-0.01, 0.01),
    #         "operation": "add",
    #         "joint_action_name": "joint_pos",
    #     },
    # )

    # base_com = EventTerm(
    #     func=wbc_mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #         "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
    #     },
    # )

    # # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 3.0),
    #     params={"velocity_range": VELOCITY_RANGE},
    # )
    
    # NOTE: soft terrain specific terrain randomization
    randomize_friction = EventTerm(
        func=g1_mdp.randomize_terrain_friction,
        mode="reset",
        params={
            "friction_range": (0.1, 1.0),
            "contact_solver_name": "physics_callback",
        },
    )

    randomize_stiffness = EventTerm(
        func=g1_mdp.randomize_terrain_stiffness,
        mode="reset",
        params={
            "stiffness_range": (0.2, 0.9),
            "contact_solver_name": "physics_callback",
        },
    )

    randomize_material_density = EventTerm(
        func=g1_soft_mdp.randomize_material_density,
        mode="reset",
        params={
            "packing_ratio_range": (0.5, 1.0),
            "bulk_density_range": (1000.0, 3000.0),
            "contact_solver_name": "physics_callback",
        },
    )