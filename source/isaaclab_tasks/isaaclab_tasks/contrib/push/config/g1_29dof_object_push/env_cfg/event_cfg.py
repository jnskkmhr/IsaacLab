# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the initial rigid-ground pushing stage."""

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

import isaaclab_tasks.core.velocity.mdp as core_mdp
from isaaclab_tasks.contrib.push import mdp
from isaaclab_tasks.contrib.push.config.g1_29dof_manipulation.mdp.events import randomize_joint_position_targets

from .scene_cfg import BOX_SIZE, ROBOT_CFG


@configclass
class EventsCfg:
    """Reset the robot before placing the box ahead of it."""

    physics_material = None
    add_base_mass = None
    reset_base = EventTermCfg(
        func=core_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTermCfg(
        func=core_mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    hold_closed_hands = EventTermCfg(
        func=randomize_joint_position_targets,
        mode="reset",
        params={
            "position_ranges": {
                name: (position, position)
                for name, position in ROBOT_CFG.init_state.joint_pos.items()
                if "_hand_" in name
            },
        },
    )
    scale_actuator_gains = None
    push_robot = None
    reset_box = EventTermCfg(
        func=mdp.reset_box,
        mode="reset",
        params={
            "distance_range": (BOX_SIZE[0] / 2 + 0.25, BOX_SIZE[0] / 2 + 1.25),
            "lateral_range": (-0.02, 0.02),
            "yaw_range": (-0.1, 0.1),
        },
    )
