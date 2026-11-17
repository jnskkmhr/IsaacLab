# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aligned robot/object resets; recycled particle states."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch import mdp as particle_mdp

from .. import mdp as push_mdp


@configclass
class EventsCfg:
    reset_base = EventTerm(
        func=particle_mdp.reset_root_state_on_terrain,
        mode="reset",
        params={"pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.2, 0.2)}, "velocity_range": {}},
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    hold_closed_hands = EventTerm(
        func=push_mdp.hold_hands,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hand_.*_joint")},
    )
    reset_object = EventTerm(
        func=push_mdp.reset_object,
        mode="reset",
        params={
            "object_name": "object",
            "robot_name": "robot",
            "forward_range": (0.9, 1.0),
            "lateral_range": (-0.05, 0.05),
            "half_height": 0.325,
        },
    )
