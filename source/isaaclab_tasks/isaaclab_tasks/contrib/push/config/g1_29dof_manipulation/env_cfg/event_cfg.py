# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint resets and waist targets for the fixed-base upper-body robot."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..mdp.events import randomize_joint_position_targets
from .action_cfg import UPPER_JOINTS
from .scene_cfg import CLOSED_HAND_JOINT_POS


@configclass
class EventsCfg:
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_JOINTS),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_fingers = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hand_.*_joint"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    hold_closed_hands = EventTerm(
        func=randomize_joint_position_targets,
        mode="reset",
        params={"position_ranges": {name: (position, position) for name, position in CLOSED_HAND_JOINT_POS.items()}},
    )

    reset_waist = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_waist_targets = EventTerm(
        func=randomize_joint_position_targets,
        mode="reset",
        params={"position_ranges": {"waist_.*_joint": (0.0, 0.0)}},
    )
    randomize_waist_targets = EventTerm(
        func=randomize_joint_position_targets,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "position_ranges": {
                "waist_yaw_joint": (-0.1, 0.1),
                "waist_roll_joint": (-0.15, 0.15),
                "waist_pitch_joint": (-0.15, 0.15),
            },
        },
    )
