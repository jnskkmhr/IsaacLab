# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint resets and waist targets for the fixed-base upper-body robot."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .. import mdp
from .action_cfg import UPPER_JOINTS
from .scene_cfg import CLOSED_HAND_JOINT_POS


@configclass
class EventsCfg:
    # Collision-checked near-torso arm pose (closed-hand dataset row 60033).
    # Keep asset default_joint_pos unchanged for checkpoint-compatible observations/actions.
    reset_joints = EventTerm(
        func=mdp.reset_joint_positions,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_JOINTS, preserve_order=True),
            "positions": (
                0.5613694,
                0.4621537,
                -0.16452092,
                -0.41065788,
                -0.51701266,
                -0.26454014,
                0.020170113,
                0.5613694,
                -0.4621537,
                0.16452092,
                -0.41065788,
                0.51701266,
                -0.26454014,
                -0.020170113,
            ),
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
        func=mdp.randomize_joint_position_targets,
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
        func=mdp.randomize_joint_position_targets,
        mode="reset",
        params={"position_ranges": {"waist_.*_joint": (0.0, 0.0)}},
    )
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object"), "pose_range": {}, "velocity_range": {}},
    )
