# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direction commands and teacher contact posture in explicit coordinate frames."""

import math
from pathlib import Path

from isaaclab.managers import CommandTermCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .. import mdp
from .scene_cfg import SAND_DEPTH


@configclass
class PushDirectionCommandCfg(CommandTermCfg):
    class_type = mdp.PushDirectionCommand
    asset_name = "robot"
    object_name = "object"
    reference_body_name = "torso_link"
    workspace_path = str(Path(__file__).resolve().parents[1] / "data/hand_workspace.npz")
    workspace_orientation_weight = 0.02
    contact_body_names = ".*(wrist_yaw|hand_.*)_link"
    contact_force_threshold = 1.0
    direction_visualizer_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/object_direction")
    hand_body_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    debug_vis = True
    resampling_time_range = (6.0, 10.0)
    angle_range = (-math.pi / 4, math.pi / 4)
    standing_probability = 0.2
    forward_speed = 0.8
    steering_gain = 1.0
    max_yaw_rate = 0.5
    neutral_distance = 0.85
    contact_distance = 0.65
    posture_transition_s = 0.5
    neutral_height = 0.80
    height_reference_offset = SAND_DEPTH
    """Rigid-floor depth below the initial sand surface, matching the MPM teacher [m]."""
    object_center_height_offset = 0.20
    contact_height_range = (0.75, 0.80)
    contact_pitch = 0.0  # The native-MPM teacher has no pitch-command input.
    # XYZW wrist poses in torso_link; palms face forward with bent elbows.
    neutral_hand_poses = [(0.20, 0.18, -0.05, 0.0, 0.0, 0.0, 1.0), (0.20, -0.18, -0.05, 0.0, 0.0, 0.0, 1.0)]
    # Object-local wrist points: cube half length 0.325 + closed-hand offset 0.130.
    contact_points = [(-0.455, 0.20, 0.30), (-0.455, -0.20, 0.30)]


@configclass
class CommandsCfg:
    object_direction = PushDirectionCommandCfg()
