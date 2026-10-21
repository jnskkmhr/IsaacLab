# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-supported base-pose goals and pose tolerances."""

import math

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.configclass import configclass


@configclass
class TerrainPoseCommandCfg(CommandTermCfg):
    """Command a world-space base position and yaw for one complete episode."""

    class_type: str = "{DIR}.commands:TerrainPoseCommand"
    asset_name: str = "robot"
    foot_body_names: str = ".*_ankle_roll_link"
    target_patch_name: str = "target"
    distance_range: tuple[float, float] = (0.75, 3.5)
    """Horizontal target distance from the base at reset, in meters."""
    heading_range: tuple[float, float] = (-math.pi, math.pi)
    base_height: float = 0.75
    """Target base Z above the sampled terrain surface, in meters."""
    position_tolerance: float = 0.2
    height_tolerance: float = 0.08
    heading_tolerance: float = 0.2
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = True
    goal_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/reach_goal",
        markers={"frame": FRAME_MARKER_CFG.markers["frame"].replace(scale=(0.3, 0.3, 0.3))},
    )
