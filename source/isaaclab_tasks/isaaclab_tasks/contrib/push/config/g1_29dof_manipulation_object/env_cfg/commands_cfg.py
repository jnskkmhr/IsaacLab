# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simultaneous forward wrist strokes in the torso frame."""

from isaaclab.envs.mdp import UniformPoseCommandCfg
from isaaclab.utils import configclass

from .. import mdp


@configclass
class ForwardPoseCommandCfg(UniformPoseCommandCfg):
    class_type = mdp.ForwardPoseCommand
    reference_body_name: str = "torso_link"
    ranges = None
    start_position: tuple[float, float, float] = (0.08, 0.18, 0.05)
    end_position: tuple[float, float, float] = (0.32, 0.18, 0.05)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    hold_time: float = 2.0
    duration: float = 4.0


@configclass
class CommandsCfg:
    left_ee_pose = ForwardPoseCommandCfg(
        asset_name="robot",
        body_name="left_wrist_yaw_link",
        resampling_time_range=(1000.0, 1000.0),
        debug_vis=True,
    )
    right_ee_pose = ForwardPoseCommandCfg(
        asset_name="robot",
        body_name="right_wrist_yaw_link",
        start_position=(0.08, -0.18, 0.05),
        end_position=(0.32, -0.18, 0.05),
        resampling_time_range=(1000.0, 1000.0),
        debug_vis=True,
    )
    left_ee_pose.goal_pose_visualizer_cfg.prim_path = "/Visuals/left_goal"
    left_ee_pose.current_pose_visualizer_cfg.prim_path = "/Visuals/left_current"
    right_ee_pose.goal_pose_visualizer_cfg.prim_path = "/Visuals/right_goal"
    right_ee_pose.current_pose_visualizer_cfg.prim_path = "/Visuals/right_current"
