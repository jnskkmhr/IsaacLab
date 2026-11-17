# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independently sampled wrist pose commands in the moving torso frame."""

from pathlib import Path

from isaaclab.envs.mdp import UniformPoseCommandCfg
from isaaclab.utils import configclass

from ..mdp.commands import DatasetPoseCommand, LinkPoseCommand


@configclass
class LinkPoseCommandCfg(UniformPoseCommandCfg):
    """Uniform pose command relative to the specified robot link."""

    class_type = LinkPoseCommand
    reference_body_name: str = "torso_link"


@configclass
class DatasetPoseCommandCfg(LinkPoseCommandCfg):
    """Read SE(3) targets from a validated dataset instead of Cartesian sampling ranges."""

    class_type = DatasetPoseCommand
    ranges = None
    dataset_path: str = str(Path(__file__).resolve().parents[1] / "data" / "hand_poses_closed.npz")
    pose_key: str = "left_pose"


@configclass
class CommandsCfg:
    # Each hand samples its own dataset row and 3–5 second timer in each environment.
    left_ee_pose = DatasetPoseCommandCfg(
        asset_name="robot",
        body_name="left_wrist_yaw_link",
        resampling_time_range=(3.0, 5.0),
        debug_vis=True,
        pose_key="left_pose",
    )
    right_ee_pose = DatasetPoseCommandCfg(
        asset_name="robot",
        body_name="right_wrist_yaw_link",
        resampling_time_range=(3.0, 5.0),
        debug_vis=True,
        pose_key="right_pose",
    )
    # Previous uniform Cartesian sampling alternative, relative to torso_link.
    # To use it, replace both dataset terms above with these terms.
    # left_ee_pose = LinkPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="left_wrist_yaw_link",
    #     resampling_time_range=(3.0, 5.0),
    #     debug_vis=True,
    #     ranges=UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.1, 0.3),
    #         pos_y=(0.15, 0.35),
    #         pos_z=(-0.1, 0.1),
    #         roll=(-0.2, 0.2),
    #         pitch=(-0.2, 0.2),
    #         yaw=(-0.2, 0.2),
    #     ),
    # )
    # right_ee_pose = LinkPoseCommandCfg(
    #     asset_name="robot",
    #     body_name="right_wrist_yaw_link",
    #     resampling_time_range=(3.0, 5.0),
    #     debug_vis=True,
    #     ranges=UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.1, 0.3),
    #         pos_y=(-0.35, -0.15),
    #         pos_z=(-0.1, 0.1),
    #         roll=(-0.2, 0.2),
    #         pitch=(-0.2, 0.2),
    #         yaw=(-0.2, 0.2),
    #     ),
    # )

    left_ee_pose.goal_pose_visualizer_cfg.prim_path = "/Visuals/left_goal"
    left_ee_pose.current_pose_visualizer_cfg.prim_path = "/Visuals/left_current"
    right_ee_pose.goal_pose_visualizer_cfg.prim_path = "/Visuals/right_goal"
    right_ee_pose.current_pose_visualizer_cfg.prim_path = "/Visuals/right_current"
