# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Commands supplied to the student and its two teachers."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.push.config.g1_29dof_locomotion.mdp.commands import UniformHeightCommand

from ..mdp.commands import BoxVelocityCommandCfg, NearestBoxFaceCommandCfg
from .scene_cfg import BOX_SIZE

STANDING_COMMAND_THRESHOLD = 0.05


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Uniform pelvis height above the flat ground in meters."""

    class_type = UniformHeightCommand
    asset_name: str = "robot"
    height_range: tuple[float, float] = (0.75, 0.75)
    current_marker_xy_offset: tuple[float, float] = (0.0, 0.45)
    """World XY offset [m] from the root for the measured-height marker; Z is unchanged."""
    goal_marker_xy_offset: tuple[float, float] = (0.0, 0.55)
    """World XY offset [m] from the root for the commanded-height marker."""
    current_height_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/height_current",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    goal_height_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/height_goal",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )


@configclass
class CommandsCfg:
    base_height = UniformHeightCommandCfg(resampling_time_range=(10.0, 10.0), debug_vis=False)
    box_velocity = BoxVelocityCommandCfg(
        asset_name="robot",
        object_name="box",
        resampling_time_range=(4.0, 6.0),
        rel_standing_envs=0.2,
        heading_command=False,
        debug_vis=True,
        vel_xy_success_threshold=0.1,
        vel_yaw_success_threshold=0.1,
        ranges=BoxVelocityCommandCfg.Ranges(lin_vel_x=(0.1, 1.0), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.2, 0.2)),
    )
    hand_pose = NearestBoxFaceCommandCfg(
        velocity_command_name="box_velocity",
        command_threshold=STANDING_COMMAND_THRESHOLD,
        standing_quaternions=((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        reference_body_name="torso_link",
        dataset_path=str(Path(__file__).resolve().parents[2] / "g1_29dof_manipulation/data/hand_poses_closed.npz"),
        pose_keys=("left_pose", "right_pose"),
        workspace_pose_count=4096,
        orientation_weight=0.1,
        approach_positions=((0.25, 0.2, 0.05), (0.25, -0.2, 0.05)),
        approach_distance=0.65,
        contact_distance=0.45,
        hand_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
        contact_offset=(0.0, 0.0, 0.0),
        box_size=BOX_SIZE,
        lateral_offset=0.2,
        top_clearance=0.05,
        wrist_surface_offset=0.145,  # Closed-hand mesh extent along wrist +X [m].
        debug_vis=False,
    )
