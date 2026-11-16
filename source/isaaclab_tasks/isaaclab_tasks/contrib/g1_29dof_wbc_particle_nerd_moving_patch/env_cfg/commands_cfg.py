# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp

from .. import mdp


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Uniform pelvis height above the initial terrain surface in meters."""

    class_type = mdp.UniformHeightCommand
    asset_name: str = "robot"
    sensor_name: str = "height_scanner"
    height_range: tuple[float, float] = (0.30, 0.80)
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
class UniformTorsoPitchCommandCfg(CommandTermCfg):
    """Forward torso lean relative to world vertical, with a zero-roll target."""

    class_type = mdp.UniformTorsoPitchCommand
    asset_name: str = "robot"
    body_name: str = "torso_link"
    pitch_range: tuple[float, float] = (0.0, math.radians(30.0))
    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.3)
    goal_visualizer_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/torso_pitch_goal")
    current_visualizer_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/torso_pitch_current")

    def __post_init__(self):
        self.goal_visualizer_cfg.markers["arrow"].scale = (0.4, 0.08, 0.08)
        self.current_visualizer_cfg.markers["arrow"].scale = (0.4, 0.08, 0.08)


@configclass
class G1CommandsCfg:
    """Command specifications for the MDP."""

    torso_pitch = UniformTorsoPitchCommandCfg(
        resampling_time_range=(10.0, 10.0), debug_vis=True, pitch_range=(0.0, math.radians(30.0))
    )

    base_height = UniformHeightCommandCfg(resampling_time_range=(10.0, 10.0), debug_vis=True, height_range=(0.5, 0.8))

    base_velocity = g1_mdp.UniformVelocityYawCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=g1_mdp.UniformVelocityYawCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )
