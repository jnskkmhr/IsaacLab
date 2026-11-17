# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp as g1_mdp

from ..mdp.commands import UniformHeightCommand


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Uniform pelvis height above the environment origin in meters."""

    class_type = UniformHeightCommand
    asset_name: str = "robot"
    height_range: tuple[float, float] = (0.60, 0.80)
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
class G1CommandsCfg:
    """Command specifications for the MDP."""

    base_height = UniformHeightCommandCfg(resampling_time_range=(10.0, 10.0), debug_vis=True)

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
