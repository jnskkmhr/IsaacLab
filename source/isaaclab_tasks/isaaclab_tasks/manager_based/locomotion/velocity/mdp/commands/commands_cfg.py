# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    pass

from isaaclab.envs.mdp import UniformVelocityCommandCfg

from .phase_command import PhaseCommand, PhaseCommandDSP, PhaseCommandFLT, PhaseCommandSSP
from .swing_command import SwingCommand
from .velocity_command import UniformLevelVelocityCommand, UniformVelocityYawCommand


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformLevelVelocityCommand

    goal_linvel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""
    goal_angvel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/angvel_goal"
    )
    """The configuration for the goal angvel visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_linvel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""
    current_angvel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/angvel_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_linvel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
    goal_angvel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
    current_linvel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)
    current_angvel_visualizer_cfg.markers["arrow"].scale = (0.4, 0.4, 0.4)


@configclass
class UniformVelocityYawCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = UniformVelocityYawCommand
    goal_heading_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/heading_goal"
    )
    """The configuration for the goal heading visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""


GAIT_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.2,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    },
)


@configclass
class PhaseCommandCfg(CommandTermCfg):
    class_type: type = PhaseCommand
    step_dt: float = 0.02
    gait_period: tuple[float, float] = (0.4, 1.0)
    sampler: Literal["uniform", "discrete"] = "uniform"
    resample: bool = True
    ss_duration_phase: float = 0.45
    ds_duration_phase: float = 0.05
    ss_duration_phase_running: float = 0.45
    flight_duration_phase_running: float = 0.05
    velocity_command_name: str = "base_velocity"


@configclass
class PhaseCommandSSPCfg(CommandTermCfg):
    class_type: type = PhaseCommandSSP
    asset_name: str = "robot"
    body_names: list[str] = [".*ankle_roll.*"]
    step_dt: float = 0.02
    gait_period: tuple[float, float] = (0.4, 1.0)
    sampler: Literal["uniform", "discrete"] = "uniform"
    resample: bool = True

    velocity_command_name: str = "base_velocity"
    gait_visualizer_cfg: VisualizationMarkersCfg = GAIT_MARKER_CFG.replace(prim_path="/Visuals/Command/gait")

    ss_duration_phase: float = 0.5


@configclass
class PhaseCommandDSPCfg(CommandTermCfg):
    class_type: type = PhaseCommandDSP
    asset_name: str = "robot"
    body_names: list[str] = [".*ankle_roll.*"]
    step_dt: float = 0.02
    gait_period: tuple[float, float] = (0.4, 1.0)
    sampler: Literal["uniform", "discrete"] = "uniform"
    resample: bool = True

    velocity_command_name: str = "base_velocity"
    gait_visualizer_cfg: VisualizationMarkersCfg = GAIT_MARKER_CFG.replace(prim_path="/Visuals/Command/gait")

    ss_duration_phase: float = 0.45
    ds_duration_phase: float = 0.05


@configclass
class PhaseCommandFLTCfg(CommandTermCfg):
    class_type: type = PhaseCommandFLT
    asset_name: str = "robot"
    body_names: list[str] = [".*ankle_roll.*"]
    step_dt: float = 0.02
    gait_period: tuple[float, float] = (0.4, 1.0)
    sampler: Literal["uniform", "discrete"] = "uniform"
    resample: bool = True

    velocity_command_name: str = "base_velocity"
    gait_visualizer_cfg: VisualizationMarkersCfg = GAIT_MARKER_CFG.replace(prim_path="/Visuals/Command/gait")

    ss_duration_phase: float = 0.5
    max_flt_duration_phase: float = 0.15
    velocity_flight_threshold: float = 1.5  # flight phase will be triggered if velocity is above this threshold
    velocity_flight_maximum: float = 2.5  # max flight duration at this speed


@configclass
class SwingCommandCfg(CommandTermCfg):
    class_type: type = SwingCommand
    foot_height: tuple[float, float] = (0.08, 0.15)
    resample: bool = True
