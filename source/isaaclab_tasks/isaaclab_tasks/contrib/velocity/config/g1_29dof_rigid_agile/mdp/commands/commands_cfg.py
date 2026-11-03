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
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    pass

from isaaclab.envs.mdp import UniformVelocityCommandCfg

from .phase_command import PhaseCommand, PhaseCommandDSP, PhaseCommandFLT, PhaseCommandSSP
from .swing_command import SwingCommand
from .velocity_command import UniformLevelVelocityCommand, UniformNullVelocityCommand, UniformVelocityYawCommand


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


@configclass
class UniformNullVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformNullVelocityCommand

    ema_smoothing_param: float = 1.0
    r"""Parameters for the exponential moving average smoothing of the command. It is used to smooth the
    velocity of the robot:

    .. math::

        \text{vel_robot_smoothed} = \text{ema_smoothing_params} \times \text{vel_robot} + (1 - \text{ema_smoothing_params}) \times \text{vel_command}

    where :math:`\text{vel_robot}` is the velocity of the robot, :math:`\text{vel_command}` is the command velocity,
    and :math:`\text{ema_smoothing_params}` is the smoothing parameter.

    1.0 means no smoothing.
    """  # noqa: E501, W605

    min_vel_norm: float = 0.1
    """Minimum velocity norm,velocity commands with a norm less than this value are set to 0"""

    bias_sampling: bool = False
    """Enable biased sampling toward lower speeds. When True, samples velocities from a piecewise
    uniform distribution that favors low speeds."""

    bias_sampling_probability: float = 0.6
    """Probability of sampling from the low-speed range when bias_sampling is True.
    For example, 0.5 means 50% of samples come from the low-speed range."""

    bias_sampling_speed: float = 0.2
    """Maximum speed for the low-speed range when bias_sampling is True (in m/s for linear velocities,
    rad/s for angular velocities). Commands are sampled from [-bias_sampling_speed, bias_sampling_speed]
    with probability bias_sampling_probability."""

    command_filter_alpha_ranges: dict[str, tuple[float, float] | None] | None = None
    """Per-axis low-pass filter coefficient ranges for smooth command transitions.

    Keys are axis names that correspond to command dimensions.
    Base class supports: 'lin_vel_x', 'lin_vel_y', 'ang_vel_z'
    Subclasses can add additional axes (e.g., 'base_height' for height commands).

    Values are tuples (min_alpha, max_alpha) for random sampling per environment.
    Alpha values should be in [0, 1] where:
    - 1.0 = no filtering (instant response)
    - 0.0 = maximum filtering (no change)
    - Typical values: 0.05-0.3 for smooth transitions

    Example:
        command_filter_alpha_ranges = {
            "lin_vel_x": (0.05, 0.2),  # Smooth linear x velocity
            "lin_vel_y": (0.05, 0.2),  # Smooth linear y velocity
            "ang_vel_z": (0.1, 0.3),   # Slightly faster angular response
            # "base_height": (0.02, 0.1),  # Height filtering (if using height commands)
        }

    If not provided or empty dict, no filtering is applied (backward compatible).
    Subclasses handle their additional axes independently.
    """
