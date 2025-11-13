from __future__ import annotations


import math
from dataclasses import MISSING

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass


import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from pathlib import Path

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import UniformVelocityCommandPlsCfg, PhaseCommandCfg


class UniformVelocityCommandPls(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: UniformVelocityCommandPlsCfg

    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandPlsCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.is_rotonly_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # get env
        self.env = env

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        temp = r.uniform_(0.0, 1.0)
        self.is_standing_env[env_ids] = temp <= self.cfg.rel_standing_envs
        self.is_rotonly_env[env_ids] = temp <= self.cfg.rel_standing_envs + self.cfg.rel_rotonly_envs

        self.env.command_manager.get_term("phase")._resample_command(env_ids)  # force resample phase command

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs and zero linear velocity for rotonly envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        rotonly_env_ids = self.is_rotonly_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0  # zero all vel
        self.vel_command_b[rotonly_env_ids, :2] = 0.0  # zero linear vel

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_linvel_viz = VisualizationMarkers(self.cfg.goal_linvel_viz_cfg)
                self.goal_angvel_viz = VisualizationMarkers(self.cfg.goal_angvel_viz_cfg)
                # -- current
                self.curr_linvel_viz = VisualizationMarkers(self.cfg.current_linvel_viz_cfg)
                self.curr_angvel_viz = VisualizationMarkers(self.cfg.current_angvel_viz_cfg)
            # set their visibility to true
            self.goal_linvel_viz.set_visibility(True)
            self.goal_angvel_viz.set_visibility(True)
            self.curr_linvel_viz.set_visibility(True)
            self.curr_angvel_viz.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_linvel_viz.set_visibility(False)
                self.goal_angvel_viz.set_visibility(False)
                self.curr_linvel_viz.set_visibility(False)
                self.curr_angvel_viz.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # -- resolve linear velocity arrows (in xy plane)
        linvel_des_scale, linvel_des_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        linvel_curr_scale, linvel_curr_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])

        # -- resolve angular velocity arrows (around z axis)
        angvel_des_scale, angvel_des_quat = self._resolve_z_angvel_to_arrow(self.command[:, 2])
        angvel_curr_scale, angvel_curr_quat = self._resolve_z_angvel_to_arrow(self.robot.data.root_ang_vel_b[:, 2])

        # display markers
        self.goal_linvel_viz.visualize(base_pos_w, linvel_des_quat, linvel_des_scale)
        self.curr_linvel_viz.visualize(base_pos_w, linvel_curr_quat, linvel_curr_scale)

        # offset angular velocity arrows slightly higher
        angvel_pos_w = base_pos_w.clone()
        angvel_pos_w[:, 2] += 0.5
        self.goal_angvel_viz.visualize(angvel_pos_w, angvel_des_quat, angvel_des_scale)
        self.curr_angvel_viz.visualize(angvel_pos_w, angvel_curr_quat, angvel_curr_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_linvel_viz.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _resolve_z_angvel_to_arrow(self, z_angvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the Z angular velocity to arrow pointing in +/- z direction."""
        # obtain default scale of the marker
        default_scale = self.goal_angvel_viz.cfg.markers["arrow"].scale
        # arrow-scale based on angular velocity magnitude
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(z_angvel.shape[0], 1)
        arrow_scale[:, 0] *= torch.abs(z_angvel) * 2.0
        # arrow direction: point up (+z) for positive angvel (ccw), down (-z) for negative angvel (cw)
        # rotate around y-axis: +90 degrees for +z, -90 degrees for -z
        pitch_angle = torch.where(z_angvel < 0, torch.full_like(z_angvel, torch.pi / 2), torch.full_like(z_angvel, -torch.pi / 2))
        zeros = torch.zeros_like(pitch_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, pitch_angle, zeros)
        # convert from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


class PhaseCommand(CommandTerm):
    """
    A phase command that repeats from [0, 1].
    """

    def __init__(self, cfg: PhaseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.step_dt = env.step_dt
        self.phase_dt = torch.ones_like(self.phase, device=self.device) / torch.empty(self.num_envs, device=self.device).uniform_(
            cfg.phase_frequency_hz_range[0],
            cfg.phase_frequency_hz_range[1]
        )
        self.phase_increment_per_step = torch.full_like(self.phase, self.step_dt) / self.phase_dt
        self.phase_max = torch.ones_like(self.phase)

    def set_phase(self, phase: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Set phase externally."""
        self.phase[env_ids] = phase

    def _update_command(self) -> None:
        self.phase += self.phase_increment_per_step  # advance by an amount of a step
        self.phase = torch.fmod(self.phase, self.phase_max)  # regulate to [0, 1]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # resample phase freq and dt
        self.phase_dt = torch.ones_like(self.phase, device=self.device) / torch.empty(self.num_envs, device=self.device).uniform_(
            self.cfg.phase_frequency_hz_range[0],
            self.cfg.phase_frequency_hz_range[1]
        )
        self.phase_increment_per_step = torch.full_like(self.phase, self.step_dt) / self.phase_dt

    def _update_metrics(self):
        pass  # no metrics

    @property
    def command(self) -> torch.Tensor:
        return self.phase