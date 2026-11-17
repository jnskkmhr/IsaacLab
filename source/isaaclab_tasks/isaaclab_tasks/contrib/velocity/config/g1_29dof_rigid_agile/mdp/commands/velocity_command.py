# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from .commands_cfg import UniformNullVelocityCommandCfg

# import logger
logger = logging.getLogger(__name__)


class UniformLevelVelocityCommand(UniformVelocityCommand):
    """
    This class inherits from `UniformVelocityCommand` to
    - apply curriclum to lin/ang velocity sampling range
    - provide debug vis for both linear and angular velocity commands
    """

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_linvel_visualizer"):
                # -- goal
                self.goal_linvel_visualizer = VisualizationMarkers(self.cfg.goal_linvel_visualizer_cfg)
                self.goal_angvel_visualizer = VisualizationMarkers(self.cfg.goal_angvel_visualizer_cfg)
                # -- current
                self.curr_linvel_visualizer = VisualizationMarkers(self.cfg.current_linvel_visualizer_cfg)
                self.curr_angvel_visualizer = VisualizationMarkers(self.cfg.current_angvel_visualizer_cfg)

            # set their visibility to true
            self.goal_linvel_visualizer.set_visibility(True)
            self.goal_angvel_visualizer.set_visibility(True)
            self.curr_linvel_visualizer.set_visibility(True)
            self.curr_angvel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_linvel_visualizer"):
                self.goal_linvel_visualizer.set_visibility(False)
                self.goal_angvel_visualizer.set_visibility(False)
                self.curr_linvel_visualizer.set_visibility(False)
                self.curr_angvel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.torch.clone()
        base_pos_w[:, 2] += 0.5

        # -- resolve linear velocity arrows (in xy plane)
        linvel_des_scale, linvel_des_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        linvel_curr_scale, linvel_curr_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b.torch[:, :2]
        )

        # -- resolve angular velocity arrows (around z axis)
        angvel_des_scale, angvel_des_quat = self._resolve_z_angvel_to_arrow(self.command[:, 2])
        angvel_curr_scale, angvel_curr_quat = self._resolve_z_angvel_to_arrow(
            self.robot.data.root_ang_vel_b.torch[:, 2]
        )

        # display markers
        self.goal_linvel_visualizer.visualize(base_pos_w, linvel_des_quat, linvel_des_scale)
        self.curr_linvel_visualizer.visualize(base_pos_w, linvel_curr_quat, linvel_curr_scale)

        # offset angular velocity arrows slightly higher
        angvel_pos_w = base_pos_w.clone()
        angvel_pos_w[:, 2] += 0.5
        self.goal_angvel_visualizer.visualize(angvel_pos_w, angvel_des_quat, angvel_des_scale)
        self.curr_angvel_visualizer.visualize(angvel_pos_w, angvel_curr_quat, angvel_curr_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_linvel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w.torch
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _resolve_z_angvel_to_arrow(self, z_angvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the Z angular velocity to arrow pointing in +/- z direction."""
        # obtain default scale of the marker
        default_scale = self.goal_angvel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale based on angular velocity magnitude
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(z_angvel.shape[0], 1)
        arrow_scale[:, 0] *= torch.abs(z_angvel) * 2.0
        # arrow direction: point up (+z) for positive angvel (ccw), down (-z) for negative angvel (cw)
        # rotate around y-axis: +90 degrees for +z, -90 degrees for -z
        pitch_angle = torch.where(
            z_angvel < 0, torch.full_like(z_angvel, torch.pi / 2), torch.full_like(z_angvel, -torch.pi / 2)
        )
        zeros = torch.zeros_like(pitch_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, pitch_angle, zeros)
        # convert from base to world frame
        base_quat_w = self.robot.data.root_quat_w.torch
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


class UniformVelocityYawCommand(UniformVelocityCommand):
    """
    This class inherits from `UniformVelocityCommand` to
    - apply curriclum to lin/ang velocity sampling range
    - provide debug vis for both linear and angular velocity commands
    """

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # check configuration
        if self.cfg.heading_command:
            logger.warning("heading_command argument is not used in this class. Set it to False.")
        if self.cfg.ranges.heading:
            logger.warning("heading range is not used in this class. Set it to None.")

        # -- metrics
        self.heading_target = torch.zeros((self.num_envs, 4), device=self.device)
        # identity quaternion, whose real part is the last element in the (x, y, z, w) convention
        self.heading_target[:, 3] = 1.0
        self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 7)."""
        return torch.cat([self.vel_command_b, self.heading_target], dim=-1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # The parent finalizes velocity errors and success rate from episode accumulators at reset.
        super()._update_metrics()
        # Preserve command-duration normalization for the additional heading metric.
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_yaw"] += (
            torch.abs(
                math_utils.quat_error_magnitude(
                    math_utils.yaw_quat(self.robot.data.root_quat_w.torch), self.heading_target
                )
            )
            / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor | slice):
        # sample velocity commands
        r = torch.empty(self.vel_command_b[env_ids].shape[0], device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # # -- heading target: reset to the current orientation
        self.heading_target[env_ids] = math_utils.yaw_quat(self.robot.data.root_quat_w.torch[env_ids])

        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

        # integrate angular velocity to get heading target
        # if current heading deviates too much from target, reset target to current heading
        current_heading = math_utils.yaw_quat(self.robot.data.root_quat_w.torch)
        heading_error = math_utils.quat_error_magnitude(current_heading, self.heading_target)
        reset_env_ids = heading_error > torch.pi / 2
        self.heading_target[reset_env_ids] = current_heading[reset_env_ids]
        # self.heading_target[standing_env_ids] = current_heading[standing_env_ids]  # BUG: yaw keep deviating??

        yaw_ang_vel_cmd = self.vel_command_b[:, 2]
        delta_yaw = yaw_ang_vel_cmd * self._env.step_dt
        delta_quat = math_utils.quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            delta_yaw,
        )
        self.heading_target[:] = math_utils.quat_unique(math_utils.quat_mul(delta_quat, self.heading_target))

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_linvel_visualizer"):
                # -- goal
                self.goal_linvel_visualizer = VisualizationMarkers(self.cfg.goal_linvel_visualizer_cfg)
                self.goal_angvel_visualizer = VisualizationMarkers(self.cfg.goal_angvel_visualizer_cfg)
                self.goal_heading_visualizer = VisualizationMarkers(self.cfg.goal_heading_visualizer_cfg)
                # -- current
                self.curr_linvel_visualizer = VisualizationMarkers(self.cfg.current_linvel_visualizer_cfg)
                self.curr_angvel_visualizer = VisualizationMarkers(self.cfg.current_angvel_visualizer_cfg)
            # set their visibility to true
            self.goal_linvel_visualizer.set_visibility(True)
            self.goal_angvel_visualizer.set_visibility(True)
            self.goal_heading_visualizer.set_visibility(True)
            self.curr_linvel_visualizer.set_visibility(True)
            self.curr_angvel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_linvel_visualizer"):
                self.goal_linvel_visualizer.set_visibility(False)
                self.goal_angvel_visualizer.set_visibility(False)
                self.goal_heading_visualizer.set_visibility(False)
                self.curr_linvel_visualizer.set_visibility(False)
                self.curr_angvel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.torch.clone()
        base_pos_w[:, 2] += 0.5

        # -- resolve linear velocity arrows (in xy plane)
        linvel_des_scale, linvel_des_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        linvel_curr_scale, linvel_curr_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b.torch[:, :2]
        )

        # -- resolve angular velocity arrows (around z axis)
        angvel_des_scale, angvel_des_quat = self._resolve_z_angvel_to_arrow(self.command[:, 2])
        angvel_curr_scale, angvel_curr_quat = self._resolve_z_angvel_to_arrow(
            self.robot.data.root_ang_vel_b.torch[:, 2]
        )

        # display markers
        self.goal_linvel_visualizer.visualize(base_pos_w, linvel_des_quat, linvel_des_scale)
        self.curr_linvel_visualizer.visualize(base_pos_w, linvel_curr_quat, linvel_curr_scale)

        # offset angular velocity arrows slightly higher
        angvel_pos_w = base_pos_w.clone()
        angvel_pos_w[:, 2] += 0.5
        self.goal_angvel_visualizer.visualize(angvel_pos_w, angvel_des_quat, angvel_des_scale)
        self.curr_angvel_visualizer.visualize(angvel_pos_w, angvel_curr_quat, angvel_curr_scale)

        heading_pos_w = base_pos_w.clone()
        heading_pos_w[:, 2] += 0.5
        self.goal_heading_visualizer.visualize(
            heading_pos_w,
            self.heading_target,
            torch.tensor([0.5, 0.5, 0.5], device=self.device).repeat(self.num_envs, 1),
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        # -- heading target: reset to the current orientation
        self.heading_target[env_ids] = math_utils.yaw_quat(self.robot.data.root_quat_w.torch[env_ids])
        return extras

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_linvel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w.torch
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _resolve_z_angvel_to_arrow(self, z_angvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the Z angular velocity to arrow pointing in +/- z direction."""
        # obtain default scale of the marker
        default_scale = self.goal_angvel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale based on angular velocity magnitude
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(z_angvel.shape[0], 1)
        arrow_scale[:, 0] *= torch.abs(z_angvel) * 2.0
        # arrow direction: point up (+z) for positive angvel (ccw), down (-z) for negative angvel (cw)
        # rotate around y-axis: +90 degrees for +z, -90 degrees for -z
        pitch_angle = torch.where(
            z_angvel < 0, torch.full_like(z_angvel, torch.pi / 2), torch.full_like(z_angvel, -torch.pi / 2)
        )
        zeros = torch.zeros_like(pitch_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, pitch_angle, zeros)
        # convert from base to world frame
        base_quat_w = self.robot.data.root_quat_w.torch
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


def resolve_env_ids(env_ids, num_envs, device):
    """Materialize a slice selector as in the native Agile command term."""
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device)[env_ids]
    return env_ids


class UniformNullVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with min velocity command and traveled distance metric."""

    def __init__(self, cfg: UniformNullVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.metrics_cumulative: dict[str, torch.Tensor] = {}
        for k, v in self.metrics.items():  # type: ignore
            if "xy" in k:
                # store vectors not norms
                self.metrics_cumulative[k] = torch.zeros(env.num_envs, 2, device=self.device)
            else:
                self.metrics_cumulative[k] = torch.zeros_like(v)

        # for performance estimate
        self.metrics["traveled_distance"] = torch.zeros(env.num_envs, device=self.device)  # type: ignore
        self.traveled_distance = torch.zeros(env.num_envs, device=self.device)
        self.start_positions = self.robot.data.root_pos_w.torch[:, :2].clone()

        self.min_vel_norm = cfg.min_vel_norm

        # smoothed velocity estimate
        self.smoothing_param = cfg.ema_smoothing_param
        self.vel_xy_smoothed = torch.zeros_like(self.robot.data.root_lin_vel_b.torch[:, :2])
        self.angvel_smoothed = torch.zeros_like(self.robot.data.root_ang_vel_b.torch[:, 2])

        # Initialize command filtering
        self._setup_command_filtering(cfg, env)

    def _setup_command_filtering(self, cfg: UniformNullVelocityCommandCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize command filtering state and configuration."""
        # Dynamically build axis mapping based on available velocity command dimensions
        # The base class has 3 velocity commands: [lin_vel_x, lin_vel_y, ang_vel_z]
        # Subclasses may override this to add more axes
        self.filter_axis_map = self._get_command_axis_mapping()

        # Store filter configuration
        self.command_filter_config = cfg.command_filter_alpha_ranges if cfg.command_filter_alpha_ranges else {}

        # Get number of command dimensions from the actual command tensor
        num_command_dims = self.vel_command_b.shape[1]

        # Initialize filter coefficients (default to 1.0 = no filtering)
        self.command_filter_alphas = torch.ones(env.num_envs, num_command_dims, device=self.device)

        # Initialize target command buffer (unfiltered commands)
        self.vel_command_target_b = torch.zeros_like(self.vel_command_b)

    def _get_command_axis_mapping(self) -> dict[str, int]:
        """Get the mapping from axis names to indices in the command tensor.

        This method can be overridden by subclasses to define their own mappings.

        Returns:
            Dictionary mapping axis names to indices in vel_command_b tensor.

        Example for subclass with additional axes:
            def _get_command_axis_mapping(self):
                base_mapping = super()._get_command_axis_mapping()
                # Add height as 4th dimension if using combined tensor
                base_mapping["base_height"] = 3
                return base_mapping
        """
        # Base velocity commands mapping
        return {
            "lin_vel_x": 0,
            "lin_vel_y": 1,
            "ang_vel_z": 2,
        }

    def _sample_filter_coefficients(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Sample new filter coefficients for specified environments.

        Args:
            env_ids: Indices of environments to sample new coefficients for.
                     If None, no operation is performed.
        """
        # Early return if no environments specified
        if env_ids is None:
            return

        if not self.command_filter_config:
            # No filtering configured - set all alphas to 1.0 (instant response)
            self.command_filter_alphas[env_ids, :] = 1.0
            return

        # Sample per-axis filter coefficients
        for axis_name, axis_idx in self.filter_axis_map.items():
            if axis_name in self.command_filter_config and self.command_filter_config[axis_name] is not None:
                alpha_range = self.command_filter_config[axis_name]
                if alpha_range is None:
                    raise ValueError(f"Alpha range is None for axis {axis_name}. Please check your configuration.")
                # Sample uniformly from the specified range
                self.command_filter_alphas[env_ids, axis_idx] = torch.empty(len(env_ids), device=self.device).uniform_(
                    alpha_range[0], alpha_range[1]
                )
            else:
                # No filtering for this axis
                self.command_filter_alphas[env_ids, axis_idx] = 1.0

    def _update_metrics(self) -> None:
        # update smoothed velocity estimate

        vel_xy = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_quat_w.torch),
            self.robot.data.root_lin_vel_w.torch[:, :3],
        )[:, :2]
        self.vel_xy_smoothed = self.smoothing_param * vel_xy + (1 - self.smoothing_param) * self.vel_xy_smoothed
        self.angvel_smoothed = (
            self.smoothing_param * self.robot.data.root_ang_vel_w.torch[:, 2]
            + (1 - self.smoothing_param) * self.angvel_smoothed
        )

        # logs data
        self.metrics_cumulative["error_vel_xy"] += self.vel_command_b[:, :2] - self.vel_xy_smoothed
        self.metrics_cumulative["error_vel_yaw"] += torch.abs(self.vel_command_b[:, 2] - self.angvel_smoothed)

        current_positions = self.robot.data.root_pos_w.torch[:, :2]
        traveled_dist = torch.norm(current_positions - self.start_positions, dim=1)

        # norm of vector sum, not sum of norms
        normalizer = torch.clamp(self._env.episode_length_buf, min=1.0)

        self.metrics = {
            k: (v / normalizer if "xy" not in k else torch.norm(v, dim=-1) / normalizer)
            for k, v in self.metrics_cumulative.items()
        }
        self.metrics["traveled_distance"] = self.traveled_distance + traveled_dist

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> dict[str, float]:
        env_ids = resolve_env_ids(env_ids, self.num_envs, self.device)
        extras = {k: v[env_ids].mean().item() for k, v in self.metrics.items()}

        self.traveled_distance[env_ids] = 0.0
        self.start_positions[env_ids] = self.robot.data.root_pos_w.torch[env_ids, :2].clone()

        # Reset filter state: copy targets directly to filtered commands
        # This is correct for reset - we want to start fresh with no history from previous episode
        self.vel_command_b[env_ids] = self.vel_command_target_b[env_ids].clone()
        # Resample filter coefficients for reset environments
        self._sample_filter_coefficients(env_ids)

        super().reset(env_ids)
        for _, v in self.metrics_cumulative.items():
            v[env_ids] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        env_ids = resolve_env_ids(env_ids, self.num_envs, self.device)
        # Reimplement parent's _resample_command with bias sampling
        r = torch.empty(len(env_ids), device=self.device)

        # Helper function for piecewise uniform sampling
        def sample_with_bias(r: torch.Tensor, full_range: tuple[float, float], bias_speed: float) -> torch.Tensor:
            """Sample with bias toward low speeds if bias_sampling is enabled."""
            if self.cfg.bias_sampling:
                # Create mask for which samples go to low-speed range
                bias_mask = r.uniform_(0.0, 1.0) < self.cfg.bias_sampling_probability

                # Initialize with full range sampling
                samples = r.uniform_(*full_range)

                # For biased samples, resample from low-speed range
                if bias_mask.any():
                    # Clamp bias_speed to not exceed the full range
                    bias_range_min = max(full_range[0], -bias_speed)
                    bias_range_max = min(full_range[1], bias_speed)
                    samples[bias_mask] = r[bias_mask].uniform_(bias_range_min, bias_range_max)

                return samples
            else:
                # Standard uniform sampling when bias is disabled
                return r.uniform_(*full_range)

        # Sample new target commands (not filtered yet)
        # -- linear velocity - x direction
        self.vel_command_target_b[env_ids, 0] = sample_with_bias(
            r, self.cfg.ranges.lin_vel_x, self.cfg.bias_sampling_speed
        )

        # -- linear velocity - y direction
        self.vel_command_target_b[env_ids, 1] = sample_with_bias(
            r, self.cfg.ranges.lin_vel_y, self.cfg.bias_sampling_speed
        )

        # -- ang vel yaw - rotation around z
        # For angular velocity, we might want to use a scaled bias speed or the same one
        self.vel_command_target_b[env_ids, 2] = sample_with_bias(
            r, self.cfg.ranges.ang_vel_z, self.cfg.bias_sampling_speed
        )

        # -- heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        # -- update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        # Sample new filter coefficients for these environments
        self._sample_filter_coefficients(env_ids)

        # Update traveled distance tracking
        current_positions = self.robot.data.root_pos_w.torch[env_ids, :2]
        self.traveled_distance[env_ids] += torch.norm(current_positions - self.start_positions[env_ids], dim=1)
        self.start_positions[env_ids] = current_positions.clone()

        # set small velocity samples to zero (apply to both target and filtered)
        too_small_envs = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self.device)
        too_small_envs[env_ids] = self.vel_command_target_b[env_ids].norm(dim=1) < self.min_vel_norm
        self.vel_command_target_b[too_small_envs] = 0
        self.vel_command_b[too_small_envs] = 0

    def _update_command(self) -> None:
        """Update command with low-pass filtering applied per-axis.

        This method is called every simulation step to smoothly transition commands.
        """
        # First apply the low-pass filter to smoothly transition to target commands
        # Filter formula: filtered = alpha * target + (1 - alpha) * previous_filtered
        # Shape: command_filter_alphas is [num_envs, 3], vel commands are [num_envs, 3]
        self.vel_command_b = (
            self.command_filter_alphas * self.vel_command_target_b
            + (1 - self.command_filter_alphas) * self.vel_command_b
        )

        # Call parent to handle heading control and standing envs
        # Parent may modify vel_command_b for heading/standing envs
        super()._update_command()

        # Re-apply standing constraint to target as well (for consistency)
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            self.vel_command_target_b[standing_env_ids, :] = 0.0
