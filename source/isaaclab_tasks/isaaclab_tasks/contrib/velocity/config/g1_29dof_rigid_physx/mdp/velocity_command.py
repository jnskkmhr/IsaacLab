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
    from isaaclab.envs import ManagerBasedEnv
logger = logging.getLogger(__name__)


class UniformVelocityYawCommand(UniformVelocityCommand):
    """
    This class inherits from `UniformVelocityCommand` to
    - apply curriclum to lin/ang velocity sampling range
    - provide debug vis for both linear and angular velocity commands
    """

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if self.cfg.heading_command:
            logger.warning("heading_command argument is not used in this class. Set it to False.")
        if self.cfg.ranges.heading:
            logger.warning("heading range is not used in this class. Set it to None.")
        self.heading_target = torch.zeros((self.num_envs, 4), device=self.device)
        self.heading_target[:, 3] = 1.0
        self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

    "\n    Properties\n    "

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 7)."""
        return torch.cat([self.vel_command_b, self.heading_target], dim=-1)

    "\n    Implementation specific functions.\n    "

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b.torch[:, :2], dim=-1)
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b.torch[:, 2]) / max_command_step
        )
        self.metrics["error_yaw"] += (
            torch.abs(
                math_utils.quat_error_magnitude(
                    math_utils.yaw_quat(self.robot.data.root_quat_w.torch), self.heading_target
                )
            )
            / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        self.heading_target[env_ids] = math_utils.yaw_quat(self.robot.data.root_quat_w.torch[env_ids])
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0
        current_heading = math_utils.yaw_quat(self.robot.data.root_quat_w.torch)
        heading_error = math_utils.quat_error_magnitude(current_heading, self.heading_target)
        reset_env_ids = heading_error > torch.pi / 2
        self.heading_target[reset_env_ids] = current_heading[reset_env_ids]
        yaw_ang_vel_cmd = self.vel_command_b[:, 2]
        delta_yaw = yaw_ang_vel_cmd * self._env.step_dt
        delta_quat = math_utils.quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device), delta_yaw
        )
        self.heading_target[:] = math_utils.quat_unique(math_utils.quat_mul(delta_quat, self.heading_target))

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_linvel_visualizer"):
                self.goal_linvel_visualizer = VisualizationMarkers(self.cfg.goal_linvel_visualizer_cfg)
                self.goal_angvel_visualizer = VisualizationMarkers(self.cfg.goal_angvel_visualizer_cfg)
                self.goal_heading_visualizer = VisualizationMarkers(self.cfg.goal_heading_visualizer_cfg)
                self.curr_linvel_visualizer = VisualizationMarkers(self.cfg.current_linvel_visualizer_cfg)
                self.curr_angvel_visualizer = VisualizationMarkers(self.cfg.current_angvel_visualizer_cfg)
            self.goal_linvel_visualizer.set_visibility(True)
            self.goal_angvel_visualizer.set_visibility(True)
            self.goal_heading_visualizer.set_visibility(True)
            self.curr_linvel_visualizer.set_visibility(True)
            self.curr_angvel_visualizer.set_visibility(True)
        elif hasattr(self, "goal_linvel_visualizer"):
            self.goal_linvel_visualizer.set_visibility(False)
            self.goal_angvel_visualizer.set_visibility(False)
            self.goal_heading_visualizer.set_visibility(False)
            self.curr_linvel_visualizer.set_visibility(False)
            self.curr_angvel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.torch.clone()
        base_pos_w[:, 2] += 0.5
        linvel_des_scale, linvel_des_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        linvel_curr_scale, linvel_curr_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b.torch[:, :2]
        )
        angvel_des_scale, angvel_des_quat = self._resolve_z_angvel_to_arrow(self.command[:, 2])
        angvel_curr_scale, angvel_curr_quat = self._resolve_z_angvel_to_arrow(
            self.robot.data.root_ang_vel_b.torch[:, 2]
        )
        self.goal_linvel_visualizer.visualize(base_pos_w, linvel_des_quat, linvel_des_scale)
        self.curr_linvel_visualizer.visualize(base_pos_w, linvel_curr_quat, linvel_curr_scale)
        angvel_pos_w = base_pos_w.clone()
        angvel_pos_w[:, 2] += 0.5
        self.goal_angvel_visualizer.visualize(angvel_pos_w, angvel_des_quat, angvel_des_scale)
        self.curr_angvel_visualizer.visualize(angvel_pos_w, angvel_curr_quat, angvel_curr_scale)
        heading_pos_w = base_pos_w.clone()
        heading_pos_w[:, 2] += 0.5
        self.goal_heading_visualizer.visualize(
            heading_pos_w,
            self.heading_target,
            torch.tensor([6.0, 1.0, 0.1], device=self.device).repeat(self.num_envs, 1),
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        self.heading_target[env_ids] = math_utils.yaw_quat(self.robot.data.root_quat_w.torch[env_ids])
        return extras

    "\n    Internal helpers.\n    "

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        default_scale = self.goal_linvel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w.torch
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return (arrow_scale, arrow_quat)

    def _resolve_z_angvel_to_arrow(self, z_angvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the Z angular velocity to arrow pointing in +/- z direction."""
        default_scale = self.goal_angvel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(z_angvel.shape[0], 1)
        arrow_scale[:, 0] *= torch.abs(z_angvel) * 2.0
        pitch_angle = torch.where(
            z_angvel < 0, torch.full_like(z_angvel, torch.pi / 2), torch.full_like(z_angvel, -torch.pi / 2)
        )
        zeros = torch.zeros_like(pitch_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, pitch_angle, zeros)
        base_quat_w = self.robot.data.root_quat_w.torch
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return (arrow_scale, arrow_quat)
