# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local SE(2) + Z targets with reach-and-hold success tracking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import TerrainPoseCommandCfg


class TerrainPoseCommand(CommandTerm):
    """Sample a supported base pose and expose its error in the base yaw frame.

    The observation is ``(dx, dy, dz, sin(yaw_error), cos(yaw_error))``. Goal Z
    is the terrain surface height plus ``cfg.base_height``. Goals remain fixed
    after arrival so the controller must continue standing until timeout.
    """

    cfg: TerrainPoseCommandCfg

    def __init__(self, cfg: TerrainPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not 0 <= cfg.distance_range[0] < cfg.distance_range[1] or cfg.hold_time <= 0:
            raise ValueError("Reach goals require an increasing nonnegative distance range and positive hold time.")
        self.robot = env.scene[cfg.asset_name]
        self.terrain = env.scene.terrain
        self.contact_sensor = env.scene[cfg.contact_sensor_name]
        self.foot_ids, _ = self.robot.find_bodies(cfg.foot_body_names)
        self.contact_ids, _ = self.contact_sensor.find_bodies(cfg.foot_body_names)
        if len(self.foot_ids) != 2 or len(self.contact_ids) != 2:
            raise ValueError("G1 reach-and-hold requires both ankle-roll bodies and their contact measurements.")
        self.valid_targets = self.terrain.flat_patches[cfg.target_patch_name]
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self._command = torch.zeros(self.num_envs, 5, device=self.device)
        self.hold_duration = torch.zeros(self.num_envs, device=self.device)
        self.goal_held = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_reached = torch.zeros_like(self.goal_held)
        self._last_status_step = -1
        for name in ("position_error", "height_error", "heading_error"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def position_error_w(self) -> torch.Tensor:
        """Current target-minus-base displacement in world coordinates, in meters."""
        return self.pos_command_w - self.robot.data.root_pos_w.torch

    @property
    def heading_error(self) -> torch.Tensor:
        """Signed target-minus-base yaw, wrapped to [-pi, pi]."""
        return wrap_to_pi(self.heading_command_w - self.robot.data.heading_w.torch)

    @property
    def episode_success(self) -> torch.Tensor:
        """Finish holding the goal at timeout, with no simultaneous failure."""
        return self.goal_held & self._env.termination_manager.time_outs & ~self._env.termination_manager.terminated

    def update_goal_status(self):
        """Count each control step once, including the final step before automatic reset.

        Called during command updates and by the settling reward and curriculum
        before reset. Command updates alone cannot evaluate the final state.
        """
        if self._last_status_step == self._env.common_step_counter:
            return
        self._last_status_step = self._env.common_step_counter
        error = self.position_error_w
        contact = (
            torch.linalg.vector_norm(self.contact_sensor.data.net_forces_w.torch[:, self.contact_ids], dim=-1)
            > self.cfg.contact_force_threshold
        )
        foot_speed = torch.linalg.vector_norm(self.robot.data.body_lin_vel_w.torch[:, self.foot_ids], dim=-1)
        settled = (
            (torch.linalg.vector_norm(error[:, :2], dim=-1) < self.cfg.position_tolerance)
            & (error[:, 2].abs() < self.cfg.height_tolerance)
            & (self.heading_error.abs() < self.cfg.heading_tolerance)
            & (
                torch.linalg.vector_norm(self.robot.data.root_lin_vel_w.torch, dim=-1)
                < self.cfg.linear_velocity_tolerance
            )
            & (
                torch.linalg.vector_norm(self.robot.data.root_ang_vel_w.torch, dim=-1)
                < self.cfg.angular_velocity_tolerance
            )
            & (foot_speed < self.cfg.foot_velocity_tolerance).all(dim=-1)
            & contact.all(dim=-1)
            & (self._env.episode_length_buf > 0)
            & ~self._env.termination_manager.terminated
        )
        self.hold_duration[:] = torch.where(settled, self.hold_duration + self._env.step_dt, 0.0)
        self.goal_held[:] = self.hold_duration >= self.cfg.hold_time
        self.goal_reached |= self.goal_held

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        # Success buffers describe the pre-reset robot, even though reset events have now run.
        completed = self._env.termination_manager.dones[env_ids] & (self._env.episode_length_buf[env_ids] > 0)
        success = self.episode_success[env_ids][completed].float()
        reached = self.goal_reached[env_ids][completed].float()
        extras = super().reset(env_ids)
        if success.numel():
            extras["success_rate"] = success.mean()
            extras["goal_reached_rate"] = reached.mean()
        return extras

    def _resample_command(self, env_ids: Sequence[int] | slice):
        env_ids = self._env.scene._ALL_INDICES[env_ids]
        candidates = self.valid_targets[self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]]
        distance = torch.linalg.vector_norm(
            candidates[..., :2] - self.robot.data.root_pos_w.torch[env_ids, None, :2], dim=-1
        )
        eligible = (distance >= self.cfg.distance_range[0]) & (distance <= self.cfg.distance_range[1])
        if not eligible.any(dim=-1).all():
            raise RuntimeError(
                "No terrain goal patch in distance_range; enlarge the range or sample more target patches."
            )
        index = torch.multinomial(eligible.float(), 1).squeeze(-1)
        self.pos_command_w[env_ids] = candidates[torch.arange(len(env_ids), device=self.device), index]
        self.pos_command_w[env_ids, 2] += self.cfg.base_height
        self.heading_command_w[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.heading_range
        )
        self.hold_duration[env_ids] = 0.0
        self.goal_held[env_ids] = False
        self.goal_reached[env_ids] = False
        self._update_command()

    def _update_command(self):
        self._command[:, :3] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w.torch), self.position_error_w)
        self._command[:, 3] = torch.sin(self.heading_error)
        self._command[:, 4] = torch.cos(self.heading_error)

    def _update_metrics(self):
        self.update_goal_status()
        error = self.position_error_w
        self.metrics["position_error"][:] = torch.linalg.vector_norm(error[:, :2], dim=-1)
        self.metrics["height_error"][:] = error[:, 2].abs()
        self.metrics["heading_error"][:] = self.heading_error.abs()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and not hasattr(self, "goal_visualizer"):
            self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
        if hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        zeros = torch.zeros_like(self.heading_command_w)
        self.goal_visualizer.visualize(
            translations=self.pos_command_w,
            orientations=quat_from_euler_xyz(zeros, zeros, self.heading_command_w),
            environment_ids=self._env.scene._ALL_INDICES,
        )
