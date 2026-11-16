# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sample pelvis height above the initial terrain surface."""

import math

import torch

from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, wrap_to_pi


class UniformHeightCommand(CommandTerm):
    """A scalar root-link height command in meters above the initial terrain surface."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        lower, upper = cfg.height_range
        if not 0.0 < lower <= upper:
            raise ValueError("height_range must satisfy 0 < lower <= upper.")
        self.robot = env.scene[cfg.asset_name]
        self.sensor = env.scene[cfg.sensor_name]
        terrain = env.scene.terrain
        self._surface_height_offset = torch.full(
            (self.num_envs,), terrain.cfg.moving_patch_terrain.particle_depth, device=self.device
        )
        # Sand-only tasks have no allocation mask. Mixed tasks keep rigid references unchanged.
        if hasattr(terrain, "mpm_env_mask"):
            self._surface_height_offset *= terrain.mpm_env_mask
        self._command = torch.full((self.num_envs, 1), lower, device=self.device)
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Desired root height above the initial terrain surface, shape (num_envs, 1)."""
        return self._command

    @property
    def terrain_height(self) -> torch.Tensor:
        """Initial terrain surface Z [m], including sand depth only in particle environments.

        Rays hit the rigid support mesh below the particles. Correct its height to the
        undeformed sand surface; rigid environments retain the measured mesh height.
        All ray misses fall back to a support height of zero before this correction.
        """
        hits = self.sensor.data.ray_hits_w.torch[..., 2]
        finite = torch.isfinite(hits)
        safe_hits = torch.where(finite, hits, torch.zeros_like(hits))
        return safe_hits.sum(dim=1) / finite.sum(dim=1).clamp_min(1) + self._surface_height_offset

    def _resample_command(self, env_ids):
        self._command[env_ids] = torch.empty_like(self._command[env_ids]).uniform_(*self.cfg.height_range)

    def _update_command(self):
        pass

    def _update_metrics(self):
        height = self.robot.data.root_pos_w.torch[:, 2] - self.terrain_height
        self.metrics["height_error"][:] = (height - self.command[:, 0]).abs()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_height_visualizer"):
                self.goal_height_visualizer = VisualizationMarkers(self.cfg.goal_height_visualizer_cfg)
                self.current_height_visualizer = VisualizationMarkers(self.cfg.current_height_visualizer_cfg)
            self.goal_height_visualizer.set_visibility(True)
            self.current_height_visualizer.set_visibility(True)
        elif hasattr(self, "goal_height_visualizer"):
            self.goal_height_visualizer.set_visibility(False)
            self.current_height_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        current_position = self.robot.data.root_pos_w.torch.clone()
        goal_position = current_position.clone()
        current_position[:, :2] += current_position.new_tensor(self.cfg.current_marker_xy_offset)
        goal_position[:, :2] += goal_position.new_tensor(self.cfg.goal_marker_xy_offset)
        goal_position[:, 2] = self.terrain_height + self.command[:, 0]
        environment_ids = self._env.scene._ALL_INDICES
        self.current_height_visualizer.visualize(current_position, environment_ids=environment_ids)
        self.goal_height_visualizer.visualize(goal_position, environment_ids=environment_ids)


class UniformTorsoPitchCommand(CommandTerm):
    """Sample torso pitch relative to world vertical; roll targets zero and yaw remains free.

    Positive pitch tilts the torso's upward axis forward. The policy receives one
    scalar in radians. Errors use the torso link, independently of pelvis orientation.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        lower, upper = cfg.pitch_range
        if not all(math.isfinite(value) for value in (lower, upper)) or not -math.pi / 2 < lower <= upper < math.pi / 2:
            raise ValueError("pitch_range must be finite and satisfy -pi/2 < lower <= upper < pi/2.")
        self.robot = env.scene[cfg.asset_name]
        body_ids, _ = self.robot.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one torso body for {cfg.body_name!r}, got {len(body_ids)}.")
        self.body_idx = body_ids[0]
        self._command = torch.full((self.num_envs, 1), lower, device=self.device)
        self.metrics["pitch_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["roll_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Desired torso pitch [rad], shape (num_envs, 1)."""
        return self._command

    @property
    def orientation_error(self) -> torch.Tensor:
        """Wrapped (roll, pitch) errors [rad], shape (num_envs, 2); yaw is unrestricted."""
        roll, pitch, _ = euler_xyz_from_quat(self.robot.data.body_quat_w.torch[:, self.body_idx])
        return torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch - self.command[:, 0])), dim=-1)

    def _resample_command(self, env_ids):
        self._command[env_ids] = torch.empty_like(self._command[env_ids]).uniform_(*self.cfg.pitch_range)

    def _update_command(self):
        pass

    def _update_metrics(self):
        error = self.orientation_error.abs()
        self.metrics["roll_error"][:] = error[:, 0]
        self.metrics["pitch_error"][:] = error[:, 1]

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
                self.current_visualizer = VisualizationMarkers(self.cfg.current_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
            self.current_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)
            self.current_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        position = self.robot.data.body_pos_w.torch[:, self.body_idx].clone()
        position += position.new_tensor(self.cfg.marker_pos_offset)
        orientation = self.robot.data.body_quat_w.torch[:, self.body_idx]
        _, _, yaw = euler_xyz_from_quat(orientation)
        target = quat_from_euler_xyz(torch.zeros_like(yaw), self.command[:, 0], yaw)
        environment_ids = self._env.scene._ALL_INDICES
        self.goal_visualizer.visualize(position, target, environment_ids=environment_ids)
        self.current_visualizer.visualize(position, orientation, environment_ids=environment_ids)
