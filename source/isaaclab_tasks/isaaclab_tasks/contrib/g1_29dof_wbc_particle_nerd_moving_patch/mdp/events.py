# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms specific to the granular-bed locomotion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import reset_root_state_uniform
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from ..env_cfg.terrain_cfg import BackgroundTerrainImporterCfg

if TYPE_CHECKING:
    from isaaclab_tasks.contrib.g1_29dof_particle_nerd_moving_patch.particle_nerd_env import G1ParticleNeRDEnv


class reset_root_state_on_terrain(reset_root_state_uniform):
    """Reset robots above the terrain height sampled at their randomized world XY.

    Pose and velocity are sampled from the current configured ranges. The sampled terrain height
    replaces the environment origin's Z offset, preserving the default root clearance [m].
    Regular terrain importers use the environment origin's Z offset instead.
    """

    def __call__(
        self,
        env: G1ParticleNeRDEnv,
        env_ids: torch.Tensor | slice,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Sample root pose and velocity, then write the terrain-adjusted state once.

        Args:
            env: Moving-patch environment with a shared terrain surface.
            env_ids: Environment indices or a slice selecting robots to reset.
            pose_range: Pose offsets [m or rad, depending on axis], cached at construction.
            velocity_range: Velocity offsets [m/s or rad/s], cached at construction.
            asset_cfg: Robot selection.
        """
        asset: Articulation = env.scene[asset_cfg.name]
        default_root_pose = asset.data.default_root_pose.torch[env_ids]
        default_root_vel = asset.data.default_root_vel.torch[env_ids]
        if default_root_pose.shape[0] == 0:
            return

        rand_samples = math_utils.sample_uniform_from_ranges(
            pose_range, ("x", "y", "z", "roll", "pitch", "yaw"), default_root_pose.shape[0], device=asset.device
        )
        positions = default_root_pose[:, :3] + rand_samples[:, :3]
        positions[:, :2] += env.scene.env_origins[env_ids][:, :2]
        if isinstance(env.scene.terrain.cfg, BackgroundTerrainImporterCfg):
            points = wp.from_torch(positions.clone().contiguous(), dtype=wp.vec3)
            env.scene.terrain.background_mesh.sample_surface_heights(points)
            positions[:, 2] += wp.to_torch(points)[:, 2]
        else:
            positions[:, 2] += env.scene.env_origins[env_ids][:, 2]

        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(default_root_pose[:, 3:7], orientations_delta)
        rand_samples = math_utils.sample_uniform_from_ranges(
            velocity_range, ("x", "y", "z", "roll", "pitch", "yaw"), default_root_pose.shape[0], device=asset.device
        )
        velocities = default_root_vel + rand_samples
        asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)


class backward_force_pulse(ManagerTermBase):
    """Apply constant world -X force at the selected body's CoM for timed pulses.

    Called every environment step. Each environment independently samples pulse
    start-to-start intervals and force magnitudes. Owns the robot's permanent
    external-wrench buffer; do not combine with another permanent-wrench event.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.body_ids = cfg.params["asset_cfg"].body_ids
        if isinstance(self.body_ids, slice) or len(self.body_ids) != 1:
            raise ValueError("backward_force_pulse requires exactly one body")
        self.force_range = cfg.params["force_range"]
        self.duration_s = cfg.params["duration_s"]
        self.interval_range_s = cfg.params["pulse_interval_range_s"]
        if not 0.0 <= self.force_range[0] <= self.force_range[1]:
            raise ValueError("force_range must contain ordered nonnegative magnitudes")
        if not 0.0 < self.duration_s < self.interval_range_s[0] <= self.interval_range_s[1]:
            raise ValueError("Pulse intervals must exceed the positive force duration")
        self._until_start = torch.zeros(self.num_envs, device=self.device)
        self._duration_steps = max(1, round(self.duration_s / env.step_dt))
        self._remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._magnitude = torch.zeros_like(self._until_start)
        self._forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self.reset()

    def reset(self, env_ids=None):
        """Cancel pulses and restart independent timers for reset environments."""
        if env_ids is None:
            env_ids = slice(None)
        count = self._until_start[env_ids].numel()
        self._until_start[env_ids] = torch.empty(count, device=self.device).uniform_(*self.interval_range_s)
        self._remaining[env_ids] = 0.0
        self._magnitude[env_ids] = 0.0
        self._forces[env_ids] = 0.0
        self.asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._forces[env_ids], body_ids=self.body_ids, env_ids=env_ids, is_global=True
        )

    def __call__(self, env, env_ids, force_range, duration_s, pulse_interval_range_s, asset_cfg):
        """Advance timers and write the force that will act during the next environment step."""
        self._remaining.sub_(1).clamp_(min=0)
        self._until_start.sub_(env.step_dt)
        starting = (self._until_start <= 1e-6).nonzero(as_tuple=False).flatten()
        self._until_start[starting] = torch.empty(starting.numel(), device=self.device).uniform_(*self.interval_range_s)
        self._magnitude[starting] = torch.empty(starting.numel(), device=self.device).uniform_(*self.force_range)
        self._remaining[starting] = self._duration_steps
        self._forces[:, 0, 0] = torch.where(self._remaining > 0, -self._magnitude, 0.0)
        self.asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._forces, body_ids=self.body_ids, is_global=True
        )
