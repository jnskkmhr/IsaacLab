# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain reset, progression and termination helpers for contact locomotion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import reset_root_state_uniform
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.core.velocity.mdp.curriculums import terrain_levels_vel as terrain_levels_vel_base

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Update terrain difficulty from walking progress after the initial placement.

    Args:
        env: Environment whose robots are reset onto generated terrain tiles.
        env_ids: Environments being reset.
        asset_cfg: Robot used to measure progress.

    Returns:
        Mean terrain level over all environments.
    """
    # Before the first reset, robot positions still refer to the scene's clone grid.
    if env.common_step_counter == 0:
        return env.scene.terrain.terrain_levels.float().mean()
    return terrain_levels_vel_base(env, env_ids, asset_cfg)


class reset_root_state_on_terrain(reset_root_state_uniform):
    """Reset robots above the terrain height sampled at their randomized world XY.

    Pose and velocity are sampled from the current configured ranges. The sampled terrain height
    replaces the environment origin's Z offset, preserving the default root clearance [m].
    Regular terrain importers use the environment origin's Z offset instead.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Sample root pose and velocity, then write the terrain-adjusted state once.

        Args:
            env: Locomotion environment with a shared terrain surface.
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
        if hasattr(env.scene.terrain, "background_mesh"):
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


def root_state_infinite(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the robot root state contains NaN or infinity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return ~torch.isfinite(asset.data.root_state_w.torch).all(dim=-1)
