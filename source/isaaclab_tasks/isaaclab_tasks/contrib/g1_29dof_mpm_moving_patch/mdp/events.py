# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms specific to the granular-bed locomotion task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.events import reset_root_state_uniform_on_ground

if TYPE_CHECKING:
    from ..mpm_env import G1MovingPatchEnv


def reset_mpm_state(env: G1MovingPatchEnv, env_ids: Sequence[int] | torch.Tensor) -> None:
    """Restore the flat granular bed for the selected environments.

    Args:
        env: Environment instance.
        env_ids: Indices of the environments to reset.
    """
    env.reset_mpm_state(env_ids)


def reset_root_state_on_terrain(
    env: G1MovingPatchEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset selected robots above the generated surface at their sampled world XY.

    Args:
        env: Moving-patch environment with a shared terrain surface.
        env_ids: Robot environment indices to reset.
        pose_range: Pose offsets [m or rad, depending on axis].
        velocity_range: Initial linear/angular velocity ranges [m/s or rad/s].
        asset_cfg: Robot selection.
    """
    reset_root_state_uniform_on_ground(env, env_ids, pose_range, velocity_range, asset_cfg)
    asset = env.scene[asset_cfg.name]
    pose = asset.data.root_state_w.torch[env_ids, :7].clone()
    points = wp.from_torch(pose[:, :3].clone().contiguous(), dtype=wp.vec3)
    env.scene.terrain.background_mesh.sample(points)
    pose[:, 2] += wp.to_torch(points)[:, 2]
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
