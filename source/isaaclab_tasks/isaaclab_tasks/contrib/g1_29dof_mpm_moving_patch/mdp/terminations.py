# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms specific to the granular-bed locomotion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from ..mpm_env import G1MovingPatchEnv


def root_outside_workspace(
    env: G1MovingPatchEnv, margin: float = 0.3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate outside terrain XY bounds inset by boundary width plus margin [m]."""
    bounds = env.scene.terrain.background_mesh.bounds
    root = env.scene[asset_cfg.name].data.root_pos_w.torch[:, :2]
    pad = env.cfg.scene.terrain.moving_patch_terrain.boundary_terrain_size + margin
    lo = torch.tensor(bounds[0, :2] + pad, device=env.device)
    hi = torch.tensor(bounds[1, :2] - pad, device=env.device)
    return ((root < lo) | (root > hi)).any(dim=1)


def root_state_not_finite(env: G1MovingPatchEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the coupled solve left the base state non-finite.

    A diverging granular solve turns a whole world's particle and body state into ``NaN``. Every
    other termination term compares against a threshold, and comparisons with ``NaN`` are false, so
    without this term the world stays broken and keeps feeding ``NaN`` observations until the
    episode times out.

    Args:
        env: Environment instance.
        asset_cfg: Configuration of the tracked articulation.

    Returns:
        Whether the base state is non-finite, shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return ~torch.isfinite(asset.data.root_state_w.torch).all(dim=-1)
