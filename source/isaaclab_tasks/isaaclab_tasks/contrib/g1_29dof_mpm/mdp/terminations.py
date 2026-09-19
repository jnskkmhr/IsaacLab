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

from ..env_cfg.scene_cfg import WALKABLE_XY_BOUNDS

if TYPE_CHECKING:
    from ..g1_mpm_env import G1MPMEnv


def root_outside_workspace(
    env: G1MPMEnv,
    margin: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base leaves the supported surface.

    Replaces the terrain out-of-bounds check of the rough tasks. The supported surface is the
    rigid approach platform followed by the granular bed; beyond it there is nothing to walk on,
    so the episode carries no useful signal.

    Args:
        env: Environment instance.
        margin: Distance inside the surface edge at which the episode ends [m].
        asset_cfg: Configuration of the tracked articulation.

    Returns:
        Whether the base is outside the supported surface, shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    position_e = asset.data.root_pos_w.torch - env.scene.env_origins
    (x_lo, x_hi), (y_lo, y_hi) = WALKABLE_XY_BOUNDS
    outside_x = (position_e[:, 0] < x_lo + margin) | (position_e[:, 0] > x_hi - margin)
    outside_y = (position_e[:, 1] < y_lo + margin) | (position_e[:, 1] > y_hi - margin)
    return outside_x | outside_y


def root_state_not_finite(env: G1MPMEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
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
