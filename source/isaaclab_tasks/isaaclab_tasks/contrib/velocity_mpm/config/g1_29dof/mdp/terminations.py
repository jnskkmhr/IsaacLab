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
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from ..streaming_mpm_env import G1MPMEnv


def root_outside_workspace(
    env: G1MPMEnv, margin: float = 0.3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate outside terrain XY bounds inset by the pool's ring width plus margin [m]."""
    bounds = env.scene.terrain.background_mesh.bounds  # type: ignore
    root = env.scene[asset_cfg.name].data.root_pos_w.torch[:, :2]
    pad = (0.0 if env.mpm_streaming is None else env.mpm_streaming.margin) + margin
    lo = torch.tensor(bounds[0, :2] + pad, device=env.device)
    hi = torch.tensor(bounds[1, :2] - pad, device=env.device)
    return ((root < lo) | (root > hi)).any(dim=1)


def root_outside_sand_bed(
    env: G1MPMEnv, margin: float = 0.3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the base leaves the sand bed of a fixed testbed.

    Beyond the bed there is nothing to walk on, so the episode carries no useful signal.

    Args:
        env: Environment instance.
        margin: Distance inside the bed edge at which the episode ends [m].
        asset_cfg: Configuration of the tracked articulation.

    Returns:
        Whether the base is outside the bed, shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    position_e = asset.data.root_pos_w.torch - env.scene.env_origins
    sand = env.cfg.scene.sand.spawn
    x_lo, y_lo = sand.lower[:2]
    x_hi, y_hi = sand.upper[:2]
    outside_x = (position_e[:, 0] < x_lo + margin) | (position_e[:, 0] > x_hi - margin)
    outside_y = (position_e[:, 1] < y_lo + margin) | (position_e[:, 1] > y_hi - margin)
    return outside_x | outside_y


def root_outside_testbed(env: G1MPMEnv, margin: float = 0.3) -> torch.Tensor:
    """Terminate outside the sand tray's interior, measured in the moving tray frame."""
    tray = env.scene["testbed"]
    tray_id = env.testbed_tray_body_id
    position = quat_apply_inverse(
        tray.data.body_quat_w.torch[:, tray_id],
        env.scene["robot"].data.root_pos_w.torch - tray.data.body_pos_w.torch[:, tray_id],
    )
    length, width, _ = env.cfg.scene.testbed.spawn.sand_bed_size
    return (position[:, 0].abs() > length / 2 - margin) | (position[:, 1].abs() > width / 2 - margin)


def root_state_infinite(env: G1MPMEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
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
