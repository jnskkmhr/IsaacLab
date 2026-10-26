# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# Adapted from Agile. Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
from scipy import interpolate

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def random_uniform_terrain_difficulty(
    difficulty: float, cfg: HfRandomUniformTerrainDifficultyCfg
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a terrain with height sampled uniformly from a specified range.

    .. image:: ../../_static/terrains/height_field/random_uniform_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The discretized height field with shape (width, length) and its origin [m] with shape (3,).
        The origin is the center of the height field at the maximum height of the central 2 m x 2 m area.

    Raises:
        ValueError: When the downsampled scale is smaller than the horizontal scale.
    """
    # check parameters
    # -- horizontal scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    height_min = int(cfg.noise_range[0] * difficulty / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] * difficulty / cfg.vertical_scale)
    height_step = max(int(cfg.noise_step * difficulty / cfg.vertical_scale), 1)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    height_field = np.rint(z_upsampled).astype(np.int16)
    # origin at the center, on the highest point of the central 2 m x 2 m area (as upstream random_uniform_terrain)
    center_x, center_y = height_field.shape[0] // 2, height_field.shape[1] // 2
    radius = max(1, int(1.0 / cfg.horizontal_scale))
    height = np.max(
        height_field[max(0, center_x - radius) : center_x + radius, max(0, center_y - radius) : center_y + radius]
    )
    origin = np.array(
        [
            (height_field.shape[0] - 1) * cfg.horizontal_scale / 2,
            (height_field.shape[1] - 1) * cfg.horizontal_scale / 2,
            height * cfg.vertical_scale,
        ]
    )
    return height_field, origin


@configclass
class HfRandomUniformTerrainDifficultyCfg(HfTerrainBaseCfg):
    """Configuration for a random uniform height field terrain."""

    function = random_uniform_terrain_difficulty

    noise_range: tuple[float, float] = MISSING
    """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""
    noise_step: float = MISSING
    """The minimum height (in m) change between two points."""
    downsampled_scale: float | None = None
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """


MEDIUM_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=100.0,
    num_rows=20,  # num different difficulties
    num_cols=15,  # num terrains per same difficulty level
    horizontal_scale=0.4,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes_small": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.15,
            grid_height_range=(0.0, 0.04),
            platform_width=0.1,
        ),
        "random_rough_small": HfRandomUniformTerrainDifficultyCfg(
            proportion=0.2,
            noise_range=(0.01, 0.2),
            noise_step=0.1,
            border_width=0.4,
        ),
        "wave_small": terrain_gen.HfWaveTerrainCfg(proportion=0.2, amplitude_range=(0.01, 0.25), num_waves=3),
    },
)
