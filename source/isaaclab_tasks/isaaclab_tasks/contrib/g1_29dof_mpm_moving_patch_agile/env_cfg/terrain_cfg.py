# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.env_cfg.terrain_cfg import (
    HfRandomUniformTerrainDifficultyCfg,
)

FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(20.0, 20.0),
    num_rows=1,
    num_cols=1,
    border_width=0.0,
    horizontal_scale=0.1,
    curriculum=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(),
    },
)

WAVE_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(20.0, 20.0),
    num_rows=1,
    num_cols=1,
    border_width=0.0,
    horizontal_scale=0.1,
    curriculum=False,
    sub_terrains={
        "waves": terrain_gen.HfWaveTerrainCfg(
            amplitude_range=(0.4, 0.4),
            num_waves=4,
            border_width=0.5,
        ),
    },
)

SLOPE_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(20.0, 20.0),
    border_width=5.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     slope_range=(0.4, 0.4), platform_width=2.0, border_width=0.25
        # ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            slope_range=(math.pi * (20.0 / 180.0), math.pi * (20.0 / 180.0)), platform_width=2.0, border_width=0.25
        ),
    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
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
        # Match Agile's effective spacing; current generators preserve child defaults.
        "random_rough_small": HfRandomUniformTerrainDifficultyCfg(
            proportion=0.2,
            horizontal_scale=0.4,
            noise_range=(0.01, 0.2),
            noise_step=0.1,
            border_width=0.4,
        ),
        "wave_small": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2, horizontal_scale=0.4, amplitude_range=(0.01, 0.25), num_waves=3
        ),
    },
)


ROUGH_TERRAINS_DIFFICULT_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(20.0, 20.0),
    border_width=5.0,
    num_rows=1,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2,
            amplitude_range=(0.4, 0.4),
            num_waves=4,
            border_width=0.25,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.1, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.4, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.4, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
