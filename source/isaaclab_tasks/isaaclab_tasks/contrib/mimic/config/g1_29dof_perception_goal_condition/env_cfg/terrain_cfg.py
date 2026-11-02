# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ascending staircases for reference-free goal policy fine-tuning."""

from isaaclab.terrains import FlatPatchSamplingCfg, MeshPyramidStairsTerrainCfg, TerrainGeneratorCfg

STAIRS_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=8,
    num_cols=9,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        f"stairs_{width:.2f}": MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.12, 0.23),
            step_width=width,
            platform_width=2.5,
            border_width=1.5,
            holes=False,
            flat_patch_sampling={
                # A standing footprint on the west ground border. The command rotates
                # this point to one of the four equivalent sides of the pyramid.
                "spawn": FlatPatchSamplingCfg(
                    num_patches=64,
                    patch_radius=[0.3, 0.0],
                    max_height_diff=0.02,
                    x_range=(-3.5, -3.0),
                    y_range=(-0.4, 0.4),
                ),
                "target": FlatPatchSamplingCfg(
                    num_patches=64,
                    patch_radius=[0.3, 0.0],
                    max_height_diff=0.02,
                    x_range=(-0.5, 0.5),
                    y_range=(-0.5, 0.5),
                    z_range=(-0.02, 0.02),
                ),
            },
        )
        for width in (0.22, 0.28, 0.34)
    },
)
