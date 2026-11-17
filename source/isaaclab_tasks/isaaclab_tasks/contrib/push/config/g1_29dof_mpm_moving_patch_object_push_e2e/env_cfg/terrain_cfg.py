# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    # Shared flat background; particle allocation follows the separate 2 x 2 m active patch.
    size=(200.0, 200.0),
    num_rows=1,
    num_cols=1,
    border_width=0.0,
    horizontal_scale=0.1,
    curriculum=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(),
    },
)
