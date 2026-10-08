# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Terrain-generator configuration for the mixed-contact G1 task."""

from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.utils import configclass

from ..mixed_env import MixedMPMObject
from ..util.terrain import MixedTerrainImporter
from .scene_cfg import G1MovingPatchSceneCfg
from .terrain_cfg import FLAT_TERRAINS_CFG


@configclass
class G1MixedTerrainSceneCfg(G1MovingPatchSceneCfg):
    """Use generator origins, terrain-type columns, and difficulty rows for both ground models."""

    def __post_init__(self) -> None:
        # Replace this generator with ROUGH_TERRAINS_CFG.copy(), as in the MPM task.
        self.terrain.terrain_generator = FLAT_TERRAINS_CFG.replace(num_rows=1, num_cols=2)
        self.terrain.class_type = MixedTerrainImporter
        self.terrain.use_terrain_origins = True
        self.clone_cfg = CloneCfg(clone_combinations=[InclusionSet(assets=["sand"]), InclusionSet(assets=[])])
        super().__post_init__()

    def configure_terrain(self) -> None:
        # At least two columns are required to separate the two contact regions.
        self.terrain.terrain_generator.num_cols = max(2, self.terrain.terrain_generator.num_cols)
        super().configure_terrain()
        self.sand.class_type = MixedMPMObject
