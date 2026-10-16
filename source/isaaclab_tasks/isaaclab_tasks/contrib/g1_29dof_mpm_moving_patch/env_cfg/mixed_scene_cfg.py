# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Terrain-generator configuration for the mixed-contact G1 task."""

import numpy as np

from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.cloner.cloner_strategies import random
from isaaclab.utils import configclass

from ..mixed_env import MixedMPMObject
from ..util.terrain import MixedTerrainImporter, PairedTerrainGenerator
from .scene_cfg import G1MovingPatchSceneCfg
from .terrain_cfg import ROUGH_TERRAINS_CFG


def sample_contact_types(weights: np.ndarray, num_clones: int) -> np.ndarray:
    """Select MPM for a single world; otherwise independently sample rigid (0) or MPM (1)."""
    if num_clones == 1:
        return np.ones(1, dtype=np.int32)
    return random(weights, num_clones)


@configclass
class G1MixedTerrainSceneCfg(G1MovingPatchSceneCfg):
    """Use generator origins, terrain-type columns, and difficulty rows for both ground models."""

    def __post_init__(self) -> None:
        # Replace this generator with ROUGH_TERRAINS_CFG.copy(), as in the MPM task.
        # self.terrain.terrain_generator = FLAT_TERRAINS_CFG.copy()
        self.terrain.terrain_generator = ROUGH_TERRAINS_CFG.copy()
        self.terrain.class_type = MixedTerrainImporter
        self.terrain.use_terrain_origins = True
        self.clone_cfg = CloneCfg(
            clone_strategy=sample_contact_types,
            clone_combinations=[InclusionSet(assets=[]), InclusionSet(assets=["sand"])],
        )
        super().__post_init__()

    def configure_terrain(self) -> None:
        # Preserve the configured grid; the generator creates one copy per contact model.
        self.terrain.terrain_generator.class_type = PairedTerrainGenerator
        super().configure_terrain()
        self.sand.class_type = MixedMPMObject
