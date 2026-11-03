# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Terrain-generator configuration for the mixed-contact G1 task."""

from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mixed_env import MixedMPMObject
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.terrain import MixedTerrainImporter

from .scene_cfg import G1MovingPatchSceneCfg
from .terrain_cfg import ROUGH_TERRAINS_CFG


@configclass
class G1MixedTerrainSceneCfg(G1MovingPatchSceneCfg):
    """Use generator origins, terrain-type columns, and difficulty rows for both ground models."""

    def __post_init__(self) -> None:
        self.terrain.terrain_generator = ROUGH_TERRAINS_CFG.copy()
        # Each contact region gets the complete Agile terrain-type distribution.
        self.terrain.terrain_generator.num_cols *= 2
        self.terrain.terrain_generator.sub_terrains = {
            f"{surface}_{name}": terrain.copy()
            for surface in ("mpm", "rigid")
            for name, terrain in ROUGH_TERRAINS_CFG.sub_terrains.items()
        }
        self.terrain.class_type = MixedTerrainImporter
        self.terrain.use_terrain_origins = True
        self.clone_cfg = CloneCfg(clone_combinations=[InclusionSet(assets=["sand"]), InclusionSet(assets=[])])
        super().__post_init__()

    def configure_terrain(self) -> None:
        # At least two columns are required to separate the two contact regions.
        self.terrain.terrain_generator.num_cols = max(2, self.terrain.terrain_generator.num_cols)
        super().configure_terrain()
        self.sand.class_type = MixedMPMObject
