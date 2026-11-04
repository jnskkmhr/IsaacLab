# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local mixed-terrain scene configuration with particles only in the sand group."""

from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.mixed_env import MixedMPMObject
from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.terrain import MixedTerrainImporter

from .scene_cfg import G1MovingPatchSceneCfg
from .terrain_cfg import ROUGH_TERRAINS_CFG, BackgroundTerrainImporterCfg

# Match the MPM mixed task without changing the sand-only task's generator.
MIXED_TERRAINS_CFG = ROUGH_TERRAINS_CFG.copy()
MIXED_TERRAINS_CFG.sub_terrains["wave"].amplitude_range = (0.1, 0.4)


@configclass
class G1MixedTerrainSceneCfg(G1MovingPatchSceneCfg):
    """Allocate half the worlds to neural sand and half to rigid ground."""

    terrain: BackgroundTerrainImporterCfg = G1MovingPatchSceneCfg().terrain.replace(
        terrain_generator=MIXED_TERRAINS_CFG
    )

    def __post_init__(self):
        self.terrain.class_type = MixedTerrainImporter
        self.terrain.use_terrain_origins = True
        self.clone_cfg = CloneCfg(clone_combinations=[InclusionSet(assets=["sand"]), InclusionSet(assets=[])])
        super().__post_init__()

    def configure_terrain(self):
        self.terrain.terrain_generator.num_cols = max(2, self.terrain.terrain_generator.num_cols)
        super().configure_terrain()
        # This asset only maps compact particle storage to scene IDs; it creates no native solver.
        self.sand.class_type = MixedMPMObject
