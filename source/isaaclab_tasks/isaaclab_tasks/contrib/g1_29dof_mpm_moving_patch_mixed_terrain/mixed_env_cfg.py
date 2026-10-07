# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Equal-sized rigid-ground and MPM-ground groups for one G1 policy."""

import math

from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.terrains import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

from ..g1_29dof_mpm_moving_patch.agents.rsl_rl_ppo_cfg import G1MovingPatchPPORunnerCfg
from ..g1_29dof_mpm_moving_patch.env_cfg.physics_cfg import MPM_ENTRY, RIGID_ENTRY, G1PhysicsProxyCfg
from ..g1_29dof_mpm_moving_patch.env_cfg.terrain_cfg import FLAT_TERRAINS_CFG
from ..g1_29dof_mpm_moving_patch.mpm_env_cfg import G1MovingPatchEnvCfg
from .mixed_env import MixedMPMObject, outside_terrain_tile, randomize_mpm_material
from .terrain import MixedTerrainImporter


@configclass
class G1MixedTerrainEnvCfg(G1MovingPatchEnvCfg):
    """Use contiguous, equal-sized MPM and rigid groups, fixed across resets."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 20.0
        self.scene.clone_cfg = CloneCfg(clone_combinations=[InclusionSet(assets=["sand"]), InclusionSet(assets=[])])
        self.scene.terrain.class_type = MixedTerrainImporter
        self.scene.terrain.terrain_generator = FLAT_TERRAINS_CFG.copy()
        self.scene.terrain.use_terrain_origins = False
        self.scene.visual_terrain.terrain_generator = FLAT_TERRAINS_CFG.copy()
        self.sim.physics = G1PhysicsProxyCfg()
        self.curriculum.terrain_levels = None
        self.terminations.terrain_out_of_bounds.func = outside_terrain_tile
        self.terminations.terrain_out_of_bounds.params = {"margin": 1.0}
        for name in ("mpm_material", "mpm_material_log"):
            term = getattr(self.events, name, None)
            if term is not None:
                term.func = randomize_mpm_material

    def validate_config(self) -> None:
        if self.scene.num_envs < 2 or self.scene.num_envs % 2:
            raise ValueError("Mixed terrain requires an even num_envs >= 2 (half MPM, half rigid).")
        num_mpm_envs = self.scene.num_envs // 2
        if not all(
            isinstance(terrain, MeshPlaneTerrainCfg)
            for terrain in self.scene.terrain.terrain_generator.sub_terrains.values()
        ):
            raise ValueError("Mixed terrain currently supports flat terrain only.")
        if self.scene.env_spacing <= max(self.scene.terrain.moving_patch_terrain.total_patch_size) + 2.0:
            raise ValueError("env_spacing must contain the moving patch and the terrain-tile reset margin.")
        # Cover the clone grid and keep each episode inside its assigned terrain tile.
        extent = (math.ceil(math.sqrt(self.scene.num_envs)) + 1) * self.scene.env_spacing
        self.scene.terrain.terrain_generator.size = (extent, extent)
        self.scene.visual_terrain.terrain_generator = self.scene.terrain.terrain_generator.copy()
        self.scene.visual_terrain.mesh_origin_offset = (
            0.0,
            0.0,
            -self.scene.terrain.moving_patch_terrain.particle_depth,
        )
        super().validate_config()
        self.scene.sand.class_type = MixedMPMObject
        env_pattern = "(" + "|".join(str(index) for index in range(num_mpm_envs)) + ")"
        self.sim.physics.solver_cfg.proxies[0].bodies = [rf"/World/envs/env_{env_pattern}/Robot/.*ankle_roll_link"]
        for entry in self.sim.physics.solver_cfg.entries:
            if entry.name == RIGID_ENTRY:
                entry.shape_label_patterns = [r"/World/ground/(terrain|rigid_surface)/.*"]
            elif entry.name == MPM_ENTRY:
                entry.solver_cfg.max_active_cell_count //= 2
                entry.solver_cfg.max_leaf_node_count //= 2
                entry.solver_cfg.max_lower_node_count //= 2
                entry.solver_cfg.max_upper_node_count = max(32, entry.solver_cfg.max_upper_node_count // 2)


@configclass
class G1MixedTerrainEnvCfg_PLAY(G1MixedTerrainEnvCfg):
    """Play one environment of each ground type without pushes or observation noise."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 2
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class G1MixedTerrainPPORunnerCfg(G1MovingPatchPPORunnerCfg):
    """Keep the existing actor/critic contract and separate mixed-terrain training logs."""

    experiment_name = "g1_29dof_mpm_mixed_terrain"
    wandb_project = "g1_29dof_mpm_mixed_terrain"
