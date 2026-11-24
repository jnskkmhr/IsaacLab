# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mixed MJWarp/MPM variant of the tow task: half of the environments tow on rigid ground, half on sand.

The flat terrain has two columns, laid out as in the mixed walking task: the terrain is rigid ground everywhere,
the first column is walked on directly and the second carries a sand layer on top of it. Only the environments
placed on sand hold sand, and an episode ends as a time-out when the robot leaves its patch, so a robot is never
on the other ground. Rigid-ground environments use a zero contact margin on the feet and the sled, and sand
environments the MPM margin.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import trimesh

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.cloner import InclusionSet
from isaaclab.cloner.cloner_strategies import sequential
from isaaclab.managers import TerminationTermCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.velocity_mpm.config.g1_29dof.mdp as mpm_mdp
from isaaclab_tasks.contrib.velocity_mpm.config.g1_29dof.streaming_mpm_mixed_env import MixedMPMObject
from isaaclab_tasks.contrib.velocity_mpm.config.g1_29dof.streaming_mpm_mixed_env_cfg import (
    MixedMPMTerrainImporter,
    MixedMPMTerrainImporterCfg,
)
from isaaclab_tasks.contrib.velocity_mpm.util.solver_setting import (
    SPARSE_MPM_MIN_TOTAL_UPPER_NODE_COUNT,
    get_mpm_solver_cfg,
)
from isaaclab_tasks.contrib.velocity_mpm.util.terrain import TexturedTerrainImporter, WarpTerrainMesh

from .env_cfg import G1TerminationsCfg, terrain_generator_cfg
from .env_cfg.scene_cfg import FOOT_CONTACT_MARGIN, SLED_FRICTION
from .tow_env_cfg import G1TowEnvCfg, G1TowEnvCfg_PLAY

MIXED_FLAT_TERRAINS_CFG = terrain_generator_cfg.FLAT_TERRAINS_CFG.replace(num_cols=2)
"""The flat terrain with two adjacent columns: the first rigid, the second sand."""

SPAWN_HEIGHT_OFFSET = 0.05
"""Extra root height at reset [m]; rigid ground with a zero margin does not yield to a sole that starts inside it."""


class TowMixedTerrainImporter(MixedMPMTerrainImporter):
    """The mixed terrain for flat tiles, which are two triangles each.

    The walking task's importer raises the sand columns vertex by vertex. A flat tile has vertices only on its
    border, shared with the neighbouring column, so both columns would come out tilted. Here whole triangles are
    raised instead, which leaves a step of one sand depth between the columns.
    """

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        """Import the generated terrain unchanged and derive the walking surface and the sand support from it."""
        if self.cfg.terrain_generator.num_cols % 2:
            raise ValueError("Mixed terrain requires an even number of columns for rigid/sand pairs.")
        if any(self.cfg.mesh_origin_offset):
            raise ValueError("Mixed terrain requires mesh_origin_offset=(0, 0, 0).")
        # Walking surface: the terrain itself in rigid columns, the top of the sand layer elsewhere.
        generator = self.cfg.terrain_generator
        triangles = np.array(mesh.triangles)
        centers = triangles.mean(axis=1)
        row = np.floor(centers[:, 0] / generator.size[0] + generator.num_rows / 2)
        column = np.floor(centers[:, 1] / generator.size[1] + generator.num_cols / 2)
        # The border around the tiles is rigid.
        inside = (row >= 0) & (row < generator.num_rows) & (column >= 0) & (column < generator.num_cols)
        triangles[inside & (column % 2 == 1), :, 2] += self.cfg.sand.particle_depth
        walking_surface = trimesh.Trimesh(
            vertices=triangles.reshape(-1, 3), faces=np.arange(3 * len(triangles)).reshape(-1, 3), process=False
        )
        self.background_mesh = WarpTerrainMesh(walking_surface.vertices, walking_surface.faces, self.device)
        # Support of the sand: the terrain under the sand columns.
        support_mesh = walking_surface.copy()
        support_mesh.apply_translation((0, 0, -self.cfg.sand.particle_depth))

        # physics_cfg.py assigns "terrain" to MJWarp and "mpm_ground" to MPM.
        TexturedTerrainImporter.import_mesh(self, name, mesh)
        TexturedTerrainImporter.import_mesh(self, "mpm_ground", support_mesh)
        UsdGeom.Imageable(
            sim_utils.get_current_stage().GetPrimAtPath(f"{self.cfg.prim_path}/mpm_ground")
        ).MakeInvisible()


@configclass
class G1TowMixedTerminationsCfg(G1TerminationsCfg):
    """Termination terms of the mixed tow task."""

    out_of_patch = TerminationTermCfg(func=mpm_mdp.root_out_of_patch, params={"margin": 0.5}, time_out=True)


@configclass
class G1TowMixedEnvCfg(G1TowEnvCfg):
    """The tow task on mixed rigid/sand flat terrain."""

    terminations: G1TowMixedTerminationsCfg = G1TowMixedTerminationsCfg()

    mpm_contact_margin: float = FOOT_CONTACT_MARGIN
    """Fixed foot and sled collider margin in sand environments [m]."""

    rigid_contact_margin: float = 0.0
    """Fixed foot and sled collider margin in rigid-ground environments [m]."""

    def __post_init__(self):
        super().__post_init__()
        terrain = self.scene.terrain
        self.scene.terrain = MixedMPMTerrainImporterCfg(
            **{f.name: getattr(terrain, f.name) for f in dataclasses.fields(terrain) if f.name != "class_type"}
        )
        self.scene.terrain.class_type = TowMixedTerrainImporter
        self.scene.terrain.terrain_generator = MIXED_FLAT_TERRAINS_CFG.copy()
        self.scene.terrain.sand.robot_spawn_height += SPAWN_HEIGHT_OFFSET
        # Contact friction is the larger of the two shapes' coefficients, so the rigid ground must not exceed the
        # sled's value for the sled to slide at ``SLED_FRICTION``, as in the rigid tow task.
        self.scene.terrain.physics_material = RigidBodyMaterialBaseCfg(
            static_friction=SLED_FRICTION, dynamic_friction=SLED_FRICTION
        )

    def validate_config(self) -> None:
        if self.foot_contact_margin_range is not None:
            raise ValueError("Mixed terrain uses mpm_contact_margin and rigid_contact_margin, not a margin range.")
        for name in ("mpm_contact_margin", "rigid_contact_margin"):
            margin = getattr(self, name)
            if not math.isfinite(margin) or margin < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        fraction = self.scene.terrain.sand_env_fraction
        if not 0.0 < fraction <= 1.0:
            raise ValueError("sand_env_fraction must be in (0, 1].")
        super().validate_config()
        # Only sand environments hold the sand asset, as in the mixed walking task.
        self.scene.sand.class_type = MixedMPMObject
        self.scene.clone_cfg.clone_combinations = [
            InclusionSet(assets=["sand"], weight=fraction),
            InclusionSet(assets=[], weight=1.0 - fraction),
        ]
        # The sparse MPM storage was sized for sand in every environment.
        sand_envs = int((sequential(np.array([fraction, 1.0 - fraction]), self.scene.num_envs) == 0).sum())
        if sand_envs == 0:
            raise ValueError("No environment tows on sand; raise num_envs or sand_env_fraction.")
        mpm = get_mpm_solver_cfg(self)
        for name in ("max_active_cell_count", "max_leaf_node_count", "max_lower_node_count"):
            setattr(mpm, name, getattr(mpm, name) // self.scene.num_envs * sand_envs)
        mpm.max_upper_node_count = max(
            SPARSE_MPM_MIN_TOTAL_UPPER_NODE_COUNT, self.mpm_upper_node_count_per_world * sand_envs
        )


@configclass
class G1TowMixedEnvCfg_PLAY(G1TowMixedEnvCfg, G1TowEnvCfg_PLAY):
    """Playback of the tow task on mixed rigid/sand flat terrain: one sand and one rigid environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2
