# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mixed MJWarp/MPM variant of the streaming G1 task: every rough terrain type once as rigid ground, once as sand.

The terrain generator is the rough one with twice the columns, so each terrain type fills two adjacent columns.
The generated terrain is rigid ground everywhere. The first column of each pair is walked on directly; the second
carries a sand layer on top of that ground. Only the environments placed on sand hold sand, and an episode ends
as a time-out when the robot leaves its patch, so a robot is never on the other ground. Rigid-ground environments
use a zero foot contact margin and sand environments the MPM margin.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch
import trimesh

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.cloner import InclusionSet
from isaaclab.cloner.cloner_strategies import sequential
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from ...util.solver_setting import SPARSE_MPM_MIN_TOTAL_UPPER_NODE_COUNT, get_mpm_solver_cfg
from ...util.terrain import MPMTerrainImporter, MPMTerrainImporterCfg, TexturedTerrainImporter, WarpTerrainMesh
from . import mdp as mpm_mdp
from .env_cfg import terrain_generator_cfg
from .env_cfg.physics_cfg import FOOT_CONTACT_MARGIN
from .env_cfg.termination_cfg import G1TerminationsCfg
from .streaming_mpm_env_cfg import G1MPMEnvCfg, G1MPMEnvCfg_PLAY
from .streaming_mpm_mixed_env import MixedMPMObject, randomize_mpm_material

SPAWN_HEIGHT_OFFSET = 0.05
"""Extra root height at reset [m].

The reset samples the terrain height under the pelvis only, so on rough ground a sole can start a few centimetres
inside the surface. Sand yields to that; rigid ground with a zero margin does not.
"""


class MixedMPMTerrainImporter(MPMTerrainImporter):
    """One rigid terrain for every column, with a sand layer on top of it in the second column of each pair.

    The generated terrain is the MJWarp collider as it is, so rigid-ground robots walk on it directly. In sand
    columns the sand rests on that same ground: the walking surface there is the terrain raised by the sand
    depth. Everything the sand needs (surface queries, streaming, its own support collider) uses that walking
    surface, as :class:`MPMTerrainImporter` does with the generated surface.
    """

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        """Import the generated terrain unchanged and derive the walking surface and the sand support from it."""
        if self.cfg.terrain_generator.num_cols % 2:
            raise ValueError("Mixed terrain requires an even number of columns for rigid/sand pairs.")
        if any(self.cfg.mesh_origin_offset):
            raise ValueError("Mixed terrain requires mesh_origin_offset=(0, 0, 0).")
        # Walking surface: the terrain itself in rigid columns, the top of the sand layer elsewhere.
        generator = self.cfg.terrain_generator
        walking_surface = mesh.copy()
        vertices = np.array(walking_surface.vertices)
        row = np.floor(vertices[:, 0] / generator.size[0] + generator.num_rows / 2)
        column = np.floor(vertices[:, 1] / generator.size[1] + generator.num_cols / 2)
        # The border around the tiles is rigid.
        inside = (row >= 0) & (row < generator.num_rows) & (column >= 0) & (column < generator.num_cols)
        vertices[inside & (column % 2 == 1), 2] += self.cfg.sand.particle_depth
        walking_surface.vertices = vertices
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

    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        """Place sand environments on the sand columns and the others on the rigid columns."""
        super()._compute_env_origins_curriculum(num_envs, origins)
        # The scene's clone plan gives sand to the first environments, by the same rule.
        fraction = self.cfg.sand_env_fraction
        sand = sequential(np.array([fraction, 1.0 - fraction]), num_envs) == 0
        self.mpm_env_mask = torch.as_tensor(sand, device=self.device)
        """Environments that walk on sand, shape ``(num_envs,)``."""
        # Each group is spread over the terrain types, on its own column of every pair.
        pairs = origins.shape[1] // 2
        for mask, column in ((self.mpm_env_mask, 1), (~self.mpm_env_mask, 0)):
            count = int(mask.sum())
            pair = torch.div(torch.arange(count, device=self.device) * pairs, max(count, 1), rounding_mode="floor")
            self.terrain_types[mask] = 2 * pair + column
        return origins[self.terrain_levels, self.terrain_types].clone()


@configclass
class MixedMPMTerrainImporterCfg(MPMTerrainImporterCfg):
    """Generated terrain whose columns alternate between rigid ground and sand."""

    class_type: type = MixedMPMTerrainImporter

    sand_env_fraction: float = 0.5
    """Share of the environments that walk on sand, in (0, 1]; the others walk on rigid ground and hold no sand.

    The first environments are the sand ones. A single environment walks on sand when this is above 0.5.
    """


@configclass
class G1MixedTerminationsCfg(G1TerminationsCfg):
    """Termination terms of the mixed task."""

    out_of_patch = TerminationTermCfg(func=mpm_mdp.root_out_of_patch, params={"margin": 0.5}, time_out=True)


@configclass
class G1MPMMixedEnvCfg(G1MPMEnvCfg):
    """The streaming G1 task on mixed rigid/sand terrain."""

    terminations: G1MixedTerminationsCfg = G1MixedTerminationsCfg()

    mpm_contact_margin: float = FOOT_CONTACT_MARGIN
    """Margin by which the sand stays clear of the robot's feet [m]; given to the MPM solver only."""

    rigid_contact_margin: float = 0.0
    """Contact margin of the feet's rigid contact, in sand and rigid-ground environments [m]."""

    def __post_init__(self):
        super().__post_init__()
        terrain = self.scene.terrain
        self.scene.terrain = MixedMPMTerrainImporterCfg(
            **{f.name: getattr(terrain, f.name) for f in dataclasses.fields(terrain) if f.name != "class_type"}
        )
        self.scene.terrain.sand.robot_spawn_height += SPAWN_HEIGHT_OFFSET
        for name in ("mpm_material", "mpm_material_log"):
            if getattr(self.events, name) is not None:
                getattr(self.events, name).func = randomize_mpm_material

    def validate_config(self) -> None:
        for name in ("mpm_contact_margin", "rigid_contact_margin"):
            margin = getattr(self, name)
            if not math.isfinite(margin) or margin < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        fraction = self.scene.terrain.sand_env_fraction
        if not 0.0 < fraction <= 1.0:
            raise ValueError("sand_env_fraction must be in (0, 1].")
        super().validate_config()
        if not self.scene.terrain.use_terrain_origins:
            raise ValueError("Mixed terrain places environments on terrain columns; keep the terrain curriculum.")
        # Only sand environments hold the sand asset.
        self.scene.sand.class_type = MixedMPMObject
        self.scene.clone_cfg.clone_combinations = [
            InclusionSet(assets=["sand"], weight=fraction),
            InclusionSet(assets=[], weight=1.0 - fraction),
        ]
        # The sparse MPM storage was sized for sand in every environment.
        sand_envs = int((sequential(np.array([fraction, 1.0 - fraction]), self.scene.num_envs) == 0).sum())
        if sand_envs == 0:
            raise ValueError("No environment walks on sand; raise num_envs or sand_env_fraction.")
        mpm = get_mpm_solver_cfg(self)
        for name in ("max_active_cell_count", "max_leaf_node_count", "max_lower_node_count"):
            setattr(mpm, name, getattr(mpm, name) // self.scene.num_envs * sand_envs)
        mpm.max_upper_node_count = max(
            SPARSE_MPM_MIN_TOTAL_UPPER_NODE_COUNT, self.mpm_upper_node_count_per_world * sand_envs
        )


@configclass
class G1MPMMixedEnvCfg_PLAY(G1MPMMixedEnvCfg, G1MPMEnvCfg_PLAY):
    """Playback of the streaming G1 task on mixed rigid/sand terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = terrain_generator_cfg.ROUGH_TERRAINS_CFG.copy()
        self.events.reset_base.params["pose_range"].update({"x": (-3.0, 3.0), "y": (-3.0, 3.0)})
