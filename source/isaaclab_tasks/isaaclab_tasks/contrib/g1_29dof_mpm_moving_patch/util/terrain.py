# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""One generated background surface and global support geometry shared by all MPM worlds."""

from __future__ import annotations

import math
from dataclasses import MISSING

import numpy as np
import trimesh
import warp as wp
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

from .kernel import sample_heights


@configclass
class MovingPatchTerrainCfg:
    """Alpha terrain: retain simulated particles and regenerate particles entering the patch."""

    moving_terrain_size: tuple[float, float] = (1.3, 1.3)
    """Width of continuously simulated terrain [m], along X and Y."""
    boundary_terrain_size: float = 0.2
    """Width of the zero-mass boundary on each side [m]."""
    tracked_body: str = "pelvis"
    """Exact robot body name used to center the patch in each environment."""
    particle_depth: float = 0.25
    """Vertical particle-layer depth below the undisturbed terrain surface [m]."""
    robot_spawn_height: float = 0.76
    """Initial robot root height above the sand surface [m]."""
    shift_step: float = 0.2
    """Patch-center quantization distance [m]."""
    voxel_size: float = 0.04
    """MPM background-grid voxel size [m]."""
    particles_per_cell: float = 1.25
    """Particle samples along each voxel dimension (IsaacLab MPMGridCfg convention)."""
    jitter: float = 0.05
    """MPMGridCfg jitter as a fraction of lattice spacing, not Newton's jitter distance."""
    material: MPMParticleMaterialCfg = MPMParticleMaterialCfg(
        density=2700.0,
        young_modulus=15.0e6,
        poisson_ratio=0.3,
        friction=math.tan(math.radians(40.0)),
        yield_pressure=1.0e12,
    )
    """Particle density and constitutive parameters, shared by initial and recycled particles."""
    floor_thickness: float = 0.1
    """Thickness of the static supporting floor below the sand [m]."""
    show_boundary_particles: bool = False
    """Render the kinematic boundary along with simulated particles."""
    visual_color: tuple[float, float, float] = (0.72, 0.55, 0.34)
    """Particle-emitter RGB color; viewer particle_color controls rendered particles."""

    @property
    def patch_size(self) -> tuple[float, float]:
        """Return the simulated-plus-boundary extent [m]."""
        x, y = self.moving_terrain_size
        return (x + 2.0 * self.boundary_terrain_size, y + 2.0 * self.boundary_terrain_size)

    def validate_geometry(self) -> None:
        """Reject invalid geometry before creating the scene."""
        for name in ("moving_terrain_size",):
            value = getattr(self, name)
            if len(value) != 2 or not all(math.isfinite(v) for v in value):
                raise ValueError(f"{name} must contain two finite values")
        positive = (
            *self.moving_terrain_size,
            self.boundary_terrain_size,
            self.particle_depth,
            self.shift_step,
            self.voxel_size,
            self.particles_per_cell,
            self.floor_thickness,
        )
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Terrain dimensions, sampling and shift_step must be positive and finite")
        if not 0 <= self.jitter <= 1:
            raise ValueError("Use jitter in [0, 1]")
        if not self.tracked_body or self.shift_step > self.boundary_terrain_size:
            raise ValueError("Set tracked_body and keep shift_step <= boundary_terrain_size")
        if self.boundary_terrain_size < 2 * self.voxel_size:
            raise ValueError("Use at least two MPM voxels of boundary terrain")


class WarpTerrainMesh:
    """Wrap terrain vertices and faces in a Warp mesh for GPU height queries.

    The mesh provides accelerated ray queries; bounds and spawn origins are
    used for particle placement. Particle rendering reads simulation positions.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, device: str):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        self.bounds = np.stack((self.vertices.min(axis=0), self.vertices.max(axis=0)))
        self.mesh = wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3, device=device),
            indices=wp.array(self.faces.flatten(), dtype=int, device=device),
        )
        self.ray_z = float(self.bounds[1, 2] + 1.0)
        self.ray_length = float(self.bounds[1, 2] - self.bounds[0, 2] + 2.0)
        self.spawn_origins = None
        self.initial_origins = None

    def sample(self, points: wp.array) -> None:
        """Replace world-space query points' Z coordinates by surface heights [m]."""
        missed = wp.zeros(1, dtype=int, device=self.mesh.device)
        wp.launch(
            sample_heights,
            dim=len(points),
            inputs=[
                self.mesh.id,  # mesh
                self.ray_z,  # ray_z
                self.ray_length,  # ray_length
                points,  # points
                missed,  # missed
            ],
            device=self.mesh.device,
        )
        if missed.numpy()[0]:
            raise ValueError("Shared terrain height query missed the surface; check terrain coverage and spawn spacing")


class BackgroundTerrainImporter(TerrainImporter):
    """Use one IsaacLab terrain with a global support collider for each subsolver."""

    def __init__(self, cfg: BackgroundTerrainImporterCfg):
        cfg.validate_geometry()
        super().__init__(cfg)

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        """Keep the generated surface for height queries and lower the supporting collider."""
        self.background_mesh = WarpTerrainMesh(mesh.vertices, mesh.faces, self.device)
        # Preserve the generated surface; only the supporting collider is lowered.
        ground_mesh = mesh.copy()
        ground_mesh.apply_translation((0, 0, -self.cfg.moving_patch_terrain.particle_depth))
        # Trimesh calls a closed mesh "watertight". Add sides and a bottom to
        # open terrain surfaces so the ground_mesh has a solid interior.
        # if not ground_mesh.is_watertight:
        #     ground_mesh.merge_vertices()
        #     n = len(ground_mesh.vertices)
        #     bottom = ground_mesh.vertices.copy()
        #     bottom[:, 2] = ground_mesh.bounds[0, 2] - self.cfg.moving_patch_terrain.floor_thickness
        #     edge_ids = trimesh.grouping.group_rows(ground_mesh.edges_sorted, require_count=1)
        #     edges = ground_mesh.edges[edge_ids]
        #     a, b = edges[:, 0], edges[:, 1]
        #     sides = np.stack((b, a, a + n, b, a + n, b + n), axis=1).reshape(-1, 3)
        #     ground_mesh = trimesh.Trimesh(
        #         vertices=np.concatenate((ground_mesh.vertices, bottom)),
        #         faces=np.concatenate((ground_mesh.faces, ground_mesh.faces[:, ::-1] + n, sides)),
        #     )
        #     ground_mesh.update_faces(ground_mesh.nondegenerate_faces())

        # physics_cfg.py assigns "terrain" to MJWarp and "mpm_ground_mesh" to MPM.
        # The coupler does not allow one collider to belong to both solvers.
        super().import_mesh(name, ground_mesh)
        super().import_mesh("mpm_support", ground_mesh)

        # for path in self.terrain_prim_paths:
        #     mpm = path.endswith("/mpm_support")
        #     # Importing enabled collision; this sets Newton contact margins.
        #     sim_utils.apply_collision_properties(
        #         path + "/mesh",
        #         [
        #             NewtonCollisionCfg(
        #                 contact_margin=self.cfg.mpm_contact_margin if mpm else self.cfg.rigid_contact_margin,
        #                 contact_gap=0.0 if mpm else self.cfg.rigid_contact_gap,
        #             )
        #         ],
        #         create_if_missing=True,
        #     )

    def _is_heightfield_collider_requested(self, cfg):
        # The installed implicit MPM collision path consumes triangle meshes.
        return False

    def configure_env_origins(self, origins=None):
        """Place robots on a regular grid over one terrain, sampling each spawn height."""
        if origins is not None:
            raise ValueError("Set use_terrain_origins=False for shared moving-patch terrain")
        super().configure_env_origins()
        self.background_mesh.initial_origins = self.env_origins.detach().cpu().numpy().copy()
        query = wp.from_torch(self.env_origins.contiguous(), dtype=wp.vec3)
        self.background_mesh.sample(query)
        self.background_mesh.spawn_origins = self.env_origins.detach().cpu().numpy().copy()
        half = np.asarray(self.cfg.moving_patch_terrain.patch_size) / 2
        xy = self.background_mesh.spawn_origins[:, :2]
        if np.any(xy - half < self.background_mesh.bounds[0, :2]) or np.any(
            xy + half > self.background_mesh.bounds[1, :2]
        ):
            raise ValueError(
                "Shared terrain must contain every initial patch; increase terrain size or reduce env_spacing"
            )


@configclass
class BackgroundTerrainImporterCfg(TerrainImporterCfg):
    """Standard generated-terrain settings plus the depth of its shared support surface."""

    class_type: type = BackgroundTerrainImporter
    use_terrain_origins: bool = False
    mpm_contact_margin: float = MISSING
    """MPM support collision margin [m], supplied by scene configuration."""
    rigid_contact_margin: float = MISSING
    """Rigid support collision margin [m], supplied by scene configuration."""
    rigid_contact_gap: float = MISSING
    """Rigid support contact detection gap [m], supplied by scene configuration."""

    moving_patch_terrain: MovingPatchTerrainCfg = MovingPatchTerrainCfg()
    """Particle sampling, material, simulated terrain and boundary settings."""

    def validate_geometry(self) -> None:
        """Validate the patch against the generator's configured terrain extent."""
        self.moving_patch_terrain.validate_geometry()
        if self.terrain_type != "generator" or self.terrain_generator is None:
            raise ValueError("Moving-patch background terrain requires terrain_type='generator'")
        if self.use_terrain_origins:
            raise ValueError("Set use_terrain_origins=False for moving-patch background terrain")
        generator = self.terrain_generator
        if len(generator.size) != 2 or not all(math.isfinite(v) and v > 0 for v in generator.size):
            raise ValueError("terrain_generator.size must contain two positive finite values")
        if generator.num_rows < 1 or generator.num_cols < 1:
            raise ValueError("Terrain generator row and column counts must be positive")
        size = (generator.size[0] * generator.num_rows, generator.size[1] * generator.num_cols)
        if any(b < p for b, p in zip(size, self.moving_patch_terrain.patch_size, strict=True)):
            raise ValueError("Generated background must contain simulated plus boundary terrain")
