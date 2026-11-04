# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain generators, particle sampling, patch boundaries and background configuration."""

import math
from dataclasses import MISSING

from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.g1_29dof_mpm_moving_patch.util.terrain import (
    BackgroundTerrainImporter,
    TexturedTerrainImporter,
)

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR


@configclass
class TexturedTerrainImporterCfg(TerrainImporterCfg):
    """Standard terrain import with a portable color texture for GL and RTX."""

    class_type: type = TexturedTerrainImporter

    mesh_origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation of imported mesh vertices along the terrain's local axes [m].

    Does not shift environment origins or the source mesh. Only applies to mesh imports.
    """
    visual_material: sim_utils.MdlFileCfg = sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_080/Ground080_4K.mdl",
        project_uvw=False,
        texture_scale=(1.0, 1.0),
    )
    color_texture: str | None = f"{ISAACLAB_ASSETS_DATA_DIR}/texture/Ground_080/Ground080_4K-PNG_Color.png"
    """Color texture for the USD preview material; None uses color."""
    texture_repeat_per_meter: float = 0.125
    """Number of texture repeats per world-space distance [1/m]."""
    color: tuple[float, float, float] = (0.72, 0.55, 0.34)
    """Untextured preview material RGB color."""


@configclass
class MovingPatchTerrainCfg:
    """Geometry and material settings for the simulated patch and its kinematic boundary."""

    simulated_terrain_size: tuple[float, float] = MISSING
    """Width of continuously simulated terrain [m], along X and Y."""
    boundary_terrain_size: float = MISSING
    """Width of the zero-mass boundary on each side [m]."""
    tracked_body: str = "base"
    """Exact robot body name used to center the patch in each environment."""
    particle_depth: float = MISSING
    """Vertical particle-layer depth below the undisturbed terrain surface [m]."""
    robot_spawn_height: float = MISSING
    """Initial robot root height above the sand surface [m]."""
    patch_discretization_step: float = MISSING
    """Patch-center quantization distance [m]."""
    voxel_size: float = MISSING
    """Particle sampling voxel size [m]; no native MPM grid is constructed."""
    particles_per_cell: float = MISSING
    """Particle samples along each voxel dimension (IsaacLab MPMGridCfg convention)."""
    jitter: float = MISSING
    """Total width of the initial particle-position jitter interval [m].

    Each X, Y, and Z coordinate receives an independent uniform offset in
    ``[-jitter / 2, jitter / 2]``, matching :class:`MPMGridCfg`.
    """
    material: MPMParticleMaterialCfg = MISSING
    """Particle density and constitutive parameters, shared by initial and recycled particles."""
    floor_thickness: float = MISSING
    """Thickness of the static supporting floor below the sand [m]."""
    show_boundary_particles: bool = False
    """Render the kinematic boundary along with simulated particles."""
    visual_color: tuple[float, float, float] = (0.72, 0.55, 0.34)
    """Particle-emitter RGB color; viewer particle_color controls rendered particles."""

    @property
    def total_patch_size(self) -> tuple[float, float]:
        """Return the simulated-plus-boundary extent [m]."""
        simulated_size_x, simulated_size_y = self.simulated_terrain_size
        return (
            simulated_size_x + 2.0 * self.boundary_terrain_size,
            simulated_size_y + 2.0 * self.boundary_terrain_size,
        )

    def validate_geometry(self) -> None:
        """Reject invalid geometry before creating the scene."""
        for dimension_name in ("simulated_terrain_size",):
            dimensions = getattr(self, dimension_name)
            if len(dimensions) != 2 or not all(math.isfinite(dimension) for dimension in dimensions):
                raise ValueError(f"{dimension_name} must contain two finite values")
        positive_dimensions = (
            *self.simulated_terrain_size,
            self.boundary_terrain_size,
            self.particle_depth,
            self.patch_discretization_step,
            self.voxel_size,
            self.particles_per_cell,
            self.floor_thickness,
        )
        if not all(math.isfinite(dimension) and dimension > 0 for dimension in positive_dimensions):
            raise ValueError("Terrain dimensions, sampling and patch_discretization_step must be positive and finite")
        if not math.isfinite(self.jitter) or self.jitter < 0.0:
            raise ValueError("Particle jitter width must be finite and non-negative [m]")
        if not self.tracked_body or self.patch_discretization_step > self.boundary_terrain_size:
            raise ValueError("Set tracked_body and keep patch_discretization_step <= boundary_terrain_size")
        if self.boundary_terrain_size < 2 * self.voxel_size:
            raise ValueError("Use at least two sampling voxels of boundary terrain")


@configclass
class BackgroundTerrainImporterCfg(TexturedTerrainImporterCfg):
    """Standard generated-terrain settings plus the depth of its shared support surface."""

    class_type: type = BackgroundTerrainImporter
    use_terrain_origins: bool = False

    moving_patch_terrain: MovingPatchTerrainCfg = MISSING
    """Particle sampling, material, simulated terrain and boundary settings."""

    def validate_geometry(self) -> None:
        """Validate the patch against the generator's configured terrain extent."""
        self.moving_patch_terrain.validate_geometry()
        if self.terrain_type != "generator" or self.terrain_generator is None:
            raise ValueError("Moving-patch background terrain requires terrain_type='generator'")
        terrain_generator_cfg = self.terrain_generator
        if len(terrain_generator_cfg.size) != 2 or not all(
            math.isfinite(dimension) and dimension > 0 for dimension in terrain_generator_cfg.size
        ):
            raise ValueError("terrain_generator.size must contain two positive finite values")
        if terrain_generator_cfg.num_rows < 1 or terrain_generator_cfg.num_cols < 1:
            raise ValueError("Terrain generator row and column counts must be positive")
        background_terrain_size = (
            terrain_generator_cfg.size[0] * terrain_generator_cfg.num_rows,
            terrain_generator_cfg.size[1] * terrain_generator_cfg.num_cols,
        )
        if any(
            background_extent < patch_extent
            for background_extent, patch_extent in zip(
                background_terrain_size, self.moving_patch_terrain.total_patch_size, strict=True
            )
        ):
            raise ValueError("Generated background must contain simulated plus boundary terrain")


FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(20.0, 20.0),
    num_rows=1,
    num_cols=1,
    border_width=0.0,
    horizontal_scale=0.1,
    curriculum=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(),
    },
)

WAVE_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(20.0, 20.0),
    num_rows=1,
    num_cols=1,
    border_width=0.0,
    horizontal_scale=0.1,
    curriculum=False,
    sub_terrains={
        "waves": terrain_gen.HfWaveTerrainCfg(
            amplitude_range=(0.4, 0.4),
            num_waves=4,
            border_width=0.5,
        ),
    },
)

SLOPE_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(20.0, 20.0),
    border_width=5.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     slope_range=(0.4, 0.4), platform_width=2.0, border_width=0.25
        # ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            slope_range=(math.pi * (20.0 / 180.0), math.pi * (20.0 / 180.0)), platform_width=2.0, border_width=0.25
        ),
    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=10,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2,
            amplitude_range=(0.05, 0.4),
            num_waves=4,
            border_width=0.25,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
    },
)


ROUGH_TERRAINS_DIFFICULT_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(20.0, 20.0),
    border_width=5.0,
    num_rows=1,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2,
            amplitude_range=(0.4, 0.4),
            num_waves=4,
            border_width=0.25,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.1, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.4, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.4, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
