# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the fixed bed from live particles using Felix's marching-cubes extractor."""

from types import SimpleNamespace

import numpy as np
import torch
import warp as wp
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer import (
    NewtonGLVisualizer,
    NewtonRTXVisualizer,
    NewtonViewerGL,
    NewtonViewerRTX,
)

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity_mpm.g1_29dof_mpm_moving_patch_particle_cache.util.felix.paged_sand_isosurface import (  # noqa: E501
    ISO_SKIRT,
    IsoSurface,
)


@configclass
class SandSurfaceCfg:
    """Display-only surface settings, independent of the physics grid."""

    resolution: float = 0.02
    """Surface grid spacing in meters."""
    smoothing: float = 1.5
    """B-spline width relative to the initial particle spacing; at least one."""
    update_interval: int = 2
    """Reconstruct every this many rendered frames, including video frames."""
    color: tuple[float, float, float] = (0.72, 0.55, 0.34)
    """Sand RGB color shared by the surface and optional particles."""


class FixedBedSurface:
    """Retain one extraction grid per visible world; read only live particle positions."""

    def __init__(self, cfg: SandSurfaceCfg, model):
        if not np.isfinite(cfg.resolution) or cfg.resolution <= 0:
            raise ValueError("Surface resolution must be finite and positive")
        if not np.isfinite(cfg.smoothing) or cfg.smoothing < 1:
            raise ValueError("Surface smoothing must be finite and at least one")
        if cfg.update_interval < 1:
            raise ValueError("Surface update_interval must be positive")
        self.cfg = cfg
        self.model = model
        self.particle_ids = {}
        self.colors = {}
        self.empty_points = wp.empty(0, dtype=wp.vec3, device=model.device)
        self.extractors = {}
        self.meshes = {}
        self.shown = set()
        self.frame = 0

    def extract(self, world: int, positions: torch.Tensor, radius: float):
        """Extract a closed sand surface without synthetic terrain or cached particles."""
        h = self.cfg.resolution
        # Quantize bounds to tiles and grow only when particles leave the previous grid.
        n = max(2, round(0.5 / h))
        tile_size = n * h
        bounds = torch.stack((positions.amin(dim=0), positions.amax(dim=0))).cpu().numpy()
        padding = max(3 * h, 3 * radius * self.cfg.smoothing)
        lower = np.floor((bounds[0] - padding) / tile_size) * tile_size
        upper = np.ceil((bounds[1] + padding) / tile_size) * tile_size
        previous = self.extractors.get(world)
        if previous is not None:
            lower = np.minimum(lower, previous[0])
            upper = np.maximum(upper, previous[1])
        if previous is None or not (np.array_equal(lower, previous[0]) and np.array_equal(upper, previous[1])):
            shape = np.rint((upper[:2] - lower[:2]) / tile_size).astype(int)
            grid = SimpleNamespace(lo=lower[:2], ncx=int(shape[0]), ncy=int(shape[1]), chunk_count=int(np.prod(shape)))
            heights = wp.zeros((int(shape[0]) * n, int(shape[1]) * n), dtype=float)
            surface = IsoSurface(grid, heights, n, h, 1, 0.0, 0.0, float(upper[2] - lower[2]), grid.chunk_count, 0)
            # Every tile is backed by live density. No pristine fill, boundary blending, or sleeping cache.
            surface.backed.fill_(0)
            ids = np.arange(grid.chunk_count, dtype=np.int32)
            self.extractors[world] = lower, upper, surface, ids
        else:
            _, _, surface, ids = previous
        q = positions.clone()
        q[:, 2] -= float(lower[2])
        mesh, nv, nt = surface.extract(
            surface.live,
            ids,
            wp.from_torch(q, dtype=wp.vec3),
            wp.ones(len(q), dtype=int),
            particle_weight=(2 * radius / h) ** 3,
            particle_spacing=2 * radius * self.cfg.smoothing,
        )
        vertices = wp.to_torch(mesh.points)[:nv].clone()
        vertices[:, 2] += float(lower[2])
        # Felix's terrain renderer adds a bottom plane and tile skirts. A finite bed
        # needs only the particle-density surface, including its actual underside.
        faces = wp.to_torch(mesh.indices)[: 3 * nt].reshape(-1, 3).long()
        skirt = (wp.to_torch(mesh.face_bits)[:nv] & int(ISO_SKIRT)) != 0
        keep = ~skirt[faces].any(dim=1) & (vertices[faces, 2] > float(lower[2]) + h).all(dim=1)
        indices = faces[keep].to(torch.int32).flatten().contiguous()
        return wp.from_torch(vertices, dtype=wp.vec3), wp.from_torch(indices), mesh.normals[:nv]

    def update(self, viewer, state, worlds, show_surface: bool, show_particles: bool):
        """Rebuild only selected worlds and hide old meshes when the selection changes."""
        visible = set(range(self.model.world_count) if worlds is None else worlds)
        active = visible if show_surface or show_particles else set()
        for world in self.shown - active:
            self._log(viewer, world, hidden=True)
        for world in sorted(active):
            if world not in self.particle_ids:
                ids = torch.where(wp.to_torch(self.model.particle_world) == world)[0]
                self.particle_ids[world] = ids
                self.colors[world] = wp.full(
                    len(ids), wp.vec3(*self.cfg.color), dtype=wp.vec3, device=self.model.device
                )
            ids = self.particle_ids[world]
            if not len(ids):
                continue
            positions = wp.to_torch(state.particle_q)[ids].contiguous()
            radii = wp.to_torch(self.model.particle_radius)[ids].contiguous()
            if show_surface and (
                world not in self.meshes or world not in self.shown or self.frame % self.cfg.update_interval == 0
            ):
                with wp.ScopedDevice(state.particle_q.device):
                    self.meshes[world] = self.extract(world, positions, float(radii[0]))
            self._log(viewer, world, hidden=not show_surface)
            viewer.log_points(
                f"/fixed_bed/world_{world}/particles",
                wp.from_torch(positions, dtype=wp.vec3) if show_particles else self.empty_points,
                radii=wp.from_torch(radii) if show_particles else None,
                colors=self.colors[world] if show_particles else None,
                hidden=not show_particles,
            )
        self.shown = active
        self.frame += 1

    def _log(self, viewer, world: int, hidden: bool):
        if world in self.meshes:
            points, indices, normals = self.meshes[world]
            viewer.log_mesh(
                f"/fixed_bed/world_{world}/surface",
                points,
                indices,
                normals=normals,
                color=self.cfg.color,
                roughness=1.0,
                dynamic=True,
                hidden=hidden,
            )
        if hidden and world in self.shown:
            # Hide a previously submitted point batch without uploading particle positions.
            viewer.log_points(f"/fixed_bed/world_{world}/particles", points=self.empty_points, hidden=True)


class FixedBedRendering:
    """Share surface rendering and GUI toggles between GL and RTX."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.surface = None
        self.register_ui_callback(self._draw_sand_controls, position="side")

    def _draw_sand_controls(self, imgui):
        if imgui.collapsing_header("Sand visualization"):
            _, self.show_particles = imgui.checkbox("Particles##fixed_bed", self.show_particles)
            _, self.show_isosurface = imgui.checkbox("Isosurface##fixed_bed", self.show_isosurface)

    def _log_particles(self, state):
        if self.surface is None or self.surface.model is not self.model:
            self.surface = FixedBedSurface(self.fixed_bed_visualizer.cfg.surface, self.model)
        self.surface.update(
            self, state, self.fixed_bed_visualizer.get_visualized_env_ids(), self.show_isosurface, self.show_particles
        )


class FixedBedViewerGL(FixedBedRendering, NewtonViewerGL):
    """Newton GL with a live sand surface."""


class FixedBedViewerRTX(FixedBedRendering, NewtonViewerRTX):
    """Newton RTX with a live sand surface on its borrowed stage."""


class FixedBedGLVisualizer(NewtonGLVisualizer):
    """Create the GL viewer with fixed-bed rendering controls."""

    def _create_viewer(self, **kwargs):
        viewer = FixedBedViewerGL(**kwargs)
        viewer.fixed_bed_visualizer = self
        viewer.show_particles = self.cfg.show_particles
        viewer.show_isosurface = self.cfg.show_isosurface
        return viewer


class FixedBedRTXVisualizer(NewtonRTXVisualizer):
    """Create the RTX viewer with fixed-bed rendering controls."""

    def _create_viewer(self, **kwargs):
        viewer = FixedBedViewerRTX(**kwargs)
        viewer.fixed_bed_visualizer = self
        viewer.show_particles = self.cfg.show_particles
        viewer.show_isosurface = self.cfg.show_isosurface
        return viewer


@configclass
class FixedBedGLVisualizerCfg(NewtonGLVisualizerCfg):
    """GL surface rendering; reconstruct only one environment by default."""

    class_type = FixedBedGLVisualizer
    show_particles: bool = False
    show_isosurface: bool = True
    surface: SandSurfaceCfg = SandSurfaceCfg()
    max_visible_envs: int | None = 1
    randomly_sample_visible_envs: bool = False


@configclass
class FixedBedRTXVisualizerCfg(NewtonRTXVisualizerCfg):
    """RTX surface rendering; authored particle geometry stays hidden."""

    class_type = FixedBedRTXVisualizer
    show_particles: bool = False
    show_isosurface: bool = True
    surface: SandSurfaceCfg = SandSurfaceCfg()
    max_visible_envs: int | None = 1
    randomly_sample_visible_envs: bool = False
