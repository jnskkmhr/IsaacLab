# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Captured Newton particle-surface extraction with persistent viewer meshes."""

from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp
from newton.geometry import ParticleSurface

from .surface_kernels import (
    check_capacity,
    check_compacted_capacity,
    classify_surface_faces,
    compact_surface_faces,
    compact_surface_vertices,
    compare_tiles,
    gather_particles,
    mark_live_tiles,
    mask_surface_slots,
    merge_cached_vertices,
    particle_glyphs,
    retain_cached_faces,
    select_surface_particles,
)


class WorldSurface:
    """GPU reconstruction for one world; all particles contribute, including sleeping particles."""

    def __init__(self, cfg, device, global_ids, local_ids, radii, eligible=None):
        self.cfg = cfg
        self.device = device
        self.global_ids = global_ids
        self.local_ids = local_ids
        count = len(global_ids)
        self.positions = wp.empty(count, dtype=wp.vec3, device=device)
        self.sizes = wp.empty(count, dtype=float, device=device)
        self.radii = radii
        self.eligible = eligible
        self.extractor = ParticleSurface(
            cfg.resolution,
            kernel_radius=cfg.kernel_radius,
            threshold=cfg.density_threshold,
            smooth_lambda=0.0,
            max_grid_cells=cfg.max_grid_cells,
            device=device,
        )
        self.points = wp.zeros(cfg.max_vertices, dtype=wp.vec3, device=device)
        self.normals = wp.zeros_like(self.points)
        self.indices = wp.zeros(cfg.max_triangles * 3, dtype=int, device=device)
        self.graphs = {}
        self.revision = 0
        self.last_step = -1
        self.particle_step = -1
        self.particle_mesh = None
        self.mesh = None
        self.cached_points = wp.zeros_like(self.points)
        self.cached_normals = wp.zeros_like(self.normals)
        self.cached_indices = wp.zeros_like(self.indices)
        self.metadata = wp.zeros(5, dtype=int, device=device)
        self.host_metadata = None
        self.cache_revision = 0
        self.refresh = wp.zeros(1, dtype=int, device=device)
        self.reset = wp.ones(1, dtype=int, device=device)
        self.live_used = wp.zeros(cfg.max_vertices, dtype=int, device=device)
        self.cached_used = wp.zeros(2 * cfg.max_vertices, dtype=int, device=device)
        self.live_offsets = wp.zeros_like(self.live_used)
        self.cached_offsets = wp.zeros_like(self.cached_used)
        self.face_group = wp.zeros(cfg.max_triangles, dtype=int, device=device)
        self.cached_face_group = wp.zeros_like(self.face_group)
        self.cache_input_counts = wp.zeros(3, dtype=int, device=device)
        self.cache_spare_points = wp.zeros_like(self.points)
        self.cache_spare_normals = wp.zeros_like(self.normals)
        self.cache_spare_indices = wp.zeros_like(self.indices)
        self.flags = wp.ones(count, dtype=int, device=device)
        self.previous_q = wp.zeros_like(self.positions)
        self.activity_fallback = wp.ones(count, dtype=int, device=device)
        self.tile_origin = None
        self.empty_points = wp.zeros(3, dtype=wp.vec3, device=device)
        self.empty_indices = wp.zeros(3, dtype=int, device=device)

    def _gather(self, source):
        wp.launch(
            gather_particles,
            len(self.positions),
            [source, self.radii, self.global_ids, self.positions, self.sizes],
            device=self.device,
        )

    def _prepare(self, source, activity):
        self._gather(source)
        wp.copy(self.old_live, self.previous_live)
        self.live.zero_()
        self.refresh.zero_()
        wp.launch(
            mark_live_tiles,
            len(self.positions),
            [
                self.positions,
                self.previous_q,
                activity,
                self.local_ids,
                self.tile_origin,
                self.tile_shape,
                self.cfg.tile_size,
                self.cfg.kernel_radius + 2 * self.cfg.resolution,
                self.live,
            ],
            device=self.device,
        )
        wp.launch(
            compare_tiles,
            len(self.live),
            [self.live, self.previous_live, self.reset, self.refresh],
            device=self.device,
        )
        wp.launch(
            select_surface_particles,
            len(self.positions),
            [
                self.positions,
                self.tile_origin,
                self.tile_shape,
                self.cfg.tile_size,
                self.cfg.kernel_radius + 2 * self.cfg.resolution,
                self.live,
                self.reset,
                self.old_live,
                self.flags,
            ],
            device=self.device,
        )

        if self.eligible is not None:
            wp.launch(
                mask_surface_slots,
                len(self.positions),
                [self.eligible, self.global_ids, self.sizes, self.flags],
                device=self.device,
            )

    def _compact(self):
        slot = 0
        used, offsets = self.live_used, self.live_offsets
        points, normals, indices = self.points, self.normals, self.indices
        self.metadata[slot : slot + 2].zero_()
        indices.zero_()
        wp.utils.array_scan(used, offsets, inclusive=True)
        wp.launch(
            compact_surface_vertices,
            len(used),
            [self.mesh.vertices, self.mesh.normals, used, offsets, points, normals, self.metadata, slot],
            device=self.device,
        )
        wp.launch(
            compact_surface_faces,
            len(self.face_group),
            [
                self.mesh.indices,
                self.mesh._counts,
                self.face_group,
                1,
                offsets,
                indices,
                self.metadata,
                slot + 1,
                0,
            ],
            device=self.device,
        )

    def _compact_cached(self):
        wp.copy(self.cache_input_counts[:2], self.metadata[2:4])
        wp.launch(
            retain_cached_faces,
            len(self.cached_face_group),
            [
                self.cached_points,
                self.cached_indices,
                self.cache_input_counts,
                self.tile_origin,
                self.tile_shape,
                self.cfg.tile_size,
                self.live,
                self.old_live,
                self.reset,
                self.cached_used,
                self.cached_face_group,
                len(self.points),
            ],
            device=self.device,
        )
        self.metadata[2:4].zero_()
        self.cache_spare_indices.zero_()
        wp.utils.array_scan(self.cached_used, self.cached_offsets, inclusive=True)
        wp.launch(
            merge_cached_vertices,
            len(self.cached_used),
            [
                self.mesh.vertices,
                self.mesh.normals,
                self.cached_points,
                self.cached_normals,
                self.cached_used,
                self.cached_offsets,
                self.cache_spare_points,
                self.cache_spare_normals,
                self.metadata,
                len(self.points),
            ],
            device=self.device,
        )
        for source, counts, groups, offset in (
            (self.mesh.indices, self.mesh._counts, self.face_group, 0),
            (self.cached_indices, self.cache_input_counts, self.cached_face_group, len(self.points)),
        ):
            wp.launch(
                compact_surface_faces,
                len(groups),
                [source, counts, groups, 2, self.cached_offsets, self.cache_spare_indices, self.metadata, 3, offset],
                device=self.device,
            )
        wp.copy(self.cached_points, self.cache_spare_points)
        wp.copy(self.cached_normals, self.cache_spare_normals)
        wp.copy(self.cached_indices, self.cache_spare_indices)

    def _extract(self):
        # Extract live and newly sleeping tiles with a density-support halo. Untouched cached
        # triangles remain on the GPU; only reset reconstructs the complete bed.
        self.mesh = self.extractor.extract(self.positions, self.sizes, particle_flags=self.flags)
        wp.launch(check_capacity, 1, [self.mesh._counts, len(self.points), len(self.indices)], device=self.device)
        self.live_used.zero_()
        self.cached_used.zero_()
        wp.launch(
            classify_surface_faces,
            len(self.face_group),
            [
                self.mesh.vertices,
                self.mesh.indices,
                self.mesh._counts,
                self.tile_origin,
                self.tile_shape,
                self.cfg.tile_size,
                self.live,
                self.reset,
                self.old_live,
                self.live_used,
                self.cached_used,
                self.face_group,
            ],
            device=self.device,
        )
        self._compact()
        if wp.get_device(self.device).is_capturing:
            wp.capture_if(self.refresh, self._compact_cached)
        else:
            self._compact_cached()
        wp.launch(check_compacted_capacity, 1, [self.metadata, len(self.points), len(self.indices)], device=self.device)
        wp.copy(self.metadata[4:5], self.refresh)
        self.reset.zero_()

    def update(self, source, step, activity=None):
        if activity is None:
            activity = self.activity_fallback
        if self.tile_origin is None:
            self._gather(source)
            q = wp.to_torch(self.positions)
            if self.eligible is not None:
                q = q[(wp.to_torch(self.eligible)[wp.to_torch(self.global_ids).long()] & 1) != 0]
            # Six scalar bounds are read once to allocate the finite-bed tile map.
            bounds = torch.stack((q.amin(dim=0), q.amax(dim=0))).cpu().numpy()
            lower = np.floor((bounds[0, :2] - 1.0) / self.cfg.tile_size) * self.cfg.tile_size
            shape = np.ceil((bounds[1, :2] + 1.0 - lower) / self.cfg.tile_size).astype(int)
            self.tile_origin = wp.vec2(*lower)
            self.tile_shape = wp.vec2i(*shape)
            self.live = wp.zeros(int(np.prod(shape)), dtype=int, device=self.device)
            self.previous_live = wp.zeros_like(self.live)
            self.old_live = wp.zeros_like(self.live)
            wp.copy(self.previous_q, self.positions)
        graph = self.graphs.get((source.ptr, activity.ptr))
        if graph is None:
            self.reset.fill_(1)
            self._prepare(source, activity)
            self._extract()
            with wp.ScopedCapture(device=self.device) as capture:
                self._prepare(source, activity)
                self._extract()
            graph = self.graphs[source.ptr, activity.ptr] = capture.graph
            self.reset.fill_(1)
        wp.capture_launch(graph)
        self.last_step = step
        self.revision += 1
        self.host_metadata = None

    def view_meshes(self):
        """Read five control integers for viewer sizes; never transfer particle/mesh arrays."""
        if self.host_metadata is None:
            self.host_metadata = self.metadata.numpy()
            if self.host_metadata[4]:
                self.cache_revision += 1
        nv, ni, cv, ci, _ = (int(v) for v in self.host_metadata)
        live = (self.points[:nv], self.indices[:ni], self.normals[:nv])
        cached = (self.cached_points[:cv], self.cached_indices[:ci], self.cached_normals[:cv])
        return live, cached

    def update_particles(self, source, activity, directions, faces, step):
        if self.particle_mesh is None:
            count = len(self.positions)
            self.particle_mesh = (
                wp.empty(count * 6, dtype=wp.vec3, device=self.device),
                wp.empty(count * 24, dtype=int, device=self.device),
                wp.empty(count * 6, dtype=wp.vec3, device=self.device),
            )
        self._gather(source)
        points, indices, normals = self.particle_mesh
        wp.launch(
            particle_glyphs,
            (len(self.positions), 6),
            [self.positions, self.sizes, activity, self.local_ids, directions, faces, points, normals, indices],
            device=self.device,
        )
        self.particle_step = step


class SandSurface:
    """One persistent GPU mesh per world, shared by window and offline views."""

    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg.sand_surface
        cfg = self.cfg
        for name in ("resolution", "kernel_radius", "density_threshold", "tile_size"):
            if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError(f"sand_surface.{name} must be finite and positive")
        for name in ("max_grid_cells", "max_vertices", "max_triangles", "update_interval"):
            if not isinstance(getattr(cfg, name), int) or getattr(cfg, name) <= 0:
                raise ValueError(f"sand_surface.{name} must be a positive integer")
        self.backend = env.sim.backend_manager.backend
        self.entry = self.backend.solver._entries["sand"]
        self.worlds = {}
        # World membership is static. Resolve it once, never in the frame-update path.
        particle_world = wp.to_torch(self.entry.solver.model.particle_world)
        global_ids = wp.to_torch(self.entry.particle_indices)
        for world in range(env.num_envs):
            local_ids = torch.where(particle_world == world)[0]
            if len(local_ids):
                self.worlds[world] = WorldSurface(
                    cfg,
                    env.device,
                    wp.from_torch(global_ids[local_ids].to(torch.int32).contiguous()),
                    wp.from_torch(local_ids.to(torch.int32).contiguous()),
                    self.backend.model.particle_radius,
                    self.backend.model.particle_flags,
                )
        self.published = {}
        self.directions = wp.array(
            [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
            dtype=wp.vec3,
            device=env.device,
        )
        self.faces = wp.array(
            [0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5],
            dtype=int,
            device=env.device,
        )

    def invalidate(self, env_ids):
        """Refresh existing buffers after reset; no CPU particle state or graph rebuild."""
        # Invalidate all display worlds without copying the GPU reset-id tensor to the CPU.
        for surface in self.worlds.values():
            surface.last_step = surface.particle_step = -1
            surface.reset.fill_(1)

    def attach(self, visualizer):
        visualizer.sand_surface = self
        visualizer.sand_show_surface = self.cfg.show_surface
        visualizer._viewer.show_particles = self.cfg.show_particles
        visualizer._viewer._log_particles = lambda state: None

        def controls(imgui):
            if imgui.collapsing_header("Sand visualization"):
                _, visualizer.sand_show_surface = imgui.checkbox("Isosurface", visualizer.sand_show_surface)
                _, visualizer._viewer.show_particles = imgui.checkbox(
                    "Awake + support particles", visualizer._viewer.show_particles
                )

        visualizer._viewer.register_ui_callback(controls, position="side")

    def render(self, visualizer):
        step = self.env.sim.get_physics_step_count()
        show_surface = visualizer.sand_show_surface
        show_particles = visualizer._viewer.show_particles
        visible_worlds = getattr(visualizer, "_env_ids", None)
        source = self.backend.state_0.particle_q
        for world, surface in self.worlds.items():
            visible = visible_worlds is None or world in visible_worlds
            if (
                visible
                and show_surface
                and (surface.last_step < 0 or step - surface.last_step >= self.cfg.update_interval)
            ):
                surface.update(source, step, self.entry.solver.particle_activity)
            if visible and show_particles and surface.particle_step != step:
                surface.update_particles(source, self.entry.solver.particle_activity, self.directions, self.faces, step)
            if surface.mesh is not None:
                live, cached = surface.view_meshes()
                for suffix, mesh, revision in (
                    ("live", live, surface.revision),
                    ("cached", cached, surface.cache_revision),
                ):
                    empty = len(mesh[1]) == 0
                    if empty:
                        mesh = (surface.empty_points, surface.empty_indices, surface.empty_points)
                    self._publish(
                        visualizer,
                        f"sand_surface/world_{world}/{suffix}",
                        mesh,
                        revision,
                        empty or not (visible and show_surface),
                        self.env.cfg.scene.sand.spawn.visual_color,
                    )
            if surface.particle_mesh is not None:
                self._publish(
                    visualizer,
                    f"sand_particles/world_{world}",
                    surface.particle_mesh,
                    surface.particle_step,
                    not (visible and show_particles),
                    (0.72, 0.48, 0.18),
                )

    def _publish(self, visualizer, name, mesh, revision, hidden, color):
        signature = revision, hidden
        key = id(visualizer), name
        if self.published.get(key) == signature:
            return
        points, indices, normals = mesh
        visualizer.log_mesh(
            name,
            points,
            indices,
            normals=normals,
            hidden=hidden,
            color=color,
            roughness=1.0,
            backface_culling=False,
            dynamic=True,
        )
        self.published[key] = signature
        visualizer._last_render_step = -1
        visualizer.current_image_view.invalidate()
