# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render sand only within the currently loaded moving patches."""

from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from .paged_sand_isosurface import ISO_SKIRT_CELLS, IsoSurface
from .surface_kernel import blend_surface_boundary


class ParticleCacheSurface:
    """Extract loaded particles and cover unloaded regions with the background terrain.

    Saved particle history remains available for physics when an area is revisited,
    but does not contribute meshes or per-frame traversal to surface rendering.
    """

    def __init__(self, patch, backend):
        smoothing = patch.terrain.particle_cache.surface_smoothing
        if not np.isfinite(smoothing) or smoothing < 1.0:
            raise ValueError("particle_cache.surface_smoothing must be finite and at least 1.0")
        self.patch = patch
        self.backend = backend
        self.extractors = {}
        self.shown = set()
        self.meshes = {}
        self.background = None
        self.background_tiles = None
        self.surface_levels = {}
        self.background_levels = None
        self.background_points = None
        self.background_normals = None
        self.particle_boundary = None
        self.textured = True
        self.send_texture_once = False
        self._textured_meshes = set()
        self._background_key = None

    def _surface_level(self, world):
        """Estimate the relaxed bed top from the template's upper particle layer."""
        cache = self.patch.cache
        key = cache.templates.keys[world]
        previous = self.surface_levels.get(world)
        if previous is None or previous[0] != key:
            begin = world * cache.template_count
            heights = cache.templates.q[begin : begin + cache.template_count].numpy()[:, 2]
            layers = max(1, round(self.patch.terrain.particle_depth / (2.0 * cache.radius)))
            top_count = max(1, round(len(heights) / layers))
            level = float(np.partition(heights, -top_count)[-top_count:].mean()) + cache.radius
            self.surface_levels[world] = key, level
        return self.surface_levels[world][1]

    def _send_texture(self, name) -> bool:
        """Whether to attach the texture to this update.

        The OVRTX renderer keeps a mesh's texture when later updates omit it and compares the image content on every
        update that supplies one, so it gets the texture once per mesh. The GL viewer removes the texture of an update
        that omits it, so it always receives it.
        """
        if not self.textured:
            return False
        if not self.send_texture_once:
            return True
        if name in self._textured_meshes:
            return False
        self._textured_meshes.add(name)
        return True

    def _texture_image(self, path):
        """Decode the color texture once at reduced size.

        The viewers decode and upload a texture given as a path on every mesh update, which costs hundreds of
        milliseconds for a 4K image; a small array keeps that to a few milliseconds.
        """
        if path is None:
            return None
        cached = getattr(self, "_texture_cache", None)
        if cached is None or cached[0] != path:
            from PIL import Image

            image = Image.open(path).convert("RGB")
            image.thumbnail((1024, 1024))
            cached = path, np.ascontiguousarray(np.asarray(image))
            self._texture_cache = cached
        return cached[1]

    def _flat_color(self, cfg):
        """Average color of the sand texture, used where the viewer cannot sample the texture."""
        if getattr(self, "_mean_color", None) is None:
            try:
                from PIL import Image

                image = Image.open(cfg.color_texture).convert("RGB")
                image.thumbnail((64, 64))
                self._mean_color = tuple(float(v) / 255.0 for v in np.asarray(image).reshape(-1, 3).mean(axis=0))
            except OSError:
                self._mean_color = tuple(cfg.color)
        return self._mean_color

    def _material(self, points, vertical=False, name=None):
        """Use the background terrain's material and world-space texture coordinates."""
        background = self.patch.background_mesh
        cfg = background.visual_cfg
        xy = wp.to_torch(points)[:, :2]
        origin = torch.as_tensor(background.bounds[0, :2], device=xy.device)
        uv = xy - origin
        if vertical:
            uv = torch.stack((uv.sum(dim=1), wp.to_torch(points)[:, 2]), dim=1)
        uvs = (uv * cfg.texture_repeat_per_meter).contiguous()
        return dict(
            uvs=wp.from_torch(uvs, dtype=wp.vec2),
            # The RTX viewer renders this textured mesh black; use the plain color until that is understood.
            texture=self._texture_image(cfg.color_texture) if self._send_texture(name) else None,
            color=(1.0, 1.0, 1.0) if self.textured else self._flat_color(cfg),
            roughness=1.0,
        )

    def _update_background(self, viewer, visible, show_surface=True, show_particles=False):
        """Exclude the regions represented by visible isosurfaces or particle batches."""
        cache, background = self.patch.cache, self.patch.background_mesh
        covered = set()
        for world in visible:
            if show_surface or show_particles:
                covered.update(cache.loaded[world])
            if show_particles:
                # Saved particles remain visible after their tiles leave simulation.
                covered.update(cache.visited[world])
                covered.update(cache.pages[world])
        covered = frozenset(covered)
        worlds = sorted(visible)
        levels = tuple(self._surface_level(world) for world in worlds)
        particle_boundary = show_particles
        signature = covered, tuple(worlds), levels, particle_boundary
        if signature == self.background_tiles:
            return
        size = cache.cfg.tile_size
        if self.background is None:
            lower = np.floor(background.bounds[0, :2] / size).astype(int)
            upper = np.ceil(background.bounds[1, :2] / size).astype(int)
            x = np.arange(lower[0], upper[0] + 1) * size
            y = np.arange(lower[1], upper[1] + 1) * size
            xx, yy = np.meshgrid(x, y, indexing="ij")
            positions = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).reshape(-1, 3).astype(np.float32)
            # Ray queries on the outermost mesh edge are numerically ambiguous.
            positions[:, :2] = np.clip(
                positions[:, :2], background.bounds[0, :2] + 1.0e-4, background.bounds[1, :2] - 1.0e-4
            )
            points = wp.array(positions, dtype=wp.vec3, device=self.patch.model.device)
            background.sample_surface_heights(points)
            ii, jj = np.meshgrid(np.arange(len(x) - 1), np.arange(len(y) - 1), indexing="ij")
            a = (ii * len(y) + jj).flatten()
            faces = np.stack((a, a + len(y), a + 1, a + 1, a + len(y), a + len(y) + 1), axis=1)
            tiles = np.stack((ii + lower[0], jj + lower[1]), axis=-1).reshape(-1, 2)
            self.background = points, faces, tiles
            self.background_shape = (len(x), len(y))
            self.background_origin = lower * size
        points, faces, tiles = self.background
        # Predict the height of untouched sand using the same relaxed template
        # that will populate it. Different worlds can have different materials.
        if self.background_levels != (tuple(worlds), levels):
            visual_points = wp.to_torch(points).clone()
            if worlds:
                origins = torch.as_tensor(background.env_origins[worlds, :2], device=visual_points.device)
                nearest = torch.cdist(visual_points[:, :2], origins).argmin(dim=1)
                visual_points[:, 2] += torch.tensor(levels, device=visual_points.device)[nearest]
            self.background_points = wp.from_torch(visual_points, dtype=wp.vec3)
            heights = visual_points[:, 2].reshape(self.background_shape)
            slope_x, slope_y = torch.gradient(heights, spacing=size)
            normals = torch.stack((-slope_x, -slope_y, torch.ones_like(heights)), dim=-1).reshape(-1, 3)
            normals = torch.nn.functional.normalize(normals, dim=-1)
            self.background_normals = wp.from_torch(normals.contiguous(), dtype=wp.vec3)
            self.background_levels = tuple(worlds), levels
        keep = np.ones(len(tiles), dtype=bool)
        # Index tile keys directly; do not rebuild the sampled terrain as the robot travels.
        lower = tiles[0]
        ny = int(tiles[-1, 1] - lower[1] + 1)
        for x, y in covered:
            index = (x - lower[0]) * ny + y - lower[1]
            if 0 <= index < len(tiles) and np.array_equal(tiles[index], (x, y)):
                keep[index] = False
        indices = wp.array(faces[keep].flatten(), dtype=int, device=self.patch.model.device)
        points = self.background_points
        viewer.log_mesh(
            "/particle_cache/background",
            points,
            indices,
            normals=self.background_normals,
            dynamic=True,
            **self._material(points, name="/particle_cache/background"),
        )
        self._update_particle_boundary(viewer, covered if particle_boundary else set())
        self.background_tiles = signature

    def _update_particle_boundary(self, viewer, covered):
        """Close the exposed side of the background mesh down to the particle-bed depth."""
        edges = []
        lower = np.rint(self.background_origin / self.patch.cache.cfg.tile_size).astype(int)
        nx, ny = self.background_shape
        for x, y in sorted(covered):
            # Orient each wall toward the hole; internal tile joins need no faces.
            for neighbor, a, b in (
                ((x - 1, y), (x, y + 1), (x, y)),
                ((x + 1, y), (x + 1, y), (x + 1, y + 1)),
                ((x, y - 1), (x, y), (x + 1, y)),
                ((x, y + 1), (x + 1, y + 1), (x, y + 1)),
            ):
                if neighbor in covered:
                    continue
                ax, ay = a[0] - lower[0], a[1] - lower[1]
                bx, by = b[0] - lower[0], b[1] - lower[1]
                if 0 <= ax < nx and 0 <= bx < nx and 0 <= ay < ny and 0 <= by < ny:
                    edges.append((ax * ny + ay, bx * ny + by))
        if edges:
            background = wp.to_torch(self.background_points)
            ids = torch.tensor(edges, device=background.device, dtype=torch.long)
            top = background[ids]
            bottom = top.clone()
            bottom[:, :, 2] -= self.patch.terrain.particle_depth
            vertices = torch.stack((top[:, 0], top[:, 1], bottom[:, 1], bottom[:, 0]), dim=1)
            normal = torch.nn.functional.normalize(
                torch.linalg.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), dim=-1
            )
            normals = normal[:, None, :].expand(-1, 4, -1).reshape(-1, 3).contiguous()
            faces = torch.tensor([0, 1, 2, 0, 2, 3], device=background.device, dtype=torch.int32)
            faces = (4 * torch.arange(len(edges), device=background.device)[:, None] + faces).to(torch.int32)
            self.particle_boundary = (
                wp.from_torch(vertices.reshape(-1, 3), dtype=wp.vec3),
                wp.from_torch(faces.flatten(), dtype=wp.int32),
                wp.from_torch(normals, dtype=wp.vec3),
            )
        if self.particle_boundary is not None:
            points, indices, normals = self.particle_boundary
            viewer.log_mesh(
                "/particle_cache/boundary",
                points,
                indices,
                normals=normals,
                hidden=not edges,
                dynamic=True,
                **self._material(points, vertical=True, name="/particle_cache/boundary"),
            )

    @staticmethod
    def _neighbors(tiles):
        return {(x + dx, y + dy) for x, y in tiles for dx in (-1, 0, 1) for dy in (-1, 0, 1)}

    def _log(self, viewer, world, mesh, hidden=False):
        points, indices, normals = mesh
        viewer.log_mesh(
            f"/particle_cache/world_{world}/surface",
            points,
            indices,
            normals=normals,
            dynamic=True,
            hidden=hidden,
            **self._material(points, name=f"/particle_cache/world_{world}/surface"),
        )

    def _positions(self, world, live, tiles):
        """Gather only the saved particle records needed by the requested tile neighborhood."""
        cache = self.patch.cache
        records = [record for tile in tiles for record in cache.pages[world].get(tile, ())]
        if records:
            ids = torch.tensor(records, device=live.device, dtype=torch.long)
            return torch.cat((live, wp.to_torch(cache.saved.q)[ids]))
        return live

    def attach(self, visualizer):
        """Publish this surface from the visualizer's frame hooks, replacing the viewer's own particle batch."""
        visualizer.sand_surface = self
        visualizer.sand_show_surface = True
        self.textured = True
        self.send_texture_once = hasattr(visualizer, "_renderers")
        viewer = visualizer._viewer
        viewer.show_particles = False
        viewer._log_particles = lambda state: None

        def controls(imgui):
            if imgui.collapsing_header("Sand visualization"):
                _, visualizer.sand_show_surface = imgui.checkbox("Isosurface", visualizer.sand_show_surface)

        viewer.register_ui_callback(controls, position="side")

    def render(self, visualizer):
        """Submit the surface for the worlds this visualizer shows."""
        self.update(
            visualizer,
            self.backend.state_0,
            getattr(visualizer, "_env_ids", None),
            show_surface=visualizer.sand_show_surface,
            show_particles=False,
        )

    def update(self, viewer, state, visible_worlds, show_surface=True, show_particles=False):
        """Submit surfaces only for loaded patches, independently of saved history size."""
        patch, cache = self.patch, self.patch.cache
        visible = set(range(cache.world_count) if visible_worlds is None else visible_worlds)
        # The background depends only on which tiles are loaded or visited, which changes with the cache revision.
        key = (int(cache.revision.numpy()[0]), frozenset(visible), show_surface, show_particles)
        if key != self._background_key:
            self._update_background(viewer, visible, show_surface, show_particles)
            self._background_key = key
        if not show_surface:
            visible = set()
        for world in self.shown - visible:
            if world in self.meshes:
                self._log(viewer, world, self.meshes[world], hidden=True)
        self.shown = visible
        layers = wp.to_torch(cache.layers)
        particle_world = wp.to_torch(patch.model.particle_world)
        positions = wp.to_torch(state.particle_q)
        frozen = wp.to_torch(cache.frozen.q)
        for world in sorted(visible):
            selected = (particle_world == world) & (layers != 0)
            live = torch.where((layers[selected] == 2)[:, None], frozen[selected], positions[selected])
            with wp.ScopedDevice(patch.model.device):
                self._update_world(viewer, world, live)

    def _update_world(self, viewer, world, live):
        cache = self.patch.cache
        loaded = cache.loaded[world]
        if loaded:
            # Neighboring saved particles support density reconstruction at the edge,
            # but only loaded tiles produce triangles or openings in the background.
            neighbors = self._neighbors(loaded)
            q = self._positions(world, live, neighbors)
            self.meshes[world] = self._extract(world, sorted(loaded), q)
            self._log(viewer, world, self.meshes[world])
        elif world in self.meshes:
            self._log(viewer, world, self.meshes.pop(world), hidden=True)

    def _extract(self, world, tiles, positions):
        terrain, cache = self.patch.terrain, self.patch.cache
        surface_level = self._surface_level(world)
        size = terrain.particle_cache.tile_size
        n = max(2, int(round(size / terrain.particle_cache.surface_resolution)))
        h = size / n
        keys = np.asarray(tiles, dtype=np.int32)
        backed_keys = keys
        # Felix's edge closure needs unoccupied neighbors to meet the untouched terrain.
        lower, upper = backed_keys.min(axis=0) - 1, backed_keys.max(axis=0) + 2
        shape = upper - lower
        # Include spray; quantization avoids reallocating for small changes in the vertical extent.
        z_min = float(positions[:, 2].min()) if len(positions) else 0.0
        z_max = float(positions[:, 2].max()) if len(positions) else z_min + terrain.particle_depth
        bottom = np.floor((z_min - 2 * h) / size) * size
        top = np.ceil((z_max + 2 * h) / size) * size
        signature = (*lower, *upper, bottom, top, n, len(tiles), surface_level)
        previous = self.extractors.get(world)
        if previous is None or previous[0] != signature:
            lo = lower * size
            grid_x = lo[0] + (np.arange(shape[0] * n) + 0.5) * h
            grid_y = lo[1] + (np.arange(shape[1] * n) + 0.5) * h
            x, y = np.meshgrid(grid_x, grid_y, indexing="ij")
            samples = np.stack((x, y, np.zeros_like(x)), axis=-1).reshape(-1, 3).astype(np.float32)
            bounds = self.patch.background_mesh.bounds
            samples[:, :2] = np.clip(samples[:, :2], bounds[0, :2] + 1.0e-4, bounds[1, :2] - 1.0e-4)
            query = wp.array(samples, dtype=wp.vec3)
            self.patch.background_mesh.sample_surface_heights(query)
            heights = (wp.to_torch(query)[:, 2] + surface_level - bottom).reshape(*x.shape).contiguous()
            world_grid = SimpleNamespace(lo=lo, ncx=int(shape[0]), ncy=int(shape[1]), chunk_count=int(np.prod(shape)))
            surface = IsoSurface(world_grid, wp.from_torch(heights), n, h, 1, 0.0, 0.0, top - bottom, len(tiles), 0)
            self.extractors[world] = signature, surface
        else:
            surface = previous[1]
        ids = ((keys[:, 0] - lower[0]) * shape[1] + keys[:, 1] - lower[1]).astype(np.int32)
        surface.backed.fill_(-1)
        surface.pristine.zero_()
        backed_ids = ((backed_keys[:, 0] - lower[0]) * shape[1] + backed_keys[:, 1] - lower[1]).astype(np.int32)
        wp.to_torch(surface.backed)[torch.as_tensor(backed_ids, device=positions.device, dtype=torch.long)] = 0
        pristine = [index for index, key in enumerate(backed_keys) if tuple(key) not in cache.visited[world]]
        if pristine:
            wp.to_torch(surface.pristine)[torch.as_tensor(backed_ids[pristine], device=positions.device)] = 1
        q = positions.clone()
        q[:, 2] -= bottom
        # Unlike the example, the task permits a surface grid coarser than the particle lattice.
        volume = 8.0 * cache.radius**3
        mesh, nv, nt = surface.extract(
            surface.live,
            ids,
            wp.from_torch(q, dtype=wp.vec3),
            wp.ones(len(q), dtype=int),
            particle_weight=volume / h**3,
            particle_spacing=2.0 * cache.radius * terrain.particle_cache.surface_smoothing,
        )
        vertices = wp.to_torch(mesh.points)[:nv].clone()
        vertices[:, 2] += bottom
        points = wp.from_torch(vertices, dtype=wp.vec3)
        wp.launch(
            blend_surface_boundary,
            dim=nv,
            inputs=[
                points,
                mesh.normals,
                mesh.face_bits,
                surface.backed,
                wp.vec2(*(lower * size)),
                wp.vec2i(*shape),
                size,
                min(size, max(3.0 * h, 2.0 * cache.radius)),
                ISO_SKIRT_CELLS * h,
                self.background_points,
                self.background_normals,
                wp.vec2(*self.background_origin),
                wp.vec2i(*self.background_shape),
            ],
            device=self.patch.model.device,
        )
        return points, mesh.indices[: 3 * nt], mesh.normals[:nv]
