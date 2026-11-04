# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render live sand and retain unloaded tile meshes in Felix Meyer's GPU mesh pool."""

from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from .paged_sand_isosurface import ISO_SKIRT_CELLS, IsoCache, IsoSurface
from .surface_kernel import blend_surface_boundary


class ParticleCacheSurface:
    """Extract loaded particles each frame and bake saved terrain only when it changes.

    Each cached tile includes particles in neighboring tiles when constructing its
    boundary. Reloading a tile drops its baked mesh; selective reset invalidates
    only that world's meshes. The bounded mesh pool is independent of physics.
    """

    def __init__(self, patch):
        self.patch = patch
        self.extractors = {}
        self.shown = set()
        self.meshes = {}
        self.static_meshes = {}
        self.baked = {}
        self.background = None
        self.background_tiles = None
        self.surface_levels = {}
        self.background_levels = None
        self.background_points = None
        self.background_normals = None

    def _surface_level(self, world):
        """Estimate the bed top from the cache's reference particle positions."""
        cache = self.patch.cache
        key, reference = cache.surface_reference(world)
        previous = self.surface_levels.get(world)
        if previous is None or previous[0] != key:
            heights = reference.numpy()[:, 2]
            layers = max(1, round(self.patch.terrain.particle_depth / (2.0 * cache.radius)))
            top_count = max(1, round(len(heights) / layers))
            level = float(np.partition(heights, -top_count)[-top_count:].mean()) + cache.radius
            self.surface_levels[world] = key, level
        return self.surface_levels[world][1]

    def _material(self, points):
        """Use the background terrain's material and world-space texture coordinates."""
        background = self.patch.background_mesh
        cfg = background.visual_cfg
        xy = wp.to_torch(points)[:, :2]
        origin = torch.as_tensor(background.bounds[0, :2], device=xy.device)
        uvs = ((xy - origin) * cfg.texture_repeat_per_meter).contiguous()
        return dict(
            uvs=wp.from_torch(uvs, dtype=wp.vec2),
            texture=cfg.color_texture,
            color=cfg.color,
            roughness=1.0,
        )

    def _update_background(self, viewer, visible):
        """Draw untouched terrain at sand height, excluding tiles with particle history."""
        cache, background = self.patch.cache, self.patch.background_mesh
        covered = frozenset(tile for world in visible for tile in cache.visited[world] | cache.pages[world].keys())
        worlds = sorted(visible)
        levels = tuple(self._surface_level(world) for world in worlds)
        signature = covered, tuple(worlds), levels
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
        # Predict the height of untouched sand using the same reference particles
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
            **self._material(points),
        )
        self.background_tiles = signature

    @staticmethod
    def _neighbors(tiles):
        return {(x + dx, y + dy) for x, y in tiles for dx in (-1, 0, 1) for dy in (-1, 0, 1)}

    def _log(self, viewer, world, mesh, cached=False, hidden=False):
        points, indices, normals = mesh
        suffix = "saved" if cached else "surface"
        viewer.log_mesh(
            f"/particle_cache/world_{world}/{suffix}",
            points,
            indices,
            normals=normals,
            dynamic=True,
            hidden=hidden,
            **self._material(points),
        )

    def _positions(self, world, live, tiles):
        """Gather only the saved particle records needed by the requested tile neighborhood."""
        cache = self.patch.cache
        records = [record for tile in tiles for record in cache.pages[world].get(tile, ())]
        if records:
            ids = torch.tensor(records, device=live.device, dtype=torch.long)
            return torch.cat((live, wp.to_torch(cache.saved.q)[ids]))
        return live

    def update(self, viewer, state, visible_worlds, show_surface=True):
        """Submit live geometry each frame and cached geometry only after a page change."""
        patch, cache = self.patch, self.patch.cache
        visible = set(range(cache.world_count) if visible_worlds is None else visible_worlds)
        self._update_background(viewer, visible)
        if not show_surface:
            visible = set()
        for world in self.shown - visible:
            if world in self.meshes:
                self._log(viewer, world, self.meshes[world], hidden=True)
            if world in self.static_meshes:
                self._log(viewer, world, self.static_meshes[world], cached=True, hidden=True)
        newly_visible = visible - self.shown
        self.shown = visible
        layers = wp.to_torch(cache.layers)
        particle_world = wp.to_torch(patch.model.particle_world)
        positions = wp.to_torch(state.particle_q)
        frozen = wp.to_torch(cache.frozen.q)
        for world in sorted(visible):
            selected = (particle_world == world) & (layers != 0)
            live = torch.where((layers[selected] == 2)[:, None], frozen[selected], positions[selected])
            with wp.ScopedDevice(patch.model.device):
                self._update_world(viewer, world, live, world in newly_visible)

    def _update_world(self, viewer, world, live, newly_visible):
        cache, cfg = self.patch.cache, self.patch.terrain.particle_cache
        loaded, pages = cache.loaded[world], cache.pages[world]
        record = self.baked.get(world)
        if (
            record is None
            or record.generation != cache.generations[world]
            or record.background_levels != self.background_levels
        ):
            if world in self.static_meshes:
                self._log(viewer, world, self.static_meshes.pop(world), cached=True, hidden=True)
            record = SimpleNamespace(
                generation=cache.generations[world],
                background_levels=self.background_levels,
                pool=IsoCache(cfg.max_visited_tiles_per_env, cfg.surface_cache_mb),
                ids={},
                signatures={},
            )
            self.baked[world] = record
        pool = record.pool
        for tile in list(record.signatures):
            if tile not in pages or tile in loaded:
                pool.drop(record.ids[tile])
                del record.signatures[tile]
        available = loaded | pages.keys()
        for tile in sorted(pages.keys() - loaded):
            neighbors = self._neighbors([tile]) & available
            signature = tuple((key, cache.page_revisions[world].get(key), key in loaded) for key in sorted(neighbors))
            if record.signatures.get(tile) == signature:
                continue
            cid = record.ids.setdefault(tile, len(record.ids))
            if cid >= cfg.max_visited_tiles_per_env:
                raise RuntimeError("Isosurface tile capacity exceeded; increase max_visited_tiles_per_env")
            if tile in record.signatures:
                pool.drop(cid)
            q = self._positions(world, live, neighbors)
            # Other loaded patches need not enlarge this tile's vertical extraction range.
            lo = torch.tensor(tile, device=q.device) * cfg.tile_size - 2 * cfg.surface_resolution
            hi = lo + cfg.tile_size + 4 * cfg.surface_resolution
            q = q[((q[:, :2] >= lo) & (q[:, :2] <= hi)).all(dim=1)]
            points, indices, normals = self._extract(("bake", world), [tile], q, neighbors)
            nv, nt = len(points), len(indices) // 3
            if max(1, -(-nv // pool.vb)) > len(pool.free_v) or max(1, -(-nt // pool.tb)) > len(pool.free_t):
                raise RuntimeError("Isosurface mesh pool exhausted; increase particle_cache.surface_cache_mb")
            mesh = SimpleNamespace(points=points, indices=indices, normals=normals, uvs=wp.zeros(nv, dtype=wp.vec2))
            pool.store(cid, mesh, nv, nt, len(pages[tile]), lambda ids: np.zeros(len(ids)))
            record.signatures[tile] = signature
        if pool.dirty or newly_visible:
            nv, nt = pool.used()
            if nt:
                self.static_meshes[world] = pool.pool.points[:nv], pool.pool.indices[: 3 * nt], pool.pool.normals[:nv]
                self._log(viewer, world, self.static_meshes[world], cached=True)
            elif world in self.static_meshes:
                self._log(viewer, world, self.static_meshes.pop(world), cached=True, hidden=True)
            pool.dirty = False
        if loaded:
            neighbors = self._neighbors(loaded) & available
            q = self._positions(world, live, neighbors)
            self.meshes[world] = self._extract(world, sorted(loaded), q, neighbors)
            self._log(viewer, world, self.meshes[world])

    def _extract(self, world, tiles, positions, backed_tiles=None):
        terrain, cache = self.patch.terrain, self.patch.cache
        world_id = world[1] if isinstance(world, tuple) else world
        surface_level = self._surface_level(world_id)
        size = terrain.particle_cache.tile_size
        n = max(2, int(round(size / terrain.particle_cache.surface_resolution)))
        h = size / n
        keys = np.asarray(tiles, dtype=np.int32)
        backed_keys = np.asarray(tiles if backed_tiles is None else sorted(backed_tiles), dtype=np.int32)
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
        pristine = [index for index, key in enumerate(backed_keys) if tuple(key) not in cache.visited[world_id]]
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
            particle_spacing=2.0 * cache.radius,
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
