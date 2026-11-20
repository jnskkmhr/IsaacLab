# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the sand of a cached Newton streaming pool as a surface that joins the surrounding terrain."""

import heapq
import logging
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from .paged_sand_isosurface import ISO_SKIRT, ISO_SKIRT_CELLS, IsoSurface

logger = logging.getLogger(__name__)


@wp.kernel
def blend_surface_boundary(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    face_bits: wp.array[int],
    backed: wp.array[int],
    patch_origin: wp.vec2,
    patch_shape: wp.vec2i,
    tile_size: float,
    blend_width: float,
    skirt_depth: float,
    background_points: wp.array[wp.vec3],
    background_normals: wp.array[wp.vec3],
    background_origin: wp.vec2,
    background_shape: wp.vec2i,
):
    """Blend only the outer surface rim; particle positions and internal tile joins are untouched."""
    vertex = wp.tid()
    p = points[vertex]
    skirt = wp.where((face_bits[vertex] & ISO_SKIRT) != 0, skirt_depth, 0.0)
    p[2] += skirt
    cell = wp.vec2i(
        int(wp.floor((p[0] - patch_origin[0]) / tile_size)),
        int(wp.floor((p[1] - patch_origin[1]) / tile_size)),
    )
    distance = blend_width
    # A band no wider than one tile needs only the surrounding tile neighborhood.
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            x, y = cell[0] + dx, cell[1] + dy
            if x >= 0 and y >= 0 and x < patch_shape[0] and y < patch_shape[1]:
                if backed[x * patch_shape[1] + y] >= 0:
                    continue
            lo = patch_origin + wp.vec2(float(x), float(y)) * tile_size
            gap = wp.vec2(
                wp.max(wp.max(lo[0] - p[0], p[0] - lo[0] - tile_size), 0.0),
                wp.max(wp.max(lo[1] - p[1], p[1] - lo[1] - tile_size), 0.0),
            )
            distance = wp.min(distance, wp.length(gap))
    if distance >= blend_width:
        return

    # Use the very same triangles and vertex normals as the background visual.
    uv = (wp.vec2(p[0], p[1]) - background_origin) / tile_size
    x = wp.clamp(int(wp.floor(uv[0])), 0, background_shape[0] - 2)
    y = wp.clamp(int(wp.floor(uv[1])), 0, background_shape[1] - 2)
    fx, fy = wp.clamp(uv[0] - float(x), 0.0, 1.0), wp.clamp(uv[1] - float(y), 0.0, 1.0)
    a = x * background_shape[1] + y
    b, c = a + background_shape[1], a + 1
    weights = wp.vec3(1.0 - fx - fy, fx, fy)
    if fx + fy > 1.0:
        a, b, c = a + background_shape[1] + 1, c, b
        weights = wp.vec3(fx + fy - 1.0, 1.0 - fx, 1.0 - fy)
    height = (
        weights[0] * background_points[a][2]
        + weights[1] * background_points[b][2]
        + weights[2] * background_points[c][2]
    )
    # Do not lift the underside of the sand bed into its upper visual surface.
    if p[2] < height - 2.0 * blend_width:
        return
    normal = wp.normalize(
        weights[0] * background_normals[a] + weights[1] * background_normals[b] + weights[2] * background_normals[c]
    )
    t = distance / blend_width
    weight = t * t * t * (t * (6.0 * t - 15.0) + 10.0)
    p[2] = wp.lerp(height, p[2], weight) - skirt
    points[vertex] = p
    normals[vertex] = wp.normalize(wp.lerp(normal, normals[vertex], weight))


class _BakedTiles:
    """Meshes of tiles that left the pool, kept so that their surface stays visible.

    Every tile owns one fixed-size slot of vertices and triangles. Slots are grouped into pages, each
    drawn as one mesh, so that storing or dropping a tile re-submits only its page. When no slot is
    free, the tile farthest from its pool is dropped and shows the plain terrain again.
    """

    VERTICES, TRIANGLES, PAGE = 2048, 4096, 16
    """Vertices and triangles per tile slot, and slots per page.

    A 0.2 m tile reconstructed on a 15 mm grid needs about 1000 vertices and 1900 triangles on rough
    terrain (at most 1500 and 2800 in a 9 m walk), and about half of that on flat ground.
    """

    def __init__(self, megabytes: int, device):
        slot_bytes = self.VERTICES * 24 + self.TRIANGLES * 12
        self.page_count = max(1, megabytes * 2**20 // (slot_bytes * self.PAGE))
        slot_count = self.page_count * self.PAGE
        self.points = wp.zeros(slot_count * self.VERTICES, dtype=wp.vec3, device=device)
        self.normals = wp.zeros(slot_count * self.VERTICES, dtype=wp.vec3, device=device)
        self.indices = wp.zeros(slot_count * self.TRIANGLES * 3, dtype=wp.int32, device=device)
        self.free = list(range(slot_count))
        self.slots = {}
        """Slot of each baked tile, keyed by ``(world, tile x, tile y)``."""
        self.dirty = set(range(self.page_count))
        """Pages whose mesh changed since they were last submitted; initially all, so every mesh exists early."""
        self.revision = 0
        self.oversized = 0

    def tiles(self, world):
        """Return the tile keys baked for a world."""
        return {(x, y) for w, x, y in self.slots if w == world}

    def store(self, key, points, normals, indices, distance) -> bool:
        """Copy one tile's mesh into its slot, dropping the farthest tile by ``distance(key)`` when full."""
        vertex_count, triangle_count = len(points), len(indices) // 3
        if vertex_count > self.VERTICES or triangle_count > self.TRIANGLES:
            if not self.oversized:
                logger.warning("A sand tile's surface exceeds the baked-mesh slot size; it shows the plain terrain.")
            self.oversized += 1
            return False
        slot = self.slots.get(key)
        if slot is None:
            if not self.free:
                self.drop(max(self.slots, key=distance))
            slot = heapq.heappop(self.free)
            self.slots[key] = slot
        first_vertex, first_index = slot * self.VERTICES, slot * self.TRIANGLES * 3
        wp.to_torch(self.points)[first_vertex : first_vertex + vertex_count] = wp.to_torch(points)
        wp.to_torch(self.normals)[first_vertex : first_vertex + vertex_count] = wp.to_torch(normals)
        target = wp.to_torch(self.indices)[first_index : first_index + self.TRIANGLES * 3]
        # Indices are relative to the page's vertices; unused triangles stay degenerate.
        target[: 3 * triangle_count] = wp.to_torch(indices) + (slot % self.PAGE) * self.VERTICES
        target[3 * triangle_count :] = 0
        self.dirty.add(slot // self.PAGE)
        self.revision += 1
        return True

    def drop(self, key) -> None:
        """Free a tile's slot; its triangles disappear when the page is next submitted."""
        slot = self.slots.pop(key)
        first_index = slot * self.TRIANGLES * 3
        wp.to_torch(self.indices)[first_index : first_index + self.TRIANGLES * 3] = 0
        heapq.heappush(self.free, slot)
        self.dirty.add(slot // self.PAGE)
        self.revision += 1

    def page(self, page):
        """Return the vertex positions, normals, triangle indices of a page, and whether it holds any tile."""
        vertices = slice(page * self.PAGE * self.VERTICES, (page + 1) * self.PAGE * self.VERTICES)
        triangles = slice(page * self.PAGE * self.TRIANGLES * 3, (page + 1) * self.PAGE * self.TRIANGLES * 3)
        used = any(slot // self.PAGE == page for slot in self.slots.values())
        return self.points[vertices], self.normals[vertices], self.indices[triangles], used


class CachedParticleSurface:
    """Extract loaded particles and cover unloaded regions with the background terrain.

    Tiles loaded in the pool are reconstructed every frame. A tile that leaves the pool is baked once
    into a mesh from its saved particles, so the marks left in the sand stay visible; it is dropped
    again when the pool returns, when its environment resets, or when the baked-mesh budget is full.
    Tiles that were never visited show the background terrain.
    """

    def __init__(
        self,
        streaming,
        backend,
        terrain_mesh,
        visual_cfg,
        *,
        resolution: float = 0.015,
        smoothing: float = 1.5,
        cache_mb: int = 64,
    ):
        """Bind the surface to a cached pool and the terrain it rests on.

        Args:
            streaming: Newton's cached streaming pool, which owns the tile cache.
            backend: Newton backend whose current state holds the particle positions.
            terrain_mesh: Undisturbed terrain surface providing ``bounds``, ``env_origins`` and
                ``sample_surface_heights``.
            visual_cfg: Terrain appearance providing ``color_texture``, ``texture_repeat_per_meter`` and ``color``.
            resolution: Approximate surface grid spacing [m]; rounded to divide each cache tile.
            smoothing: Multiplier of the particle spacing used to reconstruct the surface; at least 1.
            cache_mb: Memory for the baked meshes of tiles that left the pool [MiB]; zero disables baking.

        Raises:
            ValueError: If the smoothing multiplier is not finite or below 1.
        """
        if not np.isfinite(smoothing) or smoothing < 1.0:
            raise ValueError("The surface smoothing must be finite and at least 1.0")
        self.streaming = streaming
        self.backend = backend
        self.terrain_mesh = terrain_mesh
        self.visual_cfg = visual_cfg
        self.resolution = resolution
        self.smoothing = smoothing
        self.cache_mb = cache_mb
        self.baked = None
        """Baked meshes of tiles outside the pool; allocated when the surface is first shown."""
        self._loaded_before = {}
        self._generations = {}
        self._baked_hidden = False
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
        cache = self.streaming.cache
        key = cache.templates.keys[world]
        previous = self.surface_levels.get(world)
        if previous is None or previous[0] != key:
            begin = world * cache.template_count
            heights = cache.templates.q[begin : begin + cache.template_count].numpy()[:, 2]
            layers = max(1, round(self.streaming.terrain.particle_depth / (2.0 * cache.radius)))
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
            pixels = np.ascontiguousarray(np.asarray(image))
            if self.send_texture_once:
                # The RTX renderer encodes an image array into a file for every mesh that receives one. Every
                # mesh of this surface shares a single reduced file instead.
                self._texture_directory = tempfile.TemporaryDirectory(prefix="isaaclab-sand-")
                file = os.path.join(self._texture_directory.name, "texture.png")
                image.save(file)
                pixels = file
            cached = path, pixels
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
        background = self.terrain_mesh
        cfg = self.visual_cfg
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
        cache, background = self.streaming.cache, self.terrain_mesh
        covered = set()
        for world in visible:
            if show_surface or show_particles:
                covered.update(cache.loaded[world])
            if show_surface and self.baked is not None:
                covered.update(self.baked.tiles(world))
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
            points = wp.array(positions, dtype=wp.vec3, device=self.streaming.model.device)
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
        # When sand covers the whole terrain no triangle is left, and an empty mesh cannot be submitted: hide it.
        hidden = not keep.any()
        indices = wp.array(faces.flatten() if hidden else faces[keep].flatten(), dtype=int, device=points.device)
        points = self.background_points
        viewer.log_mesh(
            "/cached_sand/background",
            points,
            indices,
            normals=self.background_normals,
            dynamic=True,
            hidden=hidden,
            **self._material(points, name="/cached_sand/background"),
        )
        self._update_particle_boundary(viewer, covered if particle_boundary else set())
        self.background_tiles = signature

    def _update_particle_boundary(self, viewer, covered):
        """Close the exposed side of the background mesh down to the particle-bed depth."""
        edges = []
        lower = np.rint(self.background_origin / self.streaming.cache.cfg.tile_size).astype(int)
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
            bottom[:, :, 2] -= self.streaming.terrain.particle_depth
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
                "/cached_sand/boundary",
                points,
                indices,
                normals=normals,
                hidden=not edges,
                dynamic=True,
                **self._material(points, vertical=True, name="/cached_sand/boundary"),
            )

    @staticmethod
    def _neighbors(tiles):
        return {(x + dx, y + dy) for x, y in tiles for dx in (-1, 0, 1) for dy in (-1, 0, 1)}

    def _log(self, viewer, world, mesh, hidden=False):
        points, indices, normals = mesh
        viewer.log_mesh(
            f"/cached_sand/world_{world}/surface",
            points,
            indices,
            normals=normals,
            dynamic=True,
            hidden=hidden,
            **self._material(points, name=f"/cached_sand/world_{world}/surface"),
        )

    def _positions(self, world, live, tiles):
        """Gather only the saved particle records needed by the requested tile neighborhood."""
        cache = self.streaming.cache
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
        self._texture_cache = None
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
        streaming, cache = self.streaming, self.streaming.cache
        visible = set(range(cache.world_count) if visible_worlds is None else visible_worlds)
        # The background follows the loaded windows. Saved and visited tiles matter only when particles are shown,
        # so the cache revision is consulted only then.
        layers = wp.to_torch(cache.layers)
        particle_world = wp.to_torch(streaming.model.particle_world)
        positions = wp.to_torch(state.particle_q)
        frozen = wp.to_torch(cache.frozen.q)
        live = {}
        for world in sorted(visible) if show_surface else ():
            selected = (particle_world == world) & (layers != 0)
            live[world] = torch.where((layers[selected] == 2)[:, None], frozen[selected], positions[selected])
        with wp.ScopedDevice(streaming.model.device):
            self._update_baked(viewer, live, show_surface)
        key = (
            cache.tile_begins.numpy().tobytes(),
            tuple(cache.generations),
            int(cache.revision.numpy()[0]) if show_particles else 0,
            frozenset(visible),
            show_surface,
            show_particles,
            0 if self.baked is None else self.baked.revision,
        )
        if key != self._background_key:
            self._update_background(viewer, visible, show_surface, show_particles)
            self._background_key = key
        if not show_surface:
            visible = set()
        for world in self.shown - visible:
            if world in self.meshes:
                self._log(viewer, world, self.meshes[world], hidden=True)
        self.shown = visible
        for world in sorted(visible):
            with wp.ScopedDevice(streaming.model.device):
                self._update_world(viewer, world, live[world])

    def _update_baked(self, viewer, live, show_surface):
        """Bake the tiles that left a pool, drop those that no longer apply, and submit the changed pages."""
        cache = self.streaming.cache
        if self.cache_mb <= 0:
            return
        baked = self.baked
        if not show_surface:
            # Nothing is tracked while the surface is hidden; baking starts afresh when it is shown again.
            if baked is not None and not self._baked_hidden:
                for key in list(baked.slots):
                    baked.drop(key)
                self._loaded_before.clear()
                self._submit_baked(viewer)
                self._baked_hidden = True
            return
        self._baked_hidden = False
        if baked is None and self.background_points is not None:
            # Create every page's mesh up front. A mesh that first appears while the robot walks makes the
            # RTX viewer's generated frames disagree with its rendered ones for a few frames.
            baked = self.baked = _BakedTiles(self.cache_mb, self.streaming.model.device)
        generations = cache.generations
        centers = {}
        for world, particles in live.items():
            loaded = set(cache.loaded[world])
            if self._generations.get(world) != generations[world]:
                # The environment was reset: its sand is new and nothing left behind is valid.
                self._generations[world] = generations[world]
                self._loaded_before[world] = set()
                if baked is not None:
                    for tile in baked.tiles(world):
                        baked.drop((world, *tile))
            departed = self._loaded_before.get(world, set()) - loaded
            self._loaded_before[world] = loaded
            if baked is not None:
                # The pool is back on these tiles; the live surface takes over.
                for tile in loaded & baked.tiles(world):
                    baked.drop((world, *tile))
            if not departed or not loaded or baked is None:
                continue
            centers[world] = np.mean(np.asarray(sorted(loaded), dtype=np.float64), axis=0)

            def distance(key):
                center = centers.get(key[0])
                return 0.0 if center is None else float(np.hypot(key[1] - center[0], key[2] - center[1]))

            pages = cache.pages[world]
            # Neighbors baked in this same pass count as holding sand, so shared edges are not flattened.
            backed = loaded | baked.tiles(world) | departed
            for tile in sorted(departed):
                if not pages.get(tile):
                    continue
                mesh = self._extract_tile(world, tile, particles, backed)
                if mesh is not None:
                    points, indices, normals = mesh
                    baked.store((world, *tile), points, normals, indices, distance)
        self._submit_baked(viewer)

    def _submit_baked(self, viewer):
        """Submit every page of baked tiles that changed."""
        baked = self.baked
        if baked is None:
            return
        for page in sorted(baked.dirty):
            points, normals, indices, used = baked.page(page)
            name = f"/cached_sand/baked_page_{page}"
            viewer.log_mesh(
                name,
                points,
                indices,
                normals=normals,
                dynamic=True,
                hidden=not used,
                **self._material(points, name=name),
            )
        baked.dirty.clear()

    def _extract_tile(self, world, tile, live, backed):
        """Reconstruct the surface of one tile outside the pool from its saved particles and its surroundings."""
        size = self.streaming.cache.cfg.tile_size
        positions = self._positions(world, live, self._neighbors([tile]))
        lower = torch.tensor([(tile[0] - 1) * size, (tile[1] - 1) * size], device=positions.device)
        near = ((positions[:, :2] >= lower) & (positions[:, :2] < lower + 3.0 * size)).all(dim=1)
        positions = positions[near]
        if len(positions) == 0:
            return None
        return self._extract(world, [tile], positions, backed=backed, key=("baked", world))

    def _update_world(self, viewer, world, live):
        cache = self.streaming.cache
        loaded = cache.loaded[world]
        if loaded:
            # Neighboring saved particles support density reconstruction at the edge,
            # but only loaded tiles produce triangles or openings in the background.
            neighbors = self._neighbors(loaded)
            q = self._positions(world, live, neighbors)
            backed = set(loaded) if self.baked is None else set(loaded) | self.baked.tiles(world)
            mesh = self._extract(world, sorted(loaded), q, backed=backed)
            if mesh is None:
                # A diverged solve leaves non-finite particle positions; keep showing the last surface until the
                # environment resets instead of failing in the viewer.
                return
            self.meshes[world] = mesh
            self._log(viewer, world, self.meshes[world])
        elif world in self.meshes:
            self._log(viewer, world, self.meshes.pop(world), hidden=True)

    def _extract(self, world, tiles, positions, backed=None, key=None):
        """Reconstruct the surface of ``tiles`` from ``positions``.

        ``backed`` are the tiles that hold sand and therefore need no closure towards the terrain; it
        defaults to ``tiles``. ``key`` selects the reusable extractor and defaults to the world.
        """
        terrain, cache = self.streaming.terrain, self.streaming.cache
        surface_level = self._surface_level(world)
        size = cache.cfg.tile_size
        n = max(2, int(round(size / self.resolution)))
        h = size / n
        keys = np.asarray(tiles, dtype=np.int32)
        # Felix's edge closure needs unoccupied neighbors to meet the untouched terrain.
        lower, upper = keys.min(axis=0) - 1, keys.max(axis=0) + 2
        shape = upper - lower
        backed = set(tiles) if backed is None else backed
        backed_keys = np.asarray(
            [
                (x, y)
                for x in range(int(lower[0]), int(upper[0]))
                for y in range(int(lower[1]), int(upper[1]))
                if (x, y) in backed
            ],
            dtype=np.int32,
        ).reshape(-1, 2)
        # Include spray; quantization avoids reallocating for small changes in the vertical extent.
        z_min = float(positions[:, 2].min()) if len(positions) else 0.0
        z_max = float(positions[:, 2].max()) if len(positions) else z_min + terrain.particle_depth
        if not (np.isfinite(z_min) and np.isfinite(z_max)):
            return None
        bottom = np.floor((z_min - 2 * h) / size) * size
        top = np.ceil((z_max + 2 * h) / size) * size
        key = world if key is None else key
        # The extractor's buffers depend only on the grid shape; its placement is updated in place.
        signature = (*shape, top - bottom, n, len(tiles))
        placement = (*lower, bottom, surface_level)
        previous = self.extractors.get(key)
        if previous is None or previous[0] != signature or previous[2] != placement:
            lo = lower * size
            grid_x = lo[0] + (np.arange(shape[0] * n) + 0.5) * h
            grid_y = lo[1] + (np.arange(shape[1] * n) + 0.5) * h
            x, y = np.meshgrid(grid_x, grid_y, indexing="ij")
            samples = np.stack((x, y, np.zeros_like(x)), axis=-1).reshape(-1, 3).astype(np.float32)
            bounds = self.terrain_mesh.bounds
            samples[:, :2] = np.clip(samples[:, :2], bounds[0, :2] + 1.0e-4, bounds[1, :2] - 1.0e-4)
            query = wp.array(samples, dtype=wp.vec3)
            self.terrain_mesh.sample_surface_heights(query)
            heights = (wp.to_torch(query)[:, 2] + surface_level - bottom).reshape(*x.shape).contiguous()
            if previous is None or previous[0] != signature:
                grid = SimpleNamespace(lo=lo, ncx=int(shape[0]), ncy=int(shape[1]), chunk_count=int(np.prod(shape)))
                surface = IsoSurface(grid, wp.from_torch(heights), n, h, 1, 0.0, 0.0, top - bottom, len(tiles), 0)
            else:
                surface = previous[1]
                surface.world.lo = lo
                surface.heights = wp.from_torch(heights)
            self.extractors[key] = signature, surface, placement
        else:
            surface = previous[1]
        ids = ((keys[:, 0] - lower[0]) * shape[1] + keys[:, 1] - lower[1]).astype(np.int32)
        surface.backed.fill_(-1)
        surface.pristine.zero_()
        backed_ids = ((backed_keys[:, 0] - lower[0]) * shape[1] + backed_keys[:, 1] - lower[1]).astype(np.int32)
        wp.to_torch(surface.backed)[torch.as_tensor(backed_ids, device=positions.device, dtype=torch.long)] = 0
        pristine = [index for index, tile in enumerate(backed_keys) if tuple(tile) not in cache.visited[world]]
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
            particle_spacing=2.0 * cache.radius * self.smoothing,
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
            device=self.streaming.model.device,
        )
        return points, mesh.indices[: 3 * nt], mesh.normals[:nv]
