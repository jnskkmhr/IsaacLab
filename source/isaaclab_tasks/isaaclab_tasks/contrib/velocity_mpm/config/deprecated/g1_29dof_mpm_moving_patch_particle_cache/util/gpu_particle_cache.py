# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-resident particle history with a per-environment hash table of world-XY tiles.

The cache of :mod:`.particle_cache` plans paging in Python: every physics substep it reads counts back to the host,
rebuilds tile sets and loops over departing particles. This version keeps the same behavior on the device. Departed
particles are stored in a fixed pool of records chained into one list per tile; the tile-to-list map is an
open-addressing hash table per environment. Simulation slots and records are handed out by atomic recycle stacks with a
bump pointer, so resetting an environment is constant time. All launches have fixed sizes and no host reads, so the
update can be recorded in a CUDA graph with the rest of the physics step. Capacity problems set bits in a per-world
error array that :meth:`GpuParticleStateCache.check_errors` reports once per control step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .gpu_cache_kernel import (
    ERROR_RECORDS,
    ERROR_TILES,
    TILE_OFFSET,
    TILE_RANGE,
    clear_tile_tables,
    compute_tile_begins,
    finalize_update,
    generate_fresh_tiles,
    reset_worlds,
    restore_entering_tiles,
    save_outgoing,
)
from .kernel import ParticleState, apply_transitions, plan_transitions
from .particle_cache import allocate_particle_state
from .particle_sampling import sample_particle_tile
from .settled_template import SettledTemplates

if TYPE_CHECKING:
    import newton

    from .particles import MovingPatchParticles


class _WorldView:
    """Read-only ``view[world]`` access to a per-environment host mirror of the device tables."""

    def __init__(self, cache: GpuParticleStateCache, kind: str) -> None:
        self._cache = cache
        self._kind = kind

    def __getitem__(self, world: int):
        return self._cache._world_mirror(world)[self._kind]

    def __len__(self) -> int:
        return self._cache.world_count

    def __iter__(self):
        return (self[world] for world in range(len(self)))


class _LazyPages:
    """Read-only ``{tile: [record, ...]}`` that walks a tile's record list only when that tile is requested.

    Walking every list of a long run costs time proportional to everything ever saved; a renderer needs only the
    lists of the tiles around the loaded window.
    """

    def __init__(self, cache: GpuParticleStateCache, world: int, first_record: dict) -> None:
        self._cache = cache
        self._world = world
        self._first = first_record
        self._following = None
        self._chains = {}

    def _chain(self, tile) -> list[int]:
        chain = self._chains.get(tile)
        if chain is None:
            if self._following is None:
                cap = self._cache.record_capacity
                self._following = self._cache.record_next[self._world * cap : (self._world + 1) * cap].numpy()
            base = self._world * self._cache.record_capacity
            chain, record = [], self._first[tile]
            while record >= 0:
                chain.append(record)
                record = int(self._following[record - base])
            self._chains[tile] = chain
        return chain

    def __getitem__(self, tile):
        if tile not in self._first:
            raise KeyError(tile)
        return self._chain(tile)

    def get(self, tile, default=None):
        return self._chain(tile) if tile in self._first else default

    def __contains__(self, tile) -> bool:
        return tile in self._first

    def __len__(self) -> int:
        return len(self._first)

    def __iter__(self):
        return iter(self._first)

    def keys(self):
        return self._first.keys()

    def values(self):
        return (self._chain(tile) for tile in self._first)

    def items(self):
        return ((tile, self._chain(tile)) for tile in self._first)


class GpuParticleStateCache:
    """Retain departed particles and empty visited tiles until their environment resets, entirely on the GPU.

    There is one table and one record pool per environment, even when terrain coordinates overlap. Loading consumes
    saved records; a particle is never both simulated and cached. ``pages``, ``visited``, ``loaded``, ``generations``
    and ``page_revisions`` are host mirrors computed on demand for rendering and tests; they are not used by the update.
    """

    gpu_resident = True

    def __init__(self, patch: MovingPatchParticles) -> None:
        self.patch = patch
        self.cfg = patch.terrain.particle_cache
        self.device = patch.model.device
        # Keep reserve slots outside the terrain, including the sparse grid's
        # boundary cells. Inactive flags alone do not exclude quadrature points.
        self.inactive_position = wp.vec3(
            *(patch.background_mesh.bounds[0] - max(patch.terrain.total_patch_size) - patch.terrain.particle_depth)
        )
        self.worlds = patch.model.particle_world.numpy()
        self.world_count = patch.model.world_count
        slots = [np.flatnonzero(self.worlds == world) for world in range(self.world_count)]
        if len({len(world_slots) for world_slots in slots}) != 1:
            raise ValueError("The GPU particle cache needs the same number of simulation slots in every environment")
        self.slot_capacity = len(slots[0])
        self.slot_table = wp.array(np.concatenate(slots), dtype=int, device=self.device)
        self.record_capacity = self.cfg.max_cached_particles_per_env
        self.max_visited = self.cfg.max_visited_tiles_per_env
        self.table_capacity = 1 << max(4, (2 * self.max_visited - 1).bit_length())

        self.tile_shape = np.rint(np.asarray(patch.terrain.total_patch_size) / self.cfg.tile_size).astype(int)
        bounds = patch.background_mesh.bounds
        self.tile_lower = np.ceil(bounds[0, :2] / self.cfg.tile_size).astype(int)
        self.tile_upper = np.floor(bounds[1, :2] / self.cfg.tile_size).astype(int) - self.tile_shape
        if np.any(self.tile_upper < self.tile_lower):
            raise ValueError("Background terrain must contain at least one full tile-aligned particle patch")
        extent = np.max(np.abs(bounds[:, :2])) / self.cfg.tile_size + np.max(self.tile_shape) + 2
        if extent >= int(TILE_OFFSET):
            raise ValueError("Background terrain is too large for the packed tile keys of the GPU particle cache")

        count = patch.model.particle_count
        regions = patch.bodies.shape[1]
        self.body_counts = wp.array(patch.body_counts, dtype=int, device=self.device)
        # particle state
        self.saved = allocate_particle_state(self.record_capacity * self.world_count, self.device)
        self.frozen = allocate_particle_state(count, self.device)
        self.layers = wp.zeros(count, dtype=int, device=self.device)
        self.next_layers = wp.empty_like(self.layers)
        self.outgoing_ids = wp.empty_like(self.layers)
        self.outgoing_tiles = wp.empty(count, dtype=wp.vec2i, device=self.device)
        # tile table, record lists and allocators
        shape = (self.world_count, self.table_capacity)
        self.table_keys = wp.full(shape, -1, dtype=int, device=self.device)
        self.table_heads = wp.full(shape, -1, dtype=int, device=self.device)
        self.table_revisions = wp.zeros(shape, dtype=int, device=self.device)
        self.table_loaded_once = wp.zeros(shape, dtype=int, device=self.device)
        self.visited_count = wp.zeros(self.world_count, dtype=int, device=self.device)
        self.record_next = wp.full(self.record_capacity * self.world_count, -1, dtype=int, device=self.device)
        self.record_bump = wp.zeros(self.world_count, dtype=int, device=self.device)
        self.record_free_count = wp.zeros(self.world_count, dtype=int, device=self.device)
        self.record_free = wp.zeros(self.record_capacity * self.world_count, dtype=int, device=self.device)
        self.slot_bump = wp.zeros(self.world_count, dtype=int, device=self.device)
        self.slot_free_count = wp.zeros(self.world_count, dtype=int, device=self.device)
        self.slot_free = wp.zeros(count, dtype=int, device=self.device)
        # per-update bookkeeping
        self.tile_begins = wp.empty(patch.bodies.shape, dtype=wp.vec2i, device=self.device)
        self.prev_begins = wp.full(patch.bodies.shape, 1_000_000_000, dtype=wp.vec2i, device=self.device)
        self.flags = wp.zeros(8, dtype=int, device=self.device)
        self.changed = wp.zeros(1, dtype=int, device=self.device)
        self.revision = wp.zeros(1, dtype=int, device=self.device)
        self.tile_counter = wp.zeros(1, dtype=int, device=self.device)
        self.generation = wp.zeros(self.world_count, dtype=int, device=self.device)
        self.errors = wp.zeros(self.world_count, dtype=int, device=self.device)
        self.fresh_capacity = self.world_count * regions * int(np.prod(self.tile_shape))
        self.fresh = wp.empty(self.fresh_capacity, dtype=wp.vec3i, device=self.device)

        template, _, self.radius = sample_particle_tile(patch.terrain)
        self.templates = SettledTemplates(patch, template, self.radius)
        self.template_count = len(template)
        self.live = ParticleState()
        material = patch.material_state.mpm
        for name, array in {
            "grad": material.particle_qd_grad,
            "elastic": material.particle_elastic_strain,
            "plastic": material.particle_Jp,
            "stress": material.particle_stress,
            "transform": material.particle_transform,
            "mass": patch.model.particle_mass,
            "radius": patch.model.particle_radius,
            "flags": patch.model.particle_flags,
        }.items():
            setattr(self.live, name, array)
        self._mirrors: dict[int, tuple[int, dict]] = {}
        self.pages = _WorldView(self, "pages")
        self.visited = _WorldView(self, "visited")
        self.loaded = _WorldView(self, "loaded")
        self.page_revisions = _WorldView(self, "page_revisions")

    # ------------------------------------------------------------------------------------------------------------
    # Device update
    # ------------------------------------------------------------------------------------------------------------
    def update(self, state: newton.State) -> None:
        """Save departing particles and restore or generate entering tiles; ``self.changed`` tells whether any moved.

        Only launches kernels: no host reads, no allocation, so it can be recorded in a CUDA graph.
        """
        patch = self.patch
        tile = self.cfg.tile_size
        tile64 = wp.float64(tile)
        shape = wp.vec2i(*(int(v) for v in self.tile_shape))
        size = wp.vec2(*patch.terrain.simulated_terrain_size)
        regions = patch.bodies.shape[1]
        count = patch.model.particle_count
        self.live.q, self.live.qd = state.particle_q, state.particle_qd
        wp.launch(
            compute_tile_begins,
            dim=patch.bodies.shape,
            inputs=[
                self.body_counts,
                tile64,
                shape,
                wp.vec2i(*(int(v) for v in self.tile_lower)),
                wp.vec2i(*(int(v) for v in self.tile_upper)),
                patch.patch_centers,
                self.tile_begins,
            ],
            device=self.device,
        )
        self.flags.zero_()
        wp.launch(
            plan_transitions,
            dim=count,
            inputs=[
                self.live,
                self.frozen,
                patch.model.particle_world,
                patch.patch_centers,
                patch.env_reset_requested,
                size,
                self.tile_begins,
                shape,
                tile,
                self.layers,
                self.next_layers,
                self.outgoing_ids,
                self.outgoing_tiles,
                self.flags,
            ],
            device=self.device,
        )
        wp.launch(
            reset_worlds,
            dim=self.world_count,
            inputs=[
                patch.env_reset_requested,
                self.record_bump,
                self.record_free_count,
                self.slot_bump,
                self.slot_free_count,
                self.visited_count,
                self.generation,
                self.prev_begins,
                self.flags,
            ],
            device=self.device,
        )
        wp.launch(
            clear_tile_tables,
            dim=(self.world_count, self.table_capacity),
            inputs=[
                patch.env_reset_requested,
                self.table_keys,
                self.table_heads,
                self.table_revisions,
                self.table_loaded_once,
            ],
            device=self.device,
        )
        wp.launch(
            save_outgoing,
            dim=count,
            inputs=[
                self.live,
                self.frozen,
                self.layers,
                patch.model.particle_world,
                self.outgoing_ids,
                self.outgoing_tiles,
                self.flags,
                self.saved,
                self.record_capacity,
                self.record_bump,
                self.record_free_count,
                self.record_free,
                self.record_next,
                self.table_keys,
                self.table_heads,
                self.table_revisions,
                self.visited_count,
                self.max_visited,
                self.tile_counter,
                self.slot_capacity,
                self.slot_free_count,
                self.slot_free,
                self.errors,
            ],
            device=self.device,
        )
        wp.launch(
            apply_transitions,
            dim=count,
            inputs=[
                self.live,
                self.frozen,
                self.inactive_position,
                self.layers,
                self.next_layers,
                patch.particle_is_dynamic,
                patch.particle_reset_positions,
                patch.model.particle_inv_mass,
            ],
            device=self.device,
        )
        wp.launch(
            restore_entering_tiles,
            dim=self.fresh_capacity,
            inputs=[
                self.body_counts,
                self.tile_begins,
                self.prev_begins,
                shape,
                self.saved,
                self.live,
                self.frozen,
                self.table_keys,
                self.table_heads,
                self.table_revisions,
                self.table_loaded_once,
                self.visited_count,
                self.max_visited,
                self.tile_counter,
                self.record_capacity,
                self.record_free_count,
                self.record_free,
                self.record_next,
                self.slot_capacity,
                self.slot_bump,
                self.slot_free_count,
                self.slot_free,
                self.slot_table,
                patch.patch_centers,
                size,
                self.layers,
                patch.particle_is_dynamic,
                patch.particle_reset_positions,
                patch.model.particle_inv_mass,
                self.fresh,
                self.flags,
                self.errors,
            ],
            device=self.device,
        )
        mesh = patch.background_mesh
        wp.launch(
            generate_fresh_tiles,
            dim=self.fresh_capacity * self.template_count,
            inputs=[
                self.fresh,
                self.flags,
                self.templates.q,
                self.templates.elastic,
                self.templates.transform,
                self.templates.stress,
                self.templates.plastic,
                self.template_count,
                tile,
                mesh.mesh.id,
                mesh.height_query_start_z,
                mesh.height_query_max_distance,
                patch.particle_dynamic_mass,
                self.radius,
                self.live,
                self.frozen,
                patch.patch_centers,
                size,
                self.layers,
                patch.particle_is_dynamic,
                patch.particle_reset_positions,
                patch.model.particle_inv_mass,
                patch.terrain_query_miss_count,
                self.slot_capacity,
                self.slot_bump,
                self.slot_free_count,
                self.slot_free,
                self.slot_table,
                self.errors,
            ],
            device=self.device,
        )
        wp.launch(
            finalize_update,
            dim=(self.world_count, regions),
            inputs=[self.flags, self.tile_begins, self.prev_begins, self.changed, self.revision],
            device=self.device,
        )

    def check_errors(self) -> None:
        """Raise if any capacity was exceeded since the last check. Reads one small array from the device."""
        errors = self.errors.numpy()
        if not errors.any():
            return
        self.errors.zero_()
        world = int(np.flatnonzero(errors)[0])
        bits = int(errors[world])
        if bits & ERROR_RECORDS:
            raise RuntimeError(
                f"Environment {world}: particle cache capacity exceeded. Increase "
                "particle_cache.max_cached_particles_per_env."
            )
        if bits & ERROR_TILES:
            raise RuntimeError(
                f"Environment {world}: visited tile capacity exceeded. Increase "
                "particle_cache.max_visited_tiles_per_env."
            )
        raise RuntimeError(
            f"Environment {world}: accumulated terrain needs more simulation slots than are free. Increase "
            "particle_cache.simulation_capacity_multiplier."
        )

    # ------------------------------------------------------------------------------------------------------------
    # Host mirrors (rendering and tests only)
    # ------------------------------------------------------------------------------------------------------------
    @property
    def cached_particle_counts(self) -> tuple[int, ...]:
        """Saved particle count per environment (records in use)."""
        used = self.record_bump.numpy() - self.record_free_count.numpy()
        return tuple(int(v) for v in used)

    @property
    def generations(self) -> list[int]:
        """Number of resets per environment since construction."""
        return [int(v) for v in self.generation.numpy()]

    def _world_mirror(self, world: int) -> dict:
        revision = int(self.revision.numpy()[0])
        cached = self._mirrors.get(world)
        if cached is not None and cached[0] == revision:
            return cached[1]
        keys = self.table_keys.numpy()[world]
        heads = self.table_heads.numpy()[world]
        revisions = self.table_revisions.numpy()[world]
        loaded_once = self.table_loaded_once.numpy()[world]
        offset, span = int(TILE_OFFSET), int(TILE_RANGE)
        used = np.flatnonzero(keys >= 0)
        x = (keys[used] // span - offset).tolist()
        y = (keys[used] % span - offset).tolist()
        tiles = list(zip(x, y))
        stored = heads[used] >= 0
        first_record = {tile: int(head) for tile, head, has in zip(tiles, heads[used].tolist(), stored.tolist()) if has}
        page_revisions = {
            tile: int(rev) for tile, rev, has in zip(tiles, revisions[used].tolist(), stored.tolist()) if has
        }
        visited = {tile for tile, once in zip(tiles, loaded_once[used].tolist()) if once}
        pages = _LazyPages(self, world, first_record)
        begins = self.tile_begins.numpy()[world]
        loaded = {
            (int(begin[0]) + x, int(begin[1]) + y)
            for region, begin in enumerate(begins)
            if region < self.patch.body_counts[world]
            for x in range(self.tile_shape[0])
            for y in range(self.tile_shape[1])
        }
        mirror = {
            "pages": pages,
            "visited": visited,
            "loaded": loaded,
            "page_revisions": page_revisions,
        }
        self._mirrors[world] = (revision, mirror)
        return mirror
