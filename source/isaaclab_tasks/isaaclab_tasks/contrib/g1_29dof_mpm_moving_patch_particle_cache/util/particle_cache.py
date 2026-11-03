# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded GPU particle history with independent world-XY tile maps per environment.

Particle fields stay on the GPU. Python receives patch centers, outgoing integer
indices and tile coordinates, and counts. Chunk changes are not CUDA-graph captured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .felix.settled_template import SettledTemplates
from .kernel import (
    ParticleState,
    apply_transitions,
    generate_particles,
    load_particles,
    plan_transitions,
    save_particles,
)
from .particle_sampling import sample_particle_tile

if TYPE_CHECKING:
    import newton

    from .particles import MovingPatchParticles


PARTICLE_FIELDS = {
    "q": wp.vec3,
    "qd": wp.vec3,
    "grad": wp.mat33,
    "elastic": wp.mat33,
    "plastic": float,
    "stress": wp.mat33,
    "transform": wp.mat33,
    "mass": float,
    "radius": float,
    "flags": int,
}


def allocate_particle_state(count: int, device: wp.Device) -> ParticleState:
    """Allocate the particle fields needed to resume the material solve."""
    data = ParticleState()
    for name, dtype in PARTICLE_FIELDS.items():
        setattr(data, name, wp.empty(count, dtype=dtype, device=device))
    return data


class ParticleStateCache:
    """Retain departed particles and empty visited tiles until their environment resets.

    There is one cache owner per environment, even when terrain coordinates overlap.
    Loading consumes saved records; a particle is never both simulated and cached.
    Capacity is checked for all environments before any physical state is changed.
    """

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
        self.slots = [np.flatnonzero(self.worlds == world).tolist() for world in range(self.world_count)]
        self.free_slots = [list(slots) for slots in self.slots]
        capacity = self.cfg.max_cached_particles_per_env
        self.free_records = [list(range(world * capacity, (world + 1) * capacity)) for world in range(self.world_count)]
        self.pages: list[dict[tuple[int, int], list[int]]] = [{} for _ in range(self.world_count)]
        self.visited: list[set[tuple[int, int]]] = [set() for _ in range(self.world_count)]
        self.loaded: list[set[tuple[int, int]]] = [set() for _ in range(self.world_count)]
        self.generations = [0] * self.world_count
        self.page_revisions: list[dict[tuple[int, int], int]] = [{} for _ in range(self.world_count)]
        self._revision = 0
        self.saved = allocate_particle_state(capacity * self.world_count, self.device)
        self.frozen = allocate_particle_state(patch.model.particle_count, self.device)
        self.layers = wp.zeros(patch.model.particle_count, dtype=int, device=self.device)
        self.next_layers = wp.empty_like(self.layers)
        self.outgoing_ids = wp.empty_like(self.layers)
        self.outgoing_tiles = wp.empty(patch.model.particle_count, dtype=wp.vec2i, device=self.device)
        self.counts = wp.zeros(2, dtype=int, device=self.device)
        template, _, self.radius = sample_particle_tile(patch.terrain)
        self.templates = SettledTemplates(patch, template, self.radius)
        self.template_count = len(template)
        self.tile_shape = np.rint(np.asarray(patch.terrain.total_patch_size) / self.cfg.tile_size).astype(int)
        self.tile_begins = wp.empty(patch.bodies.shape, dtype=wp.vec2i, device=self.device)
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

    @property
    def cached_particle_counts(self) -> tuple[int, ...]:
        """Return saved particle counts per environment without a GPU readback."""
        return tuple(self.cfg.max_cached_particles_per_env - len(free) for free in self.free_records)

    def update(self, state: newton.State) -> bool:
        """Save departing particles and restore entering tiles; return whether solver properties changed."""
        patch = self.patch
        self.live.q, self.live.qd = state.particle_q, state.particle_qd
        centers = patch.patch_centers.numpy()
        lower = np.ceil(patch.background_mesh.bounds[0, :2] / self.cfg.tile_size).astype(int)
        upper = np.floor(patch.background_mesh.bounds[1, :2] / self.cfg.tile_size).astype(int) - self.tile_shape
        if np.any(upper < lower):
            raise ValueError("Background terrain must contain at least one full tile-aligned particle patch")
        valid = np.arange(patch.bodies.shape[1])[None, :] < patch.body_counts[:, None]
        begins = np.rint(np.where(valid[..., None], centers, 0.0) / self.cfg.tile_size - self.tile_shape / 2).astype(
            np.int32
        )
        begins = np.clip(begins, lower, upper)
        begins[~valid] = 1_000_000_000
        self.tile_begins.assign(begins)
        patch.patch_centers.assign((begins + self.tile_shape / 2) * self.cfg.tile_size)
        reset = patch.env_reset_requested.numpy().astype(bool)
        self.templates.update(reset)
        self.counts.zero_()
        wp.launch(
            plan_transitions,
            dim=patch.model.particle_count,
            inputs=[
                self.live,
                self.frozen,
                patch.model.particle_world,
                patch.patch_centers,
                patch.env_reset_requested,
                wp.vec2(*patch.terrain.simulated_terrain_size),
                self.tile_begins,
                wp.vec2i(*self.tile_shape),
                self.cfg.tile_size,
                self.layers,
                self.next_layers,
                self.outgoing_ids,
                self.outgoing_tiles,
                self.counts,
            ],
            device=self.device,
        )
        outgoing_count, changed = self.counts.numpy()
        outgoing = self.outgoing_ids[:outgoing_count].numpy() if outgoing_count else np.empty(0, dtype=int)
        tiles = self.outgoing_tiles[:outgoing_count].numpy() if outgoing_count else np.empty((0, 2), dtype=int)

        loaded_worlds = [
            {
                (x, y)
                for begin in begins[world, : patch.body_counts[world]]
                for x in range(begin[0], begin[0] + self.tile_shape[0])
                for y in range(begin[1], begin[1] + self.tile_shape[1])
            }
            for world in range(self.world_count)
        ]
        if not outgoing_count and not np.any(reset) and loaded_worlds == self.loaded:
            self._apply_transitions()
            return bool(changed)

        save_ids, save_records, load_records, load_slots, fresh_slots, fresh_tiles = [], [], [], [], [], []
        plans = []
        for world in range(self.world_count):
            loaded = loaded_worlds[world]
            if reset[world]:
                pages, visited, old_loaded = {}, set(), set()
                free_slots = list(self.slots[world])
                start = world * self.cfg.max_cached_particles_per_env
                free_records = list(range(start, start + self.cfg.max_cached_particles_per_env))
            else:
                pages = self.pages[world].copy()
                visited, old_loaded = self.visited[world], self.loaded[world]
                free_slots, free_records = list(self.free_slots[world]), list(self.free_records[world])
            selected = self.worlds[outgoing] == world
            departing, destinations = outgoing[selected], tiles[selected]
            if len(departing) > len(free_records):
                raise RuntimeError(
                    f"Environment {world}: particle cache capacity exceeded. Increase "
                    "particle_cache.max_cached_particles_per_env; no particles were recycled."
                )
            for slot, tile in zip(departing, destinations, strict=True):
                key = tuple(int(value) for value in tile)
                record = free_records.pop()
                pages[key] = [*pages.get(key, []), record]
                save_ids.append(int(slot))
                save_records.append(record)
                free_slots.append(int(slot))
            entering = sorted(loaded - old_loaded)
            next_visited = visited | loaded
            if len(next_visited | pages.keys()) > self.cfg.max_visited_tiles_per_env:
                raise RuntimeError(
                    f"Environment {world}: visited tile capacity exceeded. Increase "
                    "particle_cache.max_visited_tiles_per_env; no particles were recycled."
                )
            restored = [record for tile in entering for record in pages.get(tile, [])]
            fresh = [tile for tile in entering if tile not in visited]
            needed = len(restored) + len(fresh) * self.template_count
            if needed > len(free_slots):
                raise RuntimeError(
                    f"Environment {world}: accumulated terrain needs {needed} slots, but only "
                    f"{len(free_slots)} are free. Increase particle_cache.simulation_capacity_multiplier; "
                    "no particles were recycled."
                )
            for record in restored:
                load_records.append(record)
                load_slots.append(free_slots.pop())
            for tile in fresh:
                fresh_tiles.append(tile)
                for _ in range(self.template_count):
                    fresh_slots.append(free_slots.pop())
            for tile in entering:
                free_records.extend(pages.pop(tile, []))
            plans.append((pages, next_visited, loaded, free_slots, free_records))

        # Capacity checks have passed for every world. Save before reusing any simulation slots.
        def indices(values, dtype=int):
            return wp.array(values, dtype=dtype, device=self.device)

        if save_ids:
            wp.launch(
                save_particles,
                dim=len(save_ids),
                inputs=[self.live, self.frozen, self.layers, indices(save_ids), self.saved, indices(save_records)],
                device=self.device,
            )
        self._apply_transitions()
        common = [self.live, self.frozen]
        tail = [
            patch.model.particle_world,
            patch.patch_centers,
            wp.vec2(*patch.terrain.simulated_terrain_size),
            self.layers,
            patch.particle_is_dynamic,
            patch.particle_reset_positions,
            patch.model.particle_inv_mass,
        ]
        if load_slots:
            wp.launch(
                load_particles,
                dim=len(load_slots),
                inputs=[self.saved, indices(load_records), *common, indices(load_slots), *tail],
                device=self.device,
            )
        if fresh_slots:
            mesh = patch.background_mesh
            wp.launch(
                generate_particles,
                dim=len(fresh_slots),
                inputs=[
                    indices(fresh_slots),
                    indices(fresh_tiles, wp.vec2i),
                    self.templates.q,
                    self.templates.elastic,
                    self.templates.transform,
                    self.templates.stress,
                    self.templates.plastic,
                    self.template_count,
                    self.cfg.tile_size,
                    mesh.mesh.id,
                    mesh.height_query_start_z,
                    mesh.height_query_max_distance,
                    patch.particle_dynamic_mass,
                    self.radius,
                    *common,
                    *tail,
                    patch.terrain_query_miss_count,
                ],
                device=self.device,
            )
        for world, plan in enumerate(plans):
            self._revision += 1
            if reset[world]:
                self.generations[world] += 1
            self.page_revisions[world] = {
                tile: self.page_revisions[world][tile]
                if not reset[world] and self.pages[world].get(tile) is records
                else self._revision
                for tile, records in plan[0].items()
            }
            (
                self.pages[world],
                self.visited[world],
                self.loaded[world],
                self.free_slots[world],
                self.free_records[world],
            ) = plan
        return bool(changed or save_ids or load_slots or fresh_slots or np.any(reset))

    def _apply_transitions(self) -> None:
        patch = self.patch
        wp.launch(
            apply_transitions,
            dim=patch.model.particle_count,
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
