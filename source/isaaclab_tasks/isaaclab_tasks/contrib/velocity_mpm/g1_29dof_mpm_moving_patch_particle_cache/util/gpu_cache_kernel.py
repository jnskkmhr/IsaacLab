# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-side tile tables, allocators and transition kernels of the particle cache.

Everything here runs with fixed launch sizes and without host synchronization, so the cache update can be recorded
in a CUDA graph together with the rest of the physics step. Capacity problems set bits in a per-world error array that
the host checks once per control step.
"""

import warp as wp

from .kernel import ParticleState, copy_particle, finish_loading, surface_height

TILE_OFFSET = wp.constant(16384)
TILE_RANGE = wp.constant(32768)
"""Tile coordinates are packed as ``(x + OFFSET) * RANGE + (y + OFFSET)``, which needs ``|coordinate| < OFFSET``."""
EMPTY = wp.constant(-1)
NO_TILE = wp.constant(1_000_000_000)
"""Begin coordinate of an unused or reset tile window; no tile lies inside it."""
ERROR_RECORDS = wp.constant(1)
ERROR_TILES = wp.constant(2)
ERROR_SLOTS = wp.constant(4)

FLAG_OUTGOING = wp.constant(0)
FLAG_LAYER_CHANGES = wp.constant(1)
FLAG_FRESH = wp.constant(2)
FLAG_RESTORED = wp.constant(3)
FLAG_RESET = wp.constant(4)


@wp.func
def pack_tile(tx: int, ty: int) -> int:
    return (tx + TILE_OFFSET) * TILE_RANGE + (ty + TILE_OFFSET)


@wp.func
def tile_hash(key: int, capacity: int) -> int:
    h = wp.uint32(key) * wp.uint32(2654435761)
    return int(h >> wp.uint32(7)) & (capacity - 1)


@wp.func
def find_or_insert(keys: wp.array2d[int], visited_count: wp.array[int], world: int, key: int):
    """Return ``(entry, inserted)`` for a tile of one world; ``entry`` is -1 when the table is full."""
    capacity = keys.shape[1]
    start = tile_hash(key, capacity)
    entry = int(-1)
    inserted = int(0)
    for n in range(capacity):
        idx = (start + n) & (capacity - 1)
        old = wp.atomic_cas(keys, world, idx, EMPTY, key)
        if old == EMPTY:
            wp.atomic_add(visited_count, world, 1)
            entry = idx
            inserted = 1
            break
        if old == key:
            entry = idx
            break
    return entry, inserted


@wp.func
def take_record(
    bump: wp.array[int], free_count: wp.array[int], free_stack: wp.array[int], world: int, capacity: int
) -> int:
    """Pop the most recently freed record of this world, else the next never-used one; -1 when exhausted.

    Only one thread per world calls this, in a fixed order, so the record handed to each particle is reproducible.
    """
    n = free_count[world]
    if n > 0:
        free_count[world] = n - 1
        return free_stack[world * capacity + n - 1]
    b = bump[world]
    if b < capacity:
        bump[world] = b + 1
        return world * capacity + b
    return -1


@wp.func
def take_slot(
    bump: wp.array[int],
    free_count: wp.array[int],
    free_stack: wp.array[int],
    slot_table: wp.array[int],
    world: int,
    capacity: int,
) -> int:
    """Pop the most recently freed simulation slot of this world, else the next never-used one; -1 when none."""
    n = free_count[world]
    if n > 0:
        free_count[world] = n - 1
        return free_stack[world * capacity + n - 1]
    b = bump[world]
    if b < capacity:
        bump[world] = b + 1
        return slot_table[world * capacity + b]
    return -1


@wp.func
def inside_window(tile: wp.vec2i, begin: wp.vec2i, shape: wp.vec2i) -> bool:
    return (
        tile[0] >= begin[0] and tile[0] < begin[0] + shape[0] and tile[1] >= begin[1] and tile[1] < begin[1] + shape[1]
    )


@wp.kernel
def compute_tile_begins(
    body_counts: wp.array[int],
    tile_size: wp.float64,
    shape: wp.vec2i,
    lower: wp.vec2i,
    upper: wp.vec2i,
    centers: wp.array2d[wp.vec2],
    begins: wp.array2d[wp.vec2i],
):
    """Snap each tracked body's patch to the tile grid and keep it inside the background terrain.

    The arithmetic mirrors the host cache exactly (float32 quotient, float64 offset and product) so that patch centers
    and the layer of boundary particles are bit-identical.
    """
    w, r = wp.tid()
    half_x = wp.float64(shape[0]) * wp.float64(0.5)
    half_y = wp.float64(shape[1]) * wp.float64(0.5)
    tile_f32 = wp.float32(tile_size)
    if r < body_counts[w]:
        c = centers[w, r]
        qx = wp.float64(c[0] / tile_f32) - half_x
        qy = wp.float64(c[1] / tile_f32) - half_y
        bx = wp.clamp(int(wp.rint(qx)), lower[0], upper[0])
        by = wp.clamp(int(wp.rint(qy)), lower[1], upper[1])
        begins[w, r] = wp.vec2i(bx, by)
        cx = float((wp.float64(bx) + half_x) * tile_size)
        cy = float((wp.float64(by) + half_y) * tile_size)
        centers[w, r] = wp.vec2(cx, cy)
    else:
        begins[w, r] = wp.vec2i(NO_TILE, NO_TILE)
        cx = float((wp.float64(1.0e9) + half_x) * tile_size)
        cy = float((wp.float64(1.0e9) + half_y) * tile_size)
        centers[w, r] = wp.vec2(cx, cy)


@wp.kernel
def reset_worlds(
    reset: wp.array[int],
    record_bump: wp.array[int],
    record_free_count: wp.array[int],
    slot_bump: wp.array[int],
    slot_free_count: wp.array[int],
    visited_count: wp.array[int],
    generation: wp.array[int],
    prev_begins: wp.array2d[wp.vec2i],
    flags: wp.array[int],
):
    """Forget the history of reset worlds in constant time: no pool is refilled."""
    w = wp.tid()
    if reset[w] != 0:
        record_bump[w] = 0
        record_free_count[w] = 0
        slot_bump[w] = 0
        slot_free_count[w] = 0
        visited_count[w] = 0
        generation[w] = generation[w] + 1
        flags[FLAG_RESET] = 1
        for r in range(prev_begins.shape[1]):
            prev_begins[w, r] = wp.vec2i(NO_TILE, NO_TILE)


@wp.kernel
def clear_tile_tables(
    reset: wp.array[int],
    keys: wp.array2d[int],
    heads: wp.array2d[int],
    revisions: wp.array2d[int],
    loaded_once: wp.array2d[int],
):
    w, j = wp.tid()
    if reset[w] != 0:
        keys[w, j] = EMPTY
        heads[w, j] = EMPTY
        revisions[w, j] = 0
        loaded_once[w, j] = 0


@wp.kernel
def assign_outgoing(
    flags: wp.array[int],
    reset: wp.array[int],
    layers: wp.array[int],
    next_layers: wp.array[int],
    live: ParticleState,
    frozen: ParticleState,
    slot_table: wp.array[int],
    slot_capacity: int,
    tile_size: float,
    record_capacity: int,
    record_bump: wp.array[int],
    record_free_count: wp.array[int],
    record_free: wp.array[int],
    record_next: wp.array[int],
    keys: wp.array2d[int],
    heads: wp.array2d[int],
    revisions: wp.array2d[int],
    visited_count: wp.array[int],
    max_visited: int,
    tile_counter: wp.array[int],
    slot_free_count: wp.array[int],
    slot_free: wp.array[int],
    slot_record: wp.array[int],
    errors: wp.array[int],
):
    """Give each departing particle a record and free its slot, one thread per world in ascending slot order.

    Records, list order and the free-slot stack therefore do not depend on thread scheduling. The particle data is
    copied afterwards in parallel by ``copy_outgoing``.
    """
    w = wp.tid()
    if flags[FLAG_OUTGOING] == 0 or reset[w] != 0:
        return
    for s in range(slot_capacity):
        i = slot_table[w * slot_capacity + s]
        if layers[i] == 0 or next_layers[i] != 0:
            continue
        p = live.q[i]
        if layers[i] == 2:
            p = frozen.q[i]
        tile = wp.vec2i(int(wp.floor(p[0] / tile_size)), int(wp.floor(p[1] / tile_size)))
        record = take_record(record_bump, record_free_count, record_free, w, record_capacity)
        if record < 0:
            wp.atomic_or(errors, w, ERROR_RECORDS)
            continue
        slot_record[i] = record
        entry, inserted = find_or_insert(keys, visited_count, w, pack_tile(tile[0], tile[1]))
        if entry < 0 or visited_count[w] > max_visited:
            wp.atomic_or(errors, w, ERROR_TILES)
        if entry >= 0:
            record_next[record] = heads[w, entry]
            heads[w, entry] = record
            revisions[w, entry] = wp.atomic_add(tile_counter, 0, 1) + 1
        n = slot_free_count[w]
        slot_free[w * slot_capacity + n] = i
        slot_free_count[w] = n + 1


@wp.kernel
def copy_outgoing(
    live: ParticleState,
    frozen: ParticleState,
    layers: wp.array[int],
    slot_record: wp.array[int],
    saved: ParticleState,
):
    """Copy the state of each departing particle into the record ``assign_outgoing`` chose; saves the frozen state of
    boundary particles."""
    i = wp.tid()
    record = slot_record[i]
    if record < 0:
        return
    if layers[i] == 2:
        copy_particle(frozen, i, saved, record)
    else:
        copy_particle(live, i, saved, record)
    slot_record[i] = -1


@wp.kernel
def assign_restore(
    body_counts: wp.array[int],
    begins: wp.array2d[wp.vec2i],
    prev_begins: wp.array2d[wp.vec2i],
    shape: wp.vec2i,
    keys: wp.array2d[int],
    heads: wp.array2d[int],
    revisions: wp.array2d[int],
    loaded_once: wp.array2d[int],
    visited_count: wp.array[int],
    max_visited: int,
    tile_counter: wp.array[int],
    record_capacity: int,
    record_free_count: wp.array[int],
    record_free: wp.array[int],
    record_next: wp.array[int],
    slot_capacity: int,
    slot_bump: wp.array[int],
    slot_free_count: wp.array[int],
    slot_free: wp.array[int],
    slot_table: wp.array[int],
    fresh: wp.array[wp.vec3i],
    fresh_count: wp.array[int],
    fresh_per_world: int,
    slot_source: wp.array[int],
    flags: wp.array[int],
    errors: wp.array[int],
):
    """Choose a slot for every saved particle of each tile that entered a window, in window and tile order.

    One thread per world. A never-visited tile is queued in ``fresh``; its slots are chosen by ``assign_fresh``.
    """
    w = wp.tid()
    tiles_per_window = shape[0] * shape[1]
    regions = begins.shape[1]
    for r in range(regions):
        if r >= body_counts[w]:
            continue
        begin = begins[w, r]
        for t in range(tiles_per_window):
            tile = wp.vec2i(begin[0] + t // shape[1], begin[1] + t % shape[1])
            skip = bool(False)
            # A tile covered by several windows is handled once, by the first window that holds it.
            for other in range(r):
                if inside_window(tile, begins[w, other], shape):
                    skip = True
            # A tile that was already inside a window of the previous update is not entering.
            for other in range(regions):
                if inside_window(tile, prev_begins[w, other], shape):
                    skip = True
            if skip:
                continue
            entry, inserted = find_or_insert(keys, visited_count, w, pack_tile(tile[0], tile[1]))
            if entry < 0 or visited_count[w] > max_visited:
                wp.atomic_or(errors, w, ERROR_TILES)
            if entry < 0:
                continue
            # A tile can hold saved particles pushed there before it was ever loaded. It still needs its undeformed
            # terrain, so "has saved particles" and "has been loaded" are tracked separately, as the host cache does.
            if loaded_once[w, entry] == 0:
                loaded_once[w, entry] = 1
                n = fresh_count[w]
                fresh[w * fresh_per_world + n] = wp.vec3i(w, tile[0], tile[1])
                fresh_count[w] = n + 1
                wp.atomic_add(flags, FLAG_FRESH, 1)
            record = heads[w, entry]
            while record >= 0:
                slot = take_slot(slot_bump, slot_free_count, slot_free, slot_table, w, slot_capacity)
                if slot < 0:
                    wp.atomic_or(errors, w, ERROR_SLOTS)
                    break
                slot_source[slot] = record
                top = record_free_count[w]
                record_free[w * record_capacity + top] = record
                record_free_count[w] = top + 1
                wp.atomic_add(flags, FLAG_RESTORED, 1)
                record = record_next[record]
            heads[w, entry] = record
            revisions[w, entry] = wp.atomic_add(tile_counter, 0, 1) + 1


@wp.kernel
def copy_restored(
    saved: ParticleState,
    live: ParticleState,
    frozen: ParticleState,
    worlds: wp.array[int],
    slot_source: wp.array[int],
    centers: wp.array2d[wp.vec2],
    size: wp.vec2,
    layers: wp.array[int],
    dynamic: wp.array[int],
    anchors: wp.array[wp.vec3],
    inv_mass: wp.array[float],
):
    """Load the saved particle that ``assign_restore`` assigned to each slot."""
    i = wp.tid()
    record = slot_source[i]
    if record < 0:
        return
    copy_particle(saved, record, live, i)
    finish_loading(live, frozen, i, centers, worlds[i], size, layers, dynamic, anchors, inv_mass)
    slot_source[i] = -1


@wp.kernel
def assign_fresh(
    fresh: wp.array[wp.vec3i],
    fresh_count: wp.array[int],
    fresh_per_world: int,
    template_count: int,
    slot_capacity: int,
    slot_bump: wp.array[int],
    slot_free_count: wp.array[int],
    slot_free: wp.array[int],
    slot_table: wp.array[int],
    fresh_slots: wp.array[int],
    errors: wp.array[int],
):
    """Choose a slot for every template particle of each never-visited tile, one thread per world in queue order."""
    w = wp.tid()
    for local in range(fresh_count[w]):
        base = (w * fresh_per_world + local) * template_count
        for j in range(template_count):
            slot = take_slot(slot_bump, slot_free_count, slot_free, slot_table, w, slot_capacity)
            if slot < 0:
                wp.atomic_or(errors, w, ERROR_SLOTS)
            fresh_slots[base + j] = slot


@wp.kernel
def generate_fresh_tiles(
    fresh: wp.array[wp.vec3i],
    fresh_count: wp.array[int],
    fresh_per_world: int,
    fresh_slots: wp.array[int],
    template: wp.array[wp.vec3],
    template_elastic: wp.array[wp.mat33],
    template_transform: wp.array[wp.mat33],
    template_stress: wp.array[wp.mat33],
    template_plastic: wp.array[float],
    template_count: int,
    tile_size: float,
    mesh: wp.uint64,
    ray_z: float,
    ray_length: float,
    masses: wp.array[float],
    radius: float,
    live: ParticleState,
    frozen: ParticleState,
    centers: wp.array2d[wp.vec2],
    size: wp.vec2,
    layers: wp.array[int],
    dynamic: wp.array[int],
    anchors: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    missed: wp.array[int],
):
    """Generate the undeformed template particles of each never-visited tile from the background mesh."""
    k = wp.tid()
    f = k // template_count
    w = f // fresh_per_world
    if f % fresh_per_world >= fresh_count[w]:
        return
    j = k % template_count
    entry3 = fresh[f]
    tile = wp.vec2i(entry3[1], entry3[2])
    i = fresh_slots[k]
    if i < 0:
        return
    entry = w * template_count + j
    p = template[entry]
    # Alternating mirror tiles retain matching wall states (Felix's mirror_tensor construction).
    dx = wp.where(tile[0] % 2 == 0, 1.0, -1.0)
    dy = wp.where(tile[1] % 2 == 0, 1.0, -1.0)
    mirror = wp.diag(wp.vec3(dx, dy, 1.0))
    p = wp.vec3(wp.where(dx < 0.0, tile_size - p[0], p[0]), wp.where(dy < 0.0, tile_size - p[1], p[1]), p[2])
    x = float(tile[0]) * tile_size + p[0]
    y = float(tile[1]) * tile_size + p[1]
    hit, height = surface_height(mesh, x, y, ray_z, ray_length)
    if not hit:
        wp.atomic_add(missed, 0, 1)
    live.q[i] = wp.vec3(x, y, height + p[2])
    live.qd[i] = wp.vec3(0.0)
    live.grad[i] = wp.mat33(0.0)
    live.elastic[i] = mirror * template_elastic[entry] * mirror
    live.plastic[i] = template_plastic[entry]
    live.stress[i] = mirror * template_stress[entry] * mirror
    live.transform[i] = mirror * template_transform[entry] * mirror
    live.mass[i] = masses[i]
    live.radius[i] = radius
    live.flags[i] = 1
    finish_loading(live, frozen, i, centers, w, size, layers, dynamic, anchors, inv_mass)


@wp.kernel
def finalize_update(
    flags: wp.array[int],
    begins: wp.array2d[wp.vec2i],
    prev_begins: wp.array2d[wp.vec2i],
    changed: wp.array[int],
    revision: wp.array[int],
):
    """Remember this update's windows and publish whether any particle changed state."""
    w, r = wp.tid()
    prev_begins[w, r] = begins[w, r]
    if w == 0 and r == 0:
        total = (
            flags[FLAG_OUTGOING]
            + flags[FLAG_LAYER_CHANGES]
            + flags[FLAG_FRESH]
            + flags[FLAG_RESTORED]
            + flags[FLAG_RESET]
        )
        changed[0] = wp.where(total > 0, 1, 0)
        if total > 0:
            revision[0] = revision[0] + 1
