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
def alloc_record(
    bump: wp.array[int], free_count: wp.array[int], free_stack: wp.array[int], world: int, capacity: int
) -> int:
    """Pop a recycled record of this world, else take the next never-used one; -1 when the pool is exhausted."""
    top = wp.atomic_add(free_count, world, -1) - 1
    if top >= 0:
        return free_stack[world * capacity + top]
    wp.atomic_add(free_count, world, 1)
    b = wp.atomic_add(bump, world, 1)
    if b < capacity:
        return world * capacity + b
    wp.atomic_add(bump, world, -1)
    return -1


@wp.func
def alloc_slot(
    bump: wp.array[int],
    free_count: wp.array[int],
    free_stack: wp.array[int],
    slot_table: wp.array[int],
    world: int,
    capacity: int,
) -> int:
    """Pop a recycled simulation slot of this world, else the next never-used one; -1 when none is free."""
    top = wp.atomic_add(free_count, world, -1) - 1
    if top >= 0:
        return free_stack[world * capacity + top]
    wp.atomic_add(free_count, world, 1)
    b = wp.atomic_add(bump, world, 1)
    if b < capacity:
        return slot_table[world * capacity + b]
    wp.atomic_add(bump, world, -1)
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
def save_outgoing(
    live: ParticleState,
    frozen: ParticleState,
    layers: wp.array[int],
    worlds: wp.array[int],
    outgoing_ids: wp.array[int],
    outgoing_tiles: wp.array[wp.vec2i],
    flags: wp.array[int],
    saved: ParticleState,
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
    slot_capacity: int,
    slot_free_count: wp.array[int],
    slot_free: wp.array[int],
    errors: wp.array[int],
):
    """Move each departing particle into a record of its tile's list and free its simulation slot."""
    k = wp.tid()
    if k >= flags[FLAG_OUTGOING]:
        return
    slot = outgoing_ids[k]
    w = worlds[slot]
    tile = outgoing_tiles[k]
    record = alloc_record(record_bump, record_free_count, record_free, w, record_capacity)
    if record < 0:
        wp.atomic_or(errors, w, ERROR_RECORDS)
        return
    if layers[slot] == 2:
        copy_particle(frozen, slot, saved, record)
    else:
        copy_particle(live, slot, saved, record)
    entry, inserted = find_or_insert(keys, visited_count, w, pack_tile(tile[0], tile[1]))
    if entry < 0 or visited_count[w] > max_visited:
        wp.atomic_or(errors, w, ERROR_TILES)
    if entry >= 0:
        old = wp.atomic_exch(heads, w, entry, record)
        record_next[record] = old
        revisions[w, entry] = wp.atomic_add(tile_counter, 0, 1) + 1
    top = wp.atomic_add(slot_free_count, w, 1)
    slot_free[w * slot_capacity + top] = slot


@wp.kernel
def restore_entering_tiles(
    body_counts: wp.array[int],
    begins: wp.array2d[wp.vec2i],
    prev_begins: wp.array2d[wp.vec2i],
    shape: wp.vec2i,
    saved: ParticleState,
    live: ParticleState,
    frozen: ParticleState,
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
    centers: wp.array2d[wp.vec2],
    size: wp.vec2,
    layers: wp.array[int],
    dynamic: wp.array[int],
    anchors: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    fresh: wp.array[wp.vec3i],
    flags: wp.array[int],
    errors: wp.array[int],
):
    """Restore the saved particles of each tile that entered a window, or queue a never-visited tile."""
    tid = wp.tid()
    tiles_per_window = shape[0] * shape[1]
    regions = begins.shape[1]
    w = tid // (regions * tiles_per_window)
    r = (tid // tiles_per_window) % regions
    t = tid % tiles_per_window
    if r >= body_counts[w]:
        return
    begin = begins[w, r]
    tile = wp.vec2i(begin[0] + t // shape[1], begin[1] + t % shape[1])
    # A tile covered by several windows is handled once, by the first window that holds it.
    for other in range(r):
        if inside_window(tile, begins[w, other], shape):
            return
    # A tile that was already inside a window of the previous update is not entering.
    for other in range(regions):
        if inside_window(tile, prev_begins[w, other], shape):
            return
    entry, inserted = find_or_insert(keys, visited_count, w, pack_tile(tile[0], tile[1]))
    if entry < 0 or visited_count[w] > max_visited:
        wp.atomic_or(errors, w, ERROR_TILES)
    if entry < 0:
        return
    # A tile can hold saved particles pushed there before it was ever loaded. It still needs its undeformed terrain,
    # so "has saved particles" and "has been loaded" are tracked separately, as the host cache does.
    if wp.atomic_exch(loaded_once, w, entry, 1) == 0:
        index = wp.atomic_add(flags, FLAG_FRESH, 1)
        fresh[index] = wp.vec3i(w, tile[0], tile[1])
    record = heads[w, entry]
    while record >= 0:
        slot = alloc_slot(slot_bump, slot_free_count, slot_free, slot_table, w, slot_capacity)
        if slot < 0:
            wp.atomic_or(errors, w, ERROR_SLOTS)
            break
        following = record_next[record]
        copy_particle(saved, record, live, slot)
        finish_loading(live, frozen, slot, centers, w, size, layers, dynamic, anchors, inv_mass)
        top = wp.atomic_add(record_free_count, w, 1)
        record_free[w * record_capacity + top] = record
        wp.atomic_add(flags, FLAG_RESTORED, 1)
        record = following
    heads[w, entry] = record
    revisions[w, entry] = wp.atomic_add(tile_counter, 0, 1) + 1


@wp.kernel
def generate_fresh_tiles(
    fresh: wp.array[wp.vec3i],
    flags: wp.array[int],
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
    slot_capacity: int,
    slot_bump: wp.array[int],
    slot_free_count: wp.array[int],
    slot_free: wp.array[int],
    slot_table: wp.array[int],
    errors: wp.array[int],
):
    """Generate the undeformed template particles of each never-visited tile from the background mesh."""
    k = wp.tid()
    f = k // template_count
    if f >= flags[FLAG_FRESH]:
        return
    j = k % template_count
    entry3 = fresh[f]
    w = entry3[0]
    tile = wp.vec2i(entry3[1], entry3[2])
    i = alloc_slot(slot_bump, slot_free_count, slot_free, slot_table, w, slot_capacity)
    if i < 0:
        wp.atomic_or(errors, w, ERROR_SLOTS)
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
