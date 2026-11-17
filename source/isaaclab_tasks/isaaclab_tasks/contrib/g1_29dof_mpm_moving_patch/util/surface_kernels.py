# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU gathers, fixed-capacity mesh publication, and filtered particle glyphs."""

import warp as wp


@wp.kernel(enable_backward=False)
def gather_particles(
    source: wp.array[wp.vec3],
    radii: wp.array[float],
    ids: wp.array[int],
    positions: wp.array[wp.vec3],
    sizes: wp.array[float],
):
    i = wp.tid()
    positions[i] = source[ids[i]]
    sizes[i] = radii[ids[i]]


@wp.kernel(enable_backward=False, module="unique")
def check_capacity(counts: wp.array[int], vertex_capacity: int, index_capacity: int):
    assert counts[2] == 0, "Sand surface grid capacity exceeded: increase sand_surface.max_grid_cells"
    assert counts[0] <= vertex_capacity, "Sand surface vertex capacity exceeded: increase sand_surface.max_vertices"
    assert counts[1] <= index_capacity, "Sand surface triangle capacity exceeded: increase sand_surface.max_triangles"


# Only this one-thread guard uses debug mode: capacity failures must not silently truncate geometry.
wp.set_module_options({"mode": "debug"}, module=check_capacity.module)


@wp.kernel(enable_backward=False)
def particle_glyphs(
    q: wp.array[wp.vec3],
    radii: wp.array[float],
    activity: wp.array[int],
    local_ids: wp.array[int],
    directions: wp.array[wp.vec3],
    faces: wp.array[int],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    indices: wp.array[int],
):
    particle, vertex = wp.tid()
    n = directions.shape[0]
    active = activity[local_ids[particle]] > 0
    points[particle * n + vertex] = q[particle] + radii[particle] * directions[vertex]
    normals[particle * n + vertex] = directions[vertex]
    if vertex == 0:
        for j in range(faces.shape[0]):
            index = 0
            if active:
                index = particle * n + faces[j]
            indices[particle * faces.shape[0] + j] = index


@wp.func
def tile_index(p: wp.vec3, origin: wp.vec2, shape: wp.vec2i, size: float):
    x = int(wp.floor((p[0] - origin[0]) / size))
    y = int(wp.floor((p[1] - origin[1]) / size))
    if x < 0 or y < 0 or x >= shape[0] or y >= shape[1]:
        return -1
    return x * shape[1] + y


@wp.kernel(enable_backward=False)
def mark_live_tiles(
    q: wp.array[wp.vec3],
    previous_q: wp.array[wp.vec3],
    activity: wp.array[int],
    ids: wp.array[int],
    origin: wp.vec2,
    shape: wp.vec2i,
    size: float,
    halo: float,
    live: wp.array[int],
):
    i = wp.tid()
    p = q[i]
    if activity[ids[i]] > 0 or wp.length_sq(p - previous_q[i]) > 0.0 or tile_index(p, origin, shape, size) < 0:
        # Include both previous and current positions, including the final displacement before sleeping.
        for sample in range(2):
            center = p
            if sample == 1:
                center = previous_q[i]
            low = center - wp.vec3(halo)
            high = center + wp.vec3(halo)
            x0 = int(wp.floor((low[0] - origin[0]) / size))
            y0 = int(wp.floor((low[1] - origin[1]) / size))
            x1 = int(wp.floor((high[0] - origin[0]) / size))
            y1 = int(wp.floor((high[1] - origin[1]) / size))
            for x in range(wp.max(x0, 0), wp.min(x1 + 1, shape[0])):
                for y in range(wp.max(y0, 0), wp.min(y1 + 1, shape[1])):
                    wp.atomic_max(live, x * shape[1] + y, 1)
    previous_q[i] = p


@wp.kernel(enable_backward=False)
def compare_tiles(live: wp.array[int], previous: wp.array[int], reset: wp.array[int], refresh: wp.array[int]):
    i = wp.tid()
    value = live[i]
    if value != previous[i] or reset[0] != 0:
        wp.atomic_max(refresh, 0, 1)
    previous[i] = value


@wp.kernel(enable_backward=False)
def select_surface_particles(
    q: wp.array[wp.vec3],
    origin: wp.vec2,
    shape: wp.vec2i,
    size: float,
    halo: float,
    live: wp.array[int],
    reset: wp.array[int],
    old_live: wp.array[int],
    flags: wp.array[int],
):
    i = wp.tid()
    selected = int(0)
    if reset[0] != 0 or tile_index(q[i], origin, shape, size) < 0:
        selected = 1
    else:
        p = q[i]
        x0 = int(wp.floor((p[0] - halo - origin[0]) / size))
        y0 = int(wp.floor((p[1] - halo - origin[1]) / size))
        x1 = int(wp.floor((p[0] + halo - origin[0]) / size))
        y1 = int(wp.floor((p[1] + halo - origin[1]) / size))
        for x in range(wp.max(x0, 0), wp.min(x1 + 1, shape[0])):
            for y in range(wp.max(y0, 0), wp.min(y1 + 1, shape[1])):
                selected = wp.max(selected, wp.max(live[x * shape[1] + y], old_live[x * shape[1] + y]))
    flags[i] = selected


@wp.kernel(enable_backward=False)
def classify_surface_faces(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    counts: wp.array[int],
    origin: wp.vec2,
    shape: wp.vec2i,
    size: float,
    live: wp.array[int],
    reset: wp.array[int],
    old_live: wp.array[int],
    live_used: wp.array[int],
    cached_used: wp.array[int],
    face_group: wp.array[int],
):
    t = wp.tid()
    if t >= counts[1] / 3:
        return
    a, b, c = indices[3 * t], indices[3 * t + 1], indices[3 * t + 2]
    center = (points[a] + points[b] + points[c]) / 3.0
    tile = tile_index(center, origin, shape, size)
    is_live = tile < 0
    if tile >= 0:
        is_live = is_live or live[tile] != 0
    if is_live:
        wp.atomic_max(live_used, a, 1)
        wp.atomic_max(live_used, b, 1)
        wp.atomic_max(live_used, c, 1)
        face_group[t] = 1
    elif reset[0] != 0 or (tile >= 0 and old_live[tile] != 0):
        wp.atomic_max(cached_used, a, 1)
        wp.atomic_max(cached_used, b, 1)
        wp.atomic_max(cached_used, c, 1)
        face_group[t] = 2
    else:
        face_group[t] = 0


@wp.kernel(enable_backward=False)
def compact_surface_vertices(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    used: wp.array[int],
    offsets: wp.array[int],
    output: wp.array[wp.vec3],
    output_normals: wp.array[wp.vec3],
    metadata: wp.array[int],
    slot: int,
):
    i = wp.tid()
    if i == used.shape[0] - 1:
        metadata[slot] = offsets[i]
    if used[i] != 0:
        j = offsets[i] - 1
        if j < output.shape[0]:
            output[j] = points[i]
            output_normals[j] = normals[i]


@wp.kernel(enable_backward=False)
def compact_surface_faces(
    source: wp.array[int],
    counts: wp.array[int],
    groups: wp.array[int],
    group: int,
    offsets: wp.array[int],
    output: wp.array[int],
    metadata: wp.array[int],
    slot: int,
    vertex_offset: int,
):
    t = wp.tid()
    if t < counts[1] / 3 and groups[t] == group:
        j = wp.atomic_add(metadata, slot, 3)
        if j + 2 < output.shape[0]:
            for k in range(3):
                output[j + k] = offsets[source[3 * t + k] + vertex_offset] - 1


@wp.kernel(enable_backward=False, module="unique")
def check_compacted_capacity(metadata: wp.array[int], vertices: int, indices: int):
    assert metadata[0] <= vertices and metadata[2] <= vertices, "Increase sand_surface.max_vertices"
    assert metadata[1] <= indices and metadata[3] <= indices, "Increase sand_surface.max_triangles"


wp.set_module_options({"mode": "debug"}, module=check_compacted_capacity.module)


@wp.kernel(enable_backward=False)
def retain_cached_faces(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    counts: wp.array[int],
    origin: wp.vec2,
    shape: wp.vec2i,
    size: float,
    live: wp.array[int],
    old_live: wp.array[int],
    reset: wp.array[int],
    used: wp.array[int],
    face_group: wp.array[int],
    vertex_offset: int,
):
    t = wp.tid()
    if t >= counts[1] / 3:
        return
    a, b, c = indices[3 * t], indices[3 * t + 1], indices[3 * t + 2]
    center = (points[a] + points[b] + points[c]) / 3.0
    tile = tile_index(center, origin, shape, size)
    keep = reset[0] == 0 and tile >= 0
    if keep:
        keep = live[tile] == 0 and old_live[tile] == 0
    face_group[t] = 0
    if keep:
        face_group[t] = 2
        wp.atomic_max(used, a + vertex_offset, 1)
        wp.atomic_max(used, b + vertex_offset, 1)
        wp.atomic_max(used, c + vertex_offset, 1)


@wp.kernel(enable_backward=False)
def merge_cached_vertices(
    new_points: wp.array[wp.vec3],
    new_normals: wp.array[wp.vec3],
    old_points: wp.array[wp.vec3],
    old_normals: wp.array[wp.vec3],
    used: wp.array[int],
    offsets: wp.array[int],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    metadata: wp.array[int],
    split: int,
):
    i = wp.tid()
    if i == used.shape[0] - 1:
        metadata[2] = offsets[i]
    if used[i] != 0:
        j = offsets[i] - 1
        if j < points.shape[0]:
            if i < split:
                points[j] = new_points[i]
                normals[j] = new_normals[i]
            else:
                points[j] = old_points[i - split]
                normals[j] = old_normals[i - split]


@wp.kernel(enable_backward=False)
def mask_surface_slots(eligible: wp.array[int], ids: wp.array[int], sizes: wp.array[float], flags: wp.array[int]):
    i = wp.tid()
    if (eligible[ids[i]] & 1) == 0:
        sizes[i] = 0.0
        flags[i] = 0
