# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Join displayed particle surfaces to the displayed background terrain."""

import warp as wp

from .felix.paged_sand_isosurface import ISO_SKIRT


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
