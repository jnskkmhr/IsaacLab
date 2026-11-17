# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E741 -- Retain the reference implementation's grid-coordinate names.

"""Isosurface of the loaded material in ``mpm_paged_sand`` (drawing only).

Each chunk drawn gets a tile of (T+1) x (T+1) x NZ grid nodes at the particle spacing (T cells a side). Tiles of
neighboring awake chunks duplicate their shared face nodes, so their surfaces meet exactly. Marching cubes runs over
all tiles at once; vertices are shared per grid edge, so the mesh is watertight inside a tile. Awake chunks are
redone every frame; a sleeping chunk is baked once into a mesh cache that it keeps after unloading (until it wakes).
A tile's side faces next to chunks without particles follow the terrain, and every side face hangs a short skirt
down, so surfaces made at different times (or the terrain) never show a crack between them.
"""

from __future__ import annotations

import heapq

import numpy as np
import warp as wp

# The 0.5 level of the splatted particles lies this far (in grid cells) above the heightfield (the top of the
# particles) of the same flat material (measured). Vertices are lifted by it so flat isosurface and terrain line up.
ISO_OFFSET = -0.77
ISO_SKIRT_CELLS = 3.0  # skirt depth in grid cells
ISO_BAKES_PER_FRAME = 4  # about 1.4 ms each
# Re-logging the cache mesh (and the terrain, whose patches it hides) costs a few ms: bakes that catch up on older
# sleeping chunks are shown in batches
ISO_CACHE_RELOG_FRAMES = 25
ISO_FACE_BITS = 15  # face_bits[v] bits 0-3: on the tile's -x, +x, -y, +y face; bit 4: skirt vertex
ISO_SKIRT = wp.constant(16)

_MC_CORNERS = wp.MarchingCubes.CUBE_CORNER_OFFSETS
# per cube edge: (dx, dy, dz, axis) of the grid edge it lies on, from Warp's corner and edge tables
_MC_EDGE_OFFSETS = tuple(
    (
        *(min(_MC_CORNERS[a][d], _MC_CORNERS[b][d]) for d in range(3)),
        next(d for d in range(3) if _MC_CORNERS[a][d] != _MC_CORNERS[b][d]),
    )
    for a, b in wp.MarchingCubes.EDGE_TO_CORNERS
)


# -- terrain shading, shared with the terrain mesh -------------------------------------------------------------------


@wp.func
def height_at(heights: wp.array2d[float], i: int, j: int):
    return heights[wp.clamp(i, 0, heights.shape[0] - 1), wp.clamp(j, 0, heights.shape[1] - 1)]


@wp.func
def smooth_height(heights: wp.array2d[float], i: int, j: int, floor_z: float):
    """3x3 box average over sand columns. The lattice jitter of untouched sand would otherwise speckle the
    shading, and awake columns (below the ground) would draw a dark rim around every loaded region."""
    h = float(0.0)  # noqa: UP018 -- Warp needs a dynamic (mutable) loop variable
    n = float(0.0)  # noqa: UP018
    for di in range(-1, 2):
        for dj in range(-1, 2):
            v = height_at(heights, i + di, j + dj)
            if v > floor_z + 1.0e-4:
                h += v
                n += 1.0
    if n == 0.0:
        return floor_z
    return h / n


@wp.func
def terrain_normal(heights: wp.array2d[float], I: int, J: int, floor_z: float, col: float):
    """Normal at column corner (I, J) from smoothed central differences on the heightfield. A side whose smoothing
    window holds no terrain (all awake columns, cleared below the ground) takes the other side's height, so the
    edge of the terrain next to an awake chunk is not shaded as a steep drop."""
    xp = smooth_height(heights, I + 1, J, floor_z)
    xm = smooth_height(heights, I - 2, J, floor_z)
    yp = smooth_height(heights, I, J + 1, floor_z)
    ym = smooth_height(heights, I, J - 2, floor_z)
    empty = floor_z + 1.0e-4
    if xp <= empty:
        xp = xm
    if xm <= empty:
        xm = xp
    if yp <= empty:
        yp = ym
    if ym <= empty:
        ym = yp
    return wp.normalize(wp.vec3(-(xp - xm) / (3.0 * col), -(yp - ym) / (3.0 * col), 1.0))


# -- device side ---------------------------------------------------------------------------------------------------


@wp.func
def iso_value(
    field: wp.array4d[float],
    chunk_tile: wp.array[wp.int32],
    tile_chunk: wp.array[wp.int32],
    ncx: int,
    ncy: int,
    t: int,
    i: int,
    j: int,
    k: int,
):
    """Field value at tile-local node (i, j, k), reaching into the neighboring awake tile when off the tile."""
    n = field.shape[1] - 1
    kk = wp.clamp(k, 0, field.shape[3] - 1)
    ii = int(i)
    jj = int(j)
    tt = int(t)
    if ii < 0 or ii > n or jj < 0 or jj > n:
        cid = tile_chunk[t]
        cx = cid / ncy
        cy = cid % ncy
        if ii < 0:
            cx -= 1
            ii += n
        elif ii > n:
            cx += 1
            ii -= n
        if jj < 0:
            cy -= 1
            jj += n
        elif jj > n:
            cy += 1
            jj -= n
        nt = int(-1)  # noqa: RUF046, UP018 -- Warp needs a dynamic (mutable) variable
        if cx >= 0 and cy >= 0 and cx < ncx and cy < ncy:
            nt = chunk_tile[cx * ncy + cy]
        if nt < 0:  # no awake neighbor: one-sided
            ii = wp.clamp(i, 0, n)
            jj = wp.clamp(j, 0, n)
        else:
            tt = nt
    return field[tt, ii, jj, kk]


@wp.func
def iso_gradient(
    field: wp.array4d[float],
    chunk_tile: wp.array[wp.int32],
    tile_chunk: wp.array[wp.int32],
    ncx: int,
    ncy: int,
    t: int,
    i: int,
    j: int,
    k: int,
):
    return wp.vec3(
        iso_value(field, chunk_tile, tile_chunk, ncx, ncy, t, i + 1, j, k)
        - iso_value(field, chunk_tile, tile_chunk, ncx, ncy, t, i - 1, j, k),
        iso_value(field, chunk_tile, tile_chunk, ncx, ncy, t, i, j + 1, k)
        - iso_value(field, chunk_tile, tile_chunk, ncx, ncy, t, i, j - 1, k),
        iso_value(field, chunk_tile, tile_chunk, ncx, ncy, t, i, j, k + 1)
        - iso_value(field, chunk_tile, tile_chunk, ncx, ncy, t, i, j, k - 1),
    )


@wp.kernel
def iso_clear(field: wp.array4d[float]):
    """Empty space, except the bottom node layer (below the ground), so the surface never closes underneath."""
    t, i, j, k = wp.tid()
    field[t, i, j, k] = wp.where(k == 0, 1.0, 0.0)


@wp.func
def quadratic_weight(distance: float):
    d = wp.abs(distance)
    if d < 0.5:
        return 0.75 - d * d
    if d < 1.5:
        return 0.5 * (1.5 - d) * (1.5 - d)
    return 0.0


@wp.kernel
def iso_splat(
    q: wp.array[wp.vec3],
    occupied: wp.array[wp.int32],
    chunk_tile: wp.array[wp.int32],
    ncx: int,
    ncy: int,
    org: wp.vec3,
    h: float,
    w: float,
    particle_spacing: float,
    field: wp.array4d[float],
):
    """Quadratic B-spline splat of each loaded particle (weight: particle volume / cell volume) into every tile
    holding a copy of the node. Frozen particles count too: the face next to a sleeping chunk then comes from the
    same particles as that chunk's baked mesh."""
    p = wp.tid()
    if occupied[p] == 0:
        return
    n = field.shape[1] - 1
    g = (q[p] - org) / h
    # Keep the reference B-spline width in physical units when refining the display grid.
    ratio = wp.max(particle_spacing / h, 1.0)
    width = 1.5 * ratio
    b = wp.vec3i(int(wp.floor(g[0] - width)) + 1, int(wp.floor(g[1] - width)) + 1, int(wp.floor(g[2] - width)) + 1)
    span = int(wp.ceil(3.0 * ratio))
    weight = w / (ratio * ratio * ratio)
    for a in range(span):
        gi = b[0] + a
        if gi < 0:
            continue
        cx = gi / n
        li = gi - cx * n
        for c in range(span):
            gj = b[1] + c
            if gj < 0:
                continue
            cy = gj / n
            lj = gj - cy * n
            for d in range(span):
                gk = b[2] + d
                if gk < 0 or gk >= field.shape[3]:
                    continue
                wt = (
                    weight
                    * quadratic_weight((float(gi) - g[0]) / ratio)
                    * quadratic_weight((float(gj) - g[1]) / ratio)
                    * quadratic_weight((float(gk) - g[2]) / ratio)
                )
                # the node's copies: (cx, li) and, on the low face, (cx - 1, n); likewise in y
                for e in range(2):
                    tx = cx - e
                    ti = li + e * n
                    if e == 1 and li != 0:
                        continue
                    if tx < 0 or tx >= ncx:
                        continue
                    for s in range(2):
                        ty = cy - s
                        tj = lj + s * n
                        if s == 1 and lj != 0:
                            continue
                        if ty < 0 or ty >= ncy:
                            continue
                        t = chunk_tile[tx * ncy + ty]
                        if t >= 0:
                            wp.atomic_add(field, t, ti, tj, gk, wt)


@wp.func
def terrain_corner_height(
    heights: wp.array2d[float], backed: wp.array[wp.int32], cols_per_chunk: int, ncy: int, I: int, J: int
):
    """Height of the full-resolution terrain vertex at column corner (I, J): the mean of the adjacent columns in
    chunks without particles (``backed`` < 0). Returns (height sum, count)."""
    hs = float(0.0)  # noqa: UP018 -- Warp needs a dynamic (mutable) loop variable
    hn = float(0.0)  # noqa: UP018
    for a in range(2):
        for b in range(2):
            ci = wp.clamp(I - 1 + a, 0, heights.shape[0] - 1)
            cj = wp.clamp(J - 1 + b, 0, heights.shape[1] - 1)
            if backed[(ci / cols_per_chunk) * ncy + cj / cols_per_chunk] < 0:
                hs += heights[ci, cj]
                hn += 1.0
    return wp.vec2(hs, hn)


@wp.kernel
def iso_fill_pristine(
    heights: wp.array2d[float],
    pristine: wp.array[wp.int32],
    tile_chunk: wp.array[wp.int32],
    ncy: int,
    z0: float,
    h: float,
    offset: float,
    field: wp.array4d[float],
):
    """Retain untouched sand below particles deposited outside previously loaded tiles."""
    t, i, j, k = wp.tid()
    cid = tile_chunk[t]
    n = field.shape[1] - 1
    I, J = (cid / ncy) * n + i, (cid % ncy) * n + j
    # Both copies of a shared face must include the same untouched sand.
    fill = pristine[cid] != 0
    for a in range(2):
        for b in range(2):
            ci = wp.clamp(I - 1 + a, 0, heights.shape[0] - 1)
            cj = wp.clamp(J - 1 + b, 0, heights.shape[1] - 1)
            if pristine[(ci / n) * ncy + cj / n] != 0:
                fill = True
    if not fill:
        return
    height = 0.25 * (
        height_at(heights, I - 1, J - 1)
        + height_at(heights, I, J - 1)
        + height_at(heights, I - 1, J)
        + height_at(heights, I, J)
    )
    value = wp.clamp((height + offset - z0 - float(k) * h) / h + 0.5, 0.0, 1.0)
    field[t, i, j, k] = wp.max(field[t, i, j, k], value)


@wp.kernel
def iso_close(
    heights: wp.array2d[float],
    backed: wp.array[wp.int32],
    tile_chunk: wp.array[wp.int32],
    cols_per_chunk: int,
    ncy: int,
    z0: float,
    h: float,
    offset: float,
    field: wp.array4d[float],
):
    """On a tile face next to a chunk without particles (unloaded), the field is the terrain there (its corner
    heights, interpolated along the edge like the terrain mesh), so the isosurface ends on the terrain's edge: a berm
    or trench ramps to it within one cell instead of leaving a step."""
    t, i, j, k = wp.tid()
    n = field.shape[1] - 1
    if i != 0 and i != n and j != 0 and j != n:
        return
    cid = tile_chunk[t]
    r = n / cols_per_chunk  # grid cells per column
    gx = (cid / ncy) * n + i
    gy = (cid % ncy) * n + j
    I = gx / r
    J = gy / r
    fx = float(gx - I * r) / float(r)
    fy = float(gy - J * r) / float(r)
    acc = wp.vec2(0.0, 0.0)
    for a in range(2):
        for b in range(2):
            w = wp.where(a == 0, 1.0 - fx, fx) * wp.where(b == 0, 1.0 - fy, fy)
            if w > 0.0:
                c = terrain_corner_height(heights, backed, cols_per_chunk, ncy, I + a, J + b)
                if c[1] > 0.0:
                    acc += w * wp.vec2(c[0] / c[1], 1.0)
    if acc[1] == 0.0:
        return
    z = z0 + float(k) * h
    field[t, i, j, k] = wp.clamp((acc[0] / acc[1] + offset - z) / h + 0.5, 0.0, 1.0)


@wp.kernel
def iso_vertices(
    field: wp.array4d[float],
    chunk_tile: wp.array[wp.int32],
    tile_chunk: wp.array[wp.int32],
    backed: wp.array[wp.int32],
    ncx: int,
    ncy: int,
    org: wp.vec3,
    h: float,
    offset: float,
    skirt: float,
    v_lo: float,
    v_hi: float,
    heights: wp.array2d[float],
    col: float,
    floor_z: float,
    counts: wp.array[wp.int32],
    edge_vert: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    uvs: wp.array[wp.vec2],
    face_bits: wp.array[wp.int32],
):
    """One vertex per grid edge crossing the 0.5 level (the three edges leaving each node in +x, +y, +z). A vertex on
    a side face gets a second one ``skirt`` below it (the next index), for the skirt."""
    t, i, j, k = wp.tid()
    n = field.shape[1] - 1
    nz = field.shape[3]
    cid = tile_chunk[t]
    cx = cid / ncy
    cy = cid % ncy
    a = field[t, i, j, k]
    base = (((t * (n + 1) + i) * (n + 1) + j) * nz + k) * 3
    for axis in range(3):
        edge_vert[base + axis] = -1
        i1 = i + wp.where(axis == 0, 1, 0)
        j1 = j + wp.where(axis == 1, 1, 0)
        k1 = k + wp.where(axis == 2, 1, 0)
        if i1 > n or j1 > n or k1 >= nz:
            continue
        bv = field[t, i1, j1, k1]
        if (a >= 0.5 and bv >= 0.5) or (a < 0.5 and bv < 0.5):
            continue
        on_x = (i == 0 or i == n) and i1 == i
        on_y = (j == 0 or j == n) and j1 == j
        nv = wp.where(on_x or on_y, 2, 1)
        v = wp.atomic_add(counts, 0, nv)
        if v + nv > points.shape[0]:
            continue
        s = wp.clamp((0.5 - a) / (bv - a), 0.0, 1.0)
        p0 = org + wp.vec3(float(cx * n + i), float(cy * n + j), float(k)) * h
        p = p0 + s * h * wp.vec3(float(i1 - i), float(j1 - j), float(k1 - k)) - wp.vec3(0.0, 0.0, offset)
        g = wp.lerp(
            iso_gradient(field, chunk_tile, tile_chunk, ncx, ncy, t, i, j, k),
            iso_gradient(field, chunk_tile, tile_chunk, ncx, ncy, t, i1, j1, k1),
            s,
        )
        points[v] = p
        normals[v] = wp.normalize(-g + wp.vec3(0.0, 0.0, 1.0e-6))
        uvs[v] = wp.vec2(0.5, wp.clamp((p[2] - v_lo) / (v_hi - v_lo), 0.002, 0.998))
        # which side faces of its tile a vertex lies on: the faces' triangle edges get skirts
        bits = int(0)  # noqa: RUF046, UP018 -- Warp needs a dynamic (mutable) variable
        if on_x:
            bits = bits | wp.where(i == 0, 1, 2)
        if on_y:
            bits = bits | wp.where(j == 0, 4, 8)
        face_bits[v] = bits
        # next to a chunk without particles, the field steps to the terrain there: take the terrain's normal (as in
        # terrain_points), or the edge shades as a dark line
        if on_x or on_y:
            ncx_ = cx + wp.where(on_x, wp.where(i == 0, -1, 1), 0)
            ncy_ = cy + wp.where(on_y and not on_x, wp.where(j == 0, -1, 1), 0)
            outside = ncx_ < 0 or ncy_ < 0 or ncx_ >= ncx or ncy_ >= ncy
            if not outside:
                outside = backed[ncx_ * ncy + ncy_] < 0
            if outside:
                I = int(wp.round((p[0] - org[0]) / col))
                J = int(wp.round((p[1] - org[1]) / col))
                normals[v] = terrain_normal(heights, I, J, floor_z, col)
            points[v + 1] = p - wp.vec3(0.0, 0.0, skirt)
            normals[v + 1] = normals[v]
            uvs[v + 1] = uvs[v]
            face_bits[v + 1] = bits | ISO_SKIRT
        edge_vert[base + axis] = v


@wp.kernel
def iso_faces(
    field: wp.array4d[float],
    edge_vert: wp.array[wp.int32],
    case_to_tri_range: wp.array[wp.int32],
    tri_local_indices: wp.array[wp.int32],
    edge_offsets: wp.array2d[wp.int32],
    points: wp.array[wp.vec3],
    face_bits: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    indices: wp.array[wp.int32],
):
    """Marching cubes triangles, plus a skirt quad under every triangle edge that lies on a tile side face."""
    t, i, j, k = wp.tid()
    n = field.shape[1] - 1
    nz = field.shape[3]
    case = int(0)  # noqa: RUF046, UP018 -- Warp needs a dynamic (mutable) variable
    for c in range(8):
        if (
            field[
                t, i + wp.static(_MC_CORNERS[c][0]), j + wp.static(_MC_CORNERS[c][1]), k + wp.static(_MC_CORNERS[c][2])
            ]
            >= 0.5
        ):
            case += 1 << c
    r0 = case_to_tri_range[case]
    ntri = (case_to_tri_range[case + 1] - r0) / 3
    if ntri == 0:
        return
    for m in range(ntri):
        vs = wp.vec3i()
        for s in range(3):
            e = tri_local_indices[r0 + 3 * m + s]
            ni = i + edge_offsets[e, 0]
            nj = j + edge_offsets[e, 1]
            nk = k + edge_offsets[e, 2]
            vs[s] = edge_vert[(((t * (n + 1) + ni) * (n + 1) + nj) * nz + nk) * 3 + edge_offsets[e, 3]]
        if vs[0] < 0 or vs[1] < 0 or vs[2] < 0:
            continue
        # Warp's tables wind the triangles to face the low side of the field: flip them to face out of the material
        tri = wp.vec3i(vs[0], vs[2], vs[1])
        nskirt = int(0)  # noqa: RUF046, UP018
        for e in range(3):
            if (face_bits[tri[e]] & face_bits[tri[(e + 1) % 3]] & ISO_FACE_BITS) != 0:
                nskirt += 1
        out = wp.atomic_add(counts, 1, 1 + 2 * nskirt)
        if 3 * (out + 1 + 2 * nskirt) > indices.shape[0]:
            continue
        indices[3 * out + 0] = tri[0]
        indices[3 * out + 1] = tri[1]
        indices[3 * out + 2] = tri[2]
        q = out + 1
        for e in range(3):
            a = tri[e]
            b = tri[(e + 1) % 3]
            face = face_bits[a] & face_bits[b] & ISO_FACE_BITS
            if face == 0:
                continue
            # quad a, b, b', a' (primes: the skirt vertices below), wound to face out of the tile
            out_dir = wp.vec3(0.0, 0.0, 0.0)
            if (face & 1) != 0:
                out_dir = wp.vec3(-1.0, 0.0, 0.0)
            elif (face & 2) != 0:
                out_dir = wp.vec3(1.0, 0.0, 0.0)
            elif (face & 4) != 0:
                out_dir = wp.vec3(0.0, -1.0, 0.0)
            else:
                out_dir = wp.vec3(0.0, 1.0, 0.0)
            nrm = wp.cross(points[b] - points[a], points[b + 1] - points[a])
            if wp.dot(nrm, out_dir) >= 0.0:
                indices[3 * q + 0] = a
                indices[3 * q + 1] = b
                indices[3 * q + 2] = b + 1
                indices[3 * q + 3] = a
                indices[3 * q + 4] = b + 1
                indices[3 * q + 5] = a + 1
            else:
                indices[3 * q + 0] = a
                indices[3 * q + 1] = b + 1
                indices[3 * q + 2] = b
                indices[3 * q + 3] = a
                indices[3 * q + 4] = a + 1
                indices[3 * q + 5] = b + 1
            q += 2


@wp.kernel
def iso_cache_store_vertices(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    uvs: wp.array[wp.vec2],
    vblocks: wp.array[wp.int32],
    vb: int,
    pool_points: wp.array[wp.vec3],
    pool_normals: wp.array[wp.vec3],
    pool_uvs: wp.array[wp.vec2],
):
    """Copy a baked mesh's vertices into its blocks of the cache (blocks need not be contiguous)."""
    v = wp.tid()
    d = vblocks[v / vb] * vb + v % vb
    pool_points[d] = points[v]
    pool_normals[d] = normals[v]
    pool_uvs[d] = uvs[v]


@wp.kernel
def iso_cache_store_triangles(
    indices: wp.array[wp.int32],
    tblocks: wp.array[wp.int32],
    tb: int,
    vblocks: wp.array[wp.int32],
    vb: int,
    pool_indices: wp.array[wp.int32],
):
    t = wp.tid()
    d = tblocks[t / tb] * tb + t % tb
    for s in range(3):
        v = indices[3 * t + s]
        pool_indices[3 * d + s] = vblocks[v / vb] * vb + v % vb


# -- host side ----------------------------------------------------------------------------------------------------


class IsoMesh:
    """Output of an extraction: vertices (with normals, uvs and face bits) and triangle indices."""

    def __init__(self, verts):
        self.vcap = verts
        self.icap = 3 * 3 * verts  # ~2 triangles per vertex, plus skirts
        self.counts = wp.zeros(2, dtype=wp.int32)  # vertices, triangles
        self.points = wp.zeros(verts, dtype=wp.vec3)
        self.normals = wp.zeros(verts, dtype=wp.vec3)
        self.uvs = wp.zeros(verts, dtype=wp.vec2)
        self.face_bits = wp.zeros(verts, dtype=wp.int32)
        self.indices = wp.zeros(self.icap, dtype=wp.int32)


class IsoTiles:
    """Grid nodes and edge vertices for up to ``count`` tiles, the chunk <-> tile maps, and the output mesh."""

    def __init__(self, count, n, nz, nchunks, verts):
        self.count = count
        self.field = wp.zeros((count, n + 1, n + 1, nz), dtype=float)
        self.edge_vert = wp.zeros(count * (n + 1) ** 2 * nz * 3, dtype=wp.int32)
        self.chunk_tile = wp.full(nchunks, value=-1, dtype=wp.int32)
        self.tile_chunk = wp.zeros(count, dtype=wp.int32)
        self.ids = np.zeros(0, dtype=np.int32)
        self.mesh = IsoMesh(verts)
        self._chunk_tile = np.full(nchunks, -1, dtype=np.int32)

    def assign(self, ids):
        """Give chunk ``ids[t]`` tile ``t``."""
        if np.array_equal(ids, self.ids):
            return
        self._chunk_tile[self.ids] = -1
        self._chunk_tile[ids] = np.arange(len(ids), dtype=np.int32)
        self.chunk_tile.assign(self._chunk_tile)
        self.tile_chunk[: len(ids)].assign(ids)
        self.ids = ids


class IsoCache:
    """Baked meshes of single chunks in a fixed pool of vertex and triangle blocks. A chunk's blocks need not be
    contiguous; they are handed out lowest first, so the drawn range stays short."""

    VB, TB = 1024, 2048  # vertices per vertex block, triangles per triangle block

    def __init__(self, nchunks, mb):
        self.vb, self.tb = self.VB, self.TB
        self.blocks = max(16, int(mb * 2**20 // (self.vb * 32 + self.tb * 12)))
        self.pool = None  # allocated at the first bake
        self.free_v = list(range(self.blocks))
        self.free_t = list(range(self.blocks))
        self.entries = {}  # cid -> (vertex blocks, triangle blocks)
        self.counts = {}  # cid -> (vertices, triangles, particles) of its baked mesh
        self.baked = np.zeros(nchunks, dtype=bool)
        self.baked_wp = wp.zeros(nchunks, dtype=wp.int32)  # what the terrain hides, updated on re-log
        self.evicted = 0
        # drawing state
        self.dirty = True  # changed since the last re-log
        self.urgent = False  # a dropped mesh must go now (live isosurface takes over)
        self.age = 0  # frames since the last re-log
        self.shown = False
        self.prev_state = None  # chunk states at the last update: older sleepers are caught up as a backlog

    def store(self, cid, mesh, nv, ntri, particles, distance):
        """Copy ``mesh`` into free blocks, evicting the chunks farthest away by ``distance(cids)`` to make room."""
        vb, tb = self.vb, self.tb
        need_v, need_t = max(1, -(-nv // vb)), max(1, -(-ntri // tb))
        if need_v > self.blocks or need_t > self.blocks:
            return
        while len(self.free_v) < need_v or len(self.free_t) < need_t:
            cached = np.fromiter(self.entries, dtype=np.int64)
            if len(cached) == 0:
                return
            self.drop(int(cached[np.argmax(distance(cached))]), urgent=False)
            self.evicted += 1
        if self.pool is None:
            self.pool = IsoMesh(0)
            self.pool.points = wp.zeros(self.blocks * vb, dtype=wp.vec3)
            self.pool.normals = wp.zeros(self.blocks * vb, dtype=wp.vec3)
            self.pool.uvs = wp.zeros(self.blocks * vb, dtype=wp.vec2)
            self.pool.indices = wp.zeros(self.blocks * tb * 3, dtype=wp.int32)
        vblocks = [heapq.heappop(self.free_v) for _ in range(need_v)]
        tblocks = [heapq.heappop(self.free_t) for _ in range(need_t)]
        vb_wp, tb_wp = wp.array(vblocks, dtype=wp.int32), wp.array(tblocks, dtype=wp.int32)
        pool = self.pool
        if nv:
            wp.launch(
                iso_cache_store_vertices,
                dim=nv,
                inputs=[mesh.points, mesh.normals, mesh.uvs, vb_wp, vb],
                outputs=[pool.points, pool.normals, pool.uvs],
            )
        for b in tblocks:  # unused triangles stay degenerate
            pool.indices[b * tb * 3 : (b + 1) * tb * 3].zero_()
        if ntri:
            wp.launch(
                iso_cache_store_triangles,
                dim=ntri,
                inputs=[mesh.indices, tb_wp, tb, vb_wp, vb],
                outputs=[pool.indices],
            )
        self.entries[cid] = (vblocks, tblocks)
        self.counts[cid] = (nv, ntri, particles)
        self.baked[cid] = True
        self.dirty = True

    def drop(self, cid, urgent=True):
        vblocks, tblocks = self.entries.pop(cid)
        self.counts.pop(cid, None)
        tb = self.tb
        for b in tblocks:
            self.pool.indices[b * tb * 3 : (b + 1) * tb * 3].zero_()
            heapq.heappush(self.free_t, b)
        for b in vblocks:
            heapq.heappush(self.free_v, b)
        self.baked[cid] = False
        self.dirty = True
        self.urgent |= urgent

    def used(self):
        """Vertex and triangle count of the pool range in use (zero when empty)."""
        used_v = [b for blocks, _ in self.entries.values() for b in blocks]
        used_t = [b for _, blocks in self.entries.values() for b in blocks]
        if not used_t:
            return 0, 0
        return (max(used_v) + 1) * self.vb, (max(used_t) + 1) * self.tb


class IsoSurface:
    """Isosurface grid: one tile per awake chunk at the particle spacing, from below the ground to ``v_hi`` (the top
    of the height texture; higher spray is not drawn); plus a one-tile grid for baking chunks into the cache."""

    def __init__(self, world, heights, cols_per_chunk, col, particles_per_cell, floor_z, v_lo, v_hi, tiles, cache_mb):
        self.world = world
        self.heights = heights
        self.cols_per_chunk = cols_per_chunk
        self.col = col
        self.floor_z = floor_z
        self.v_lo, self.v_hi = v_lo, v_hi
        self.n = cols_per_chunk * particles_per_cell
        self.h = col / particles_per_cell
        self.z0 = -self.h
        self.nz = int(np.ceil((v_hi - self.z0) / self.h)) + 1
        nchunks = world.chunk_count
        self.live = IsoTiles(tiles, self.n, self.nz, nchunks, 1 << 16)
        self.bake_tiles = IsoTiles(1, self.n, self.nz, nchunks, 1 << 14)
        self.backed = wp.full(nchunks, value=-1, dtype=wp.int32)  # >= 0: the chunk has particles (loaded)
        self.pristine = wp.zeros(nchunks, dtype=wp.int32)
        mc = wp.MarchingCubes
        self.case_to_tri_range = wp.array(mc.CASE_TO_TRI_RANGE, dtype=wp.int32)
        self.tri_local_indices = wp.array(mc.TRI_LOCAL_INDICES, dtype=wp.int32)
        self.edge_offsets = wp.array(_MC_EDGE_OFFSETS, dtype=wp.int32)
        self.cache = IsoCache(nchunks, cache_mb)
        # drawing state of the live mesh
        self.shown = False
        self.warned = False

    def extract(self, tiles, ids, q, occupied, particle_weight=1.0, particle_spacing=0.0):
        """Marching cubes over the tiles of chunks ``ids`` from particles ``q``; returns the output mesh (in
        ``tiles.mesh``) and its vertex and triangle counts."""
        tiles.assign(ids)
        n, h, nz = self.n, self.h, self.nz
        org = wp.vec3(self.world.lo[0], self.world.lo[1], self.z0)
        ncx, ncy = self.world.ncx, self.world.ncy
        nodes = (len(ids), n + 1, n + 1, nz)
        field, chunk_tile, tile_chunk = tiles.field, tiles.chunk_tile, tiles.tile_chunk
        offset = ISO_OFFSET * h
        wp.launch(iso_clear, dim=nodes, inputs=[field])
        wp.launch(
            iso_splat,
            dim=q.shape[0],
            inputs=[q, occupied, chunk_tile, ncx, ncy, org, h, particle_weight, particle_spacing, field],
        )
        wp.launch(
            iso_fill_pristine,
            dim=nodes,
            inputs=[self.heights, self.pristine, tile_chunk, ncy, self.z0, h, offset, field],
        )
        wp.launch(
            iso_close,
            dim=nodes,
            inputs=[self.heights, self.backed, tile_chunk, self.cols_per_chunk, ncy, self.z0, h, offset, field],
        )
        for _attempt in range(2):
            out = tiles.mesh
            out.counts.zero_()
            wp.launch(
                iso_vertices,
                dim=nodes,
                inputs=[field, chunk_tile, tile_chunk, self.backed, ncx, ncy, org, h, offset, ISO_SKIRT_CELLS * h,
                        self.v_lo, self.v_hi, self.heights, self.col, self.floor_z, out.counts, tiles.edge_vert,
                        out.points, out.normals, out.uvs, out.face_bits],
            )  # fmt: skip
            wp.launch(
                iso_faces,
                dim=(len(ids), n, n, nz - 1),
                inputs=[field, tiles.edge_vert, self.case_to_tri_range, self.tri_local_indices, self.edge_offsets,
                        out.points, out.face_bits, out.counts, out.indices],
            )  # fmt: skip
            nv, ntri = (int(c) for c in out.counts.numpy())
            if nv <= out.vcap and 3 * ntri <= out.icap:
                break
            tiles.mesh = IsoMesh(int(1.5 * max(nv, ntri / 3)))  # grow and redo
        if nv > out.vcap or 3 * ntri > out.icap:
            raise RuntimeError("Isosurface mesh capacity exhausted after resizing")
        return out, nv, ntri

    def bake(self, cid, q, occupied, distance):
        """Extract chunk ``cid`` alone and store its mesh in the cache."""
        mesh, nv, ntri = self.extract(self.bake_tiles, np.array([cid], dtype=np.int32), q, occupied)
        self.cache.store(cid, mesh, nv, ntri, self.world.count(cid), distance)
