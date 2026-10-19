# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline triangle-mesh distance sampling using Warp spatial queries, without MPM."""

import numpy as np
import warp as wp


@wp.kernel
def sample_distance(
    mesh: wp.uint64, origin: wp.vec3, spacing: float, max_distance: float, values: wp.array3d(dtype=float)
):
    i, j, k = wp.tid()
    point = origin + spacing * wp.vec3(float(i), float(j), float(k))
    result = wp.mesh_query_point_sign_parity(mesh, point, max_distance, 5, 0.1)
    if result.result:
        closest = wp.mesh_eval_position(mesh, result.face, result.u, result.v)
        values[i, j, k] = result.sign * wp.length(point - closest)
    else:
        values[i, j, k] = wp.inf


def sample_closed_mesh(vertices, triangles, spacing=0.005, padding=0.1, device="cpu"):
    """Sample a closed triangle mesh in its local frame.

    Each undirected edge must belong to two faces. The caller must additionally
    supply a non-self-intersecting surface; this routine does not repair meshes.
    Five deterministic parity rays classify interior points, matching Newton's
    parity-query convention. No material solver or force model is involved.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    raw_triangles = np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite with shape (N, 3).")
    if raw_triangles.ndim != 2 or raw_triangles.shape[1] != 3 or not np.issubdtype(raw_triangles.dtype, np.integer):
        raise ValueError("Triangle indices must be integers with shape (M, 3).")
    if len(raw_triangles) < 4 or raw_triangles.min() < 0 or raw_triangles.max() >= len(vertices):
        raise ValueError("Triangle indices must address the supplied vertices.")
    triangles = raw_triangles.astype(np.int32)
    edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    area = np.linalg.norm(
        np.cross(
            vertices[triangles[:, 1]] - vertices[triangles[:, 0]], vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
        ),
        axis=1,
    )
    if np.any(counts != 2) or np.any(area <= 0):
        raise ValueError("SDF generation requires a closed mesh with nondegenerate triangles.")
    if not np.isfinite(spacing) or not np.isfinite(padding) or spacing <= 0 or padding <= 0:
        raise ValueError("Spacing and padding must be finite and positive.")
    lower = vertices.min(axis=0).astype(float) - padding
    upper = vertices.max(axis=0).astype(float) + padding
    shape = tuple((np.ceil((upper - lower) / spacing).astype(int) + 1).tolist())
    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(triangles.reshape(-1), dtype=wp.int32, device=device),
    )
    values = wp.empty(shape, dtype=float, device=device)
    wp.launch(
        sample_distance,
        dim=shape,
        inputs=[mesh.id, wp.vec3(*lower), spacing, float(np.linalg.norm(upper - lower) + spacing), values],
        device=device,
    )
    result = values.numpy()
    if not np.isfinite(result).all():
        raise ValueError("Mesh distance query failed inside the requested volume.")
    return result, lower, spacing
