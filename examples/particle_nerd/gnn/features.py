# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA neighborhood and rigid-geometry features; no host particle processing."""

import numpy as np
import torch
import warp as wp


@wp.kernel
def count_edges(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    radius: float,
    counts: wp.array(dtype=int),
):
    i = wp.tid()
    query = wp.hash_grid_query(grid, x[i], radius)
    j = int(0)
    n = int(0)
    while wp.hash_grid_query_next(query, j):
        if i != j and wp.length_sq(x[j] - x[i]) < radius * radius:
            n += 1
    counts[i] = n


@wp.kernel
def fill_edges(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    radius: float,
    counts: wp.array(dtype=int),
    offsets: wp.array(dtype=int),
    src: wp.array(dtype=wp.int64),
    dst: wp.array(dtype=wp.int64),
    valid: wp.array(dtype=float),
    overflow: wp.array(dtype=int),
):
    i = wp.tid()
    slot = offsets[i] - counts[i]
    if offsets[i] > src.shape[0]:
        wp.atomic_max(overflow, 0, 1)
    query = wp.hash_grid_query(grid, x[i], radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if i != j and wp.length_sq(x[j] - x[i]) < radius * radius:
            if slot < src.shape[0]:
                src[slot] = wp.int64(j)
                dst[slot] = wp.int64(i)
                valid[slot] = 1.0
            slot += 1


@wp.kernel
def edge_features(
    x: wp.array(dtype=wp.vec3),
    hist: wp.array2d(dtype=float),
    radius: float,
    src: wp.array(dtype=wp.int64),
    dst: wp.array(dtype=wp.int64),
    out: wp.array2d(dtype=float),
):
    e = wp.tid()
    i, j = src[e], dst[e]
    dx = (x[i] - x[j]) / radius
    for k in range(3):
        out[e, k] = dx[k]
        out[e, k + 4] = hist[i, hist.shape[1] - 3 + k] - hist[j, hist.shape[1] - 3 + k]
    out[e, 3] = wp.length(dx)


@wp.kernel
def node_features(
    hist: wp.array2d(dtype=float),
    material: wp.array(dtype=float),
    counts: wp.array(dtype=int),
    out: wp.array2d(dtype=float),
):
    i = wp.tid()
    h = hist.shape[1]
    for k in range(h):
        out[i, k] = hist[i, k]
    for k in range(material.shape[0]):
        out[i, h + k] = material[k]
    out[i, h + 13] = 0.0
    out[i, h + 14] = 0.0
    out[i, h + 15] = -1.0
    out[i, h + 16] = float(counts[i]) / 64.0


@wp.kernel
def boundary_features(
    x: wp.array(dtype=wp.vec3),
    hist: wp.array2d(dtype=float),
    pose: wp.array(dtype=wp.transform),
    velocity: wp.array(dtype=wp.spatial_vector),
    mass: float,
    radius: float,
    out: wp.array3d(dtype=float),
    influence: wp.array3d(dtype=float),
):
    i, b = wp.tid()
    point = x[i]
    distance = point[2]
    normal = wp.vec3(0.0, 0.0, 1.0)
    surface_v = wp.vec3(0.0)
    inverse_mass = float(0.0)
    if b == 1:
        transform = pose[0]
        local = wp.transform_point(wp.transform_inverse(transform), point)
        half = wp.vec3(0.06, 0.06, 0.04)
        q = wp.vec3(wp.abs(local[0]), wp.abs(local[1]), wp.abs(local[2])) - half
        positive = wp.max(q, wp.vec3(0.0))
        length = wp.length(positive)
        distance = length + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
        local_n = wp.vec3(0.0)
        if length > 1.0e-8:
            for k in range(3):
                local_n[k] = wp.sign(local[k]) * positive[k] / length
        else:
            axis = int(0)
            if q[1] > q[axis]:
                axis = 1
            if q[2] > q[axis]:
                axis = 2
            local_n[axis] = 1.0
            if local[axis] < 0.0:
                local_n[axis] = -1.0
        normal = wp.transform_vector(transform, local_n)
        closest = point - distance * normal
        v = velocity[0]
        surface_v = wp.vec3(v[0], v[1], v[2]) + wp.cross(
            wp.vec3(v[3], v[4], v[5]), closest - wp.transform_get_translation(transform)
        )
        inverse_mass = 1.0 / mass
    out[i, b, 0] = wp.clamp(distance / radius, -2.0, 2.0)
    for k in range(3):
        out[i, b, k + 1] = normal[k]
        out[i, b, k + 4] = surface_v[k] - hist[i, hist.shape[1] - 3 + k]
    out[i, b, 7] = inverse_mass
    out[i, b, 8] = 0.6
    influence[i, b, 0] = wp.max(0.0, 1.0 - wp.max(distance, 0.0) / radius)


def material_features(manifest, particle_mass, particle_radius, dt=0.02):
    m = manifest["material"]
    return np.asarray(
        [
            m["density"] / 1600.0,
            np.log10(m["young_modulus"]) / 15.0,
            m["poisson_ratio"],
            m["friction"],
            np.log1p(m["yield_pressure"]) / np.log1p(1e15),
            m["tensile_yield_ratio"],
            np.log1p(m["yield_stress"]) / 15.0,
            m["hardening"],
            m["dilatancy"],
            np.log1p(m["viscosity"]) / 10.0,
            particle_mass / 0.05,
            particle_radius / 0.02,
            dt / 0.02,
        ],
        dtype=np.float32,
    )


class Features:
    """Fixed CUDA buffers. Compact training graphs may synchronize one edge count.

    Inference keeps fixed-capacity buffers and does not transfer state to CPU.
    Overflow is reported in a CUDA diagnostic flag and invalidates that step.
    """

    def __init__(
        self,
        particles,
        radius=0.07,
        history=3,
        capacity_per_particle=96,
        device="cuda:0",
    ):
        wp.init()
        self.device = str(device)
        self.n, self.radius, self.history = particles, radius, history
        self.capacity = particles * capacity_per_particle
        with wp.ScopedDevice(self.device):
            self.grid = wp.HashGrid(64, 64, 64)
            self.counts = wp.zeros(particles, dtype=int)
            self.offsets = wp.zeros(particles, dtype=int)
            self.src = wp.zeros(self.capacity, dtype=wp.int64)
            self.dst = wp.zeros(self.capacity, dtype=wp.int64)
            self.valid = wp.zeros(self.capacity)
            self.overflow = wp.zeros(1, dtype=int)
            self.edges = wp.zeros((self.capacity, 7))
            self.nodes = wp.zeros((particles, 3 * history + 17))
            self.boundary = wp.zeros((particles, 2, 9))
            self.influence = wp.zeros((particles, 2, 1))

    def prepare(self, x, history, pose, velocity, material, mass, compact=False):
        stream = wp.stream_from_torch(torch.cuda.current_stream(x.device))
        with wp.ScopedStream(stream):
            wx = wp.from_torch(x.detach(), dtype=wp.vec3)
            wh = wp.from_torch(history.detach())
            self.grid.build(wx, self.radius)
            wp.launch(count_edges, self.n, [self.grid.id, wx, self.radius, self.counts])
            wp.utils.array_scan(self.counts, self.offsets, inclusive=True)
            self.src.zero_()
            self.dst.zero_()
            self.valid.zero_()
            self.overflow.zero_()
            wp.launch(
                fill_edges,
                self.n,
                [
                    self.grid.id,
                    wx,
                    self.radius,
                    self.counts,
                    self.offsets,
                    self.src,
                    self.dst,
                    self.valid,
                    self.overflow,
                ],
            )
            count = self.capacity
            if compact:
                count = int(wp.to_torch(self.offsets)[-1].item())
                if count > self.capacity:
                    raise RuntimeError(
                        f"Graph capacity {self.capacity} exceeded by {count}; increase capacity, never truncate"
                    )
            wp.launch(
                edge_features,
                count,
                [wx, wh, self.radius, self.src, self.dst, self.edges],
            )
            wp.launch(
                node_features,
                self.n,
                [wh, wp.from_torch(material), self.counts, self.nodes],
            )
            wp.launch(
                boundary_features,
                (self.n, 2),
                [
                    wx,
                    wh,
                    wp.from_torch(pose.reshape(1, 7), dtype=wp.transform),
                    wp.from_torch(velocity.reshape(1, 6), dtype=wp.spatial_vector),
                    float(mass),
                    self.radius,
                    self.boundary,
                    self.influence,
                ],
            )
        return (
            wp.to_torch(self.nodes),
            wp.to_torch(self.edges)[:count],
            wp.to_torch(self.boundary),
            wp.to_torch(self.influence),
            wp.to_torch(self.src)[:count],
            wp.to_torch(self.dst)[:count],
            wp.to_torch(self.counts).clamp_min(1).float(),
            wp.to_torch(self.valid)[:count],
        )
