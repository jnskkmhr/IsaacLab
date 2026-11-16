# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified SDF collider queries, full-bed packing & rate-of-change, and Warp ContinuousConv for ParticleNeRD-Local.

Key design principles:
  1. Semi-implicit Euler predictor first (`integrate_pos_vel`), so the network predicts only a local residual.
  2. Unified 3D SDF collider representation (`sample_sdf_and_normal`): boxes, spheres, cylinders, and arbitrary
     concave meshes are all queried through a single trilinear 3D SDF volume (or exact analytical fallback)
     with smooth C1 gradients everywhere.
  3. Full-bed local packing fraction `phi` and its recent rate of change `dphi/dt` (`-div(v)`) reconstructed
     on the complete particle bed before any active-particle compaction.
  4. Translation invariance: global coordinates `(x, y, z)` are never passed into token MLPs.
"""

from functools import lru_cache

import numpy as np
import torch
import warp as wp
from torch import nn

wp.set_module_options({"enable_backward": True})


def material_features(manifest: dict, particle_mass: float, particle_radius: float, dt: float = 0.005) -> np.ndarray:
    """Normalize the 13-element continuum material and timestep feature vector."""
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


_GRAVITY_TENSORS: dict[tuple, torch.Tensor] = {}


def integrate_pos_vel(
    pos: torch.Tensor,
    vel: torch.Tensor,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    dt: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Semi-implicit Euler / Verlet predictor step before learned residual correction."""
    key = (pos.device, pos.dtype, tuple(float(v) for v in gravity))
    g = _GRAVITY_TENSORS.get(key)
    if g is None:
        g = torch.tensor(gravity, device=pos.device, dtype=pos.dtype)
        _GRAVITY_TENSORS[key] = g
    vel_pred = vel + dt * g
    pos_pred = pos + 0.5 * dt * (vel + vel_pred)
    return pos_pred, vel_pred


@lru_cache(maxsize=32)
def build_collider_sdf_grid(
    kind: int,
    size_tuple: tuple[float, float, float],
    resolution: int = 48,
    padding: float = 0.08,
    device: str = "cuda:0",
) -> tuple[torch.Tensor, tuple[float, float, float], float]:
    """Bake any primitive shape (1=box, 2=sphere, 4=cylinder) into a unified local 3D SDF grid.

    Every collider in the scene can be queried identically via trilinear SDF interpolation
    (`sample_sdf_surface`), eliminating shape-specific branches in the neural pipeline.
    """
    sx, sy, sz = size_tuple
    max_half = max(sx, sy, sz) + padding
    spacing = float(2.0 * max_half / (resolution - 1))
    origin = (-max_half, -max_half, -max_half)
    axis = torch.linspace(-max_half, max_half, resolution, device=device, dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    pts = torch.stack((gx, gy, gz), dim=-1)

    if kind == 1:  # Box with half-extents (sx, sy, sz)
        q = pts.abs() - torch.tensor([sx, sy, sz], device=device, dtype=torch.float32)
        outside = q.clamp_min(0.0)
        dist = torch.sqrt((outside * outside).sum(dim=-1) + 1e-8) - 1e-4 + q.amax(dim=-1).clamp_max(0.0)
    elif kind == 2:  # Sphere with radius sx
        dist = torch.sqrt((pts * pts).sum(dim=-1) + 1e-8) - sx
    elif kind == 4:  # Cylinder along local z with radius sx and half-height sz
        radial = torch.sqrt(pts[..., 0] ** 2 + pts[..., 1] ** 2 + 1e-8)
        q = torch.stack((radial - sx, pts[..., 2].abs() - sz), dim=-1)
        outside = q.clamp_min(0.0)
        dist = torch.sqrt((outside * outside).sum(dim=-1) + 1e-8) - 1e-4 + q.amax(dim=-1).clamp_max(0.0)
    else:  # Plane z = 0
        dist = pts[..., 2]
    return dist.contiguous(), origin, spacing


@wp.func
def sample_sdf_distance(p: wp.vec3, values: wp.array3d(dtype=float), origin: wp.vec3, spacing: float) -> float:
    """Trilinearly sample signed distance from a local 3D SDF grid with smooth exterior extrapolation."""
    grid = (p - origin) / spacing
    bounded = wp.vec3(0.0)
    base = wp.vec3i(0)
    fraction = wp.vec3(0.0)
    for a in range(3):
        bounded[a] = wp.clamp(grid[a], 0.0, float(values.shape[a] - 1))
        base[a] = wp.min(int(wp.floor(bounded[a])), values.shape[a] - 2)
        fraction[a] = bounded[a] - float(base[a])
    result = float(0.0)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                w = wp.vec3(float(i), float(j), float(k))
                weight = float(1.0)
                for a in range(3):
                    weight *= (1.0 - w[a]) * (1.0 - fraction[a]) + w[a] * fraction[a]
                result += weight * values[base[0] + i, base[1] + j, base[2] + k]
    diff = grid - bounded
    ext = wp.sqrt(wp.dot(diff, diff) + 1.0e-8) - 1.0e-4
    return result + spacing * wp.max(ext, 0.0)


@wp.func
def sample_sdf_surface(p: wp.vec3, values: wp.array3d(dtype=float), origin: wp.vec3, spacing: float):
    """Return signed distance and smooth central-difference outward normal from a 3D SDF grid."""
    d = sample_sdf_distance(p, values, origin, spacing)
    gradient = wp.vec3(0.0)
    for a in range(3):
        offset = wp.vec3(0.0)
        offset[a] = spacing * 0.5
        gradient[a] = (
            sample_sdf_distance(p + offset, values, origin, spacing)
            - sample_sdf_distance(p - offset, values, origin, spacing)
        ) / spacing
    grad_len = wp.sqrt(wp.dot(gradient, gradient) + 1.0e-8)
    return d, gradient / grad_len


@wp.func
def smooth_primitive_or_sdf_surface(
    local: wp.vec3,
    kind: int,
    size: wp.vec3,
    sdf: wp.array3d(dtype=float),
    sdf_origin: wp.vec3,
    sdf_spacing: float,
):
    """Smooth C1 signed distance and outward normal for SDF grids (kind=3) or analytical shapes."""
    if kind == 3:
        return sample_sdf_surface(local, sdf, sdf_origin, sdf_spacing)
    if kind == 1:
        q = wp.vec3(wp.abs(local[0]) - size[0], wp.abs(local[1]) - size[1], wp.abs(local[2]) - size[2])
        outside = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        out_sq = wp.dot(outside, outside)
        length = wp.sqrt(out_sq + 1.0e-8)
        distance = (length - 1.0e-4) + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
        axis = int(0)
        if q[1] > q[axis]:
            axis = 1
        if q[2] > q[axis]:
            axis = 2
        normal = wp.vec3(0.0)
        for a in range(3):
            sign = float(1.0)
            if local[a] < 0.0:
                sign = -1.0
            if out_sq > 1.0e-8:
                normal[a] = sign * outside[a] / length
            elif a == axis:
                normal[a] = sign
        return distance, normal
    if kind == 2:
        length = wp.sqrt(wp.dot(local, local) + 1.0e-8)
        distance = length - size[0]
        normal = wp.vec3(0.0, 0.0, 1.0)
        if length > 1.0e-4:
            normal = local / length
        return distance, normal
    if kind == 5:
        z_cap = wp.clamp(local[2], -size[1], size[1])
        diff = wp.vec3(local[0], local[1], local[2] - z_cap)
        length = wp.sqrt(wp.dot(diff, diff) + 1.0e-8)
        distance = length - size[0]
        normal = wp.vec3(0.0, 0.0, 1.0)
        if length > 1.0e-4:
            normal = diff / length
        return distance, normal
    if kind == 4:
        radial = wp.sqrt(local[0] * local[0] + local[1] * local[1] + 1.0e-8)
        q2 = wp.vec2(radial - size[0], wp.abs(local[2]) - size[2])
        outside2 = wp.vec2(wp.max(q2[0], 0.0), wp.max(q2[1], 0.0))
        out_sq2 = wp.dot(outside2, outside2)
        length2 = wp.sqrt(out_sq2 + 1.0e-8)
        distance = (length2 - 1.0e-4) + wp.min(wp.max(q2[0], q2[1]), 0.0)
        direction = wp.vec2(1.0, 0.0)
        if radial > 1.0e-4:
            direction = wp.vec2(local[0], local[1]) / radial
        weights = wp.vec2(1.0, 0.0)
        if out_sq2 > 1.0e-8:
            weights = outside2 / length2
        elif q2[1] > q2[0]:
            weights = wp.vec2(0.0, 1.0)
        sign = float(1.0)
        if local[2] < 0.0:
            sign = -1.0
        return distance, wp.vec3(direction[0] * weights[0], direction[1] * weights[0], sign * weights[1])
    return local[2], wp.vec3(0.0, 0.0, 1.0)


@wp.func
def ball_to_cube_radial(d: wp.vec3, inv_radius: float) -> wp.vec3:
    """Volume-preserving ball-to-cube mapping in [-1, 1]^3."""
    s = d * inv_radius
    r2 = wp.dot(s, s)
    if r2 <= 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)
    r = wp.sqrt(r2)
    max_c = wp.max(wp.abs(s[0]), wp.max(wp.abs(s[1]), wp.abs(s[2])))
    if max_c <= 1.0e-8:
        return wp.vec3(0.0, 0.0, 0.0)
    return s * (r / max_c)


@wp.func
def poly6_window(r_norm_sq: float) -> float:
    """Cubic poly6 window (1 - (r/R)^2)^3 for r <= R."""
    if r_norm_sq >= 1.0:
        return 0.0
    t = 1.0 - r_norm_sq
    return t * t * t


@wp.kernel
def cconv_and_packing_rate_kernel(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    radius: float,
    max_neighbors: int,
    particles_per_env: int,
    basis_out: wp.array3d(dtype=float),
    packing_out: wp.array2d(dtype=float),
):
    """Full-bed ContinuousConv basis [N, 64, 4] + 6D packing state `[count, phi, dphi_dt, v_rel_x, v_rel_y, v_rel_z]`.

    `dphi_dt` measures the instantaneous local compaction rate `-sum_j grad(w_ij) . (v_j - v_i)`:
      - `dphi_dt > 0`: local neighborhood is compressing / compacting under load,
      - `dphi_dt < 0`: local neighborhood is dilating / loosening / shearing.
    """
    i = wp.tid()
    x_i = pos[i]
    v_i = vel[i]
    inv_r = 1.0 / radius
    inv_r2 = inv_r * inv_r
    r_sq = radius * radius

    group = (i // particles_per_env) if particles_per_env > 0 else -1
    query = wp.hash_grid_query(grid, x_i, radius, group)
    neighbor = int(0)
    count = int(0)
    w_sum = float(0.0)
    dphi_dt = float(0.0)
    v_rel_sum = wp.vec3(0.0, 0.0, 0.0)

    while wp.hash_grid_query_next(query, neighbor):
        if neighbor == i:
            continue
        d = pos[neighbor] - x_i
        dist_sq = wp.dot(d, d)
        if dist_sq < r_sq:
            r_norm_sq = dist_sq * inv_r2
            t = 1.0 - r_norm_sq
            w = t * t * t
            dv = vel[neighbor] - v_i
            # Time derivative of poly6 weight w(r_ij(t)): dw/dt = -6 * (1 - r^2/R^2)^2 / R^2 * (d . dv)
            dw_dt = -6.0 * (t * t) * inv_r2 * wp.dot(d, dv)
            w_sum += w
            dphi_dt += dw_dt
            v_rel_sum += w * dv

            u = ball_to_cube_radial(d, inv_r)
            gx = wp.clamp(1.5 * (u[0] + 1.0), 0.0, 2.9999)
            gy = wp.clamp(1.5 * (u[1] + 1.0), 0.0, 2.9999)
            gz = wp.clamp(1.5 * (u[2] + 1.0), 0.0, 2.9999)
            ix = int(gx)
            iy = int(gy)
            iz = int(gz)
            fx = gx - float(ix)
            fy = gy - float(iy)
            fz = gz - float(iz)

            for dx in range(2):
                wx = (1.0 - fx) if dx == 0 else fx
                for dy in range(2):
                    wy = (1.0 - fy) if dy == 0 else fy
                    for dz in range(2):
                        wz = (1.0 - fz) if dz == 0 else fz
                        wt = w * wx * wy * wz
                        cell = (ix + dx) * 16 + (iy + dy) * 4 + (iz + dz)
                        basis_out[i, cell, 0] += wt
                        basis_out[i, cell, 1] += wt * dv[0]
                        basis_out[i, cell, 2] += wt * dv[1]
                        basis_out[i, cell, 3] += wt * dv[2]
            count += 1
            if count >= max_neighbors:
                break

    scale = 1.0 / 16.0
    for cell in range(64):
        for c in range(4):
            basis_out[i, cell, c] = basis_out[i, cell, c] * scale

    packing_out[i, 0] = float(count) / 32.0
    packing_out[i, 1] = w_sum / 16.0
    packing_out[i, 2] = wp.clamp(dphi_dt * 0.05, -5.0, 5.0)
    inv_w = 1.0 / wp.max(w_sum, 1.0e-4)
    packing_out[i, 3] = v_rel_sum[0] * inv_w
    packing_out[i, 4] = v_rel_sum[1] * inv_w
    packing_out[i, 5] = v_rel_sum[2] * inv_w


@wp.kernel
def cconv_obstacle_basis_kernel(
    pairs: wp.array3d(dtype=float),
    influence: wp.array2d(dtype=float),
    obs_basis_out: wp.array3d(dtype=float),
):
    """Continuous 3D convolution [4, 4, 4] over collider surface points within `support`."""
    i = wp.tid()
    num_colliders = pairs.shape[1]

    for j in range(num_colliders):
        inf = influence[i, j]
        if inf > 0.0:
            gap_norm = pairs[i, j, 0]
            nx = pairs[i, j, 1]
            ny = pairs[i, j, 2]
            nz = pairs[i, j, 3]
            vx = pairs[i, j, 4]
            vy = pairs[i, j, 5]
            vz = pairs[i, j, 6]
            mu = pairs[i, j, 7]

            r_norm = 1.0 - inf
            w = poly6_window(r_norm * r_norm)
            d = wp.vec3(-gap_norm * nx, -gap_norm * ny, -gap_norm * nz)
            u = ball_to_cube_radial(d, 1.0)
            gx = wp.clamp(1.5 * (u[0] + 1.0), 0.0, 2.9999)
            gy = wp.clamp(1.5 * (u[1] + 1.0), 0.0, 2.9999)
            gz = wp.clamp(1.5 * (u[2] + 1.0), 0.0, 2.9999)
            ix = int(gx)
            iy = int(gy)
            iz = int(gz)
            fx = gx - float(ix)
            fy = gy - float(iy)
            fz = gz - float(iz)

            for dx in range(2):
                wx = (1.0 - fx) if dx == 0 else fx
                for dy in range(2):
                    wy = (1.0 - fy) if dy == 0 else fy
                    for dz in range(2):
                        wz = (1.0 - fz) if dz == 0 else fz
                        wt = w * wx * wy * wz
                        cell = (ix + dx) * 16 + (iy + dy) * 4 + (iz + dz)
                        obs_basis_out[i, cell, 0] += wt
                        obs_basis_out[i, cell, 1] += wt * gap_norm
                        obs_basis_out[i, cell, 2] += wt * nx
                        obs_basis_out[i, cell, 3] += wt * ny
                        obs_basis_out[i, cell, 4] += wt * nz
                        obs_basis_out[i, cell, 5] += wt * vx
                        obs_basis_out[i, cell, 6] += wt * vy
                        obs_basis_out[i, cell, 7] += wt * vz
                        obs_basis_out[i, cell, 8] += wt * mu


@wp.kernel
def particle_translation_invariant_features_kernel(
    history: wp.array3d(dtype=float),
    material: wp.array(dtype=float),
    dt: float,
    gravity: wp.vec3,
    nodes: wp.array2d(dtype=float),
):
    """Build 22D particle features with NO global position: velocities (3*(H-1)), material (12), dt (1), gravity (3)."""
    i = wp.tid()
    h = history.shape[1]
    vel_dim = 3 * (h - 1)
    for t in range(h - 1):
        for a in range(3):
            nodes[i, t * 3 + a] = (history[i, t + 1, a] - history[i, t, a]) / dt
    for k in range(12):
        nodes[i, vel_dim + k] = material[k]
    nodes[i, vel_dim + 12] = dt / 0.02
    for a in range(3):
        nodes[i, vel_dim + 13 + a] = gravity[a] / 9.81


@wp.kernel
def unified_collider_features_kernel(
    history: wp.array3d(dtype=float),
    poses: wp.array2d(dtype=float),
    velocities: wp.array2d(dtype=float),
    kinds: wp.array(dtype=int),
    sizes: wp.array2d(dtype=float),
    friction: wp.array(dtype=float),
    material: wp.array(dtype=float),
    dt: float,
    support: float,
    far_support_scale: float,
    far_gap_weight: float,
    far_influence_weight: float,
    sdf: wp.array3d(dtype=float),
    sdf_origin: wp.vec3,
    sdf_spacing: float,
    pairs: wp.array3d(dtype=float),
    influence: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    h = history.shape[1]
    x = wp.vec3(history[i, h - 1, 0], history[i, h - 1, 1], history[i, h - 1, 2])
    previous = wp.vec3(history[i, h - 2, 0], history[i, h - 2, 1], history[i, h - 2, 2])
    center = wp.vec3(poses[j, 0], poses[j, 1], poses[j, 2])
    rotation = wp.quat(poses[j, 3], poses[j, 4], poses[j, 5], poses[j, 6])
    local = wp.quat_rotate_inv(rotation, x - center)
    size_vec = wp.vec3(sizes[j, 0], sizes[j, 1], sizes[j, 2])
    distance, normal_local = smooth_primitive_or_sdf_surface(local, kinds[j], size_vec, sdf, sdf_origin, sdf_spacing)
    normal = wp.quat_rotate(rotation, normal_local)
    surface = x - distance * normal
    linear = wp.vec3(velocities[j, 0], velocities[j, 1], velocities[j, 2])
    angular = wp.vec3(velocities[j, 3], velocities[j, 4], velocities[j, 5])
    relative = linear + wp.cross(angular, surface - center) - (x - previous) / dt
    gap = distance - 0.02 * material[11]
    gap_s = gap / support
    inv_far = 1.0 / wp.max(far_support_scale, 1.0)
    near_gap_w = 1.0 - far_gap_weight
    pairs[i, j, 0] = near_gap_w * wp.tanh(gap_s) + far_gap_weight * wp.tanh(inv_far * gap_s)
    for a in range(3):
        pairs[i, j, 1 + a] = normal[a]
        pairs[i, j, 4 + a] = relative[a]
    pairs[i, j, 7] = friction[j]
    g_pos = wp.max(gap_s, 0.0)
    th_short = wp.tanh(g_pos)
    th_long = wp.tanh(inv_far * g_pos)
    near_inf_w = 1.0 - far_influence_weight
    influence[i, j] = near_inf_w * (1.0 - th_short * th_short) + far_influence_weight * (1.0 - th_long * th_long)


class _DifferentiableLocalFeatures(torch.autograd.Function):
    """Differentiable translation-invariant particle + unified SDF collider features."""

    @staticmethod
    def forward(
        ctx,
        history,
        poses,
        velocities,
        kinds,
        sizes,
        friction,
        material,
        dt,
        support,
        far_support_scale,
        far_gap_weight,
        far_influence_weight,
        gravity,
        sdf,
        sdf_origin,
        sdf_spacing,
    ):
        n, h = history.shape[:2]
        device = str(history.device)
        grad = wp.zeros(history.shape, dtype=float, device=device) if history.requires_grad else None
        hist = wp.from_torch(history, requires_grad=history.requires_grad, grad=grad)
        nodes = wp.empty((n, 16 + 3 * (h - 1)), dtype=float, device=device, requires_grad=history.requires_grad)
        pairs = wp.empty((n, len(poses), 8), dtype=float, device=device, requires_grad=history.requires_grad)
        influence = wp.empty((n, len(poses)), dtype=float, device=device, requires_grad=history.requires_grad)
        tape = wp.Tape()
        torch_stream = torch.cuda.current_stream(history.device)
        warp_stream = wp.get_stream(device)
        if warp_stream.cuda_stream != torch_stream.cuda_stream:
            warp_stream = wp.stream_from_torch(torch_stream)
        with wp.ScopedStream(warp_stream), tape:
            wp.launch(
                particle_translation_invariant_features_kernel,
                n,
                [hist, wp.from_torch(material), dt, wp.vec3(*gravity), nodes],
                device=device,
            )
            if len(poses):
                wp.launch(
                    unified_collider_features_kernel,
                    (n, len(poses)),
                    [
                        hist,
                        wp.from_torch(poses),
                        wp.from_torch(velocities),
                        wp.from_torch(kinds),
                        wp.from_torch(sizes),
                        wp.from_torch(friction),
                        wp.from_torch(material),
                        dt,
                        support,
                        far_support_scale,
                        far_gap_weight,
                        far_influence_weight,
                        wp.from_torch(sdf),
                        wp.vec3(*sdf_origin),
                        sdf_spacing,
                        pairs,
                        influence,
                    ],
                    device=device,
                )
        ctx.history = hist
        ctx.outputs = (nodes, pairs, influence)
        ctx.tape = tape
        ctx.device = history.device
        return tuple(wp.to_torch(x) for x in ctx.outputs)

    @staticmethod
    def backward(ctx, *gradients):
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(ctx.device))):
            grads = {out: wp.from_torch(g.contiguous()) for out, g in zip(ctx.outputs, gradients) if g is not None}
            ctx.tape.zero()
            ctx.tape.backward(grads=grads)
            gradient = wp.to_torch(ctx.history.grad).clone()
        return (gradient,) + (None,) * 15


@lru_cache(maxsize=8)
def empty_sdf_tensors(device: str):
    return (torch.zeros((2, 2, 2), device=device, dtype=torch.float32), (0.0, 0.0, 0.0), 1.0)


def features_particle_nerd_local(
    history: torch.Tensor,
    poses: torch.Tensor,
    velocities: torch.Tensor,
    kinds: torch.Tensor,
    sizes: torch.Tensor,
    friction: torch.Tensor,
    material: torch.Tensor,
    dt: float = 0.005,
    support: float = 0.07,
    far_support_scale: float = 5.0,
    far_gap_weight: float = 0.3,
    far_influence_weight: float = 0.15,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    sdf: tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predictor integration (`pos_pred`, `vel_pred`) + translation-invariant particle & SDF collider features."""
    if sdf is None:
        sdf = empty_sdf_tensors(str(history.device))
    pos_cur = history[:, -1]
    vel_cur = (history[:, -1] - history[:, -2]) / dt
    pos_pred, vel_pred = integrate_pos_vel(pos_cur, vel_cur, gravity=gravity, dt=dt)
    if not history.requires_grad:
        n, h = history.shape[:2]
        device = str(history.device)
        nodes_t = torch.empty((n, 16 + 3 * (h - 1)), dtype=torch.float32, device=history.device)
        pairs_t = torch.empty((n, len(poses), 8), dtype=torch.float32, device=history.device)
        inf_t = torch.empty((n, len(poses)), dtype=torch.float32, device=history.device)
        torch_stream = torch.cuda.current_stream(history.device)
        warp_stream = wp.get_stream(device)
        if warp_stream.cuda_stream != torch_stream.cuda_stream:
            warp_stream = wp.stream_from_torch(torch_stream)
        with wp.ScopedStream(warp_stream):
            wp.launch(
                particle_translation_invariant_features_kernel,
                n,
                [
                    wp.from_torch(history, requires_grad=False),
                    wp.from_torch(material, requires_grad=False),
                    dt,
                    wp.vec3(*gravity),
                    wp.from_torch(nodes_t, requires_grad=False),
                ],
                device=device,
            )
            if len(poses):
                wp.launch(
                    unified_collider_features_kernel,
                    (n, len(poses)),
                    [
                        wp.from_torch(history, requires_grad=False),
                        wp.from_torch(poses, requires_grad=False),
                        wp.from_torch(velocities, requires_grad=False),
                        wp.from_torch(kinds, requires_grad=False),
                        wp.from_torch(sizes, requires_grad=False),
                        wp.from_torch(friction, requires_grad=False),
                        wp.from_torch(material, requires_grad=False),
                        dt,
                        support,
                        far_support_scale,
                        far_gap_weight,
                        far_influence_weight,
                        wp.from_torch(sdf[0], requires_grad=False),
                        wp.vec3(*sdf[1]),
                        sdf[2],
                        wp.from_torch(pairs_t, requires_grad=False),
                        wp.from_torch(inf_t, requires_grad=False),
                    ],
                    device=device,
                )
        return pos_pred, vel_pred, nodes_t, pairs_t, inf_t
    nodes_t, pairs_t, inf_t = _DifferentiableLocalFeatures.apply(
        history,
        poses,
        velocities,
        kinds,
        sizes,
        friction,
        material,
        dt,
        support,
        far_support_scale,
        far_gap_weight,
        far_influence_weight,
        gravity,
        *sdf,
    )
    return pos_pred, vel_pred, nodes_t, pairs_t, inf_t


class LocalNeighborhoodAggregator(nn.Module):
    """Full-bed Continuous 3D Convolution (`[4, 4, 4]`) and 6D local packing + compaction-rate extractor."""

    def __init__(
        self,
        fluid_in_channels: int = 4,
        obs_in_channels: int = 9,
        filters: int = 64,
        radius: float = 0.072,
        max_neighbors: int = 48,
    ):
        super().__init__()
        self.fluid_in_channels = fluid_in_channels
        self.obs_in_channels = obs_in_channels
        self.filters = filters
        self.radius = float(radius)
        self.max_neighbors = int(max_neighbors)
        num_cells = 64

        self.fluid_linear = nn.Linear(num_cells * fluid_in_channels, filters, bias=True)
        self.obs_linear = nn.Linear(num_cells * obs_in_channels, filters, bias=True)
        nn.init.uniform_(self.fluid_linear.weight, -0.02, 0.02)
        nn.init.zeros_(self.fluid_linear.bias)
        nn.init.uniform_(self.obs_linear.weight, -0.02, 0.02)
        nn.init.zeros_(self.obs_linear.bias)

        self._grids: dict[tuple[str, int, bool], wp.HashGrid] = {}
        self._group_arrays: dict[tuple[str, int, int], wp.array] = {}
        self._eval_buffers: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _get_grid(self, device: str, n: int, with_groups: bool = False) -> wp.HashGrid:
        key = (device, n, with_groups)
        grid = self._grids.get(key)
        if grid is None:
            grid = wp.HashGrid(32, 32, 16, device=device)
            grid.reserve(n, with_groups=with_groups)
            self._grids[key] = grid
        return grid

    def _get_groups(self, device: str, num_envs: int, m_pts: int, torch_device: torch.device) -> wp.array:
        key = (device, num_envs, m_pts)
        grp = self._group_arrays.get(key)
        if grp is None:
            env_ids = torch.arange(num_envs, device=torch_device, dtype=torch.int32).repeat_interleave(m_pts)
            grp = wp.from_torch(env_ids, dtype=wp.int32, requires_grad=False)
            self._group_arrays[key] = grp
        return grp

    @torch.no_grad()
    def compute_bases(
        self,
        pos_pred: torch.Tensor,
        vel_pred: torch.Tensor,
        pairs: torch.Tensor,
        influence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute fluid [N, 64*4], obstacle [N, 64*9], and 6D packing `[count, phi, dphi_dt, vx, vy, vz]` on GPU."""
        n = pos_pred.shape[0]
        device = str(pos_pred.device)
        pos_f = pos_pred.detach().float()
        pos_c = (pos_f - pos_f.mean(dim=0, keepdim=True)).contiguous()
        vel_c = vel_pred.detach().float().contiguous()
        pairs_c = pairs.detach().float().contiguous()
        inf_c = influence.detach().float().contiguous()

        if not self.training:
            buf_key = (device, n, torch.is_inference_mode_enabled())
            bufs = self._eval_buffers.get(buf_key)
            if bufs is None:
                fluid_basis = torch.zeros((n, 64, self.fluid_in_channels), dtype=torch.float32, device=pos_pred.device)
                obs_basis = torch.zeros((n, 64, self.obs_in_channels), dtype=torch.float32, device=pos_pred.device)
                packing = torch.zeros((n, 6), dtype=torch.float32, device=pos_pred.device)
                self._eval_buffers[buf_key] = (fluid_basis, obs_basis, packing)
            else:
                fluid_basis, obs_basis, packing = bufs
                fluid_basis.zero_()
                obs_basis.zero_()
                packing.zero_()
        else:
            fluid_basis = torch.zeros((n, 64, self.fluid_in_channels), dtype=torch.float32, device=pos_pred.device)
            obs_basis = torch.zeros((n, 64, self.obs_in_channels), dtype=torch.float32, device=pos_pred.device)
            packing = torch.zeros((n, 6), dtype=torch.float32, device=pos_pred.device)

        grid = self._get_grid(device, n)
        pos_wp = wp.from_torch(pos_c, dtype=wp.vec3, requires_grad=False)
        vel_wp = wp.from_torch(vel_c, dtype=wp.vec3, requires_grad=False)
        fluid_wp = wp.from_torch(fluid_basis, requires_grad=False)
        obs_wp = wp.from_torch(obs_basis, requires_grad=False)
        pack_wp = wp.from_torch(packing, requires_grad=False)

        torch_stream = torch.cuda.current_stream(pos_pred.device)
        warp_stream = wp.get_stream(device)
        if warp_stream.cuda_stream != torch_stream.cuda_stream:
            warp_stream = wp.stream_from_torch(torch_stream)

        with wp.ScopedStream(warp_stream):
            grid.build(pos_wp, self.radius)
            wp.launch(
                cconv_and_packing_rate_kernel,
                n,
                [grid.id, pos_wp, vel_wp, self.radius, self.max_neighbors, 0, fluid_wp, pack_wp],
                device=device,
            )
            if pairs_c.shape[1] > 0:
                wp.launch(
                    cconv_obstacle_basis_kernel,
                    n,
                    [
                        wp.from_torch(pairs_c, requires_grad=False),
                        wp.from_torch(inf_c, requires_grad=False),
                        obs_wp,
                    ],
                    device=device,
                )
        return fluid_basis.view(n, -1), obs_basis.view(n, -1), packing

    @torch.no_grad()
    def compute_bases_batched(
        self,
        pos_pred_env: torch.Tensor,
        vel_pred_env: torch.Tensor,
        pairs_env: torch.Tensor,
        influence_env: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-launch grouped HashGrid continuous-convolution bases + packing rate for [E, M, 3] particles."""
        num_envs, m_pts, _ = pos_pred_env.shape
        n_tot = num_envs * m_pts
        n_col = pairs_env.shape[2]
        device = str(pos_pred_env.device)

        pos_f = pos_pred_env.detach().float()
        pos_c = (pos_f - pos_f.mean(dim=1, keepdim=True)).reshape(n_tot, 3).contiguous()
        vel_c = vel_pred_env.detach().float().reshape(n_tot, 3).contiguous()
        pairs_c = pairs_env.detach().float().reshape(n_tot, n_col, 8).contiguous()
        inf_c = influence_env.detach().float().reshape(n_tot, n_col).contiguous()

        buf_key = (device, n_tot, torch.is_inference_mode_enabled())
        bufs = self._eval_buffers.get(buf_key)
        if bufs is None:
            fluid_basis = torch.zeros(
                (n_tot, 64, self.fluid_in_channels), dtype=torch.float32, device=pos_pred_env.device
            )
            obs_basis = torch.zeros((n_tot, 64, self.obs_in_channels), dtype=torch.float32, device=pos_pred_env.device)
            packing = torch.zeros((n_tot, 6), dtype=torch.float32, device=pos_pred_env.device)
            self._eval_buffers[buf_key] = (fluid_basis, obs_basis, packing)
        else:
            fluid_basis, obs_basis, packing = bufs
            fluid_basis.zero_()
            obs_basis.zero_()
            packing.zero_()

        grid = self._get_grid(device, n_tot, with_groups=True)
        groups_wp = self._get_groups(device, num_envs, m_pts, pos_pred_env.device)
        pos_wp = wp.from_torch(pos_c, dtype=wp.vec3, requires_grad=False)
        vel_wp = wp.from_torch(vel_c, dtype=wp.vec3, requires_grad=False)
        fluid_wp = wp.from_torch(fluid_basis, requires_grad=False)
        obs_wp = wp.from_torch(obs_basis, requires_grad=False)
        pack_wp = wp.from_torch(packing, requires_grad=False)

        torch_stream = torch.cuda.current_stream(pos_pred_env.device)
        warp_stream = wp.get_stream(device)
        if warp_stream.cuda_stream != torch_stream.cuda_stream:
            warp_stream = wp.stream_from_torch(torch_stream)

        with wp.ScopedStream(warp_stream):
            grid.build(pos_wp, self.radius, groups=groups_wp)
            wp.launch(
                cconv_and_packing_rate_kernel,
                n_tot,
                [grid.id, pos_wp, vel_wp, self.radius, self.max_neighbors, m_pts, fluid_wp, pack_wp],
                device=device,
            )
            if n_col > 0:
                wp.launch(
                    cconv_obstacle_basis_kernel,
                    n_tot,
                    [
                        wp.from_torch(pairs_c, requires_grad=False),
                        wp.from_torch(inf_c, requires_grad=False),
                        obs_wp,
                    ],
                    device=device,
                )
        return (
            fluid_basis.view(num_envs, m_pts, -1),
            obs_basis.view(num_envs, m_pts, -1),
            packing.view(num_envs, m_pts, 6),
        )

    def forward(
        self,
        fluid_basis_flat: torch.Tensor,
        obs_basis_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = self.fluid_linear.weight.dtype
        return self.fluid_linear(fluid_basis_flat.to(dt)), self.obs_linear(obs_basis_flat.to(dt))


@wp.kernel
def particle_translation_invariant_features_batched_kernel(
    history: wp.array3d(dtype=float),
    material: wp.array2d(dtype=float),
    particles_per_env: int,
    dt: float,
    gravity: wp.vec3,
    nodes: wp.array2d(dtype=float),
):
    i = wp.tid()
    env = i // particles_per_env
    h = history.shape[1]
    vel_dim = 3 * (h - 1)
    for t in range(h - 1):
        for a in range(3):
            nodes[i, t * 3 + a] = (history[i, t + 1, a] - history[i, t, a]) / dt
    for k in range(12):
        nodes[i, vel_dim + k] = material[env, k]
    nodes[i, vel_dim + 12] = dt / 0.02
    for a in range(3):
        nodes[i, vel_dim + 13 + a] = gravity[a] / 9.81


@wp.kernel
def unified_collider_features_batched_kernel(
    history: wp.array3d(dtype=float),
    poses: wp.array3d(dtype=float),
    velocities: wp.array3d(dtype=float),
    kinds: wp.array(dtype=int),
    sizes: wp.array3d(dtype=float),
    friction: wp.array2d(dtype=float),
    material: wp.array2d(dtype=float),
    particles_per_env: int,
    dt: float,
    support: float,
    far_support_scale: float,
    far_gap_weight: float,
    far_influence_weight: float,
    sdf: wp.array3d(dtype=float),
    sdf_origin: wp.vec3,
    sdf_spacing: float,
    pairs: wp.array3d(dtype=float),
    influence: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    env = i // particles_per_env
    h = history.shape[1]
    x = wp.vec3(history[i, h - 1, 0], history[i, h - 1, 1], history[i, h - 1, 2])
    previous = wp.vec3(history[i, h - 2, 0], history[i, h - 2, 1], history[i, h - 2, 2])
    center = wp.vec3(poses[env, j, 0], poses[env, j, 1], poses[env, j, 2])
    rotation = wp.quat(poses[env, j, 3], poses[env, j, 4], poses[env, j, 5], poses[env, j, 6])
    local = wp.quat_rotate_inv(rotation, x - center)
    size_vec = wp.vec3(sizes[env, j, 0], sizes[env, j, 1], sizes[env, j, 2])
    distance, normal_local = smooth_primitive_or_sdf_surface(local, kinds[j], size_vec, sdf, sdf_origin, sdf_spacing)
    normal = wp.quat_rotate(rotation, normal_local)
    surface = x - distance * normal
    linear = wp.vec3(velocities[env, j, 0], velocities[env, j, 1], velocities[env, j, 2])
    angular = wp.vec3(velocities[env, j, 3], velocities[env, j, 4], velocities[env, j, 5])
    relative = linear + wp.cross(angular, surface - center) - (x - previous) / dt
    gap = distance - 0.02 * material[env, 11]
    gap_s = gap / support
    inv_far = 1.0 / wp.max(far_support_scale, 1.0)
    near_gap_w = 1.0 - far_gap_weight
    pairs[i, j, 0] = near_gap_w * wp.tanh(gap_s) + far_gap_weight * wp.tanh(inv_far * gap_s)
    for a in range(3):
        pairs[i, j, 1 + a] = normal[a]
        pairs[i, j, 4 + a] = relative[a]
    pairs[i, j, 7] = friction[env, j]
    g_pos = wp.max(gap_s, 0.0)
    th_short = wp.tanh(g_pos)
    th_long = wp.tanh(inv_far * g_pos)
    near_inf_w = 1.0 - far_influence_weight
    influence[i, j] = near_inf_w * (1.0 - th_short * th_short) + far_influence_weight * (1.0 - th_long * th_long)


_FEAT_BATCH_BUFFERS: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def features_particle_nerd_local_batched(
    history_env: torch.Tensor,
    poses_env: torch.Tensor,
    velocities_env: torch.Tensor,
    kinds: torch.Tensor,
    sizes: torch.Tensor,
    friction: torch.Tensor,
    material: torch.Tensor,
    dt: float = 0.005,
    support: float = 0.07,
    far_support_scale: float = 5.0,
    far_gap_weight: float = 0.3,
    far_influence_weight: float = 0.15,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    sdf: tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-launch batched predictor and collider features for [E, M, H, 3] particle histories."""
    pos_cur = history_env[:, :, -1]
    vel_cur = (history_env[:, :, -1] - history_env[:, :, -2]) / dt
    pos_pred, vel_pred = integrate_pos_vel(pos_cur, vel_cur, gravity=gravity, dt=dt)

    num_envs, m_pts, h_len, _ = history_env.shape
    n_tot = num_envs * m_pts
    n_col = len(kinds)
    device = str(history_env.device)
    hist_flat = history_env.reshape(n_tot, h_len, 3).contiguous()
    sizes_env = (sizes.unsqueeze(0).expand(num_envs, -1, -1) if sizes.ndim == 2 else sizes).contiguous()
    friction_env = (friction.unsqueeze(0).expand(num_envs, -1) if friction.ndim == 1 else friction).contiguous()
    material_env = (material.unsqueeze(0).expand(num_envs, -1) if material.ndim == 1 else material).contiguous()
    buf_key = (device, num_envs, m_pts, h_len, n_col, torch.is_inference_mode_enabled())
    cached = _FEAT_BATCH_BUFFERS.get(buf_key)
    if cached is None:
        nodes_t = torch.empty((num_envs, m_pts, 16 + 3 * (h_len - 1)), dtype=torch.float32, device=history_env.device)
        pairs_t = torch.empty((num_envs, m_pts, n_col, 8), dtype=torch.float32, device=history_env.device)
        inf_t = torch.empty((num_envs, m_pts, n_col), dtype=torch.float32, device=history_env.device)
        _FEAT_BATCH_BUFFERS[buf_key] = (nodes_t, pairs_t, inf_t)
    else:
        nodes_t, pairs_t, inf_t = cached
    if sdf is None:
        sdf = empty_sdf_tensors(device)

    torch_stream = torch.cuda.current_stream(history_env.device)
    warp_stream = wp.get_stream(device)
    if warp_stream.cuda_stream != torch_stream.cuda_stream:
        warp_stream = wp.stream_from_torch(torch_stream)
    with wp.ScopedStream(warp_stream):
        wp.launch(
            particle_translation_invariant_features_batched_kernel,
            n_tot,
            [
                wp.from_torch(hist_flat, requires_grad=False),
                wp.from_torch(material_env, requires_grad=False),
                m_pts,
                dt,
                wp.vec3(*gravity),
                wp.from_torch(nodes_t.view(n_tot, -1), requires_grad=False),
            ],
            device=device,
        )
        if n_col > 0:
            wp.launch(
                unified_collider_features_batched_kernel,
                (n_tot, n_col),
                [
                    wp.from_torch(hist_flat, requires_grad=False),
                    wp.from_torch(poses_env.contiguous(), requires_grad=False),
                    wp.from_torch(velocities_env.contiguous(), requires_grad=False),
                    wp.from_torch(kinds.contiguous(), requires_grad=False),
                    wp.from_torch(sizes_env, requires_grad=False),
                    wp.from_torch(friction_env, requires_grad=False),
                    wp.from_torch(material_env, requires_grad=False),
                    m_pts,
                    dt,
                    support,
                    far_support_scale,
                    far_gap_weight,
                    far_influence_weight,
                    wp.from_torch(sdf[0], requires_grad=False),
                    wp.vec3(*sdf[1]),
                    sdf[2],
                    wp.from_torch(pairs_t.view(n_tot, n_col, 8), requires_grad=False),
                    wp.from_torch(inf_t.view(n_tot, n_col), requires_grad=False),
                ],
                device=device,
            )
    return pos_pred, vel_pred, nodes_t, pairs_t, inf_t


@wp.kernel
def active_particle_score_batched_kernel(
    pos_in: wp.array(dtype=wp.vec3),
    history: wp.array(dtype=wp.mat33),
    dyn_poses: wp.array3d(dtype=float),
    dyn_sizes: wp.array3d(dtype=float),
    capacity: int,
    n_dyn: int,
    dt: float,
    scores: wp.array(dtype=float),
    min_gaps: wp.array(dtype=float),
):
    tid = wp.tid()
    env = tid // capacity
    x = pos_in[tid]
    h = history[tid]
    x_prev = wp.vec3(h[1, 0], h[1, 1], h[1, 2])
    speed = wp.length(x - x_prev) / wp.max(dt, 1.0e-6)
    min_gap = float(1.0e6)
    for c in range(n_dyn):
        center = wp.vec3(dyn_poses[env, c, 0], dyn_poses[env, c, 1], dyn_poses[env, c, 2])
        rot = wp.quat(dyn_poses[env, c, 3], dyn_poses[env, c, 4], dyn_poses[env, c, 5], dyn_poses[env, c, 6])
        local = wp.quat_rotate_inv(rot, x - center)
        half_ext = wp.vec3(dyn_sizes[env, c, 0], dyn_sizes[env, c, 1], dyn_sizes[env, c, 2])
        qx = wp.abs(local[0]) - half_ext[0]
        qy = wp.abs(local[1]) - half_ext[1]
        qz = wp.abs(local[2]) - half_ext[2]
        out_d = wp.length(wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0)))
        in_d = wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)
        d_c = out_d + in_d
        if d_c < min_gap:
            min_gap = d_c
    min_gaps[tid] = min_gap
    scores[tid] = min_gap - 0.12 * wp.min(speed, 2.0)


@wp.kernel
def extract_sub_history_local_kernel(
    topk: wp.array2d(dtype=wp.int64),
    pos_in: wp.array(dtype=wp.vec3),
    history: wp.array(dtype=wp.mat33),
    capacity: int,
    hist_sub: wp.array4d(dtype=float),
):
    env, k = wp.tid()
    p_idx = env * capacity + int(topk[env, k])
    h = history[p_idx]
    x = pos_in[p_idx]
    for a in range(3):
        hist_sub[env, k, 0, a] = h[0, a]
        hist_sub[env, k, 1, a] = h[1, a]
    hist_sub[env, k, 2, 0] = x[0]
    hist_sub[env, k, 2, 1] = x[1]
    hist_sub[env, k, 2, 2] = x[2]


@wp.kernel
def advance_stationary_history_local_kernel(
    pos_in: wp.array(dtype=wp.vec3),
    history: wp.array(dtype=wp.mat33),
):
    tid = wp.tid()
    x = pos_in[tid]
    h = history[tid]
    history[tid] = wp.mat33(
        h[1, 0],
        h[1, 1],
        h[1, 2],
        x[0],
        x[1],
        x[2],
        x[0],
        x[1],
        x[2],
    )


@wp.kernel
def scatter_sub_step_local_kernel(
    topk: wp.array2d(dtype=wp.int64),
    next_pos_sub: wp.array3d(dtype=float),
    next_vel_sub: wp.array3d(dtype=float),
    gamma: wp.array2d(dtype=float),
    tray_lower: wp.array2d(dtype=float),
    tray_upper: wp.array2d(dtype=float),
    capacity: int,
    dt: float,
    pos_in: wp.array(dtype=wp.vec3),
    pos_out: wp.array(dtype=wp.vec3),
    vel_out: wp.array(dtype=wp.vec3),
    history: wp.array(dtype=wp.mat33),
):
    env, k = wp.tid()
    p_idx = env * capacity + int(topk[env, k])
    x_cur = pos_in[p_idx]
    x_pred = wp.vec3(next_pos_sub[env, k, 0], next_pos_sub[env, k, 1], next_pos_sub[env, k, 2])
    v_pred = wp.vec3(next_vel_sub[env, k, 0], next_vel_sub[env, k, 1], next_vel_sub[env, k, 2])
    g = gamma[env, k]
    x_raw = x_cur + g * (x_pred - x_cur)
    x_new = wp.vec3(
        wp.clamp(x_raw[0], tray_lower[env, 0], tray_upper[env, 0]),
        wp.clamp(x_raw[1], tray_lower[env, 1], tray_upper[env, 1]),
        wp.clamp(x_raw[2], tray_lower[env, 2], tray_upper[env, 2]),
    )
    v_eff = (x_new - x_cur) / wp.max(dt, 1.0e-6)
    pos_out[p_idx] = x_new
    vel_out[p_idx] = v_eff
    h = history[p_idx]
    x_prev = x_new - dt * (g * v_pred)
    x_prev_clamped = wp.vec3(
        wp.clamp(x_prev[0], tray_lower[env, 0], tray_upper[env, 0]),
        wp.clamp(x_prev[1], tray_lower[env, 1], tray_upper[env, 1]),
        wp.clamp(x_prev[2], tray_lower[env, 2], tray_upper[env, 2]),
    )
    history[p_idx] = wp.mat33(
        h[0, 0],
        h[0, 1],
        h[0, 2],
        x_prev_clamped[0],
        x_prev_clamped[1],
        x_prev_clamped[2],
        x_new[0],
        x_new[1],
        x_new[2],
    )


@wp.kernel
def relax_amorphous_bed_kernel(
    grid: wp.uint64,
    pos_in: wp.array(dtype=wp.vec3),
    min_dist: float,
    lo: wp.vec3,
    hi: wp.vec3,
    pos_out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    xi = pos_in[i]
    disp = wp.vec3(0.0, 0.0, -0.0004)
    query = wp.hash_grid_query(grid, xi, min_dist)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if j == i:
            continue
        d = xi - pos_in[j]
        dist = wp.length(d)
        if dist < min_dist and dist > 1.0e-6:
            overlap = min_dist - dist
            disp += (d / dist) * (0.35 * overlap)
    x_new = xi + disp
    pos_out[i] = wp.vec3(
        wp.clamp(x_new[0], lo[0], hi[0]),
        wp.clamp(x_new[1], lo[1], hi[1]),
        wp.clamp(x_new[2], lo[2], hi[2]),
    )
