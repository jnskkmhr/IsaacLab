# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NVIDIA Warp Continuous 3D Convolution and explicit physics predictor for WorldParticle.

Replaces Open3D's `FixedRadiusSearch` and `ContinuousConv` (`ops.continuous_conv`)
with allocation-free `wp.HashGrid` kernels that map spherical neighborhoods to a
`[4, 4, 4]` trilinear filter cube with a poly6 window.
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


def integrate_pos_vel(
    pos: torch.Tensor,
    vel: torch.Tensor,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    dt: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit semi-implicit Euler / Verlet predictor matching WorldParticle `integrate_pos_vel`."""
    g = pos.new_tensor(gravity)
    vel_pred = vel + dt * g
    pos_pred = pos + 0.5 * dt * (vel + vel_pred)
    return pos_pred, vel_pred


@lru_cache(maxsize=8)
def empty_sdf_tensors(device: str):
    return (torch.zeros((2, 2, 2), device=device, dtype=torch.float32), (0.0, 0.0, 0.0), 1.0)


@wp.func
def cylinder_surface(local: wp.vec3, radius: float, half_height: float):
    """Signed distance and outward normal of a capped cylinder along local z (smooth backward Jacobian)."""
    radial = wp.sqrt(local[0] * local[0] + local[1] * local[1] + 1.0e-8)
    q = wp.vec2(radial - radius, wp.abs(local[2]) - half_height)
    outside = wp.vec2(wp.max(q[0], 0.0), wp.max(q[1], 0.0))
    out_sq = wp.dot(outside, outside)
    length = wp.sqrt(out_sq + 1.0e-8)
    distance = (length - 1.0e-4) + wp.min(wp.max(q[0], q[1]), 0.0)
    direction = wp.vec2(1.0, 0.0)
    if radial > 1.0e-4:
        direction = wp.vec2(local[0], local[1]) / radial
    weights = wp.vec2(1.0, 0.0)
    if out_sq > 1.0e-8:
        weights = outside / length
    elif q[1] > q[0]:
        weights = wp.vec2(0.0, 1.0)
    sign = float(1.0)
    if local[2] < 0.0:
        sign = -1.0
    return distance, wp.vec3(direction[0] * weights[0], direction[1] * weights[0], sign * weights[1])


@wp.func
def ball_to_cube_radial(d: wp.vec3, inv_radius: float) -> wp.vec3:
    """Volume-preserving ball-to-cube mapping (Ummenhofer et al. ContinuousConv) in [-1, 1]^3."""
    s = d * inv_radius
    r2 = wp.dot(s, s)
    if r2 <= 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)
    r = wp.sqrt(r2)
    ax = wp.abs(s[0])
    ay = wp.abs(s[1])
    az = wp.abs(s[2])
    max_c = wp.max(ax, wp.max(ay, az))
    if max_c <= 1.0e-8:
        return wp.vec3(0.0, 0.0, 0.0)
    # Concentric ray stretching with volume-preserving radial blend
    ray = s * (r / max_c)
    return ray


@wp.func
def poly6_window(r_norm_sq: float) -> float:
    """Cubic poly6 window (1 - (r/R)^2)^3 for r <= R."""
    if r_norm_sq >= 1.0:
        return 0.0
    t = 1.0 - r_norm_sq
    return t * t * t


@wp.kernel
def particle_features_kernel(
    history: wp.array3d(dtype=float),
    material: wp.array(dtype=float),
    dt: float,
    origin: wp.vec3,
    gravity: wp.vec3,
    nodes: wp.array2d(dtype=float),
):
    i = wp.tid()
    h = history.shape[1]
    for a in range(3):
        nodes[i, a] = history[i, h - 1, a] - origin[a]
    for t in range(h - 1):
        for a in range(3):
            nodes[i, 3 + t * 3 + a] = (history[i, t + 1, a] - history[i, t, a]) / dt
    for k in range(12):
        nodes[i, 3 * h + k] = material[k]
    nodes[i, 3 * h + 12] = dt / 0.02
    for a in range(3):
        nodes[i, 3 * h + 13 + a] = gravity[a] / 9.81


@wp.kernel
def collider_features_kernel(
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
    normal = wp.vec3(0.0, 0.0, 1.0)
    distance = local[2]
    if kinds[j] == 1:
        q = wp.vec3(wp.abs(local[0]) - sizes[j, 0], wp.abs(local[1]) - sizes[j, 1], wp.abs(local[2]) - sizes[j, 2])
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
    elif kinds[j] == 2:
        length = wp.sqrt(wp.dot(local, local) + 1.0e-8)
        distance = length - sizes[j, 0]
        if length > 1.0e-4:
            normal = local / length
    elif kinds[j] == 5:
        z_cap = wp.clamp(local[2], -sizes[j, 1], sizes[j, 1])
        diff = wp.vec3(local[0], local[1], local[2] - z_cap)
        length = wp.sqrt(wp.dot(diff, diff) + 1.0e-8)
        distance = length - sizes[j, 0]
        if length > 1.0e-4:
            normal = diff / length
    elif kinds[j] == 4:
        distance, normal = cylinder_surface(local, sizes[j, 0], sizes[j, 2])
    normal = wp.quat_rotate(rotation, normal)
    surface = x - distance * normal
    linear = wp.vec3(velocities[j, 0], velocities[j, 1], velocities[j, 2])
    angular = wp.vec3(velocities[j, 3], velocities[j, 4], velocities[j, 5])
    relative = linear + wp.cross(angular, surface - center) - (x - previous) / dt
    gap = distance - 0.02 * material[11]
    gap_s = gap / support
    if far_gap_weight > 0.0 or far_influence_weight > 0.0:
        inv_far = 1.0 / wp.max(far_support_scale, 1.0)
        near_gap_w = 1.0 - far_gap_weight
        pairs[i, j, 0] = near_gap_w * wp.tanh(gap_s) + far_gap_weight * wp.tanh(inv_far * gap_s)
        g_pos = wp.max(gap_s, 0.0)
        th_short = wp.tanh(g_pos)
        th_long = wp.tanh(inv_far * g_pos)
        near_inf_w = 1.0 - far_influence_weight
        influence[i, j] = near_inf_w * (1.0 - th_short * th_short) + far_influence_weight * (1.0 - th_long * th_long)
    else:
        pairs[i, j, 0] = wp.clamp(gap_s, -1.0, 1.0)
        influence[i, j] = wp.max(0.0, 1.0 - wp.max(gap, 0.0) / support)
    for a in range(3):
        pairs[i, j, 1 + a] = normal[a]
        pairs[i, j, 4 + a] = relative[a]
    pairs[i, j, 7] = friction[j]


class _DifferentiableFeatures(torch.autograd.Function):
    """Backpropagate through Warp particle and collider geometry to predicted particle history."""

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
        origin,
        gravity,
    ):
        n, h = history.shape[:2]
        device = str(history.device)
        grad = wp.zeros(history.shape, dtype=float, device=device) if history.requires_grad else None
        hist = wp.from_torch(history, requires_grad=history.requires_grad, grad=grad)
        nodes = wp.empty((n, 16 + 3 * h), dtype=float, device=device, requires_grad=history.requires_grad)
        pairs = wp.empty((n, len(poses), 8), dtype=float, device=device, requires_grad=history.requires_grad)
        influence = wp.empty((n, len(poses)), dtype=float, device=device, requires_grad=history.requires_grad)
        tape = wp.Tape()
        torch_stream = torch.cuda.current_stream(history.device)
        warp_stream = wp.get_stream(device)
        if warp_stream.cuda_stream != torch_stream.cuda_stream:
            warp_stream = wp.stream_from_torch(torch_stream)
        with wp.ScopedStream(warp_stream), tape:
            wp.launch(
                particle_features_kernel,
                n,
                [hist, wp.from_torch(material), dt, wp.vec3(*origin), wp.vec3(*gravity), nodes],
                device=device,
            )
            if len(poses):
                wp.launch(
                    collider_features_kernel,
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
        return (gradient,) + (None,) * 13


def features_differentiable(
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
    far_gap_weight: float = 0.0,
    far_influence_weight: float = 0.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute particle and collider features on GPU (differentiable w.r.t. `history`)."""
    if not history.requires_grad:
        n, h = history.shape[:2]
        device = str(history.device)
        nodes_t = torch.empty((n, 16 + 3 * h), dtype=torch.float32, device=history.device)
        pairs_t = torch.empty((n, len(poses), 8), dtype=torch.float32, device=history.device)
        inf_t = torch.empty((n, len(poses)), dtype=torch.float32, device=history.device)
        torch_stream = torch.cuda.current_stream(history.device)
        warp_stream = wp.get_stream(device)
        if warp_stream.cuda_stream != torch_stream.cuda_stream:
            warp_stream = wp.stream_from_torch(torch_stream)
        with wp.ScopedStream(warp_stream):
            wp.launch(
                particle_features_kernel,
                n,
                [
                    wp.from_torch(history, requires_grad=False),
                    wp.from_torch(material, requires_grad=False),
                    dt,
                    wp.vec3(*origin),
                    wp.vec3(*gravity),
                    wp.from_torch(nodes_t, requires_grad=False),
                ],
                device=device,
            )
            if len(poses):
                wp.launch(
                    collider_features_kernel,
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
                        wp.from_torch(pairs_t, requires_grad=False),
                        wp.from_torch(inf_t, requires_grad=False),
                    ],
                    device=device,
                )
        return nodes_t, pairs_t, inf_t
    return _DifferentiableFeatures.apply(
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
        origin,
        gravity,
    )


@wp.kernel
def cconv_fluid_basis_kernel(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    radius: float,
    max_neighbors: int,
    particles_per_env: int,
    basis_out: wp.array3d(dtype=float),
    packing_out: wp.array2d(dtype=float),
):
    """Accumulate [4, 4, 4] continuous-convolution trilinear basis and local packing over HashGrid neighbors."""
    i = wp.tid()
    x_i = pos[i]
    v_i = vel[i]
    inv_r = 1.0 / radius
    r_sq = radius * radius

    group = (i // particles_per_env) if particles_per_env > 0 else -1
    query = wp.hash_grid_query(grid, x_i, radius, group)
    neighbor = int(0)
    count = int(0)
    w_sum = float(0.0)
    v_rel_sum = wp.vec3(0.0, 0.0, 0.0)

    while wp.hash_grid_query_next(query, neighbor):
        if neighbor == i:
            continue
        d = pos[neighbor] - x_i
        dist_sq = wp.dot(d, d)
        if dist_sq < r_sq:
            w = poly6_window(dist_sq * inv_r * inv_r)
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

            dv = vel[neighbor] - v_i
            w_sum += w
            v_rel_sum += w * dv

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
    inv_w = 1.0 / wp.max(w_sum, 1.0e-4)
    packing_out[i, 2] = v_rel_sum[0] * inv_w
    packing_out[i, 3] = v_rel_sum[1] * inv_w
    packing_out[i, 4] = v_rel_sum[2] * inv_w


@wp.kernel
def cconv_obstacle_basis_kernel(
    pairs: wp.array3d(dtype=float),
    influence: wp.array2d(dtype=float),
    obs_basis_out: wp.array3d(dtype=float),
):
    """Continuous 3D convolution [4, 4, 4] over analytical collider surface points within `support`."""
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


class ContinuousConvWarp(nn.Module):
    """Warp HashGrid Continuous Convolution layer (`[4, 4, 4]` trilinear filter grid)."""

    def __init__(
        self,
        fluid_in_channels: int = 4,
        obs_in_channels: int = 9,
        filters: int = 64,
        kernel_size: tuple[int, int, int] = (4, 4, 4),
        radius: float = 0.072,
        max_neighbors: int = 48,
    ):
        super().__init__()
        if kernel_size != (4, 4, 4):
            raise ValueError("ContinuousConvWarp uses a 4x4x4 trilinear filter cube")
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

        self._grids: dict[tuple[str, int], wp.HashGrid] = {}
        self._group_arrays: dict[tuple[str, int, int], wp.array] = {}
        self._eval_buffers: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _get_grid(self, device: str, n: int) -> wp.HashGrid:
        key = (device, n)
        grid = self._grids.get(key)
        if grid is None:
            grid = wp.HashGrid(64, 64, 64, device=device)
            grid.reserve(n)
            self._grids[key] = grid
        return grid

    def _get_groups(
        self, device: str, num_envs: int, m_pts: int, torch_device: torch.device
    ) -> tuple[torch.Tensor, wp.array]:
        key = (device, num_envs, m_pts)
        grp = self._group_arrays.get(key)
        if grp is None:
            env_ids = torch.arange(num_envs, device=torch_device, dtype=torch.int32).repeat_interleave(m_pts)
            grp = (env_ids, wp.from_torch(env_ids, dtype=wp.int32, requires_grad=False))
            self._group_arrays[key] = grp
        return grp

    @torch.no_grad()
    def compute_bases(
        self,
        pos_pred: torch.Tensor,
        vel_pred: torch.Tensor,
        pairs: torch.Tensor,
        influence: torch.Tensor,
        center_positions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute fluid [N, 64, 4], obstacle [N, 64, 9], and local packing [N, 5] tensors on GPU."""
        n = pos_pred.shape[0]
        device = str(pos_pred.device)

        pos_f = pos_pred.detach().float()
        pos_c = (pos_f - pos_f.mean(dim=0, keepdim=True)).contiguous() if center_positions else pos_f.contiguous()
        vel_c = vel_pred.detach().float().contiguous()
        pairs_c = pairs.detach().float().contiguous()
        inf_c = influence.detach().float().contiguous()

        if not self.training:
            buf_key = (device, n)
            bufs = self._eval_buffers.get(buf_key)
            if bufs is None:
                fluid_basis = torch.zeros((n, 64, self.fluid_in_channels), dtype=torch.float32, device=pos_pred.device)
                obs_basis = torch.zeros((n, 64, self.obs_in_channels), dtype=torch.float32, device=pos_pred.device)
                packing = torch.zeros((n, 5), dtype=torch.float32, device=pos_pred.device)
                self._eval_buffers[buf_key] = (fluid_basis, obs_basis, packing)
            else:
                fluid_basis, obs_basis, packing = bufs
                fluid_basis.zero_()
                obs_basis.zero_()
                packing.zero_()
        else:
            fluid_basis = torch.zeros((n, 64, self.fluid_in_channels), dtype=torch.float32, device=pos_pred.device)
            obs_basis = torch.zeros((n, 64, self.obs_in_channels), dtype=torch.float32, device=pos_pred.device)
            packing = torch.zeros((n, 5), dtype=torch.float32, device=pos_pred.device)

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
                cconv_fluid_basis_kernel,
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
        center_positions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-launch grouped HashGrid continuous-convolution bases for [E, M, 3] particles."""
        num_envs, m_pts, _ = pos_pred_env.shape
        n_tot = num_envs * m_pts
        n_col = pairs_env.shape[2]
        device = str(pos_pred_env.device)

        pos_f = pos_pred_env.detach().float()
        if center_positions:
            pos_f = pos_f - pos_f.mean(dim=1, keepdim=True)
        pos_c = pos_f.reshape(n_tot, 3).contiguous()
        vel_c = vel_pred_env.detach().float().reshape(n_tot, 3).contiguous()
        pairs_c = pairs_env.detach().float().reshape(n_tot, n_col, 8).contiguous()
        inf_c = influence_env.detach().float().reshape(n_tot, n_col).contiguous()

        buf_key = (device, n_tot)
        bufs = self._eval_buffers.get(buf_key)
        if bufs is None:
            fluid_basis = torch.zeros(
                (n_tot, 64, self.fluid_in_channels), dtype=torch.float32, device=pos_pred_env.device
            )
            obs_basis = torch.zeros((n_tot, 64, self.obs_in_channels), dtype=torch.float32, device=pos_pred_env.device)
            packing = torch.zeros((n_tot, 5), dtype=torch.float32, device=pos_pred_env.device)
            self._eval_buffers[buf_key] = (fluid_basis, obs_basis, packing)
        else:
            fluid_basis, obs_basis, packing = bufs
            fluid_basis.zero_()
            obs_basis.zero_()
            packing.zero_()

        grid = self._get_grid(device, n_tot)
        _, groups_wp = self._get_groups(device, num_envs, m_pts, pos_pred_env.device)
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
                cconv_fluid_basis_kernel,
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
            packing.view(num_envs, m_pts, 5),
        )

    def forward(
        self,
        fluid_basis_flat: torch.Tensor,
        obs_basis_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project continuous-convolution bases with learned [4, 4, 4, C_in, C_out] filters."""
        dt = self.fluid_linear.weight.dtype
        conv_fluid = self.fluid_linear(fluid_basis_flat.to(dt))
        conv_obs = self.obs_linear(obs_basis_flat.to(dt))
        return conv_fluid, conv_obs


def features_world_particle(
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
    far_gap_weight: float = 0.0,
    far_influence_weight: float = 0.0,
    use_actual_velocity: bool = False,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    sdf: tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute explicit predictor states (pos_pred, vel_pred) and differentiable Warp particle/collider features."""
    del sdf
    pos_cur = history[:, -1]
    vel_cur = (history[:, -1] - history[:, -2]) / dt
    pos_pred, vel_pred = integrate_pos_vel(pos_cur, vel_cur, gravity=gravity, dt=dt)
    history_feat = (
        history if use_actual_velocity else torch.cat([history[:, 1:], pos_pred.unsqueeze(1)], dim=1).contiguous()
    )
    nodes, pairs, influence = features_differentiable(
        history_feat,
        poses,
        velocities,
        kinds,
        sizes,
        friction,
        material,
        dt=dt,
        support=support,
        far_support_scale=far_support_scale,
        far_gap_weight=far_gap_weight,
        far_influence_weight=far_influence_weight,
        origin=origin,
        gravity=gravity,
    )
    return pos_pred, vel_pred, nodes, pairs, influence


@wp.kernel
def particle_features_batched_kernel(
    history: wp.array3d(dtype=float),
    material: wp.array2d(dtype=float),
    particles_per_env: int,
    dt: float,
    origin: wp.vec3,
    gravity: wp.vec3,
    nodes: wp.array2d(dtype=float),
):
    i = wp.tid()
    env = i // particles_per_env
    h = history.shape[1]
    for a in range(3):
        nodes[i, a] = history[i, h - 1, a] - origin[a]
    for t in range(h - 1):
        for a in range(3):
            nodes[i, 3 + t * 3 + a] = (history[i, t + 1, a] - history[i, t, a]) / dt
    for k in range(12):
        nodes[i, 3 * h + k] = material[env, k]
    nodes[i, 3 * h + 12] = dt / 0.02
    for a in range(3):
        nodes[i, 3 * h + 13 + a] = gravity[a] / 9.81


@wp.kernel
def collider_features_batched_kernel(
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
    normal = wp.vec3(0.0, 0.0, 1.0)
    distance = local[2]
    if kinds[j] == 1:
        q = wp.vec3(
            wp.abs(local[0]) - sizes[env, j, 0],
            wp.abs(local[1]) - sizes[env, j, 1],
            wp.abs(local[2]) - sizes[env, j, 2],
        )
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
    elif kinds[j] == 2:
        length = wp.sqrt(wp.dot(local, local) + 1.0e-8)
        distance = length - sizes[env, j, 0]
        if length > 1.0e-4:
            normal = local / length
    elif kinds[j] == 4:
        distance, normal = cylinder_surface(local, sizes[env, j, 0], sizes[env, j, 2])
    normal = wp.quat_rotate(rotation, normal)
    surface = x - distance * normal
    linear = wp.vec3(velocities[env, j, 0], velocities[env, j, 1], velocities[env, j, 2])
    angular = wp.vec3(velocities[env, j, 3], velocities[env, j, 4], velocities[env, j, 5])
    relative = linear + wp.cross(angular, surface - center) - (x - previous) / dt
    gap = distance - 0.02 * material[env, 11]
    gap_s = gap / support
    if far_gap_weight > 0.0 or far_influence_weight > 0.0:
        inv_far = 1.0 / wp.max(far_support_scale, 1.0)
        near_gap_w = 1.0 - far_gap_weight
        pairs[i, j, 0] = near_gap_w * wp.tanh(gap_s) + far_gap_weight * wp.tanh(inv_far * gap_s)
        g_pos = wp.max(gap_s, 0.0)
        th_short = wp.tanh(g_pos)
        th_long = wp.tanh(inv_far * g_pos)
        near_inf_w = 1.0 - far_influence_weight
        influence[i, j] = near_inf_w * (1.0 - th_short * th_short) + far_influence_weight * (1.0 - th_long * th_long)
    else:
        pairs[i, j, 0] = wp.clamp(gap_s, -1.0, 1.0)
        influence[i, j] = wp.max(0.0, 1.0 - wp.max(gap, 0.0) / support)
    for a in range(3):
        pairs[i, j, 1 + a] = normal[a]
        pairs[i, j, 4 + a] = relative[a]
    pairs[i, j, 7] = friction[env, j]


def features_world_particle_batched(
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
    far_gap_weight: float = 0.0,
    far_influence_weight: float = 0.0,
    use_actual_velocity: bool = False,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-launch batched predictor and collider features for [E, M, H, 3] particle histories."""
    pos_cur = history_env[:, :, -1]
    vel_cur = (history_env[:, :, -1] - history_env[:, :, -2]) / dt
    pos_pred, vel_pred = integrate_pos_vel(pos_cur, vel_cur, gravity=gravity, dt=dt)
    history_feat = (
        history_env
        if use_actual_velocity
        else torch.cat([history_env[:, :, 1:], pos_pred.unsqueeze(2)], dim=2).contiguous()
    )

    num_envs, m_pts, h_len, _ = history_feat.shape
    n_tot = num_envs * m_pts
    n_col = len(kinds)
    device = str(history_feat.device)
    hist_flat = history_feat.reshape(n_tot, h_len, 3).contiguous()
    sizes_env = (sizes.unsqueeze(0).expand(num_envs, -1, -1) if sizes.ndim == 2 else sizes).contiguous()
    friction_env = (friction.unsqueeze(0).expand(num_envs, -1) if friction.ndim == 1 else friction).contiguous()
    material_env = (material.unsqueeze(0).expand(num_envs, -1) if material.ndim == 1 else material).contiguous()
    nodes_t = torch.empty((num_envs, m_pts, 16 + 3 * h_len), dtype=torch.float32, device=history_feat.device)
    pairs_t = torch.empty((num_envs, m_pts, n_col, 8), dtype=torch.float32, device=history_feat.device)
    inf_t = torch.empty((num_envs, m_pts, n_col), dtype=torch.float32, device=history_feat.device)

    torch_stream = torch.cuda.current_stream(history_feat.device)
    warp_stream = wp.get_stream(device)
    if warp_stream.cuda_stream != torch_stream.cuda_stream:
        warp_stream = wp.stream_from_torch(torch_stream)
    with wp.ScopedStream(warp_stream):
        wp.launch(
            particle_features_batched_kernel,
            n_tot,
            [
                wp.from_torch(hist_flat, requires_grad=False),
                wp.from_torch(material_env, requires_grad=False),
                m_pts,
                dt,
                wp.vec3(*origin),
                wp.vec3(*gravity),
                wp.from_torch(nodes_t.view(n_tot, -1), requires_grad=False),
            ],
            device=device,
        )
        if n_col:
            wp.launch(
                collider_features_batched_kernel,
                (n_tot, n_col),
                [
                    wp.from_torch(hist_flat, requires_grad=False),
                    wp.from_torch(poses_env.contiguous(), requires_grad=False),
                    wp.from_torch(velocities_env.contiguous(), requires_grad=False),
                    wp.from_torch(kinds, requires_grad=False),
                    wp.from_torch(sizes_env, requires_grad=False),
                    wp.from_torch(friction_env, requires_grad=False),
                    wp.from_torch(material_env, requires_grad=False),
                    m_pts,
                    dt,
                    support,
                    far_support_scale,
                    far_gap_weight,
                    far_influence_weight,
                    wp.from_torch(pairs_t.view(n_tot, n_col, 8), requires_grad=False),
                    wp.from_torch(inf_t.view(n_tot, n_col), requires_grad=False),
                ],
                device=device,
            )
    return pos_pred, vel_pred, nodes_t, pairs_t, inf_t
