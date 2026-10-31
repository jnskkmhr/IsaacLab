# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp particle/collider features, including gradients for rollout training.

Collider kinds: 0 = plane (local z=0), 1 = box, 2 = sphere. Poses are
world xyz + xyzw quaternion. Velocities are world linear then angular,
with linear velocity measured at the collider pose origin.
"""

import numpy as np
import torch
import warp as wp

from particle_nerd.shovel.sdf import surface as sdf_surface
from particle_nerd.shovel.sdf import tensors as sdf_tensors


def material_features(manifest, particle_mass, particle_radius, dt=0.02):
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

# Existing simulation modules disable backward globally; rollout training needs it here.
wp.set_module_options({"enable_backward": True})


@wp.kernel
def particle_features(
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
        nodes[i, a] = history[i, h - 1, a] - origin[a]  # metres / 1 metre
    for t in range(h - 1):
        for a in range(3):
            nodes[i, 3 + t * 3 + a] = (history[i, t + 1, a] - history[i, t, a]) / dt
    for k in range(12):
        nodes[i, 3 * h + k] = material[k]
    nodes[i, 3 * h + 12] = dt / 0.02
    for a in range(3):
        nodes[i, 3 * h + 13 + a] = gravity[a] / 9.81


@wp.kernel
def collider_features(
    history: wp.array3d(dtype=float),
    poses: wp.array2d(dtype=float),
    velocities: wp.array2d(dtype=float),
    kinds: wp.array(dtype=int),
    sizes: wp.array2d(dtype=float),
    friction: wp.array(dtype=float),
    material: wp.array(dtype=float),
    dt: float,
    support: float,
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
    normal = wp.vec3(0.0, 0.0, 1.0)
    distance = local[2]
    if kinds[j] == 1:
        q = wp.vec3(wp.abs(local[0]) - sizes[j, 0], wp.abs(local[1]) - sizes[j, 1], wp.abs(local[2]) - sizes[j, 2])
        outside = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        length = wp.length(outside)
        distance = length + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
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
            if length > 1.0e-8:
                normal[a] = sign * outside[a] / length
            elif a == axis:
                normal[a] = sign
    elif kinds[j] == 2:
        length = wp.length(local)
        distance = length - sizes[j, 0]
        if length > 1.0e-8:
            normal = local / length
    elif kinds[j] == 3:
        distance, normal = sdf_surface(local, sdf, sdf_origin, sdf_spacing)
    normal = wp.quat_rotate(rotation, normal)
    surface = x - distance * normal
    linear = wp.vec3(velocities[j, 0], velocities[j, 1], velocities[j, 2])
    angular = wp.vec3(velocities[j, 3], velocities[j, 4], velocities[j, 5])
    relative = linear + wp.cross(angular, surface - center) - (x - previous) / dt
    gap = distance - 0.02 * material[11]
    pairs[i, j, 0] = wp.clamp(gap / support, -1.0, 1.0)
    for a in range(3):
        pairs[i, j, 1 + a] = normal[a]
        pairs[i, j, 4 + a] = relative[a]
    pairs[i, j, 7] = friction[j]
    influence[i, j] = wp.max(0.0, 1.0 - wp.max(gap, 0.0) / support)


def _launch(
    history,
    poses,
    velocities,
    kinds,
    sizes,
    friction,
    material,
    dt,
    support,
    origin,
    gravity,
    sdf,
    sdf_origin,
    sdf_spacing,
):
    n, h = history.shape[:2]
    device = str(history.device)
    # A private adjoint buffer avoids adding the same gradient through Warp and Torch.
    grad = wp.zeros(history.shape, dtype=float, device=device) if history.requires_grad else None
    hist = wp.from_torch(history, requires_grad=history.requires_grad, grad=grad)
    nodes = wp.empty((n, 16 + 3 * h), dtype=float, device=device, requires_grad=history.requires_grad)
    pairs = wp.empty((n, len(poses), 8), dtype=float, device=device, requires_grad=history.requires_grad)
    influence = wp.empty((n, len(poses)), dtype=float, device=device, requires_grad=history.requires_grad)
    tape = wp.Tape()
    torch_stream = torch.cuda.current_stream(history.device)
    warp_stream = wp.get_stream(str(history.device))
    # Reuse the active Warp stream: destroying an alias can unregister the
    # owning stream before subsequent Warp graph capture.
    if warp_stream.cuda_stream != torch_stream.cuda_stream:
        warp_stream = wp.stream_from_torch(torch_stream)
    with wp.ScopedStream(warp_stream), tape:
        wp.launch(
            particle_features,
            n,
            [hist, wp.from_torch(material), dt, wp.vec3(*origin), wp.vec3(*gravity), nodes],
            device=device,
        )
        if len(poses):
            wp.launch(
                collider_features,
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
                    wp.from_torch(sdf),
                    wp.vec3(*sdf_origin),
                    sdf_spacing,
                    pairs,
                    influence,
                ],
                device=device,
            )
    return hist, (nodes, pairs, influence), tape


class _Features(torch.autograd.Function):
    """Backpropagate through Warp geometry to predicted particle histories."""

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
        origin,
        gravity,
        sdf,
        sdf_origin,
        sdf_spacing,
    ):
        ctx.history, ctx.outputs, ctx.tape = _launch(
            history,
            poses,
            velocities,
            kinds,
            sizes,
            friction,
            material,
            dt,
            support,
            origin,
            gravity,
            sdf,
            sdf_origin,
            sdf_spacing,
        )
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


def features(
    history,
    poses,
    velocities,
    kinds,
    sizes,
    friction,
    material,
    dt=0.02,
    support=0.07,
    origin=(0.0, 0.0, 0.0),
    gravity=(0.0, 0.0, -9.81),
    sdf=None,
):
    """GPU feature tensors; gradients are supported for particle history only."""
    if history.ndim != 3 or history.shape[-1] != 3 or history.shape[1] < 2:
        raise ValueError("Particle history must have shape [N, H>=2, 3]")
    b = len(kinds)
    if poses.shape != (b, 7) or velocities.shape != (b, 6) or sizes.shape != (b, 3) or friction.shape != (b,):
        raise ValueError("Collider poses/velocities/kinds/sizes/friction must share their collider count")
    if dt <= 0 or support <= 0 or material.shape != (13,):
        raise ValueError("Positive timestep/support and the 13-value material vector are required")
    tensors = (history, poses, velocities, kinds, sizes, friction, material)
    if not history.is_cuda or any(t.device != history.device or not t.is_contiguous() for t in tensors):
        raise ValueError("Features require contiguous tensors on the same CUDA device")
    if kinds.dtype != torch.int32 or any(
        t.dtype != torch.float32 for t in (history, poses, velocities, sizes, friction, material)
    ):
        raise ValueError("Collider kinds must be int32; physical inputs must be float32")
    if sdf is None:
        sdf = sdf_tensors(str(history.device), "none")
    return _Features.apply(*tensors, dt, support, origin, gravity, *sdf)
