# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Primitive collider distances for object-scene particle selection and stabilization."""

import torch


def boundary_boxes(lower, upper, device):
    """Build 5 boundary box descriptors (1 floor + 4 walls) for intrinsic bed bounds [lower, upper]."""
    x_lo, y_lo, z_lo = float(lower[0]), float(lower[1]), float(lower[2])
    x_hi, y_hi, _z_hi = float(upper[0]), float(upper[1]), float(upper[2])
    lx, ly = x_hi - x_lo, y_hi - y_lo
    cx, cy = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    wall = 0.1
    wall_h = 0.4
    floor_h = 0.1
    poses = torch.tensor(
        [
            [cx, cy, z_lo - 0.5 * floor_h, 0.0, 0.0, 0.0, 1.0],
            [x_hi + 0.5 * wall, cy, z_lo + 0.5 * wall_h, 0.0, 0.0, 0.0, 1.0],
            [x_lo - 0.5 * wall, cy, z_lo + 0.5 * wall_h, 0.0, 0.0, 0.0, 1.0],
            [cx, y_hi + 0.5 * wall, z_lo + 0.5 * wall_h, 0.0, 0.0, 0.0, 1.0],
            [cx, y_lo - 0.5 * wall, z_lo + 0.5 * wall_h, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    sizes = torch.tensor(
        [
            [0.5 * lx + wall, 0.5 * ly + wall, 0.5 * floor_h],
            [0.5 * wall, 0.5 * ly + wall, 0.5 * wall_h],
            [0.5 * wall, 0.5 * ly + wall, 0.5 * wall_h],
            [0.5 * lx + wall, 0.5 * wall, 0.5 * wall_h],
            [0.5 * lx + wall, 0.5 * wall, 0.5 * wall_h],
        ],
        dtype=torch.float32,
        device=device,
    )
    friction = torch.full((5,), 0.85, dtype=torch.float32, device=device)
    return poses, sizes, friction


def primitive_surface(local, sizes, kinds):
    """Return signed distance and outward normal in the collider frame.

    Shape IDs and size conventions match ``features.py``: box=1 (half extents),
    sphere=2 (radius in x), cylinder=4 (radius in x, half height in z).
    The penultimate dimension indexes colliders; preceding dimensions broadcast.
    """
    q = local.abs() - sizes
    outside = q.clamp_min(0.0)
    length = outside.norm(dim=-1)
    distance = length + q.amax(dim=-1).clamp_max(0.0)
    sign = torch.where(local < 0.0, -1.0, 1.0)
    face = torch.zeros_like(q).scatter_(-1, q.argmax(dim=-1, keepdim=True), 1.0)
    normal = torch.where(length[..., None] > 1e-8, sign * outside / length[..., None].clamp_min(1e-8), sign * face)

    radius = local.norm(dim=-1)
    sphere_normal = local / radius[..., None].clamp_min(1e-8)
    fallback = torch.zeros_like(local)
    fallback[..., 0] = 1.0
    sphere_normal = torch.where((radius > 1e-8)[..., None], sphere_normal, fallback)
    distance = torch.where(kinds == 2, radius - sizes[..., 0], distance)
    normal = torch.where((kinds == 2)[..., None], sphere_normal, normal)

    radial = local[..., :2].norm(dim=-1)
    radial_xy = local[..., :2] / radial[..., None].clamp_min(1e-8)
    radial_xy = torch.where((radial > 1e-8)[..., None], radial_xy, fallback[..., :2])
    radial_normal = torch.cat((radial_xy, torch.zeros_like(radial[..., None])), dim=-1)
    cap_normal = torch.zeros_like(local)
    cap_normal[..., 2] = sign[..., 2]
    cq = torch.stack((radial - sizes[..., 0], local[..., 2].abs() - sizes[..., 2]), dim=-1)
    co = cq.clamp_min(0.0)
    cl = co.norm(dim=-1)
    cylinder_distance = cl + cq.amax(dim=-1).clamp_max(0.0)
    cylinder_normal = torch.where(
        (cl > 1e-8)[..., None],
        (co[..., :1] * radial_normal + co[..., 1:] * cap_normal) / cl[..., None].clamp_min(1e-8),
        torch.where((cq[..., 0] > cq[..., 1])[..., None], radial_normal, cap_normal),
    )
    return (
        torch.where(kinds == 4, cylinder_distance, distance),
        torch.where((kinds == 4)[..., None], cylinder_normal, normal),
    )
