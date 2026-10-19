# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid geometry queries shared by particle learning and force reconstruction.

Signed distance is positive outside a body. Normals point out of the body.
All poses use xyzw quaternions and all returned vectors use world axes.
Shape names select geometric evaluation only; they are not learned categories.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from particle_nerd.force_estimation.reconstruction import box_contacts


def body_contacts(position, pose, size, shape="box", sdf=None):
    """Return distance, outward normal, and nearest surface point for each sample."""
    position, pose, size = (np.asarray(a, dtype=float) for a in (position, pose, size))
    if sdf is None and shape == "box":
        return box_contacts(position, pose, size)
    rotation = Rotation.from_quat(pose[3:]).as_matrix()
    local = (position - pose[:3]) @ rotation
    if sdf is not None:
        distance, normal = sdf.query(local)
    elif shape == "sphere":
        length = np.linalg.norm(local, axis=1)
        distance = length - size[0] / 2
        normal = local / np.maximum(length[:, None], 1e-15)
        normal[length < 1e-15] = [1, 0, 0]
    elif shape == "cylinder":
        radial = np.linalg.norm(local[:, :2], axis=1)
        direction = local[:, :2] / np.maximum(radial[:, None], 1e-15)
        direction[radial < 1e-15] = [1, 0]
        q = np.c_[radial - size[0] / 2, np.abs(local[:, 2]) - size[2] / 2]
        positive = np.maximum(q, 0)
        length = np.linalg.norm(positive, axis=1)
        distance = length + np.minimum(np.max(q, axis=1), 0)
        weights = positive / np.maximum(length[:, None], 1e-15)
        inside = length < 1e-15
        weights[inside] = np.eye(2)[np.argmax(q[inside], axis=1)]
        normal = np.c_[direction * weights[:, :1], np.where(local[:, 2] >= 0, 1, -1) * weights[:, 1]]
    else:
        raise ValueError(f"Unsupported rigid geometry: {shape}")
    world_normal = normal @ rotation.T
    return distance, world_normal, position - distance[:, None] * world_normal


@dataclass
class SampledSDF:
    """Local signed-distance samples with trilinear distance/gradient interpolation.

    Outside the sampled volume, queries raise instead of silently clamping.
    Closest points are a first-order approximation to the sampled isosurface.
    """

    values: np.ndarray
    origin: np.ndarray
    spacing: float

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.origin = np.asarray(self.origin, dtype=float)
        if self.values.ndim != 3 or min(self.values.shape) < 3 or self.spacing <= 0:
            raise ValueError("SDF requires a 3D volume and positive spacing.")
        self.gradient = np.stack(np.gradient(self.values, self.spacing), axis=-1)

    def query(self, local):
        coordinate = (local - self.origin) / self.spacing
        if np.any(coordinate < 0) or np.any(coordinate > np.asarray(self.values.shape) - 1):
            raise ValueError("Rigid SDF query lies outside the sampled volume.")
        distance = map_coordinates(self.values, coordinate.T, order=1, prefilter=False)
        gradient = np.stack(
            [map_coordinates(self.gradient[..., i], coordinate.T, order=1, prefilter=False) for i in range(3)], axis=-1
        )
        norm = np.linalg.norm(gradient, axis=1)
        normal = gradient / np.maximum(norm[:, None], 1e-15)
        normal[norm < 1e-15] = [1, 0, 0]
        return distance, normal

    @classmethod
    def from_mesh(cls, vertices, triangles, spacing=0.005, padding=0.1, device="cpu"):
        """Sample a closed local triangle mesh once; reuse the field at inference."""
        from particle_nerd.force_estimation.mesh_sdf import sample_closed_mesh

        return cls(*sample_closed_mesh(vertices, triangles, spacing, padding, device))

    @classmethod
    def from_primitive(cls, shape, size, spacing=0.005, padding=0.5):
        """Sample an analytic shape; useful for independently checking interpolation error."""
        size = np.asarray(size)
        lower = -size / 2 - padding
        shape_grid = np.ceil((size + 2 * padding) / spacing).astype(int) + 1
        coordinates = [lower[i] + np.arange(shape_grid[i]) * spacing for i in range(3)]
        points = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1)
        distance, _, _ = body_contacts(points.reshape(-1, 3), [0, 0, 0, 0, 0, 0, 1], size, shape)
        return cls(distance.reshape(tuple(shape_grid)), lower, spacing)


def body_contacts_torch(position, pose, size, shape="box"):
    """Differentiable analytic query for rollout training; same geometric convention."""
    import torch

    rotation = torch.as_tensor(Rotation.from_quat(pose[3:]).as_matrix(), dtype=position.dtype, device=position.device)
    origin = torch.as_tensor(pose[:3], dtype=position.dtype, device=position.device)
    half = torch.as_tensor(np.asarray(size) / 2, dtype=position.dtype, device=position.device)
    local = (position - origin) @ rotation
    if shape == "box":
        q = local.abs() - half
        positive = q.clamp_min(0)
        length = positive.norm(dim=-1)
        distance = length + q.max(dim=-1).values.clamp_max(0)
        normal = local.sign() * positive / length[:, None].clamp_min(1e-12)
        face = torch.nn.functional.one_hot(q.argmax(dim=-1), 3).to(position.dtype)
        sign = torch.where(local >= 0, 1.0, -1.0)
        normal = torch.where((length < 1e-12)[:, None], face * sign, normal)
    elif shape == "sphere":
        length = local.norm(dim=-1)
        distance = length - half[0]
        normal = local / length[:, None].clamp_min(1e-12)
        normal = torch.where((length < 1e-12)[:, None], torch.tensor([1.0, 0, 0], device=position.device), normal)
    elif shape == "cylinder":
        radial = local[:, :2].norm(dim=-1)
        direction = local[:, :2] / radial[:, None].clamp_min(1e-12)
        direction = torch.where((radial < 1e-12)[:, None], torch.tensor([1.0, 0], device=position.device), direction)
        q = torch.stack((radial - half[0], local[:, 2].abs() - half[2]), dim=-1)
        positive = q.clamp_min(0)
        length = positive.norm(dim=-1)
        distance = length + q.max(dim=-1).values.clamp_max(0)
        weights = positive / length[:, None].clamp_min(1e-12)
        weights = torch.where(
            (length < 1e-12)[:, None], torch.nn.functional.one_hot(q.argmax(dim=-1), 2).to(position.dtype), weights
        )
        normal = torch.cat(
            (direction * weights[:, :1], torch.where(local[:, 2:] >= 0, 1.0, -1.0) * weights[:, 1:]), dim=-1
        )
    else:
        raise ValueError(f"Unsupported rigid geometry: {shape}")
    world_normal = normal @ rotation.T
    return distance, world_normal, position - distance[:, None] * world_normal
