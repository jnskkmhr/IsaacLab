# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent signed-distance, frame, and sampled-surface checks."""

import numpy as np
from scipy.spatial.transform import Rotation

from particle_nerd.force_estimation.geometry import SampledSDF, body_contacts


def test_rotated_cylinder_side_cap_corner_and_interior():
    # A 90-degree rotation maps the cylinder axis onto world x. The cases
    # distinguish side, cap, rounded distance outside the rim, and interior.
    rotation = Rotation.from_euler("y", 90, degrees=True)
    matrix = rotation.as_matrix()
    local = np.array([[2.0, 0, 0], [0, 0, 3], [2, 0, 3], [0.5, 0, 0]])
    origin = np.array([0.3, -0.4, 0.8])
    expected_normal = np.array([[1.0, 0, 0], [0, 0, 1], [1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [1, 0, 0]])
    distance, normal, closest = body_contacts(
        local @ matrix.T + origin, np.r_[origin, rotation.as_quat()], [2, 2, 4], "cylinder"
    )
    np.testing.assert_allclose(distance, [1, 1, np.sqrt(2), -0.5], atol=1e-12)
    np.testing.assert_allclose(normal, expected_normal @ matrix.T, atol=1e-12)
    np.testing.assert_allclose((closest - origin) @ matrix, [[1, 0, 0], [0, 0, 2], [1, 0, 2], [1, 0, 0]], atol=1e-12)


def test_sampled_sphere_distance_and_normal_converge_under_refinement():
    rng = np.random.default_rng(72)
    normal = rng.normal(size=(100, 3))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    radius = rng.uniform(0.06, 0.14, 100)
    points = radius[:, None] * normal
    errors = []
    for spacing in (0.02, 0.01):
        sdf = SampledSDF.from_primitive("sphere", [0.2, 0.2, 0.2], spacing, 0.06)
        distance, predicted_normal = sdf.query(points)
        errors.append(np.sqrt(np.mean((distance - (radius - 0.1)) ** 2)))
        if spacing == 0.01:
            np.testing.assert_allclose(predicted_normal, normal, atol=0.012)
    assert errors[1] < errors[0] * 0.4
    assert errors[1] < 0.0005


def test_differentiable_geometry_matches_independent_queries_and_gradients():
    import torch

    from particle_nerd.force_estimation.geometry import body_contacts_torch

    rotation = Rotation.from_euler("xyz", [0.3, -0.4, 0.2])
    pose = np.r_[[0.1, 0.2, -0.1], rotation.as_quat()]
    points = np.array([[0.8, 0.4, 0.3], [0.2, 0.8, -0.3], [0.4, 0.3, 0.9]])
    for shape in ("box", "sphere", "cylinder"):
        position = torch.tensor(points, dtype=torch.float64, requires_grad=True)
        expected = body_contacts(points, pose, [0.6, 0.6, 0.8], shape)
        actual = body_contacts_torch(position, pose, [0.6, 0.6, 0.8], shape)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a.detach().numpy(), b, atol=1e-12)
        assert torch.autograd.gradcheck(lambda x: body_contacts_torch(x, pose, [0.6, 0.6, 0.8], shape), (position,))


def test_mesh_sdf_preserves_a_concave_notch_and_interior_sign():
    """A closed L-shaped prism must retain its notch, not become its convex hull."""
    polygon = np.array([[-1, -1], [1, -1], [1, 0], [0, 0], [0, 1], [-1, 1]]) * 0.05
    vertices = np.concatenate([np.c_[polygon, np.full(6, z)] for z in (-0.025, 0.025)])
    cap = np.array([[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]])
    triangles = [*cap[:, ::-1], *(cap + 6)]
    for i in range(6):
        j = (i + 1) % 6
        triangles.extend([[i, j, j + 6], [i, j + 6, i + 6]])
    sdf = SampledSDF.from_mesh(vertices, np.array(triangles), spacing=0.005, padding=0.04)
    points = np.array([[0.015, 0.03, 0], [0.02, -0.015, 0], [-0.03, 0.02, 0], [-0.075, 0.02, 0]])
    distance, normal = sdf.query(points)
    np.testing.assert_allclose(distance, [0.015, -0.015, -0.02, 0.025], atol=1e-7)
    np.testing.assert_allclose(normal[[0, 3]], [[1, 0, 0], [-1, 0, 0]], atol=1e-5)
