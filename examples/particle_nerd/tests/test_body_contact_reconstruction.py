# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent collision momentum and free-flight checks for body-contact reconstruction."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from body_contact_reconstruction import reconstruct_body_contact


@pytest.mark.parametrize("shape", ["box", "sphere", "cylinder"])
def test_normal_impact_has_reduced_mass_impulse_and_action_reaction(shape):
    result = reconstruct_body_contact(
        np.array([[0, 0, 0.0]]),
        np.zeros((1, 3)),
        np.array([1.0]),
        np.array([0, 0, 0.05, 0, 0, 0, 1.0]),
        np.array([0, 0, -1.0, 0, 0, 0]),
        2.0,
        np.eye(3),
        np.array([0.1, 0.1, 0.1]),
        dt=0.01,
        contact_offset=0,
        effective_mass_scale=1,
        body_shape=shape,
    )
    # Perfectly inelastic collision: impulse = reduced mass * closing speed.
    expected_impulse = (2 * 1 / (2 + 1)) * (1 + 9.81 * 0.01)
    assert result.converged
    np.testing.assert_allclose(result.body_wrench, [0, 0, expected_impulse / 0.01, 0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(result.particle_impulse.sum(axis=0), [0, 0, -expected_impulse], atol=1e-7)


def test_no_contact_preserves_free_flight():
    result = reconstruct_body_contact(
        np.array([[0, 0, -2.0]]),
        np.zeros((1, 3)),
        np.array([1.0]),
        np.array([0, 0, 0.05, 0, 0, 0, 1.0]),
        np.zeros(6),
        2.0,
        np.eye(3),
        np.array([0.1, 0.1, 0.1]),
        dt=0.01,
    )
    np.testing.assert_allclose(result.body_wrench, 0, atol=1e-10)
    np.testing.assert_allclose(result.body_velocity, [0, 0, -0.0981, 0, 0, 0], atol=1e-10)


def test_sliding_impact_obeys_coulomb_limit_and_contact_moment():
    result = reconstruct_body_contact(
        np.array([[0, 0, 0.0]]),
        np.zeros((1, 3)),
        np.array([1.0]),
        np.array([0, 0, 0.05, 0, 0, 0, 1.0]),
        np.array([10.0, 0, -1.0, 0, 0, 0]),
        2.0,
        np.eye(3),
        np.array([0.1, 0.1, 0.1]),
        dt=0.01,
        gravity=(0, 0, 0),
        contact_offset=0,
        effective_mass_scale=1,
        friction=0.5,
    )
    normal_impulse = 2 / 3
    tangent_impulse = -0.5 * normal_impulse
    expected = np.array([tangent_impulse, 0, normal_impulse, 0, -0.05 * tangent_impulse, 0])
    assert result.converged
    np.testing.assert_allclose(result.body_wrench * 0.01, expected, atol=1e-7)
    np.testing.assert_allclose(result.particle_impulse.sum(axis=0), -expected[:3], atol=1e-7)


def test_affine_material_velocity_is_reproduced_at_a_displaced_surface():
    from body_contact_reconstruction import estimate_velocity_gradient

    positions = np.array([[x, y, z] for x in (-0.03, 0.03) for y in (-0.03, 0.03) for z in (-0.03, 0.03)])
    gradient = np.array([[1.0, 2.0, -0.5], [-2.0, 0.0, 0.2], [0.1, -0.2, 0.5]])
    velocity = positions @ gradient + np.array([0.3, -0.1, 0.4])
    fitted = estimate_velocity_gradient(positions, velocity, radius=0.15, regularization=0)
    displacement = np.array([0.005, -0.01, 0.008])
    predicted = velocity + np.einsum("i,nij->nj", displacement, fitted)
    expected = (positions + displacement) @ gradient + np.array([0.3, -0.1, 0.4])
    np.testing.assert_allclose(predicted, expected, atol=1e-8)


def test_penetration_stabilization_adds_a_separating_velocity_target():
    dt, penetration, beta = 0.01, 0.001, 0.01
    result = reconstruct_body_contact(
        np.array([[0, 0, 0.0]]),
        np.zeros((1, 3)),
        np.array([1.0]),
        np.array([0, 0, 0.05 - penetration, 0, 0, 0, 1.0]),
        np.zeros(6),
        2.0,
        np.eye(3),
        np.array([0.1, 0.1, 0.1]),
        dt=dt,
        contact_offset=0,
        effective_mass_scale=1,
        stabilization=beta,
    )
    expected_impulse = (2 / 3) * (9.81 * dt + beta * penetration / dt)
    np.testing.assert_allclose(result.body_wrench[2] * dt, expected_impulse, atol=1e-7)


def test_rotating_the_scene_rotates_the_predicted_wrench():
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_rotvec([0.2, -0.3, 0.4])
    matrix = rotation.as_matrix()
    particle = np.array([[0.02, 0.01, 0]])
    center = np.array([0, 0, 0.05])
    velocity = np.array([1.0, 0.3, -0.2, 0, 0, 0])
    inertia = np.diag([0.003, 0.005, 0.006])
    common = dict(
        body_mass=2.0,
        body_inertia=inertia,
        body_size=np.array([0.1, 0.1, 0.1]),
        dt=0.01,
        gravity=(0, 0, 0),
        contact_offset=0,
        effective_mass_scale=1,
        friction=0.5,
    )
    original = reconstruct_body_contact(
        particle, np.zeros((1, 3)), np.ones(1), np.r_[center, [0, 0, 0, 1]], velocity, **common
    )
    rotated = reconstruct_body_contact(
        particle @ matrix.T,
        np.zeros((1, 3)),
        np.ones(1),
        np.r_[matrix @ center, rotation.as_quat()],
        np.r_[matrix @ velocity[:3], matrix @ velocity[3:]],
        **common,
    )
    assert original.converged and rotated.converged
    np.testing.assert_allclose(
        rotated.body_wrench, np.r_[matrix @ original.body_wrench[:3], matrix @ original.body_wrench[3:]], atol=1e-6
    )


def test_sliding_friction_opposes_contact_slip_with_anisotropic_inertia():
    # Off-center contact creates unequal effective masses in the tangent plane.
    # Coulomb sliding still requires friction to oppose slip, independently of inertia.
    point = np.array([0.5, 0.4, -1.0])
    result = reconstruct_body_contact(
        point[None],
        np.zeros((1, 3)),
        np.array([1.0]),
        np.array([0, 0, 0, 0, 0, 0, 1.0]),
        np.array([2.0, 1.0, -0.2, 0, 0, 0]),
        2.0,
        np.diag([0.3, 0.6, 0.9]),
        np.array([2.0, 2.0, 2.0]),
        dt=0.01,
        gravity=(0, 0, 0),
        contact_offset=0,
        effective_mass_scale=1,
        friction=0.4,
    )
    impulse = -result.particle_impulse[0]
    body_contact_velocity = result.body_velocity[:3] + np.cross(result.body_velocity[3:], point)
    # The compliance represents an opposite impulse on one unit of effective material mass.
    relative_velocity = body_contact_velocity - result.particle_impulse[0]
    assert result.converged
    assert np.dot(impulse[:2], relative_velocity[:2]) < 0
    np.testing.assert_allclose(np.linalg.norm(impulse[:2]), 0.4 * impulse[2], atol=1e-8)
    np.testing.assert_allclose(
        impulse[:2] / np.linalg.norm(impulse[:2]),
        -relative_velocity[:2] / np.linalg.norm(relative_velocity[:2]),
        atol=1e-8,
    )


def test_contact_impulse_history_recovers_static_support_and_clears_on_separation():
    # Finite contact compliance acts on impulse changes. Repeated stationary
    # contact must recover the box weight without measured force initialization.
    previous = np.zeros((1, 3))
    args = (
        np.array([[0.0, 0.0, -1.0]]),
        np.zeros((1, 3)),
        np.array([1.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        np.zeros(6),
        2.0,
        np.eye(3),
        np.array([2.0, 2.0, 2.0]),
    )
    for _ in range(80):
        result = reconstruct_body_contact(
            *args,
            dt=0.01,
            contact_offset=0.0,
            effective_mass_scale=1.0,
            previous_particle_impulse=previous,
            history_weight=1.0,
        )
        previous = result.particle_impulse
    np.testing.assert_allclose(result.body_wrench[:3], [0, 0, 19.62], atol=1e-6)
    separated = reconstruct_body_contact(
        np.array([[0.0, 0.0, -2.0]]),
        *args[1:],
        dt=0.01,
        contact_offset=0.0,
        effective_mass_scale=1.0,
        previous_particle_impulse=previous,
        history_weight=1.0,
    )
    np.testing.assert_allclose(separated.body_wrench, 0.0, atol=1e-10)
    np.testing.assert_allclose(separated.particle_impulse, 0.0, atol=1e-10)


def test_warp_gradient_preserves_irregular_neighborhood_fit():
    """Acceleration must preserve the regularized CPU fit, including isolated samples."""
    import pytest

    wp = pytest.importorskip("warp")
    from body_contact_reconstruction import estimate_velocity_gradient

    rng = np.random.default_rng(182)
    positions = np.r_[rng.uniform(-0.08, 0.08, (300, 3)), [[2.0, 1.0, 0.0]]]
    velocities = rng.normal(size=positions.shape)
    expected = estimate_velocity_gradient(positions, velocities)
    actual = estimate_velocity_gradient(positions, velocities, device=str(wp.get_preferred_device()))
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-8)
    np.testing.assert_array_equal(actual[-1], np.zeros((3, 3)))
