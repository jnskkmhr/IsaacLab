# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent free-flight momentum and direct rigid-floor contact checks."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from newton_rigid import NewtonRigidStepper


def test_applied_material_force_advances_momentum_exactly_once():
    e = dict(
        body_pose=np.array([[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]]),
        body_velocity=np.array([[[0.2, -0.1, 0.3, 0.0, 0.0, 0.0]]]),
        body_mass=np.array([2.0]),
        body_size=np.array([[0.1, 0.1, 0.1]]),
        body_shapes=["box"],
        body_com=np.zeros((1, 3)),
        body_inertia=np.array([np.eye(3) * 0.01]),
        body_friction=np.array([0.6]),
        gravity=np.array([0.0, 0.0, -9.81]),
        dt=0.01,
    )
    stepper = NewtonRigidStepper(e)
    force = np.array([[4.0, 2.0, -1.0, 0.0, 0.0, 0.0]])
    for _ in range(20):
        pose, velocity = stepper.step(force)
    acceleration = np.array([2.0, 1.0, -10.31])
    expected_velocity = np.array([0.2, -0.1, 0.3]) + 0.2 * acceleration
    expected_position = np.array([0.0, 0.0, 1.0]) + 0.2 * np.array([0.2, -0.1, 0.3]) + 0.021 * acceleration
    np.testing.assert_allclose(velocity[0, :3], expected_velocity, atol=2e-6)
    np.testing.assert_allclose(pose[0, :3], expected_position, atol=2e-6)
    np.testing.assert_allclose(stepper.direct_contact_force, 0, atol=5e-5)


def test_free_sphere_is_supported_by_static_rigid_floor():
    e = dict(
        body_pose=np.array([[[0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]]]),
        body_velocity=np.zeros((1, 2, 6)),
        body_mass=np.array([0.0, 1.0]),
        body_size=np.array([[1.0, 1.0, 0.1], [0.1, 0.1, 0.1]]),
        body_shapes=["box", "sphere"],
        body_com=np.zeros((2, 3)),
        body_inertia=np.array([np.zeros((3, 3)), np.eye(3) * 0.001]),
        body_friction=np.array([0.6, 0.6]),
        gravity=np.array([0.0, 0.0, -9.81]),
        dt=0.005,
    )
    stepper = NewtonRigidStepper(e)
    contact_forces = []
    for _ in range(300):
        pose, velocity = stepper.step(np.zeros((2, 6)))
        contact_forces.append(stepper.direct_contact_force[1, 2])
    np.testing.assert_allclose(pose[0], e["body_pose"][0, 0], atol=1e-7)
    assert abs(pose[1, 2] - 0.05) < 0.001
    assert np.linalg.norm(velocity[1]) < 0.005
    assert abs(np.mean(contact_forces[-50:]) - 9.81) < 0.01
