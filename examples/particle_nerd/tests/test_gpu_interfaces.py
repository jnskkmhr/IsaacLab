# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical contracts at the extracted GPU force and coupled-runner boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from particle_nerd.force_estimation.estimator import ContactEstimator
from particle_nerd.runner.simulation import CoupledSimulation


@pytest.mark.parametrize(
    "relaxation,solver,friction",
    [
        (0.1, "jacobi", 0.5),
        (0.025, "jacobi", 0.5),
        (0.075, "jacobi", 0.5),
        (0.075, "body_velocity", 0.5),
        (0.075, "body_velocity", 0.0),
    ],
)
def test_gpu_impact_conserves_momentum_and_obeys_coulomb_friction(relaxation, solver, friction):
    # Two masses (2 kg body, 1 kg effective particle) closing at 1 m/s:
    # normal impulse = reduced mass * closing speed = 2/3 N s.
    wp.init()
    estimator = ContactEstimator(
        [0.1],
        2,
        np.eye(3),
        [0.1, 0.1, 0.1],
        0.01,
        gravity=(0, 0, 0),
        friction=friction,
        jacobi_relaxation=relaxation,
        contact_solver=solver,
    )

    def a(value, dtype):
        return wp.array(value, dtype=dtype, device="cuda:0")

    force = estimator.estimate(
        a([[0, 0, 0]], wp.vec3),
        a([[0, 0, 0]], wp.vec3),
        a([[0, 0, 0.05, 0, 0, 0, 1]], wp.transform),
        a([[10, 0, -1, 0, 0, 0]], wp.spatial_vector),
        a([[0] * 6], wp.spatial_vector),
    ).numpy()[0]
    np.testing.assert_allclose(force * 0.01, [-friction * 2 / 3, 0, 2 / 3, 0, friction / 30, 0], atol=1e-7)
    frame, impulse = estimator.frames.numpy()[0], estimator.impulses.numpy()[0]
    np.testing.assert_allclose(-frame.T @ impulse, -force[:3] * 0.01, atol=1e-7)
    assert estimator.diagnostics()["converged"]


@pytest.mark.parametrize("solver", ["jacobi", "body_velocity"])
def test_coupled_runner_keeps_free_flight_on_gpu_and_applies_external_force_once(monkeypatch, solver):
    # An inert particle field far from the box isolates the rigid integration
    # contract from learned-model error. No future box states are supplied.
    predictor = SimpleNamespace(
        device="cuda:0",
        dt=0.02,
        x=torch.tensor([[10.0, 10.0, 10.0]], device="cuda"),
        v=torch.zeros((1, 3), device="cuda"),
    )

    def stationary_particles(pose, velocity):
        return predictor.x, predictor.v

    predictor.step = stationary_particles
    simulation = CoupledSimulation(
        predictor,
        dict(body_size_m=[0.12, 0.12, 0.08], body_mass_kg=2),
        np.array([0.1]),
        [0, 0, 1, 0, 0, 0, 1],
        [0] * 6,
        contact_warm_start=solver == "jacobi",
        contact_solver=solver,
        use_cuda_graph=True,
    )
    np.testing.assert_allclose(simulation.rigid.state.body_q.numpy()[0], [0, 0, 1, 0, 0, 0, 1])
    external = wp.array([[4, 0, 0, 0, 0, 0]], dtype=wp.spatial_vector, device="cuda:0")
    simulation.step(external)  # Compile before the no-host-transfer guard.

    def forbidden(*args, **kwargs):
        raise AssertionError("Simulation step transferred state to the host")

    with monkeypatch.context() as m:
        for name in ["cpu", "numpy", "item"]:
            m.setattr(torch.Tensor, name, forbidden)
        m.setattr(wp.array, "numpy", forbidden)
        for _ in range(4):
            simulation.step(external)
    wp.synchronize()
    np.testing.assert_allclose(simulation.rigid.state.body_qd.numpy()[0, :3], [0.2, 0, -0.981], atol=2e-5)
    # 20 semi-implicit Euler rigid steps of 5 ms.
    np.testing.assert_allclose(
        simulation.rigid.state.body_q.numpy()[0, :3], [2 * 0.005**2 * 210, 0, 1 - 9.81 * 0.005**2 * 210], atol=2e-5
    )
    np.testing.assert_allclose(simulation.estimator.wrench.numpy(), 0, atol=1e-10)


def test_fixed_material_velocity_supports_weight_and_stops_normal_impact():
    # A stationary, central contact must balance gravity and remove downward
    # momentum. Finite effective-particle inertia would underpredict both.
    wp.init()
    mass, dt = 7.5, 0.02
    estimator = ContactEstimator(
        [0.1],
        mass,
        np.eye(3),
        [0.1, 0.1, 0.1],
        dt,
        fixed_material_velocity=True,
        warm_start=True,
        contact_solver="body_velocity",
    )
    position = wp.array([[0, 0, 0]], dtype=wp.vec3, device="cuda:0")
    material_velocity = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
    pose = wp.array([[0, 0, 0.05, 0, 0, 0, 1]], dtype=wp.transform, device="cuda:0")
    velocity = wp.zeros(1, dtype=wp.spatial_vector, device="cuda:0")
    external = wp.zeros(1, dtype=wp.spatial_vector, device="cuda:0")
    for downward_speed in [0.0, 1.0]:
        velocity.assign(np.array([[0, 0, -downward_speed, 0, 0, 0]], dtype=np.float32))
        before = velocity.numpy().copy()
        force = estimator.estimate(position, material_velocity, pose, velocity, external).numpy()[0]
        np.testing.assert_allclose(force, [0, 0, mass * (9.81 + downward_speed / dt), 0, 0, 0], atol=2e-4)
        np.testing.assert_array_equal(velocity.numpy(), before)
        np.testing.assert_array_equal(material_velocity.numpy(), 0)
        assert estimator.diagnostics()["converged"]


def test_warm_start_preserves_impact_under_frame_changes_and_contact_release():
    # Known two-mass impact protects against double-applying the cached impulse,
    # stale tangent frames, and ghost forces after a contact disappears.
    wp.init()
    estimator = ContactEstimator(
        [0.1],
        2,
        np.eye(3),
        [0.1, 0.1, 0.1],
        0.01,
        gravity=(0, 0, 0),
        friction=0.5,
        warm_start=True,
    )
    position = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
    material = wp.zeros_like(position)
    pose = wp.empty(1, dtype=wp.transform, device="cuda:0")
    velocity = wp.empty(1, dtype=wp.spatial_vector, device="cuda:0")
    external = wp.zeros(1, dtype=wp.spatial_vector, device="cuda:0")
    reference = np.array([-1 / 3, 0, 2 / 3, 0, 1 / 60, 0])
    cold_iterations = None
    for angle in [0.0, 0.0, 0.7, 1.8]:
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pose.assign(
            np.array([[*(rotation @ [0, 0, 0.05]), 0, np.sin(angle / 2), 0, np.cos(angle / 2)]], dtype=np.float32)
        )
        velocity.assign(np.array([[*(rotation @ [10, 0, -1]), 0, 0, 0]], dtype=np.float32))
        impulse = estimator.estimate(position, material, pose, velocity, external).numpy()[0] * 0.01
        np.testing.assert_allclose(impulse, np.r_[rotation @ reference[:3], rotation @ reference[3:]], atol=2e-6)
        diagnostics = estimator.diagnostics()
        assert diagnostics["converged"]
        if cold_iterations is None:
            cold_iterations = diagnostics["iterations"]
        elif angle == 0.0:
            assert diagnostics["iterations"] < cold_iterations
    pose.assign(np.array([[0, 0, 10, 0, 0, 0, 1]], dtype=np.float32))
    np.testing.assert_array_equal(estimator.estimate(position, material, pose, velocity, external).numpy(), 0)
    pose.assign(np.array([[0, 0, 0.05, 0, 0, 0, 1]], dtype=np.float32))
    velocity.assign(np.array([[10, 0, -1, 0, 0, 0]], dtype=np.float32))
    for reset in [False, True]:
        if reset:
            estimator.reset()
        impulse = estimator.estimate(position, material, pose, velocity, external).numpy()[0] * 0.01
        np.testing.assert_allclose(impulse, reference, atol=1e-7)
        assert estimator.diagnostics()["iterations"] == cold_iterations


def test_dense_contact_with_rounded_inertia_preserves_sticking_impact():
    # Float32 asset accumulation can leave I_xz != I_zx. The physical tensor is
    # symmetric. At a strict supported tolerance this discrepancy used to send
    # a converged body-velocity solution into divergent Jacobi refinement.
    wp.init()
    inertia = np.array(
        [[0.13237396, 0, 0.13422221], [0, 1.08860445, 0], [0.13422222, 0, 1.17173827]], dtype=np.float32
    ).astype(np.float64)
    count, mass, dt = 534, 20.0, 0.02
    velocity = np.array([0.1, 0.1, -1.0])
    arm = np.array([-0.04, 0.02, -0.05])
    cross = np.array([[0, -arm[2], arm[1]], [arm[2], 0, -arm[0]], [-arm[1], arm[0], 0]])
    # Independent sticking-impact solution: equal final contact velocities,
    # linear/angular momentum balance, and 534 independent 18 kg material masses.
    mobility = np.eye(3) * (1 / mass + 1 / (count * 18)) - cross @ np.linalg.inv((inertia + inertia.T) * 0.5) @ cross
    impulse = -np.linalg.solve(mobility, velocity)
    assert impulse[2] > 0 and np.linalg.norm(impulse[:2]) < 0.5 * impulse[2]
    estimator = ContactEstimator(
        np.full(count, 0.1),
        mass,
        inertia,
        [0.1, 0.1, 0.1],
        dt,
        gravity=(0, 0, 0),
        friction=0.5,
        effective_mass_scale=180,
        jacobi_relaxation=0.075,
        contact_solver="body_velocity",
        tolerance=1e-12,
    )

    def a(value, dtype):
        return wp.array(value, dtype=dtype, device="cuda:0")

    wrench = estimator.estimate(
        a(np.tile([-0.04, 0.02, 0], (count, 1)), wp.vec3),
        a(np.zeros((count, 3)), wp.vec3),
        a([[0, 0, 0.05, 0, 0, 0, 1]], wp.transform),
        a([[*velocity, 0, 0, 0]], wp.spatial_vector),
        a([[0] * 6], wp.spatial_vector),
    ).numpy()[0]
    assert estimator.diagnostics()["converged"]
    np.testing.assert_allclose(wrench * dt, np.r_[impulse, np.cross(arm, impulse)], atol=3e-6)
