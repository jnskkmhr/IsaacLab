# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical contracts at the extracted GPU force and coupled-runner boundaries."""

from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from particle_nerd.force_estimation.estimator import ContactEstimator
from particle_nerd.runner.simulation import CoupledSimulation


def test_gpu_impact_conserves_momentum_and_obeys_coulomb_friction():
    # Two masses (2 kg body, 1 kg effective particle) closing at 1 m/s:
    # normal impulse = reduced mass * closing speed = 2/3 N s.
    wp.init()
    estimator = ContactEstimator([0.1], 2, np.eye(3), [0.1, 0.1, 0.1], 0.01, gravity=(0, 0, 0), friction=0.5)

    def a(value, dtype):
        return wp.array(value, dtype=dtype, device="cuda:0")

    force = estimator.estimate(
        a([[0, 0, 0]], wp.vec3),
        a([[0, 0, 0]], wp.vec3),
        a([[0, 0, 0.05, 0, 0, 0, 1]], wp.transform),
        a([[10, 0, -1, 0, 0, 0]], wp.spatial_vector),
        a([[0] * 6], wp.spatial_vector),
    ).numpy()[0]
    np.testing.assert_allclose(force * 0.01, [-1 / 3, 0, 2 / 3, 0, 1 / 60, 0], atol=1e-7)
    frame, impulse = estimator.frames.numpy()[0], estimator.impulses.numpy()[0]
    np.testing.assert_allclose(-frame.T @ impulse, -force[:3] * 0.01, atol=1e-7)
    assert estimator.diagnostics()["converged"]


def test_coupled_runner_keeps_free_flight_on_gpu_and_applies_external_force_once(monkeypatch):
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
        predictor, dict(body_size_m=[0.12, 0.12, 0.08], body_mass_kg=2), np.array([0.1]), [0, 0, 1, 0, 0, 0, 1], [0] * 6
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
