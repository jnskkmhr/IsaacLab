# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guard world isolation, graph replay and partial reset at the public runner boundary."""

import numpy as np
import pytest
import torch
from particle_nerd_batch.model import ParticleTransolver
from particle_nerd_batch.simulation import Simulation


@pytest.fixture(scope="module")
def scene():
    torch.manual_seed(12)
    model = ParticleTransolver(hidden=16, heads=2, slices=4, layers=1)
    torch.nn.init.normal_(model.decoder[-1].weight, std=0.005)
    saved = dict(
        architecture="particle_transolver_v1",
        model=model.state_dict(),
        config=dict(model=model.config, dt=0.02, features=dict(support=0.07)),
    )
    manifest = dict(body_size_m=[0.12, 0.12, 0.08], body_mass_kg=5.0, geometry="box", body_contact_friction=0.6)
    grid = (
        np.stack(
            np.meshgrid(
                np.linspace(-0.075, 0.075, 5),
                np.linspace(-0.075, 0.075, 5),
                np.linspace(0.025, 0.075, 3),
                indexing="ij",
            ),
            axis=-1,
        )
        .reshape(-1, 3)
        .astype(np.float32)
    )
    history = np.tile(grid[None, :, None, :], (3, 1, 3, 1))
    history[1, :, :, 0] += 0.01
    material = np.zeros((3, 13), dtype=np.float32)
    material[:, 0] = 1
    material[:, 3] = 0.48
    material[:, 10] = 0.2
    material[:, 11] = 0.5
    poses = np.tile([0, 0, 0.105, 0, 0, 0, 1], (3, 1)).astype(np.float32)
    poses[1, 0] = 0.025
    poses[2, 2] = 0.6  # No contacts: must not inherit another world's compacted IDs.
    initial = dict(
        history=history,
        material=material,
        particle_mass=np.full((3, len(grid)), 0.01, dtype=np.float32),
        body_pose=poses,
        body_velocity=np.zeros((3, 6), dtype=np.float32),
    )
    return saved, manifest, initial


def test_batched_worlds_match_independent_simulations(scene):
    saved, manifest, initial = scene
    batch = Simulation(saved, manifest, **initial, capture=True)
    independent = [
        Simulation(saved, manifest, **{k: v[e : e + 1] for k, v in initial.items()}, capture=False) for e in range(3)
    ]
    external = torch.zeros((3, 6), device="cuda")
    external[1, 0] = 2.0
    for _ in range(4):
        batch.step(external)
        for e, sim in enumerate(independent):
            sim.step(external[e : e + 1])
    for e, sim in enumerate(independent):
        torch.testing.assert_close(batch.positions[e], sim.positions[0], atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(batch.body_pose[e], sim.body_pose[0], atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(batch.contact_wrench[e], sim.contact_wrench[0], atol=0.02, rtol=2e-3)
    health = batch.diagnostics()
    assert health["finite"] and health["converged"] and not health["rigid_overflow"]
    assert health["contacts"][0] > 0 and health["contacts"][2] == 0


def test_graph_replay_and_partial_reset_preserve_other_worlds(scene):
    saved, manifest, initial = scene
    captured = Simulation(saved, manifest, **initial, capture=True, rigid_dt=0.02 / 3)
    eager = Simulation(saved, manifest, **initial, capture=False, rigid_dt=0.02 / 3)
    external = torch.zeros((3, 6), device="cuda")
    external[1, 0] = 3.0
    for _ in range(3):
        captured.step(external)
        eager.step(external)
    before = captured.positions.clone()
    before_pose = captured.body_pose.clone()
    mask = torch.tensor([False, True, False], device="cuda")
    captured.reset(mask)
    eager.reset(mask)
    torch.testing.assert_close(captured.positions[[0, 2]], before[[0, 2]], atol=0, rtol=0)
    torch.testing.assert_close(captured.body_pose[[0, 2]], before_pose[[0, 2]], atol=0, rtol=0)
    torch.testing.assert_close(captured.positions[1], torch.as_tensor(initial["history"][1, :, -1], device="cuda"))
    external[1, 0] = -4.0
    for _ in range(3):
        captured.step(external)
        eager.step(external)
    torch.testing.assert_close(captured.positions, eager.positions, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(captured.body_pose, eager.body_pose, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(captured.contact_wrench, eager.contact_wrench, atol=1e-4, rtol=1e-5)


def test_active_compaction_preserves_world_isolation(scene):
    saved, manifest, initial = scene
    batch = Simulation(saved, manifest, **initial, capture=True, active_particles=32, anchor_particles=16)
    eager = Simulation(saved, manifest, **initial, capture=False, active_particles=32, anchor_particles=16)
    independent = [
        Simulation(
            saved,
            manifest,
            **{k: v[e : e + 1] for k, v in initial.items()},
            capture=False,
            active_particles=32,
            anchor_particles=16,
        )
        for e in range(3)
    ]
    external = torch.zeros((3, 6), device="cuda")
    external[1, 0] = 2.0
    for _ in range(3):
        batch.step(external)
        eager.step(external)
        for e, sim in enumerate(independent):
            sim.step(external[e : e + 1])
    torch.testing.assert_close(batch.positions, eager.positions, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batch.contact_wrench, eager.contact_wrench, atol=1e-4, rtol=1e-4)
    for e, sim in enumerate(independent):
        torch.testing.assert_close(batch.positions[e], sim.positions[0], atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(batch.body_pose[e], sim.body_pose[0], atol=2e-5, rtol=1e-4)
        torch.testing.assert_close(batch.contact_wrench[e], sim.contact_wrench[0], atol=0.02, rtol=2e-3)
    health = batch.diagnostics()
    assert health["finite"] and health["converged"] and health["force_model"] == "analytical"
    assert health["contacts"][0] > 0 and health["contacts"][2] == 0.0


def test_proximity_and_material_heuristic_determines_active_particles(scene):
    saved, manifest, initial = scene
    mat_custom = initial["material"].copy()
    mat_custom[0, 3] = 0.85  # High internal friction -> larger stress-cone reach
    mat_custom[1, 3] = 0.15  # Low internal friction -> smaller stress-cone reach
    init_custom = dict(initial, material=mat_custom, body_pose=initial["body_pose"].copy())
    init_custom["body_pose"][0] = [0, 0, 0.135, 0, 0, 0, 1]
    init_custom["body_pose"][1] = [0, 0, 0.135, 0, 0, 0, 1]

    sim = Simulation(saved, manifest, **init_custom, capture=True, active_particles="auto", anchor_particles=16)
    sim.step()
    diag = sim.diagnostics()
    assert diag["finite"] and diag["converged"]
    counts = diag["heuristic_active_counts"]
    assert counts[0] >= counts[1] >= counts[2]
