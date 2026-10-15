# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Particle and rigid-object numbering must not change physical predictions."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particle_model import ParticleGraph, graph_inputs


def test_particle_and_body_permutations_preserve_predictions():
    torch.manual_seed(13)
    torch.set_num_threads(1)
    rng = np.random.default_rng(9)
    x = rng.uniform(-0.03, 0.03, (12, 3)).astype(np.float32)
    velocity = rng.normal(0, 0.1, (12, 9)).astype(np.float32)
    e = dict(
        body_size=np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]),
        body_shapes=["sphere", "box"],
        body_mass=np.array([2.0, 0]),
        body_friction=np.array([0.5, 0.6]),
        moving_body=0,
        material=dict(density=1600.0, friction=0.4, young_modulus=1e6, poisson_ratio=0.3),
        particle_radius=np.full(12, 0.008),
        particle_mass=np.full(12, 0.006),
        dt=0.005,
        gravity=[0, 0, -9.81],
    )
    poses = np.array([[0, 0, 0.06, 0, 0, 0, 1], [0, 0, -0.12, 0, 0, 0, 1.0]])
    body_v = rng.normal(0, 0.1, (2, 6))
    model = ParticleGraph(hidden=16, passes=2)
    # Exercise a nontrivial decoder, rather than the zero-initialized free step.
    torch.nn.init.normal_(model.decoder[-1].weight, std=0.1)
    before = model(*graph_inputs(x, velocity, e, poses, body_v, "cpu"))[0].detach().numpy()
    permutation = rng.permutation(12)
    changed = dict(e)
    for key in ("particle_radius", "particle_mass"):
        changed[key] = e[key][permutation]
    for key in ("body_size", "body_mass", "body_friction"):
        changed[key] = e[key][::-1].copy()
    changed["body_shapes"] = e["body_shapes"][::-1]
    changed["moving_body"] = 1
    after = (
        model(
            *graph_inputs(
                x[permutation], velocity[permutation], changed, poses[::-1].copy(), body_v[::-1].copy(), "cpu"
            )
        )[0]
        .detach()
        .numpy()
    )
    np.testing.assert_allclose(after, before[permutation], atol=1e-7)


def test_static_projection_uses_physical_mobility_and_updates_velocity():
    from evaluate_particles import project_particles

    episode = dict(
        body_mass=np.array([0.0, 2.0]),
        body_size=np.array([[1.0, 1.0, 0.1], [0.1, 0.1, 0.1]]),
        body_shapes=["box", "sphere"],
        dt=0.01,
    )
    poses = np.array([[0, 0, -0.05, 0, 0, 0, 1.0], [0, 0, 0.2, 0, 0, 0, 1.0]])
    positions = np.array([[0.0, 0.0, -0.01], [0.0, 0.0, 0.24]])
    projected, velocity = project_particles(positions, positions, poses, episode, mode="static")
    np.testing.assert_allclose(projected, [[0, 0, 0], [0, 0, 0.24]], atol=1e-12)
    np.testing.assert_allclose(velocity, [[0, 0, 1], [0, 0, 0]], atol=1e-12)
    projected, velocity = project_particles(positions, positions, poses, episode, mode="all")
    np.testing.assert_allclose(projected, [[0, 0, 0], [0, 0, 0.25]], atol=1e-12)
    np.testing.assert_allclose(velocity, [[0, 0, 1], [0, 0, 1]], atol=1e-12)


def test_activation_checkpointing_preserves_predictions_and_parameter_gradients():
    import copy

    torch.manual_seed(37)
    model = ParticleGraph(hidden=16, passes=3)
    torch.nn.init.normal_(model.decoder[-1].weight, std=0.2)
    copied = copy.deepcopy(model)
    copied.config["activation_checkpointing"] = True
    edges = torch.tensor([[i, j] for i in range(6) for j in range(6) if i != j]).T
    inputs = (
        torch.randn(6, 3) * 0.01,
        torch.randn(6, 9),
        torch.randn(6, 6),
        torch.randn(6, 2, 9),
        edges,
        0.005,
        torch.tensor([0.0, 0.0, -9.81]),
    )
    outputs = []
    gradients = []
    for network in (model, copied):
        position, velocity = network(*inputs)
        (position.square().sum() + velocity.square().sum()).backward()
        outputs.append(position.detach())
        gradients.append(torch.cat([p.grad.flatten() for p in network.parameters()]))
    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(gradients[0], gradients[1])


def test_chunked_inference_preserves_edges_across_chunk_boundaries():
    """Workspace limits must not turn the graph into disconnected subgraphs."""
    torch.manual_seed(71)
    model = ParticleGraph(hidden=16, passes=3).eval()
    torch.nn.init.normal_(model.decoder[-1].weight, std=0.2)
    # Asymmetric directed edges span every node chunk; the last particle is isolated.
    edges = torch.tensor([[0, 4, 1, 5, 2, 6, 3, 0, 6], [4, 1, 5, 2, 6, 3, 0, 6, 0]])
    inputs = (
        torch.randn(8, 3) * 0.01,
        torch.randn(8, 9),
        torch.randn(8, 6),
        torch.randn(8, 2, 9),
        edges,
        0.005,
        torch.tensor([0.0, 0.0, -9.81]),
    )
    with torch.no_grad():
        expected = model(*inputs)
        actual = model.predict_chunked(*inputs, chunk_size=3)
    for reference, prediction in zip(expected, actual, strict=True):
        torch.testing.assert_close(prediction, reference, atol=2e-7, rtol=2e-6)
