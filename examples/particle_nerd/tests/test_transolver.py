# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-collider geometry, permutation symmetry, rollout gradients, GPU state update."""

import torch
import warp as wp

from particle_nerd.transolver.features import features
from particle_nerd.transolver.inference import Predictor
from particle_nerd.transolver.model import ParticleTransolver


def scene():
    wp.init()
    history = torch.tensor([[[0.0, 0.0, 0.03]] * 3, [[0.15, 0.0, 0.03]] * 3], device="cuda:0")
    poses = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 1], [0.15, 0, 0.12, 0, 0, 0, 1], [3, 0, 0, 0, 0, 0, 1]],
        dtype=torch.float32,
        device="cuda:0",
    )
    velocities = torch.zeros(3, 6, device="cuda:0")
    kinds = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda:0")
    sizes = torch.tensor([[0, 0, 0], [0.06, 0.06, 0.04], [0.1, 0, 0]], device="cuda:0")
    friction = torch.full((3,), 0.6, device="cuda:0")
    material = torch.zeros(13, device="cuda:0")
    material[11] = 0.5
    return history, poses, velocities, kinds, sizes, friction, material


def test_multiple_colliders_and_particle_order_do_not_require_semantic_ids():
    args = scene()
    nodes, pairs, weights = features(*args)
    torch.testing.assert_close(weights[:, 0], torch.full((2,), 5 / 7, device="cuda:0"))
    assert weights[1, 1] > 0
    torch.testing.assert_close(weights[:, 2], torch.zeros(2, device="cuda:0"))
    torch.testing.assert_close(pairs[1, 1, 1:4], torch.tensor([0.0, 0.0, -1.0], device="cuda:0"))
    torch.manual_seed(42)
    model = ParticleTransolver(hidden=32, heads=4, slices=8, layers=2).cuda().eval()
    torch.nn.init.normal_(model.decoder[-1].weight, std=0.1)
    output = model(nodes, pairs, weights)
    torch.testing.assert_close(
        model(nodes.flip(0), pairs.flip(0).flip(1), weights.flip(0).flip(1)), output.flip(0), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(model(nodes, pairs[:, :2], weights[:, :2]), output, atol=1e-6, rtol=1e-5)
    changed = list(args)
    changed[2] = args[2].clone()
    changed[2][1, 0] = 1.0
    assert (model(*features(*changed)) - output).abs().max() > 1e-6
    assert model(nodes[:1], pairs[:1], weights[:1]).shape == (1, 3)


def test_rollout_geometry_gradient_matches_finite_difference():
    args = list(scene())
    args[0] = args[0].clone().requires_grad_(True)

    def objective(history):
        nodes, pairs, weights = features(history, *args[1:])
        return nodes.square().sum() * 0.001 + (pairs.square().sum(-1) * weights).sum()

    loss = objective(args[0])
    loss.backward()
    eps = 1e-5
    plus, minus = args[0].detach().clone(), args[0].detach().clone()
    plus[0, -1, 2] += eps
    minus[0, -1, 2] -= eps
    finite_difference = (objective(plus) - objective(minus)) / (2 * eps)
    torch.testing.assert_close(args[0].grad[0, -1, 2], finite_difference, atol=0.03, rtol=0.01)


def test_multicollider_inference_keeps_state_on_gpu_and_derives_velocity(monkeypatch):
    history, poses, velocities, kinds, sizes, friction, material = scene()
    model = ParticleTransolver(hidden=32, heads=4, slices=8, layers=2)
    with torch.no_grad():
        torch.nn.init.normal_(model.decoder[-1].weight, std=0.1)
        model.decoder[-1].bias.copy_(torch.tensor([0.1, 0.0, 0.0]))
    config = dict(dt=0.02, model=model.config, features=dict(support=0.07), body_size_m=[0.12, 0.12, 0.08])
    saved = dict(architecture="particle_transolver_v1", config=config, model=model.state_dict())
    predictor = Predictor(saved, history, material, kinds, sizes, friction)
    eager = Predictor(saved, history, material, kinds, sizes, friction, use_cuda_graph=False)
    predictor.step(poses, velocities)
    eager.step(poses, velocities)
    before = predictor.x.clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU state access during inference")

    with monkeypatch.context() as m:
        for name in ("cpu", "numpy", "item"):
            m.setattr(torch.Tensor, name, forbidden)
        m.setattr(wp.array, "numpy", forbidden)
        predictor.step(poses, velocities)
    torch.testing.assert_close(predictor.v, (predictor.x - before) / 0.02, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(predictor.history[:, -1], predictor.x)
    eager.step(poses, velocities)
    for _ in range(3):
        poses[1, 0] += 0.01
        velocities[1, 0] += 0.1
        predictor.step(poses, velocities)
        eager.step(poses, velocities)
        torch.testing.assert_close(predictor.x, eager.x)
        torch.testing.assert_close(predictor.v, eager.v)
        torch.testing.assert_close(predictor.history, eager.history)


def test_balanced_loss_is_independent_of_stationary_population_and_handles_empty_groups():
    from particle_nerd.transolver.train import position_loss

    target = torch.tensor([[[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    prediction = target + torch.tensor([[[0.002, 0.0, 0.0], [0.001, 0.0, 0.0]]])
    initial = torch.zeros(2, 3)
    balanced, _, _ = position_loss(prediction, target, initial, 0.02, "balanced")
    torch.testing.assert_close(balanced, torch.tensor(0.00625))
    indices = torch.tensor([0] + [1] * 100)
    repeated, _, _ = position_loss(prediction[:, indices], target[:, indices], initial[indices], 0.02, "balanced")
    torch.testing.assert_close(repeated, balanced)
    for index, expected in ((0, 0.01), (1, 0.0025)):
        selected = prediction[:, index : index + 1].clone().requires_grad_(True)
        loss, _, _ = position_loss(selected, target[:, index : index + 1], initial[:1], 0.02, "balanced")
        torch.testing.assert_close(loss, torch.tensor(expected))
        loss.backward()
        assert torch.isfinite(selected.grad).all()
        assert selected.grad.abs().sum() > 0
