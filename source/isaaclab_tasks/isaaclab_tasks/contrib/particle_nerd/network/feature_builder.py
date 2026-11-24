# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inputs of the Particle NeRD network, built from the particle history and the collider state.

- :func:`material_features` normalizes the material and time-step vector.
- :func:`integrate_pos_vel` is the predictor: it advances every particle under gravity alone.
- :func:`compute_features` (one environment, differentiable) and :func:`compute_features_batched` (all
  environments, inference only) launch the Warp kernels that return the particle features, the particle-collider
  pair features and the influence of every collider on every particle.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from ..nerd_kernels import (
    collider_features,
    collider_features_batched,
    particle_features,
    particle_features_batched,
    warp_stream,
)


def material_features(manifest: dict, particle_mass: float, particle_radius: float, dt: float = 0.005) -> torch.Tensor:
    """Normalize the 13-element continuum material and timestep feature vector (float32, on the CPU)."""
    m = manifest["material"]
    return torch.tensor(
        [
            m["density"] / 1600.0,
            math.log10(m["young_modulus"]) / 15.0,
            m["poisson_ratio"],
            m["friction"],
            math.log1p(m["yield_pressure"]) / math.log1p(1e15),
            m["tensile_yield_ratio"],
            math.log1p(m["yield_stress"]) / 15.0,
            m["hardening"],
            m["dilatancy"],
            math.log1p(m["viscosity"]) / 10.0,
            particle_mass / 0.05,
            particle_radius / 0.02,
            dt / 0.02,
        ],
        dtype=torch.float32,
    )


def integrate_pos_vel(
    pos: torch.Tensor,
    vel: torch.Tensor,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    dt: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Semi-implicit Euler predictor step under gravity.

    The gravity increment is added per axis as a scalar, so the step creates no tensor from host memory and can
    be recorded in a CUDA graph.
    """
    vel_pred = vel.clone()
    for axis in range(3):
        vel_pred[..., axis] += dt * gravity[axis]
    pos_pred = pos + 0.5 * dt * (vel + vel_pred)
    return pos_pred, vel_pred


class _DifferentiableFeatures(torch.autograd.Function):
    """Particle and collider features with gradients with respect to the position history (training only)."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        history: torch.Tensor,
        poses: torch.Tensor,
        velocities: torch.Tensor,
        kinds: torch.Tensor,
        sizes: torch.Tensor,
        friction: torch.Tensor,
        material: torch.Tensor,
        dt: float,
        support: float,
        far_support_scale: float,
        far_gap_weight: float,
        far_influence_weight: float,
        gravity: tuple[float, float, float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, h = history.shape[:2]
        device = str(history.device)
        grad = wp.zeros(history.shape, dtype=float, device=device) if history.requires_grad else None
        hist = wp.from_torch(history, requires_grad=history.requires_grad, grad=grad)
        nodes = wp.empty((n, 16 + 3 * (h - 1)), dtype=float, device=device, requires_grad=history.requires_grad)
        pairs = wp.empty((n, len(poses), 8), dtype=float, device=device, requires_grad=history.requires_grad)
        influence = wp.empty((n, len(poses)), dtype=float, device=device, requires_grad=history.requires_grad)
        tape = wp.Tape()
        with wp.ScopedStream(warp_stream(history.device)), tape:
            wp.launch(
                particle_features,
                n,
                [hist, wp.from_torch(material), dt, wp.vec3(*gravity), nodes],
                device=device,
            )
            if len(poses):
                wp.launch(
                    collider_features,
                    (n, len(poses)),
                    [
                        hist,
                        wp.from_torch(poses),
                        wp.from_torch(velocities),
                        wp.from_torch(kinds),
                        wp.from_torch(sizes),
                        wp.from_torch(friction),
                        wp.from_torch(material),
                        dt,
                        support,
                        far_support_scale,
                        far_gap_weight,
                        far_influence_weight,
                        pairs,
                        influence,
                    ],
                    device=device,
                )
        ctx.history = hist
        ctx.outputs = (nodes, pairs, influence)
        ctx.tape = tape
        ctx.device = history.device
        return tuple(wp.to_torch(x) for x in ctx.outputs)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, *gradients: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, ...]:
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(ctx.device))):
            grads = {out: wp.from_torch(g.contiguous()) for out, g in zip(ctx.outputs, gradients) if g is not None}
            ctx.tape.zero()
            ctx.tape.backward(grads=grads)
            gradient = wp.to_torch(ctx.history.grad).clone()
        return (gradient,) + (None,) * 12


def compute_features(
    history: torch.Tensor,
    poses: torch.Tensor,
    velocities: torch.Tensor,
    kinds: torch.Tensor,
    sizes: torch.Tensor,
    friction: torch.Tensor,
    material: torch.Tensor,
    dt: float = 0.005,
    support: float = 0.07,
    far_support_scale: float = 5.0,
    far_gap_weight: float = 0.3,
    far_influence_weight: float = 0.15,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predicted state and network inputs of one environment.

    Returns the predicted position and velocity [N, 3], the particle features [N, 22], the particle-collider pair
    features [N, C, 8] (normalized gap, outward normal, relative surface velocity, friction) and the influence of
    every collider on every particle [N, C]. Differentiable with respect to ``history`` when it requires a gradient.
    """
    pos_cur = history[:, -1]
    vel_cur = (history[:, -1] - history[:, -2]) / dt
    pos_pred, vel_pred = integrate_pos_vel(pos_cur, vel_cur, gravity=gravity, dt=dt)
    if not history.requires_grad:
        n, h = history.shape[:2]
        device = str(history.device)
        nodes_t = torch.empty((n, 16 + 3 * (h - 1)), dtype=torch.float32, device=history.device)
        pairs_t = torch.empty((n, len(poses), 8), dtype=torch.float32, device=history.device)
        inf_t = torch.empty((n, len(poses)), dtype=torch.float32, device=history.device)
        with wp.ScopedStream(warp_stream(history.device)):
            wp.launch(
                particle_features,
                n,
                [
                    wp.from_torch(history, requires_grad=False),
                    wp.from_torch(material, requires_grad=False),
                    dt,
                    wp.vec3(*gravity),
                    wp.from_torch(nodes_t, requires_grad=False),
                ],
                device=device,
            )
            if len(poses):
                wp.launch(
                    collider_features,
                    (n, len(poses)),
                    [
                        wp.from_torch(history, requires_grad=False),
                        wp.from_torch(poses, requires_grad=False),
                        wp.from_torch(velocities, requires_grad=False),
                        wp.from_torch(kinds, requires_grad=False),
                        wp.from_torch(sizes, requires_grad=False),
                        wp.from_torch(friction, requires_grad=False),
                        wp.from_torch(material, requires_grad=False),
                        dt,
                        support,
                        far_support_scale,
                        far_gap_weight,
                        far_influence_weight,
                        wp.from_torch(pairs_t, requires_grad=False),
                        wp.from_torch(inf_t, requires_grad=False),
                    ],
                    device=device,
                )
        return pos_pred, vel_pred, nodes_t, pairs_t, inf_t
    nodes_t, pairs_t, inf_t = _DifferentiableFeatures.apply(
        history,
        poses,
        velocities,
        kinds,
        sizes,
        friction,
        material,
        dt,
        support,
        far_support_scale,
        far_gap_weight,
        far_influence_weight,
        gravity,
    )
    return pos_pred, vel_pred, nodes_t, pairs_t, inf_t


def compute_features_batched(
    history_env: torch.Tensor,
    poses_env: torch.Tensor,
    velocities_env: torch.Tensor,
    kinds: torch.Tensor,
    sizes: torch.Tensor,
    friction: torch.Tensor,
    material: torch.Tensor,
    dt: float = 0.005,
    support: float = 0.07,
    far_support_scale: float = 5.0,
    far_gap_weight: float = 0.3,
    far_influence_weight: float = 0.15,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`compute_features` for [E, M, H, 3] histories in one launch (inference only)."""
    pos_cur = history_env[:, :, -1]
    vel_cur = (history_env[:, :, -1] - history_env[:, :, -2]) / dt
    pos_pred, vel_pred = integrate_pos_vel(pos_cur, vel_cur, gravity=gravity, dt=dt)

    num_envs, m_pts, h_len, _ = history_env.shape
    n_tot = num_envs * m_pts
    n_col = len(kinds)
    device = str(history_env.device)
    hist_flat = history_env.reshape(n_tot, h_len, 3).contiguous()
    sizes_env = (sizes.unsqueeze(0).expand(num_envs, -1, -1) if sizes.ndim == 2 else sizes).contiguous()
    friction_env = (friction.unsqueeze(0).expand(num_envs, -1) if friction.ndim == 1 else friction).contiguous()
    material_env = (material.unsqueeze(0).expand(num_envs, -1) if material.ndim == 1 else material).contiguous()
    nodes_t = torch.empty((num_envs, m_pts, 16 + 3 * (h_len - 1)), dtype=torch.float32, device=history_env.device)
    pairs_t = torch.empty((num_envs, m_pts, n_col, 8), dtype=torch.float32, device=history_env.device)
    inf_t = torch.empty((num_envs, m_pts, n_col), dtype=torch.float32, device=history_env.device)

    with wp.ScopedStream(warp_stream(history_env.device)):
        wp.launch(
            particle_features_batched,
            n_tot,
            [
                wp.from_torch(hist_flat, requires_grad=False),
                wp.from_torch(material_env, requires_grad=False),
                m_pts,
                dt,
                wp.vec3(*gravity),
                wp.from_torch(nodes_t.view(n_tot, -1), requires_grad=False),
            ],
            device=device,
        )
        if n_col > 0:
            wp.launch(
                collider_features_batched,
                (n_tot, n_col),
                [
                    wp.from_torch(hist_flat, requires_grad=False),
                    wp.from_torch(poses_env.contiguous(), requires_grad=False),
                    wp.from_torch(velocities_env.contiguous(), requires_grad=False),
                    wp.from_torch(kinds.contiguous(), requires_grad=False),
                    wp.from_torch(sizes_env, requires_grad=False),
                    wp.from_torch(friction_env, requires_grad=False),
                    wp.from_torch(material_env, requires_grad=False),
                    m_pts,
                    dt,
                    support,
                    far_support_scale,
                    far_gap_weight,
                    far_influence_weight,
                    wp.from_torch(pairs_t.view(n_tot, n_col, 8), requires_grad=False),
                    wp.from_torch(inf_t.view(n_tot, n_col), requires_grad=False),
                ],
                device=device,
            )
    return pos_pred, vel_pred, nodes_t, pairs_t, inf_t
