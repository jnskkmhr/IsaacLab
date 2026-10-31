# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched Transolver slice attention for multi-environment particle simulation.

Reference: mnabian/physicsnemo at fa1d68a67cc77baa176db8b4c1c5ef2ecdce00f7,
physicsnemo/nn/module/physics_attention.py. See NOTICE.md for attribution.
No PhysicsNeMo installation or pretrained crash weights are required.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class PhysicsAttention(nn.Module):
    """Aggregate N particles into S learned slices, attend, then distribute back."""

    def __init__(self, hidden, heads, slices):
        super().__init__()
        self.heads, self.width = heads, hidden // heads
        self.project_geometry = nn.Linear(hidden, hidden)
        self.project_values = nn.Linear(hidden, hidden)
        self.project_slices = nn.Linear(self.width, slices)
        nn.init.orthogonal_(self.project_slices.weight)
        self.temperature = nn.Parameter(torch.full((heads, 1, 1), 0.5))
        self.qkv = nn.Linear(self.width, 3 * self.width, bias=False)
        self.output = nn.Linear(hidden, hidden)

    def forward(self, x, multiplicity=None):
        single_environment = x.ndim == 2
        if single_environment:
            x = x.unsqueeze(0)
            if multiplicity is not None and multiplicity.ndim == 1:
                multiplicity = multiplicity.unsqueeze(0)
        environments, n, _ = x.shape
        geometry = self.project_geometry(x).view(environments, n, self.heads, self.width).transpose(1, 2)
        values = self.project_values(x).view(environments, n, self.heads, self.width).transpose(1, 2)
        logits = self.project_slices(geometry)
        weights = (logits.float() / self.temperature.clamp(0.5, 5)).softmax(-1)
        weighted = weights if multiplicity is None else weights * multiplicity[:, None, :, None].float()
        normalized = weighted / (weighted.sum(-2, keepdim=True) + 0.01)
        tokens = normalized.transpose(-2, -1).to(values.dtype) @ values
        q, k, v = self.qkv(tokens).chunk(3, -1)
        attended = F.scaled_dot_product_attention(q, k, v)
        points = weights.to(attended.dtype) @ attended
        output = self.output(points.transpose(1, 2).reshape(environments, n, -1))
        return output[0] if single_environment else output


class Block(nn.Module):
    def __init__(self, hidden, heads, slices):
        super().__init__()
        self.norm_attention, self.norm_mlp = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.attention = PhysicsAttention(hidden, heads, slices)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))

    def forward(self, x, multiplicity=None):
        x = x + self.attention(self.norm_attention(x), multiplicity=multiplicity)
        return x + self.mlp(self.norm_mlp(x))


class ParticleTransolver(nn.Module):
    """Variable N particles and C colliders; output [..., N, 3] displacement in metres."""

    def __init__(self, hidden=128, heads=4, slices=64, layers=4, history=3, displacement_scale=0.02):
        super().__init__()
        if hidden <= 0 or heads <= 0 or hidden % heads or min(slices, layers) < 1 or history < 2:
            raise ValueError("Positive dimensions, hidden divisible by heads, and history >= 2 required")
        self.config = dict(
            hidden=hidden,
            heads=heads,
            slices=slices,
            layers=layers,
            history=history,
            displacement_scale=displacement_scale,
        )
        self.particle_encoder = nn.Sequential(
            nn.Linear(16 + 3 * history, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden)
        )
        self.collider_encoder = nn.Sequential(nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([Block(hidden, heads, slices) for _ in range(layers)])
        self.decoder = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode(self, particles, colliders, influence, multiplicity=None):
        x = self.particle_encoder(particles)
        x = x + (self.collider_encoder(colliders) * influence[..., None]).sum(-2)
        for block in self.blocks:
            if self.training and multiplicity is None:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x, multiplicity=multiplicity)
        return x

    def forward(self, particles, colliders, influence, multiplicity=None):
        x = self.encode(particles, colliders, influence, multiplicity=multiplicity)
        return self.config["displacement_scale"] * self.decoder(x).float()
