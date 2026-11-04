# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standard Transolver slice attention, adapted to material particles.

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
        single = x.ndim == 2
        if single:
            x = x.unsqueeze(0)
            if multiplicity is not None:
                multiplicity = multiplicity.unsqueeze(0)
        b, n, _ = x.shape
        values = self.project_values(x).view(b, n, self.heads, self.width).transpose(1, 2)
        if not self.training:
            w_g = self.project_geometry.weight.view(self.heads, self.width, -1)
            b_g = self.project_geometry.bias.view(self.heads, self.width)
            w_s = self.project_slices.weight
            b_s = self.project_slices.bias
            w_gs = torch.einsum("sw,hwd->hsd", w_s, w_g).reshape(self.heads * w_s.shape[0], -1)
            b_gs = (torch.einsum("sw,hw->hs", w_s, b_g) + b_s.unsqueeze(0)).reshape(-1)
            logits = F.linear(x, w_gs, b_gs).view(b, n, self.heads, -1).transpose(1, 2)
            inv_temp = (1.0 / self.temperature.clamp(0.5, 5)).to(logits.dtype)
            weights = (logits * inv_temp).softmax(dim=-1, dtype=values.dtype)
            w_t = weights.transpose(-2, -1)
        else:
            geometry = self.project_geometry(x).view(b, n, self.heads, self.width).transpose(1, 2)
            logits = self.project_slices(geometry)
            weights = (logits.float() / self.temperature.clamp(0.5, 5)).softmax(-1)
            w_t = weights.transpose(-2, -1).to(values.dtype)
        # Normalize slice sums using the factored identity (W^T (m * V)) / (W^T m + 0.01)
        # to avoid materializing [B, H, N, S] float32 intermediate tensors.
        if multiplicity is None:
            denom = weights.sum(-2, dtype=torch.float32).unsqueeze(-1) + 0.01
            tokens = ((w_t @ values).float() / denom).to(values.dtype)
        else:
            m_col = multiplicity[:, None, :, None].to(values.dtype)
            denom = (w_t @ m_col).float() + 0.01
            tokens = ((w_t @ (values * m_col)).float() / denom).to(values.dtype)
        q, k, v = self.qkv(tokens).chunk(3, -1)
        attended = F.scaled_dot_product_attention(q, k, v)
        points = weights.to(attended.dtype) @ attended
        out = self.output(points.transpose(1, 2).reshape(b, n, -1))
        return out[0] if single else out


class Block(nn.Module):
    def __init__(self, hidden, heads, slices):
        super().__init__()
        self.norm_attention, self.norm_mlp = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.attention = PhysicsAttention(hidden, heads, slices)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))

    def forward(self, x, multiplicity=None):
        if not self.training:
            dt = x.dtype
            x = x + self.attention(self.norm_attention(x).to(dt), multiplicity=multiplicity)
            return x + self.mlp(self.norm_mlp(x).to(dt))
        x = x + self.attention(self.norm_attention(x), multiplicity=multiplicity)
        return x + self.mlp(self.norm_mlp(x))


class ParticleTransolver(nn.Module):
    """Variable N particles and B colliders; output [N, 3] displacement in metres.

    Collider features are embedded independently by a shared encoder and summed
    with distance weights into each particle token. No object-ID embeddings,
    particle-neighbor graph, velocity head, or region selector.
    """

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
        """Return particle tokens shared by displacement and body-impulse heads."""
        x = self.particle_encoder(particles)
        # Collider ordering is immaterial; simultaneous contacts are retained.
        if not self.training:
            inf = influence.to(x.dtype)
            h_col = (self.collider_encoder[1](self.collider_encoder[0](colliders)) * inf[..., None]).sum(-2)
            fc1 = self.collider_encoder[2]
            x = x + F.linear(h_col, fc1.weight) + inf.sum(-1, keepdim=True) * fc1.bias.to(x.dtype)
        else:
            x = x + (self.collider_encoder(colliders) * influence[..., None]).sum(-2)
        for block in self.blocks:
            x = (
                checkpoint(block, x, multiplicity, use_reentrant=False)
                if self.training
                else block(x, multiplicity=multiplicity)
            )
        return x

    def forward(self, particles, colliders, influence, multiplicity=None):
        x = self.encode(particles, colliders, influence, multiplicity=multiplicity)
        return self.config["displacement_scale"] * self.decoder(x).float()
