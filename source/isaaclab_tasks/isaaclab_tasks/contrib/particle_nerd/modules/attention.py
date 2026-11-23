# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transolver slice attention: particles are pooled into slices, the slices attend to each other, and the result
is distributed back to the particles.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .normalization import RMSNorm
from .positional_encoder import ParticleRotaryEmbedding


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    c = cos.to(dtype=x.dtype)
    s = sin.to(dtype=x.dtype)
    return x * c + _rotate_half(x) * s


class FeedForwardSwiGLU(nn.Module):
    """SwiGLU feed-forward network with 2/3 hidden-dimension scaling, rounded up to a multiple of 8."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        h = int(2 * hidden_dim / 3)
        h = 8 * ((h + 7) // 8)
        self.w12 = nn.Linear(dim, 2 * h)
        self.w3 = nn.Linear(h, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class PhysicsAttention(nn.Module):
    """Transolver slice attention: pool N particles into S spatially localized slices, attend, distribute back.

    ``multiplicity`` [B, N] weights every token when the slices are pooled. The solver uses it to let a few
    background particles stand in for all particles outside the active set.
    """

    def __init__(self, hidden: int, heads: int, slices: int):
        super().__init__()
        self.heads, self.width = heads, hidden // heads
        self.slices = int(slices)
        rope_dim = max(6, (self.width // 6) * 6)
        self.rope3d = ParticleRotaryEmbedding(dim=min(rope_dim, self.width))
        self.project_geometry = nn.Linear(hidden, hidden)
        self.project_values = nn.Linear(hidden, hidden)
        self.project_slices = nn.Linear(self.width, slices)
        nn.init.orthogonal_(self.project_slices.weight)
        self.temperature = nn.Parameter(torch.full((heads, 1, 1), 0.5))
        self.geom_q_norm = RMSNorm(self.width)
        self.slice_k_norm = RMSNorm(self.width)
        self.q_norm = RMSNorm(self.width)
        self.k_norm = RMSNorm(self.width)
        self.deslice_q_norm = RMSNorm(self.width)
        self.deslice_k_norm = RMSNorm(self.width)
        # Learned per-head spatial locality strength in scale-invariant [0, 1]^3 cloud coordinates
        self.spatial_locality = nn.Parameter(torch.full((heads, 1, 1), 1.8))
        g = round(slices ** (1.0 / 3.0))
        if g * g * g == slices:
            u = torch.linspace(0.5 / g, 1.0 - 0.5 / g, g)
            gx, gy, gz = torch.meshgrid(u, u, u, indexing="ij")
            unit_grid = torch.stack([gx, gy, gz], dim=-1).reshape(1, slices, 3)
        else:
            u = torch.linspace(0.05, 0.95, slices)
            unit_grid = torch.stack(
                [u, torch.remainder(u * 3.7, 1.0), torch.remainder(u * 7.3, 1.0)], dim=-1
            ).unsqueeze(0)
        self.register_buffer("unit_grid", unit_grid, persistent=False)
        self.qkv = nn.Linear(self.width, 3 * self.width, bias=False)
        self.deslice_q = nn.Linear(self.width, self.width, bias=False)
        self.deslice_k = nn.Linear(self.width, self.width, bias=False)
        self.output = nn.Linear(hidden, hidden)

    def precompute_rope_cache(
        self, pos: torch.Tensor, dtype: torch.dtype, multiplicity: torch.Tensor | None = None
    ) -> tuple:
        """Position-dependent tables shared by every block of one forward pass."""
        if pos.ndim == 2:
            pos = pos.unsqueeze(0)
        b = pos.shape[0]
        pos_f = pos.float()
        if multiplicity is not None:
            m_w = multiplicity.float().unsqueeze(-1)
            mean_pos = (pos_f * m_w).sum(dim=1, keepdim=True) / m_w.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            mean_pos = pos_f.mean(dim=1, keepdim=True)
        pos_rel = pos_f - mean_pos
        p_cos, p_sin = self.rope3d.cos_sin(pos_rel[:, None, :, :], self.width)
        p_cos = p_cos.to(dtype=dtype)
        p_sin = p_sin.to(dtype=dtype)
        pos_rel_dt = pos_rel.to(dtype=dtype)
        p_min = pos_rel.amin(dim=1, keepdim=True)
        p_max = pos_rel.amax(dim=1, keepdim=True)
        p_span = (p_max - p_min).clamp_min(1e-3)
        pos_unit = (pos_rel - p_min) / p_span
        proto_pos = p_min + self.unit_grid * p_span
        dist_sq = torch.cdist(pos_unit, self.unit_grid.expand(b, -1, -1)).square().unsqueeze(1)
        pr_cos, pr_sin = self.rope3d.cos_sin(proto_pos[:, None, :, :], self.width)
        pr_cos = pr_cos.to(dtype=dtype)
        pr_sin = pr_sin.to(dtype=dtype)
        return pos_rel, pos_rel_dt, p_cos, p_sin, dist_sq, pr_cos, pr_sin

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        rope_cache: tuple | None = None,
        multiplicity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        single = x.ndim == 2
        if single:
            x = x.unsqueeze(0)
            pos = pos.unsqueeze(0)
            if multiplicity is not None and multiplicity.ndim == 1:
                multiplicity = multiplicity.unsqueeze(0)
        b, n, _ = x.shape
        values = self.project_values(x).view(b, n, self.heads, self.width).transpose(1, 2)

        if rope_cache is None:
            pos_rel, pos_rel_dt, p_cos, p_sin, dist_sq, pr_cos, pr_sin = self.precompute_rope_cache(
                pos, values.dtype, multiplicity=multiplicity
            )
        else:
            pos_rel, pos_rel_dt, p_cos, p_sin, dist_sq, pr_cos, pr_sin = rope_cache

        # Slice assignment: feature similarity to learned slice prototypes plus a spatial locality term
        geom = self.geom_q_norm(self.project_geometry(x).view(b, n, self.heads, self.width).transpose(1, 2)).to(
            values.dtype
        )
        geom_rope = apply_rope(geom, p_cos, p_sin)  # [B, H, N, W]
        slice_w = self.slice_k_norm(
            self.project_slices.weight.unsqueeze(0).unsqueeze(0).expand(b, self.heads, -1, -1)
        ).to(values.dtype)
        slice_k = apply_rope(slice_w, pr_cos, pr_sin)  # [B, H, S, W]
        feat_logits = (geom_rope @ slice_k.transpose(-2, -1)).float() * (self.width**-0.5)
        if self.project_slices.bias is not None:
            feat_logits = feat_logits + self.project_slices.bias.view(1, 1, 1, -1).float()
        loc_scale = F.softplus(self.spatial_locality).float() * 3.0
        logits = feat_logits - loc_scale * dist_sq
        inv_temp = (1.0 / self.temperature.clamp(0.5, 5.0)).to(logits.dtype)
        weights = (logits * inv_temp).softmax(dim=-1, dtype=values.dtype)
        w_t = weights.transpose(-2, -1)

        # Slice tokens [B, H, S, W] and slice centroids [B, H, S, 3]
        if multiplicity is None:
            w_sum = weights.sum(-2, dtype=torch.float32).unsqueeze(-1)
            denom = w_sum + 0.01
            tokens = ((w_t @ values).float() / denom).to(values.dtype)
            slice_pos_num = torch.bmm(w_t.reshape(b, self.heads * self.slices, n), pos_rel_dt).view(
                b, self.heads, self.slices, 3
            )
            slice_pos = slice_pos_num.float() / w_sum.clamp_min(1e-6)
        else:
            m_col = multiplicity[:, None, :, None].to(values.dtype)
            w_sum = (w_t @ m_col).float()
            denom = w_sum + 0.01
            tokens = ((w_t @ (values * m_col)).float() / denom).to(values.dtype)
            pos_m = pos_rel_dt * multiplicity[:, :, None].to(values.dtype)
            slice_pos_num = torch.bmm(w_t.reshape(b, self.heads * self.slices, n), pos_m).view(
                b, self.heads, self.slices, 3
            )
            slice_pos = slice_pos_num.float() / w_sum.clamp_min(1e-6)

        # 1. Slice-to-slice self-attention (RoPE encodes the vector between slice centroids)
        s_cos, s_sin = self.rope3d.cos_sin(slice_pos, self.width)
        q, k, v = self.qkv(tokens).chunk(3, -1)
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)
        q = apply_rope(q, s_cos, s_sin)
        k = apply_rope(k, s_cos, s_sin)
        attended = F.scaled_dot_product_attention(q, k, v)

        # 2. Particle-to-slice deslicing (RoPE encodes the vector from a particle to a slice centroid)
        dq = self.deslice_q(values)
        dk = self.deslice_k(attended)
        dq = self.deslice_q_norm(dq).to(attended.dtype)
        dk = self.deslice_k_norm(dk).to(attended.dtype)
        q_p = apply_rope(dq, p_cos, p_sin)  # [B, H, N, W]
        k_s = apply_rope(dk, s_cos, s_sin)  # [B, H, S, W]
        points_soft = weights.to(attended.dtype) @ attended
        points_rope = F.scaled_dot_product_attention(q_p, k_s, attended)
        points = points_soft + points_rope
        out = self.output(points.transpose(1, 2).reshape(b, n, -1))
        return out[0] if single else out


class TransolverBlock(nn.Module):
    """Pre-norm Transolver block: slice attention followed by a SwiGLU feed-forward network."""

    def __init__(self, hidden: int, heads: int, slices: int):
        super().__init__()
        self.norm_attention = RMSNorm(hidden)
        self.norm_mlp = RMSNorm(hidden)
        self.attention = PhysicsAttention(hidden, heads, slices)
        self.mlp = FeedForwardSwiGLU(hidden, 4 * hidden)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        rope_cache: tuple | None = None,
        multiplicity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.training:
            dt = x.dtype
            x = x + self.attention(self.norm_attention(x).to(dt), pos, rope_cache=rope_cache, multiplicity=multiplicity)
            return x + self.mlp(self.norm_mlp(x).to(dt))
        x = x + self.attention(self.norm_attention(x), pos, rope_cache=rope_cache, multiplicity=multiplicity)
        return x + self.mlp(self.norm_mlp(x))
