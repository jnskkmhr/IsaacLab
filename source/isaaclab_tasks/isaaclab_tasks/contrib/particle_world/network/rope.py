# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""3D Rotary Position Embeddings (RoPE) for continuous 3D Lagrangian particle coordinates.

Adapted from MIT CSAIL WorldParticle (`models/encodings/rope.py`).
"""

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate paired dimensions (x_2i, x_2i+1) -> (-x_2i+1, x_2i)."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)


def freqs_to_cos_sin(
    freqs: torch.Tensor,
    head_dim: int,
    rope_dim: int,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert [B, N, rope_dim // 2] frequencies to [B, 1, N, head_dim] cos/sin tensors."""
    # Repeat each frequency for the (2i, 2i+1) pair -> [B, N, rope_dim]
    angles = torch.repeat_interleave(freqs, 2, dim=-1)
    if unsqueeze_dim is not None:
        angles = angles.unsqueeze(unsqueeze_dim)
    cos = angles.cos()
    sin = angles.sin()
    if rope_dim < head_dim:
        pad_shape = (*cos.shape[:-1], head_dim - rope_dim)
        ones = torch.ones(pad_shape, dtype=cos.dtype, device=cos.device)
        zeros = torch.zeros(pad_shape, dtype=sin.dtype, device=sin.device)
        cos = torch.cat([cos, ones], dim=-1)
        sin = torch.cat([sin, zeros], dim=-1)
    return cos, sin


def apply_rotary_emb_cossin(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cos_k: torch.Tensor | None = None,
    sin_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply precomputed 3D rotary cos/sin embeddings to query and key tensors [B, H, N, Hd]."""
    cos_q = cos.to(dtype=q.dtype)
    sin_q = sin.to(dtype=q.dtype)
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    if cos_k is None or sin_k is None:
        cos_k = cos.to(dtype=k.dtype)
        sin_k = sin.to(dtype=k.dtype)
    else:
        cos_k = cos_k.to(dtype=k.dtype)
        sin_k = sin_k.to(dtype=k.dtype)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed


class ParticleRotaryEmbedding(nn.Module):
    """Continuous 3D rotary positional encoding across (x, y, z) particle coordinates."""

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        max_freq: float = 64.0,
        coord_scale: float = 1.0,
    ):
        super().__init__()
        if dim % 6 != 0:
            raise ValueError(f"ParticleRotaryEmbedding dim must be divisible by 6 for 3D coordinates, got {dim}")
        self.dim = dim
        self.theta = theta
        self.max_freq = max_freq
        self.coord_scale = coord_scale
        axes_dim = dim // 3
        freqs = 1.0 / (theta ** (torch.arange(0, axes_dim, 2, dtype=torch.float32) / axes_dim))
        self.register_buffer("freqs", freqs * max_freq, persistent=False)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """Compute 3D RoPE frequencies of shape [B, N, dim // 2] from 3D coordinates [B, N, 3]."""
        scaled = pos.float() * self.coord_scale
        # scaled: [B, N, 3], self.freqs: [axes_dim // 2] -> [B, N, 3, axes_dim // 2]
        angles = scaled.unsqueeze(-1) * self.freqs
        return angles.reshape(*pos.shape[:-1], -1)
