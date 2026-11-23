# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rotary position embedding for continuous 3D coordinates."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ParticleRotaryEmbedding(nn.Module):
    """3D rotary position embedding (RoPE) over continuous coordinates.

    RoPE rotates queries and keys by the angle of their own position, so the attention inner product depends only
    on the relative displacement and stays translation invariant.
    """

    def __init__(self, dim: int, min_freq: float = 1.0 / 64.0, max_freq: float = 20.0):
        super().__init__()
        if dim % 6 != 0:
            raise ValueError(f"3D RoPE dim must be divisible by 6, got {dim}")
        num_bands = dim // 6
        freqs = torch.exp(
            torch.linspace(torch.log(torch.tensor(min_freq)), torch.log(torch.tensor(max_freq)), num_bands)
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def cos_sin(self, pos: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute [..., head_dim] cosine and sine tables from [..., 3] positions."""
        # pos: [..., 3], freqs: [F] -> angles: [..., 3, F] -> [..., 3*F]
        angles = (pos.float().unsqueeze(-1) * self.freqs).reshape(*pos.shape[:-1], -1)
        angles = torch.cat([angles, angles], dim=-1)  # [..., 6*F]
        rope_dim = angles.shape[-1]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        if rope_dim < head_dim:
            pad = head_dim - rope_dim
            cos = F.pad(cos, (0, pad), value=1.0)
            sin = F.pad(sin, (0, pad), value=0.0)
        return cos, sin
