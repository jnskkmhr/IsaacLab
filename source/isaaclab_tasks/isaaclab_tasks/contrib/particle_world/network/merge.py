# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Token Merging (ToMe) k-partite soft matching and O(N) super-particle position tracking.

Adapted from MIT CSAIL WorldParticle (`models/layers/merge.py`) and Meta ToMe.
Uses static strided slicing (zero boolean-mask CPU sync) and propagates 3D
super-particle coordinates through `merge_wavg` in O(N) memory.
"""

from collections.abc import Callable

import torch


def do_nothing(x: torch.Tensor, mode: str | None = None) -> torch.Tensor:
    return x


def _split_k_partite(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split [B, T, C] into sources A and sinks B (every k-th token) without boolean indexing."""
    if k == 2:
        # Fast strided views for bipartite matching
        b = x[:, 1::2, :]
        a = x[:, 0::2, :]
        return a, b
    bsz, num_tokens, c_dim = x.shape
    n_groups = num_tokens // k
    t_rnd = n_groups * k
    xg = x[:, :t_rnd, :].view(bsz, n_groups, k, c_dim)
    b = xg[:, :, k - 1, :]
    a_main = xg[:, :, : k - 1, :].reshape(bsz, n_groups * (k - 1), c_dim)
    if t_rnd < num_tokens:
        a = torch.cat([a_main, x[:, t_rnd:, :]], dim=1)
    else:
        a = a_main
    return a, b


def k_partite_soft_matching(
    metric: torch.Tensor,
    k: int,
) -> tuple[Callable, Callable]:
    """Partition tokens into sinks B (every k-th token) and sources A, merging A into B by cosine similarity."""
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    num_tokens = metric.shape[1]
    tb = num_tokens // k
    ta = num_tokens - tb
    if ta <= 0 or tb <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric_f = metric.float()
        metric_n = metric_f / (metric_f.norm(dim=-1, keepdim=True) + 1e-12)
        a, b = _split_k_partite(metric_n, k)
        scores = torch.matmul(a, b.transpose(-1, -2))  # [B, TA, TB]
        dst_idx = scores.argmax(dim=-1, keepdim=True)  # [B, TA, 1]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src_all, dst_all = _split_k_partite(x, k)
        b_dim, _, c_dim = src_all.shape
        return dst_all.scatter_reduce(-2, dst_idx.expand(b_dim, ta, c_dim), src_all, reduce=mode)

    def unmerge(y: torch.Tensor) -> torch.Tensor:
        b_dim, _, c_dim = y.shape
        src_rec = y.gather(dim=-2, index=dst_idx.expand(b_dim, ta, c_dim))
        if k == 2:
            out = torch.empty(b_dim, num_tokens, c_dim, device=y.device, dtype=y.dtype)
            out[:, 1::2, :] = y
            out[:, 0::2, :] = src_rec
            return out
        n_groups = tb
        t_rnd = n_groups * k
        a_main_len = n_groups * (k - 1)
        src_main = src_rec[:, :a_main_len, :].view(b_dim, n_groups, k - 1, c_dim)
        dst_main = y.unsqueeze(2)
        merged_groups = torch.cat([src_main, dst_main], dim=2).view(b_dim, t_rnd, c_dim)
        if t_rnd < num_tokens:
            return torch.cat([merged_groups, src_rec[:, a_main_len:, :]], dim=1)
        return merged_groups

    return merge, unmerge


def merge_wavg(
    merge: Callable,
    x: torch.Tensor,
    size: torch.Tensor | None = None,
    pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Merge token features `x` and optional 3D coordinates `pos` using token-size weighted average in O(N) memory."""
    if size is None:
        size = torch.ones_like(x[..., :1], dtype=torch.float32)
    size_f = size.float()
    x_merged = merge(x.float() * size_f, mode="sum")
    pos_merged = merge(pos.float() * size_f, mode="sum") if pos is not None else None
    size_merged = merge(size_f, mode="sum").clamp_min(1e-6)
    x_out = (x_merged / size_merged).to(dtype=x.dtype)
    pos_out = (pos_merged / size_merged).to(dtype=pos.dtype) if pos_merged is not None else None
    return x_out, size_merged, pos_out
