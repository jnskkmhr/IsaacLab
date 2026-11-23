# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""WorldParticle attention, SwiGLU feed-forward, RMSNorm, and ToMe super-particle blocks.

Uses PyTorch's built-in FlashAttention-2 / cuDNN fused `F.scaled_dot_product_attention`
with optional external `flash_attn` support.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .merge import k_partite_soft_matching, merge_wavg
from .rope import apply_rotary_emb_cossin

EPS = 1e-6


class RMSNorm(nn.Module):
    """Root-Mean-Square normalization layer."""

    def __init__(self, dim: int, eps: float = EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x_norm = x.float() * torch.rsqrt(norm + self.eps)
        return (self.weight * x_norm).to(dtype=x.dtype)


class FeedForwardSwiGLU(nn.Module):
    """SwiGLU feed-forward network with 2/3 hidden-dim scaling aligned to 8."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = 8 * ((hidden_dim + 7) // 8)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.dropout(F.silu(x1) * x2))


class FeedForwardGeLU(nn.Module):
    """Standard GeLU feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.gelu(self.w1(x))))


def _norm_layer(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "rms_norm":
        return RMSNorm(dim, eps=EPS)
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, eps=EPS)
    raise ValueError(f"Unsupported norm_type: {norm_type!r}")


def _ffn_layer(activation: str, dim: int, hidden_dim: int, dropout: float, bias: bool) -> nn.Module:
    if activation == "swiglu":
        return FeedForwardSwiGLU(dim, hidden_dim=hidden_dim, dropout=dropout, bias=bias)
    if activation == "gelu":
        return FeedForwardGeLU(dim, hidden_dim=hidden_dim, dropout=dropout, bias=bias)
    raise ValueError(f"Unsupported activation: {activation!r}")


class MultiHeadAttention(nn.Module):
    """Multi-head self/cross-attention with 3D RoPE, QK-normalization, and Flash SDPA."""

    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        kv_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = True,
        norm_type: str = "rms_norm",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.is_self_attn = kv_dim is None
        kv_dim = query_dim if kv_dim is None else kv_dim

        if self.is_self_attn:
            self.in_proj = nn.Linear(query_dim, 3 * query_dim, bias=bias)
        else:
            self.q_proj = nn.Linear(query_dim, query_dim, bias=bias)
            self.kv_proj = nn.Linear(kv_dim, 2 * query_dim, bias=bias)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=bias)

        if qk_norm:
            self.q_norm = _norm_layer(norm_type, query_dim)
            self.k_norm = _norm_layer(norm_type, query_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        self.dropout_p = dropout

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        rope_ctx_cos: torch.Tensor | None = None,
        rope_ctx_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, src_len, _ = query.shape
        if self.is_self_attn:
            q, k, v = self.in_proj(query).chunk(3, dim=-1)
            ctx_len = src_len
        else:
            assert kv is not None
            ctx_len = kv.shape[1]
            q = self.q_proj(query)
            k, v = self.kv_proj(kv).chunk(2, dim=-1)

        q = self.q_norm(q).to(dtype=v.dtype)
        k = self.k_norm(k).to(dtype=v.dtype)

        q = q.view(bs, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, ctx_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, ctx_len, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rotary_emb_cossin(q, k, rope_cos, rope_sin, rope_ctx_cos, rope_ctx_sin)

        attn_out = F.scaled_dot_product_attention(
            query=q.to(dtype=v.dtype),
            key=k.to(dtype=v.dtype),
            value=v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bs, src_len, -1)
        return self.out_proj(attn_out)


class MultiHeadAttentionBlock(nn.Module):
    """Pre-norm cross/self-attention block with optional super-particle self-attention and SwiGLU FFN."""

    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        kv_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        activation: str = "swiglu",
        norm_type: str = "rms_norm",
        qk_norm: bool = True,
    ):
        super().__init__()
        self.is_self_attn = kv_dim is None
        self.query_norm = _norm_layer(norm_type, query_dim)
        self.kv_norm = nn.Identity() if self.is_self_attn else _norm_layer(norm_type, kv_dim)
        self.attn = MultiHeadAttention(
            query_dim=query_dim,
            num_heads=num_heads,
            kv_dim=kv_dim,
            dropout=dropout,
            bias=bias,
            qk_norm=qk_norm,
            norm_type=norm_type,
        )
        self.ffn_norm = _norm_layer(norm_type, query_dim)
        self.ffn = _ffn_layer(activation, query_dim, ffn_hidden_dim, dropout, bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        rope_ctx_cos: torch.Tensor | None = None,
        rope_ctx_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.query_norm(query)
        kv_in = None if self.is_self_attn else self.kv_norm(kv)
        query = query + self.dropout(self.attn(q, kv_in, rope_cos, rope_sin, rope_ctx_cos, rope_ctx_sin))
        query = query + self.dropout(self.ffn(self.ffn_norm(query)))
        return query


class MultiHeadAttentionWithTOME(nn.Module):
    """Self-attention block with 3D RoPE and k-partite Token Merging into super-particles."""

    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
        activation: str = "swiglu",
        norm_type: str = "rms_norm",
        qk_norm: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.query_norm = _norm_layer(norm_type, query_dim)
        self.in_proj = nn.Linear(query_dim, 3 * query_dim, bias=bias)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=bias)
        if qk_norm:
            self.q_norm = _norm_layer(norm_type, query_dim)
            self.k_norm = _norm_layer(norm_type, query_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        self.ffn_norm = _norm_layer(norm_type, query_dim)
        self.ffn = _ffn_layer(activation, query_dim, ffn_hidden_dim, dropout, bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout_p = dropout

    def forward(
        self,
        query: torch.Tensor,
        pos: torch.Tensor,
        size_token: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        k_partite: int = 2,
        merge_tokens: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, src_len, _ = query.shape
        q_in = self.query_norm(query)
        q, k, v = self.in_proj(q_in).chunk(3, dim=-1)
        q = self.q_norm(q).to(dtype=v.dtype)
        k = self.k_norm(k).to(dtype=v.dtype)

        q = q.view(bs, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rotary_emb_cossin(q, k, rope_cos, rope_sin)

        attn_out = F.scaled_dot_product_attention(
            query=q.to(dtype=v.dtype),
            key=k.to(dtype=v.dtype),
            value=v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(bs, src_len, -1)
        x_attn = self.out_proj(attn_out)
        metric = k.mean(dim=1)  # [B, N, head_dim]

        query = query + self.dropout(x_attn)
        if merge_tokens and k_partite >= 2 and src_len >= k_partite:
            merge_fn, _ = k_partite_soft_matching(metric, k=k_partite)
            query, size_token, pos = merge_wavg(merge_fn, query, size_token, pos)
        elif size_token is None:
            size_token = torch.ones_like(query[..., :1], dtype=torch.float32)

        query = query + self.dropout(self.ffn(self.ffn_norm(query)))
        return query, size_token, pos
