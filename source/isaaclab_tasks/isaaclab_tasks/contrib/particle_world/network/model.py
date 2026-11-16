# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""WorldParticle network: Warp Continuous Convolution Tokenizer + 3D-RoPE ToMe Encoder + Cross-Attn Decoder.

Adapted from MIT CSAIL WorldParticle:
  "WorldParticle: Unified World Simulation of Lagrangian Particle Dynamics via Transformer"
  https://worldparticle.github.io/WorldParticle_Web/
  https://github.com/WorldParticle/WorldParticle
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadAttentionBlock, MultiHeadAttentionWithTOME, RMSNorm
from .cconv_warp import ContinuousConvWarp, features_world_particle, features_world_particle_batched
from .merge import k_partite_soft_matching, merge_wavg
from .rope import ParticleRotaryEmbedding, freqs_to_cos_sin


class LocalFeatureExtractor(nn.Module):
    """Stage 1: Tokenize particles via Warp Continuous 3D Convolution (fluid + obstacle + center dense)."""

    def __init__(
        self,
        node_dim: int = 25,
        collider_dim: int = 8,
        cconv_dim: int = 64,
        d_model: int = 128,
        radius: float = 0.072,
        translation_invariant: bool = False,
    ):
        super().__init__()
        self.translation_invariant = bool(translation_invariant)
        self.cconv = ContinuousConvWarp(
            fluid_in_channels=4,
            obs_in_channels=9,
            filters=cconv_dim,
            kernel_size=(4, 4, 4),
            radius=radius,
        )
        # Center particle + local packing (5 channels) dense branch
        self.dense_fluid = nn.Sequential(
            nn.Linear(node_dim + 5, 2 * cconv_dim),
            nn.GELU(),
            nn.Linear(2 * cconv_dim, cconv_dim),
        )
        # Distance-weighted analytical collider branch
        self.collider_encoder = nn.Sequential(
            nn.Linear(collider_dim, cconv_dim),
            nn.GELU(),
            nn.Linear(cconv_dim, cconv_dim),
        )
        # Fuse [conv_obs, conv_fluid, dense_fluid + collider] -> d_model
        self.in_proj = nn.Sequential(
            nn.Linear(3 * cconv_dim, d_model),
            RMSNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        colliders: torch.Tensor,
        influence: torch.Tensor,
        fluid_basis: torch.Tensor,
        obs_basis: torch.Tensor,
        packing: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = self.dense_fluid[0].weight.dtype
        conv_fluid, conv_obs = self.cconv(fluid_basis, obs_basis)
        if self.translation_invariant:
            # Upstream WorldParticle (`super_particle_layers.py:352`):
            # fluid_feats = [1, vel, other_feats] without raw world coordinates (x, y, z).
            ones = torch.ones_like(nodes[..., :3], dtype=dt)
            nodes_in = torch.cat([ones, nodes[..., 3:].to(dt)], dim=-1)
        else:
            nodes_in = nodes.to(dt)
        center_in = torch.cat([nodes_in, packing.to(dt)], dim=-1)
        dense_out = self.dense_fluid(center_in)
        col_out = (self.collider_encoder(colliders.to(dt)) * influence.to(dt).unsqueeze(-1)).sum(dim=-2)
        dense_out = dense_out + col_out
        cat_feat = torch.cat([conv_obs, conv_fluid, dense_out], dim=-1)
        tokens = self.in_proj(cat_feat)
        return tokens, cat_feat


class SuperParticleEncoder(nn.Module):
    """Stage 2: Iterative k-partite Token Merging with 3D RoPE self-attention into Super-Particles."""

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        slot_attn_iters: int = 3,
        pre_merge_k: int = 16,
        k_partite: int = 2,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pre_merge_k = pre_merge_k
        self.k_partite = k_partite

        rope_dim = max(6, (self.head_dim // 6) * 6)
        self.rope_dim = min(rope_dim, self.head_dim)
        self.rope3d = ParticleRotaryEmbedding(dim=self.rope_dim)

        self.pre_metric_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.layers = nn.ModuleList(
            [
                MultiHeadAttentionWithTOME(
                    query_dim=d_model,
                    num_heads=num_heads,
                    ffn_hidden_dim=d_model * ffn_multiplier,
                    dropout=dropout,
                    activation="swiglu",
                    norm_type="rms_norm",
                    qk_norm=True,
                )
                for _ in range(slot_attn_iters)
            ]
        )

    def _compute_rope(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = self.rope3d(pos)
        return freqs_to_cos_sin(freqs, head_dim=self.head_dim, rope_dim=self.rope_dim, unsqueeze_dim=1)

    def forward(
        self,
        tokens: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge [B, N, D] particle tokens into [B, N_super, D] super-particles and [B, N_super, 3] 3D positions."""
        x = tokens
        p = pos
        size_token = torch.ones_like(x[..., :1], dtype=torch.float32)

        # Initial k0-partite soft matching so large particle clouds (e.g. N=20,646)
        # condense to ~1,290 tokens before quadratic self-attention layers.
        if self.pre_merge_k >= 2 and x.shape[1] >= self.pre_merge_k:
            metric0 = self.pre_metric_proj(x)
            merge0, _ = k_partite_soft_matching(metric0, k=self.pre_merge_k)
            x, size_token, p = merge_wavg(merge0, x, size_token, p)

        for layer in self.layers:
            cos, sin = self._compute_rope(p)
            x, size_token, p = layer(
                x,
                p,
                size_token=size_token,
                rope_cos=cos,
                rope_sin=sin,
                k_partite=self.k_partite,
                merge_tokens=True,
            )
        return x, p, size_token


class SuperParticleDecoder(nn.Module):
    """Stage 3: 3D-RoPE Cross-Attention from all N particles to Super-Particles + SwiGLU FFN."""

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_decoder_layers: int = 3,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        rope_dim = max(6, (self.head_dim // 6) * 6)
        self.rope_dim = min(rope_dim, self.head_dim)
        self.rope3d = ParticleRotaryEmbedding(dim=self.rope_dim)

        self.super_self_blocks = nn.ModuleList(
            [
                MultiHeadAttentionBlock(
                    query_dim=d_model,
                    num_heads=num_heads,
                    ffn_hidden_dim=d_model * ffn_multiplier,
                    kv_dim=None,
                    dropout=dropout,
                    activation="swiglu",
                    norm_type="rms_norm",
                    qk_norm=True,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.cross_blocks = nn.ModuleList(
            [
                MultiHeadAttentionBlock(
                    query_dim=d_model,
                    num_heads=num_heads,
                    ffn_hidden_dim=d_model * ffn_multiplier,
                    kv_dim=d_model,
                    dropout=dropout,
                    activation="swiglu",
                    norm_type="rms_norm",
                    qk_norm=True,
                )
                for _ in range(num_decoder_layers)
            ]
        )

    def _compute_rope(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = self.rope3d(pos)
        return freqs_to_cos_sin(freqs, head_dim=self.head_dim, rope_dim=self.rope_dim, unsqueeze_dim=1)

    def forward(
        self,
        particle_tokens: torch.Tensor,
        particle_pos: torch.Tensor,
        super_tokens: torch.Tensor,
        super_pos: torch.Tensor,
    ) -> torch.Tensor:
        q_cos, q_sin = self._compute_rope(particle_pos)
        k_cos, k_sin = self._compute_rope(super_pos)

        q = particle_tokens
        kv = super_tokens
        for self_blk, cross_blk in zip(self.super_self_blocks, self.cross_blocks, strict=True):
            kv = self_blk(kv, None, k_cos, k_sin)
            q = cross_blk(q, kv, q_cos, q_sin, k_cos, k_sin)
        return q


class WorldParticleNetwork(nn.Module):
    """MIT CSAIL WorldParticle Lagrangian particle dynamics transformer implemented in Warp + PyTorch.

    Combines:
      1. Explicit gravity predictor `integrate_pos_vel` (with gated ballistic residual),
      2. Warp HashGrid Continuous 3D Convolution (`conv0_fluid`, `conv0_obstacle`, `dense_fluid`),
      3. Iterative k-partite Token Merging (`MultiHeadAttentionWithTOME`) with 3D RoPE,
      4. 3D-RoPE Super-Particle Cross-Attention Decoder with local geometry skip and `velhead`.
    """

    def __init__(
        self,
        d_model: int = 128,
        cconv_dim: int = 64,
        heads: int = 4,
        slot_attn_iters: int = 3,
        num_decoder_layers: int = 3,
        pre_merge_k: int = 16,
        k_partite: int = 2,
        history: int = 3,
        radius: float = 0.072,
        displacement_scale: float = 0.02,
        velhead: bool = True,
        translation_invariant: bool = False,
        far_support_scale: float = 5.0,
        far_gap_weight: float = 0.0,
        far_influence_weight: float = 0.0,
        use_actual_velocity: bool = False,
    ):
        super().__init__()
        if d_model <= 0 or heads <= 0 or d_model % heads != 0:
            raise ValueError("d_model must be positive and divisible by heads")
        self.config = dict(
            d_model=d_model,
            cconv_dim=cconv_dim,
            heads=heads,
            slot_attn_iters=slot_attn_iters,
            num_decoder_layers=num_decoder_layers,
            pre_merge_k=pre_merge_k,
            k_partite=k_partite,
            history=history,
            radius=radius,
            displacement_scale=displacement_scale,
            velhead=velhead,
            translation_invariant=translation_invariant,
            far_support_scale=float(far_support_scale),
            far_gap_weight=float(far_gap_weight),
            far_influence_weight=float(far_influence_weight),
            use_actual_velocity=bool(use_actual_velocity),
        )
        node_dim = 16 + 3 * history
        self.tokenizer = LocalFeatureExtractor(
            node_dim=node_dim,
            collider_dim=8,
            cconv_dim=cconv_dim,
            d_model=d_model,
            radius=radius,
            translation_invariant=translation_invariant,
        )
        self.encoder = SuperParticleEncoder(
            d_model=d_model,
            num_heads=heads,
            slot_attn_iters=slot_attn_iters,
            pre_merge_k=pre_merge_k,
            k_partite=k_partite,
        )
        self.decoder = SuperParticleDecoder(
            d_model=d_model,
            num_heads=heads,
            num_decoder_layers=num_decoder_layers,
        )
        # Skip-connect local ContinuousConv features [3 * cconv_dim] into the output head
        head_in = d_model + 3 * cconv_dim
        out_dim = 8 if velhead else 5
        self.output_head = nn.Sequential(
            RMSNorm(head_in),
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)
        with torch.no_grad():
            # Initialize inertial and collider predictor gates to sigmoid(-3.0) ≈ 0.047
            self.output_head[-1].bias[-2:] = -3.0

    def _encode_decode(
        self,
        nodes: torch.Tensor,
        colliders: torch.Tensor,
        influence: torch.Tensor,
        fluid_basis: torch.Tensor,
        obs_basis: torch.Tensor,
        packing: torch.Tensor,
        pos_pred: torch.Tensor,
    ) -> torch.Tensor:
        single = nodes.ndim == 2
        tokens, local_cat = self.tokenizer(nodes, colliders, influence, fluid_basis, obs_basis, packing)
        tokens_b = tokens.unsqueeze(0) if single else tokens
        pos_b = pos_pred.unsqueeze(0) if single else pos_pred
        if self.config.get("translation_invariant", False):
            pos_b = pos_b.float() - pos_b.float().mean(dim=1, keepdim=True)
        super_tokens, super_pos, _ = self.encoder(tokens_b, pos_b)
        decoded = self.decoder(tokens_b, pos_b, super_tokens, super_pos)
        if single:
            decoded = decoded.squeeze(0)
        combined = torch.cat([decoded, local_cat], dim=-1)
        return self.output_head(combined).float()

    def step_from_history(
        self,
        history: torch.Tensor,
        poses: torch.Tensor,
        velocities: torch.Tensor,
        kinds: torch.Tensor,
        sizes: torch.Tensor,
        friction: torch.Tensor,
        material: torch.Tensor,
        dt: float = 0.005,
        support: float = 0.07,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        sdf: tuple | None = None,
        return_vel: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict next particle positions [N, 3] (and optional velocity [N, 3]) using WorldParticle."""
        pos_cur = history[:, -1]
        vel_cur = (history[:, -1] - history[:, -2]) / dt
        pos_pred, vel_pred, nodes, colliders, influence = features_world_particle(
            history,
            poses,
            velocities,
            kinds,
            sizes,
            friction,
            material,
            dt=dt,
            support=support,
            far_support_scale=self.config.get("far_support_scale", 5.0),
            far_gap_weight=self.config.get("far_gap_weight", 0.0),
            far_influence_weight=self.config.get("far_influence_weight", 0.0),
            use_actual_velocity=self.config.get("use_actual_velocity", False),
            origin=origin,
            gravity=gravity,
            sdf=sdf,
        )
        fluid_basis, obs_basis, packing = self.tokenizer.cconv.compute_bases(
            pos_pred,
            vel_pred,
            colliders,
            influence,
            center_positions=bool(self.config.get("translation_invariant", False)),
        )
        if self.training:
            raw_out = checkpoint(
                self._encode_decode,
                nodes,
                colliders,
                influence,
                fluid_basis,
                obs_basis,
                packing,
                pos_pred,
                use_reentrant=False,
            )
        else:
            raw_out = self._encode_decode(
                nodes,
                colliders,
                influence,
                fluid_basis,
                obs_basis,
                packing,
                pos_pred,
            )
        scale = self.config["displacement_scale"]
        gates = torch.sigmoid(raw_out[:, -2:])
        g_inert = gates[:, 0:1]
        g_col = gates[:, 1:2]
        if self.config["velhead"]:
            dx_corr = raw_out[:, 0:3]
            dv_corr = raw_out[:, 3:6]
            corr = scale * dx_corr + (dt * scale * 10.0) * dv_corr
        else:
            dv_corr = torch.zeros_like(pos_cur)
            corr = scale * raw_out[:, 0:3]

        dx_inert = vel_cur * dt
        v_surf = colliders[:, :, 4:7] + vel_cur.unsqueeze(1)
        dx_col = (influence.unsqueeze(-1) * v_surf).sum(dim=1) * dt
        pos_next = pos_cur + g_inert * dx_inert + g_col * dx_col + corr
        if return_vel:
            # Upstream WorldParticle `velhead` velocity prediction (without double-counting dv_corr)
            vel_next = g_inert * vel_cur + g_col * (dx_col / dt) + (scale * 10.0) * dv_corr
            return pos_next, vel_next
        return pos_next

    def step_from_history_batched(
        self,
        history_env: torch.Tensor,
        poses_env: torch.Tensor,
        velocities_env: torch.Tensor,
        kinds: torch.Tensor,
        sizes: torch.Tensor,
        friction: torch.Tensor,
        material: torch.Tensor,
        dt: float = 0.005,
        support: float = 0.07,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        return_vel: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict next particle positions [E, M, 3] across E parallel environments in one batched pass."""
        pos_cur = history_env[:, :, -1]
        vel_cur = (history_env[:, :, -1] - history_env[:, :, -2]) / dt
        pos_pred, vel_pred, nodes, colliders, influence = features_world_particle_batched(
            history_env,
            poses_env,
            velocities_env,
            kinds,
            sizes,
            friction,
            material,
            dt=dt,
            support=support,
            far_support_scale=self.config.get("far_support_scale", 5.0),
            far_gap_weight=self.config.get("far_gap_weight", 0.0),
            far_influence_weight=self.config.get("far_influence_weight", 0.0),
            use_actual_velocity=self.config.get("use_actual_velocity", False),
            origin=origin,
            gravity=gravity,
        )
        fluid_basis, obs_basis, packing = self.tokenizer.cconv.compute_bases_batched(
            pos_pred,
            vel_pred,
            colliders,
            influence,
            center_positions=bool(self.config.get("translation_invariant", False)),
        )
        raw_out = self._encode_decode(
            nodes,
            colliders,
            influence,
            fluid_basis,
            obs_basis,
            packing,
            pos_pred,
        )
        scale = self.config["displacement_scale"]
        gates = torch.sigmoid(raw_out[..., -2:])
        g_inert = gates[..., 0:1]
        g_col = gates[..., 1:2]
        if self.config["velhead"]:
            dx_corr = raw_out[..., 0:3]
            dv_corr = raw_out[..., 3:6]
            corr = scale * dx_corr + (dt * scale * 10.0) * dv_corr
        else:
            dv_corr = torch.zeros_like(pos_cur)
            corr = scale * raw_out[..., 0:3]

        dx_inert = vel_cur * dt
        v_surf = colliders[:, :, :, 4:7] + vel_cur.unsqueeze(2)
        dx_col = (influence.unsqueeze(-1) * v_surf).sum(dim=2) * dt
        pos_next = pos_cur + g_inert * dx_inert + g_col * dx_col + corr
        if return_vel:
            vel_next = g_inert * vel_cur + g_col * (dx_col / dt) + (scale * 10.0) * dv_corr
            return pos_next, vel_next
        return pos_next

    def forward(
        self,
        nodes: torch.Tensor,
        colliders: torch.Tensor,
        influence: torch.Tensor,
        fluid_basis: torch.Tensor,
        obs_basis: torch.Tensor,
        packing: torch.Tensor,
        pos_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Direct forward pass returning displacement [N, 3]."""
        raw_out = self._encode_decode(nodes, colliders, influence, fluid_basis, obs_basis, packing, pos_pred)
        scale = self.config["displacement_scale"]
        if self.config["velhead"]:
            dx_corr, dv_corr = raw_out.chunk(2, dim=-1)
            return scale * dx_corr + (0.05 * scale) * dv_corr
        return scale * raw_out
