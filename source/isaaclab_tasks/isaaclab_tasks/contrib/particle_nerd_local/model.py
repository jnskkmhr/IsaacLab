# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ParticleNeRD-Local: Translation-invariant local-to-global Transolver with predictor-corrector & stress head.

Addresses all 6 physical requirements from Gilles' review and WorldParticle's drift-free formulation:
  1. Strict translation invariance: global particle positions `(x, y, z)` never enter the network MLPs.
  2. Semi-implicit Euler predictor + learned residual correction `(dx_corr, dv_corr)` with explicit velocity state.
  3. Full-bed local packing fraction `phi` and its recent rate of change `dphi/dt` (`-div(v)`) reconstructed on
     the complete bed before particle selection.
  4. Explicit external loading conditioning (`collider_loading`: gravity, collider inverse mass/inertia,
     prescribed/free motion flag, and external applied wrench).
  5. Direct local geometry & CConv skip connection into shared displacement, velocity, and continuum stress
     heads (`pressure >= 0` and deviatoric stress `tau_dev`).
  6. Consistent stress-to-contact boundary traction integration for force and torque reconstruction, plus
     contact-aware particle selection that preserves stationary particles near loaded interfaces.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .features import LocalNeighborhoodAggregator, features_particle_nerd_local, features_particle_nerd_local_batched


class ParticleRotaryEmbedding(nn.Module):
    """3D Rotary Position Embedding (RoPE) over continuous 3D coordinates (x, y, z).

    Because RoPE rotates Q by angle(pos_q) and K by angle(pos_k), the attention inner product
    depends ONLY on the relative 3D displacement (pos_q - pos_k), preserving strict global
    translation invariance while encoding full 3D topology across particles and slices.
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
        """Compute [..., head_dim] cosine and sine tables from [..., 3] 3D positions."""
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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    c = cos.to(dtype=x.dtype)
    s = sin.to(dtype=x.dtype)
    return x * c + _rotate_half(x) * s


class RMSNorm(nn.Module):
    """Root-Mean-Square normalization without additive mean-shift bias (matching WorldParticle)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True)
        return (self.weight * x.float() * torch.rsqrt(norm + self.eps)).to(dtype=x.dtype)


class FeedForwardSwiGLU(nn.Module):
    """SwiGLU feed-forward network with 2/3 hidden-dim scaling aligned to 8 (matching WorldParticle)."""

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
    """Aggregate N particles into S learned 3D-localized slices with 3D-RoPE, attend, and distribute back."""

    def __init__(self, hidden: int, heads: int, slices: int, spatial_slice_assign: bool = True):
        super().__init__()
        self.heads, self.width = heads, hidden // heads
        self.slices = int(slices)
        self.spatial_slice_assign = bool(spatial_slice_assign)
        rope_dim = max(6, (self.width // 6) * 6)
        self.rope3d = ParticleRotaryEmbedding(dim=min(rope_dim, self.width))
        self.project_geometry = nn.Linear(hidden, hidden)
        self.project_values = nn.Linear(hidden, hidden)
        self.project_slices = nn.Linear(self.width, slices)
        nn.init.orthogonal_(self.project_slices.weight)
        self.temperature = nn.Parameter(torch.full((heads, 1, 1), 0.5))
        if self.spatial_slice_assign:
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
        single = pos.ndim == 2
        if single:
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
        if self.spatial_slice_assign:
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
        return pos_rel, pos_rel_dt, p_cos, p_sin, None, None, None

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

        if self.spatial_slice_assign:
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
        else:
            if not self.training:
                w_g = self.project_geometry.weight.view(self.heads, self.width, -1)
                b_g = self.project_geometry.bias.view(self.heads, self.width)
                w_s = self.project_slices.weight
                b_s = self.project_slices.bias
                w_gs = torch.einsum("sw,hwd->hsd", w_s, w_g).reshape(self.heads * w_s.shape[0], -1)
                b_gs = (torch.einsum("sw,hw->hs", w_s, b_g) + b_s.unsqueeze(0)).reshape(-1)
                logits = F.linear(x, w_gs, b_gs).view(b, n, self.heads, -1).transpose(1, 2)
                inv_temp = (1.0 / self.temperature.clamp(0.5, 5.0)).to(logits.dtype)
                weights = (logits * inv_temp).softmax(dim=-1, dtype=values.dtype)
                w_t = weights.transpose(-2, -1)
            else:
                geometry = self.project_geometry(x).view(b, n, self.heads, self.width).transpose(1, 2)
                logits = self.project_slices(geometry)
                weights = (logits.float() / self.temperature.clamp(0.5, 5.0)).softmax(-1)
                w_t = weights.transpose(-2, -1).to(values.dtype)

        # Compute slice tokens [B, H, S, W] and 3D slice centroids `slice_pos` [B, H, S, 3]
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

        # 1. 3D-RoPE Slice-to-Slice Self-Attention (encodes relative 3D vector between slice centroids p_s1 - p_s2)
        s_cos, s_sin = self.rope3d.cos_sin(slice_pos, self.width)
        q, k, v = self.qkv(tokens).chunk(3, -1)
        if self.spatial_slice_assign:
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)
        q = apply_rope(q, s_cos, s_sin)
        k = apply_rope(k, s_cos, s_sin)
        attended = F.scaled_dot_product_attention(q, k, v)

        # 2. 3D-RoPE Particle-to-Slice Deslicing (encodes relative 3D vector x_i - p_s)
        dq = self.deslice_q(values)
        dk = self.deslice_k(attended)
        if self.spatial_slice_assign:
            dq = self.deslice_q_norm(dq).to(attended.dtype)
            dk = self.deslice_k_norm(dk).to(attended.dtype)
        q_p = apply_rope(dq, p_cos, p_sin)  # [B, H, N, W]
        k_s = apply_rope(dk, s_cos, s_sin)  # [B, H, S, W]
        points_soft = weights.to(attended.dtype) @ attended
        points_rope = F.scaled_dot_product_attention(q_p, k_s, attended)
        points = points_soft + points_rope
        out = self.output(points.transpose(1, 2).reshape(b, n, -1))
        return out[0] if single else out


class Block(nn.Module):
    def __init__(
        self, hidden: int, heads: int, slices: int, spatial_slice_assign: bool = True, use_rms_swiglu: bool = True
    ):
        super().__init__()
        self.norm_attention = RMSNorm(hidden) if use_rms_swiglu else nn.LayerNorm(hidden)
        self.norm_mlp = RMSNorm(hidden) if use_rms_swiglu else nn.LayerNorm(hidden)
        self.attention = PhysicsAttention(hidden, heads, slices, spatial_slice_assign=spatial_slice_assign)
        self.mlp = (
            FeedForwardSwiGLU(hidden, 4 * hidden)
            if use_rms_swiglu
            else nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))
        )

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


class ConsistentStressContactHead(nn.Module):
    """Shared continuum stress head (compressive pressure p >= 0 + deviatoric stress) & boundary traction integrator."""

    def __init__(self, in_dim: int, hidden: int = 128, characteristic_pressure: float = 1500.0):
        super().__init__()
        self.characteristic_pressure = float(characteristic_pressure)
        self.stress_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),  # [raw_pressure, tau_x, tau_y, tau_z]
        )
        nn.init.zeros_(self.stress_mlp[-1].weight)
        nn.init.zeros_(self.stress_mlp[-1].bias)
        with torch.no_grad():
            # Start with slight positive hydrostatic baseline when in contact
            self.stress_mlp[-1].bias[0] = -1.0

    def forward(
        self,
        head_features: torch.Tensor,
        pos_pred: torch.Tensor,
        vel_pred: torch.Tensor,
        colliders: torch.Tensor,
        influence: torch.Tensor,
        packing: torch.Tensor,
        collider_poses: torch.Tensor,
        particle_radius: float = 0.01,
        support: float = 0.07,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict per-particle pressure [N, 1], deviatoric stress [N, 3], and integrated wrenches [C, 6]."""
        raw = self.stress_mlp(head_features).float()
        # Local packing `phi = packing[:, 1]` and compaction rate `dphi_dt = packing[:, 2]`
        phi_pack = packing[:, 1:2].clamp_min(0.0)
        compaction_boost = 1.0 + 0.5 * F.relu(packing[:, 2:3])
        # Overburden depth above tray floor + strictly additive collider bearing bulb (never shadows under object)
        depth_est = (-pos_pred[:, 2:3]).clamp_min(0.0)
        p_overburden = 1600.0 * 9.81 * depth_est * (phi_pack / 0.35).clamp(0.5, 1.4)
        # Dynamic collider proximity & Boussinesq sub-collider bearing bulb (strictly additive >= 0)
        gap_norm = colliders[:, :, 0]  # [N, C]
        normals = F.normalize(colliders[:, :, 1:4], dim=-1, eps=1e-8)  # [N, C, 3] outward from collider
        v_rel = colliders[:, :, 4:7]  # [N, C, 3] v_collider_surf - v_particle
        mu = colliders[:, :, 7:8].clamp_min(0.0)  # [N, C, 1]
        v_surf = v_rel + vel_pred.unsqueeze(1)  # [N, C, 3] world surface velocity of collider
        v_surf_speed = v_surf.norm(dim=-1)  # [N, C]
        v_approach = F.relu((v_rel * normals).sum(dim=-1))  # [N, C]
        # Sub-collider Boussinesq bulb under collider 0
        dx0 = pos_pred[:, 0:1] - collider_poses[0, 0]
        dy0 = pos_pred[:, 1:2] - collider_poses[0, 1]
        dz0 = (collider_poses[0, 2] - pos_pred[:, 2:3]).clamp_min(0.0)
        under_obj = (pos_pred[:, 2:3] <= collider_poses[0, 2] + 0.04).float()
        boussinesq_bulb = (
            torch.exp(-0.5 * ((dx0 / 0.14) ** 2 + (dy0 / 0.11) ** 2)) / (1.0 + (dz0 / 0.12) ** 2) * under_obj
        )
        obj_activity = 1.0 + 6.0 * v_surf_speed[:, 0:1] + 6.0 * v_approach[:, 0:1]
        bearing_boost = boussinesq_bulb * obj_activity
        pressure = p_overburden + self.characteristic_pressure * phi_pack * compaction_boost * (
            0.15 * F.softplus(raw[:, 0:1]) + bearing_boost * F.softplus(raw[:, 0:1] + 1.5)
        )
        deviatoric = self.characteristic_pressure * phi_pack * (0.4 + 1.5 * bearing_boost) * raw[:, 1:4]

        # Consistent stress-to-contact boundary traction integration over each collider b = 0..C-1
        # Only integrate over the 1-layer boundary contact shell (gap_norm <= 2 * r_p / support)
        area_eff = float((2.0 * particle_radius) ** 2)

        gap_m = gap_norm * support
        shell_width = max(2.2 * float(particle_radius), 0.025)
        gate_raw = (1.0 - F.relu(gap_m) / shell_width).clamp(0.0, 1.0)
        contact_weight = (gate_raw * gate_raw * gate_raw) * (influence > 0.0).float()  # [N, C]

        # Unilateral normal traction on collider b along -n_ib: f_n,ib = p_i * A_eff * contact_weight
        fn_mag = pressure * area_eff * contact_weight  # [N, C]
        f_normal = -fn_mag.unsqueeze(-1) * normals  # [N, C, 3]

        # Deviatoric + tangential slip traction projected onto tangent plane and bounded by ||f_t|| <= mu * f_n
        v_rel_n = (v_rel * normals).sum(dim=-1, keepdim=True)
        v_rel_t = v_rel - v_rel_n * normals
        tau_exp = (
            deviatoric.unsqueeze(1) * area_eff * contact_weight.unsqueeze(-1)
            - 15.0 * area_eff * contact_weight.unsqueeze(-1) * v_rel_t
        )
        tau_tangent = tau_exp - (tau_exp * normals).sum(dim=-1, keepdim=True) * normals
        tau_norm = torch.sqrt((tau_tangent * tau_tangent).sum(dim=-1, keepdim=True) + 1e-8)
        max_friction = mu * fn_mag.unsqueeze(-1)
        f_tangent = max_friction * (tau_tangent / tau_norm) * torch.tanh(tau_norm / (max_friction + 1e-6))

        f_particle_on_collider = f_normal + f_tangent  # [N, C, 3]
        # Reconstruct contact surface point s_ib = x_i - (gap_m + r_p) * n_ib and lever arm r_ib = s_ib - c_b
        dist_m = gap_m + particle_radius
        surface_pt = pos_pred.unsqueeze(1) - dist_m.unsqueeze(-1) * normals  # [N, C, 3]
        arm = surface_pt - collider_poses[None, :, :3]  # [N, C, 3]
        torque_particle_on_collider = torch.linalg.cross(arm, f_particle_on_collider, dim=-1)  # [N, C, 3]

        wrenches = torch.cat(
            (f_particle_on_collider.sum(dim=0), torque_particle_on_collider.sum(dim=0)),
            dim=-1,
        )  # [C, 6]
        return pressure, deviatoric, wrenches


class ParticleNeRDLocalNetwork(nn.Module):
    """Translation-invariant ParticleNeRD with full-bed CConv + packing rate, SDF colliders, and stress head."""

    def __init__(
        self,
        hidden: int = 128,
        cconv_dim: int = 64,
        heads: int = 4,
        slices: int = 64,
        layers: int = 4,
        history: int = 3,
        radius: float = 0.072,
        displacement_scale: float = 0.02,
        velhead: bool = True,
        far_support_scale: float = 5.0,
        far_gap_weight: float = 0.3,
        far_influence_weight: float = 0.15,
        translation_invariant: bool = True,
        spatial_slice_assign: bool = True,
        use_rms_swiglu: bool = True,
        use_cloud_context: bool = False,
        update_mode: str = "gated",
    ):
        super().__init__()
        if hidden <= 0 or heads <= 0 or hidden % heads != 0 or min(slices, layers) < 1 or history < 2:
            raise ValueError("Positive dimensions, hidden divisible by heads, and history >= 2 required")
        if update_mode not in ("gated", "ballistic"):
            raise ValueError("update_mode must be 'gated' or 'ballistic'")
        if update_mode == "ballistic" and not velhead:
            raise ValueError("The ballistic update requires the velocity head")
        self.config = dict(
            hidden=hidden,
            cconv_dim=cconv_dim,
            heads=heads,
            slices=slices,
            layers=layers,
            history=history,
            radius=radius,
            displacement_scale=displacement_scale,
            velhead=velhead,
            translation_invariant=bool(translation_invariant),
            far_support_scale=float(far_support_scale),
            far_gap_weight=float(far_gap_weight),
            far_influence_weight=float(far_influence_weight),
            spatial_slice_assign=bool(spatial_slice_assign),
            use_rms_swiglu=bool(use_rms_swiglu),
            use_cloud_context=bool(use_cloud_context),
            update_mode=update_mode,
        )
        node_dim = 16 + 3 * (history - 1)
        self.local_agg = LocalNeighborhoodAggregator(
            fluid_in_channels=4,
            obs_in_channels=9,
            filters=cconv_dim,
            radius=radius,
        )
        # Center particle (translation-invariant 22D: velocities + material + dt + gravity) + 6D packing = 28D
        self.particle_encoder = nn.Sequential(
            nn.Linear(node_dim + 6, 2 * cconv_dim),
            nn.GELU(),
            nn.Linear(2 * cconv_dim, cconv_dim),
        )
        # Pairwise collider geometry (8D: gap, normal(3), v_rel(3), mu) + explicit external loading (10D)
        self.collider_encoder = nn.Sequential(
            nn.Linear(8, cconv_dim),
            nn.GELU(),
            nn.Linear(cconv_dim, cconv_dim),
        )
        self.loading_encoder = nn.Sequential(
            nn.Linear(10, cconv_dim),
            nn.GELU(),
            nn.Linear(cconv_dim, cconv_dim),
        )
        # Fuse [conv_obs, conv_fluid, dense_particle + collider] -> hidden
        self.in_proj = nn.Sequential(
            RMSNorm(3 * cconv_dim) if use_rms_swiglu else nn.LayerNorm(3 * cconv_dim),
            nn.Linear(3 * cconv_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden,
                    heads,
                    slices,
                    spatial_slice_assign=spatial_slice_assign,
                    use_rms_swiglu=use_rms_swiglu,
                )
                for _ in range(layers)
            ]
        )

        # Direct local geometry skip into the output heads:
        #   global slice-attended token `hidden`
        # + local CConv/dense features `3 * cconv_dim`
        # + nearest-collider raw local geometry `8` (gap, normal, v_rel, mu)
        # + local packing & compaction rate `6` (count, phi, dphi_dt, v_rel_mean)
        head_in = hidden + 3 * cconv_dim + 8 + 6
        base_dim = 8 if velhead else 5
        # The ballistic update appends one logit (explicit-gravity "support" gate) after the original outputs.
        out_dim = base_dim + (1 if update_mode == "ballistic" else 0)
        self.motion_head = nn.Sequential(
            RMSNorm(head_in) if use_rms_swiglu else nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)
        with torch.no_grad():
            self.motion_head[-1].bias[base_dim - 2 : base_dim] = -3.0
            if update_mode == "ballistic":
                self.motion_head[-1].bias[base_dim] = 3.0  # starts fully supported: no explicit gravity

        if self.config["use_cloud_context"]:
            self.cloud_context_encoder = nn.Sequential(
                nn.Linear(22, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            nn.init.zeros_(self.cloud_context_encoder[-1].weight)
            nn.init.zeros_(self.cloud_context_encoder[-1].bias)
            self.cloud_head_skip = nn.Sequential(
                nn.Linear(22 + 3 * cconv_dim + 8 + 6, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
            )
            nn.init.zeros_(self.cloud_head_skip[-1].weight)
            nn.init.zeros_(self.cloud_head_skip[-1].bias)

        self.stress_head = ConsistentStressContactHead(in_dim=head_in, hidden=hidden)

    @staticmethod
    def _compute_cloud_context(
        pos: torch.Tensor,
        nodes: torch.Tensor,
        colliders: torch.Tensor,
        multiplicity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute 22D translation-invariant statistical cloud moments, velocity activity, and (e_r, e_theta)."""
        single = pos.ndim == 2
        if single:
            pos = pos.unsqueeze(0)
            nodes = nodes.unsqueeze(0)
            colliders = colliders.unsqueeze(0)
            if multiplicity is not None and multiplicity.ndim == 1:
                multiplicity = multiplicity.unsqueeze(0)
        pos_f = pos.float()
        vel_f = nodes[:, :, 3:6].float()
        col_f = colliders.float()
        if multiplicity is not None:
            m_w = multiplicity.float().unsqueeze(-1)
            m_sum = m_w.sum(dim=1, keepdim=True).clamp_min(1.0)
            p_mean = (pos_f * m_w).sum(dim=1, keepdim=True) / m_sum
            v_mean = (vel_f * m_w).sum(dim=1, keepdim=True) / m_sum
        else:
            p_mean = pos_f.mean(dim=1, keepdim=True)
            v_mean = vel_f.mean(dim=1, keepdim=True)
        p_min_eff = torch.quantile(pos_f, 0.01, dim=1, keepdim=True)
        p_max_eff = torch.quantile(pos_f, 0.99, dim=1, keepdim=True)
        span = (p_max_eff - p_min_eff).clamp_min(0.04)

        d_xy = pos_f[:, :, :2] - p_mean[:, :, :2]
        r_dist = torch.sqrt((d_xy * d_xy).sum(dim=-1, keepdim=True) + 1e-8)
        e_r = d_xy / (r_dist + 1e-5)
        e_theta = torch.cat((-e_r[:, :, 1:2], e_r[:, :, 0:1]), dim=-1)

        u_xy = (d_xy / (0.5 * span[:, :, :2])).clamp(-2.5, 2.5)
        u_z = ((pos_f[:, :, 2:3] - p_min_eff[:, :, 2:3]) / span[:, :, 2:3]).clamp(-0.5, 1.5)
        r_norm = torch.sqrt((u_xy * u_xy).sum(dim=-1, keepdim=True) + 1e-6)
        rel_xy = (d_xy / 0.30).clamp(-4.0, 4.0)
        h_base = ((pos_f[:, :, 2:3] - p_min_eff[:, :, 2:3]) / 0.30).clamp(-0.5, 4.0)
        d_top = ((p_max_eff[:, :, 2:3] - pos_f[:, :, 2:3]) / 0.30).clamp(-0.5, 4.0)
        aspect_x = (span[:, :, 2:3] / span[:, :, 0:1]).clamp(0.0, 4.0).expand_as(u_z) - 0.25
        aspect_y = (span[:, :, 2:3] / span[:, :, 1:2]).clamp(0.0, 4.0).expand_as(u_z) - 0.25
        l_xy = (span[:, :, :2] / 0.50).clamp(0.0, 5.0).expand_as(u_xy)
        l_z = (span[:, :, 2:3] / 0.50).clamp(0.0, 5.0).expand_as(u_z)

        # Cylindrical radial, azimuthal, and vertical relative velocities
        dv_xy = vel_f[:, :, :2] - v_mean[:, :, :2]
        v_r = ((dv_xy * e_r).sum(dim=-1, keepdim=True) / 1.5).clamp(-4.0, 4.0)
        v_theta = ((dv_xy * e_theta).sum(dim=-1, keepdim=True) / 1.5).clamp(-4.0, 4.0)
        v_z_rel = ((vel_f[:, :, 2:3] - v_mean[:, :, 2:3]) / 1.5).clamp(-4.0, 4.0)
        v_rel_cyl = torch.cat((v_r, v_theta, v_z_rel), dim=-1)
        v_mean_cloud = (v_mean / 2.0).clamp(-4.0, 4.0).expand_as(vel_f)

        is_floor = col_f[:, :, :, 3] > 0.7
        floor_gaps = torch.where(is_floor, col_f[:, :, :, 0], torch.full_like(col_f[:, :, :, 0], 1.0))
        floor_gap_i = floor_gaps.amin(dim=-1, keepdim=True)
        cloud_floor_min = torch.quantile(floor_gap_i, 0.05, dim=1, keepdim=True).expand_as(floor_gap_i)
        if col_f.shape[2] >= 5:
            wall_min_per_p = col_f[:, :, -4:, 0].amin(dim=-1, keepdim=True)
            wall_min = wall_min_per_p.amin(dim=1, keepdim=True).expand_as(floor_gap_i)
        else:
            wall_min = torch.ones_like(floor_gap_i)

        # Purely learned cloud context: w_open = 1.0 everywhere (no hardcoded wall or aspect-ratio gates)
        w_open = torch.ones_like(floor_gap_i)

        ctx = torch.cat(
            (
                u_xy,
                u_z,
                r_norm,
                rel_xy,
                h_base,
                d_top,
                aspect_x,
                aspect_y,
                l_xy,
                l_z,
                v_rel_cyl,
                v_mean_cloud,
                floor_gap_i,
                cloud_floor_min,
                wall_min,
            ),
            dim=-1,
        )
        if single:
            return ctx[0], w_open[0], e_r[0], e_theta[0]
        return ctx, w_open, e_r, e_theta

    @staticmethod
    def select_contact_and_packing_aware_particles(
        pos_pred: torch.Tensor,
        vel_pred: torch.Tensor,
        colliders: torch.Tensor,
        influence: torch.Tensor,
        packing: torch.Tensor,
        active_k: int,
    ) -> torch.Tensor:
        """Select K active particles using full-bed contact proximity, compaction rate, and speed.

        Preserves stationary particles near loaded interfaces (`influence > 0`) and compressing regions
        (`|dphi_dt| > 0`) without any background anchor heuristic.
        """
        n_pts = pos_pred.shape[0]
        k_act = min(int(active_k), n_pts)
        max_inf = influence.amax(dim=-1)
        compaction_mag = packing[:, 2].abs()
        speed = vel_pred.norm(dim=-1)
        score = -4.0 * max_inf - 0.5 * compaction_mag - 0.5 * speed
        _, topk = torch.topk(score, k=k_act, dim=0, largest=False, sorted=False)
        return topk

    def _encode_With_local_skip(
        self,
        pos_pred: torch.Tensor,
        nodes: torch.Tensor,
        colliders: torch.Tensor,
        influence: torch.Tensor,
        fluid_basis: torch.Tensor,
        obs_basis: torch.Tensor,
        packing: torch.Tensor,
        collider_loading: torch.Tensor | None = None,
        multiplicity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = self.particle_encoder[0].weight.dtype
        conv_fluid, conv_obs = self.local_agg(fluid_basis, obs_basis)
        nodes_in = torch.cat([nodes.to(dt), packing.to(dt)], dim=-1)
        dense_p = self.particle_encoder(nodes_in)

        if influence.shape[-1] > 6:
            dyn_inf = influence[..., :-5]
            dyn_scale = (dyn_inf.sum(dim=-1, keepdim=True) / 1.25).clamp_min(1.0)
            inf_eff = torch.cat((dyn_inf / dyn_scale, influence[..., -5:]), dim=-1)
        else:
            inf_eff = influence

        col_emb = self.collider_encoder(colliders.to(dt))
        if collider_loading is not None:
            col_emb = col_emb + self.loading_encoder(collider_loading.to(dt)).unsqueeze(0)
        col_weighted = (col_emb * inf_eff.to(dt).unsqueeze(-1)).sum(dim=-2)

        local_cat = torch.cat([conv_obs, conv_fluid, dense_p + col_weighted], dim=-1)
        x = self.in_proj(local_cat)
        cloud_ctx = None
        w_open = None
        e_r = None
        e_theta = None
        if self.config.get("use_cloud_context", False):
            cloud_ctx_f, w_open_f, e_r_f, e_theta_f = self._compute_cloud_context(
                pos_pred, nodes, colliders, multiplicity=multiplicity
            )
            cloud_ctx = cloud_ctx_f.to(dt)
            w_open = w_open_f.to(dt)
            e_r = e_r_f.float()
            e_theta = e_theta_f.float()
            x = x + w_open * self.cloud_context_encoder(cloud_ctx)
        if not self.training and len(self.blocks) > 0:
            rope_cache = self.blocks[0].attention.precompute_rope_cache(pos_pred, x.dtype, multiplicity=multiplicity)
            for block in self.blocks:
                x = block(x, pos_pred, rope_cache=rope_cache, multiplicity=multiplicity)
        else:
            for block in self.blocks:
                x = checkpoint(block, x, pos_pred, use_reentrant=False) if self.training else block(x, pos_pred)
        # Direct local geometry skip: smooth influence-weighted collider geometry (avoids discontinuous argmax)
        if self.config.get("spatial_slice_assign", True):
            nearest_geom = (colliders.to(dt) * inf_eff.to(dt).unsqueeze(-1)).sum(dim=-2)
        else:
            best_col = inf_eff.argmax(dim=-1, keepdim=True).unsqueeze(-1).expand(*inf_eff.shape[:-1], 1, 8)
            nearest_geom = torch.gather(colliders.to(dt), -2, best_col).squeeze(-2)
        head_features = torch.cat([x, local_cat, nearest_geom, packing.to(dt)], dim=-1)
        raw_motion = self.motion_head(head_features).float()
        if cloud_ctx is not None and w_open is not None and e_r is not None and e_theta is not None:
            skip_in = torch.cat([cloud_ctx, local_cat, nearest_geom, packing.to(dt)], dim=-1)
            cyl_out = self.cloud_head_skip(skip_in).float()
            dx_r = cyl_out[..., 0:1]
            dx_th = cyl_out[..., 1:2]
            dx_z = cyl_out[..., 2:3]
            dv_r = cyl_out[..., 3:4]
            dv_th = cyl_out[..., 4:5]
            dv_z = cyl_out[..., 5:6]
            g_skip = cyl_out[..., 6:8]
            dx_xy = dx_r * e_r + dx_th * e_theta
            dv_xy = dv_r * e_r + dv_th * e_theta
            cart_skip = torch.cat((dx_xy, dx_z, dv_xy, dv_z, g_skip, cyl_out[..., 8:]), dim=-1)
            raw_motion = raw_motion + w_open.float() * cart_skip
        return raw_motion, head_features

    @staticmethod
    def _hard_gate(logit: torch.Tensor) -> torch.Tensor:
        """Sigmoid stretched so that it saturates exactly at 0 and 1 (a plain sigmoid never reaches either)."""
        return (1.2 * torch.sigmoid(logit) - 0.1).clamp(0.0, 1.0)

    def _ballistic_update(
        self,
        raw_out: torch.Tensor,
        pos_cur: torch.Tensor,
        vel_cur: torch.Tensor,
        dx_col: torch.Tensor,
        dt: float,
        gravity: tuple[float, float, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predictor-corrector update with exact inertial motion and explicit gravity for unsupported particles.

        Three hard gates (exactly 0 or 1 is reachable): inertial retention ``g_inert``, collider coupling ``g_col``
        and support ``s``. A particle with ``g_inert = 1``, ``g_col = 0`` and ``s = 0`` follows exact semi-implicit
        free flight, so its momentum is conserved and it accelerates at ``gravity``; a supported particle
        (``s = 1``) receives no gravity term and moves only through the learned corrections.
        """
        scale = self.config["displacement_scale"]
        g_inert = self._hard_gate(raw_out[..., 6:7])
        g_col = self._hard_gate(raw_out[..., 7:8])
        free = 1.0 - self._hard_gate(raw_out[..., 8:9])
        dv_learned = (scale * 10.0) * raw_out[..., 3:6]
        dx_learned = scale * raw_out[..., 0:3] + dt * dv_learned
        # Build the gravity vector from scalar arithmetic: no host-to-device copy, so CUDA graph capture is safe.
        zero = torch.zeros_like(free)
        gravity_vec = torch.cat((zero + gravity[0], zero + gravity[1], zero + gravity[2]), dim=-1)
        dv_gravity = free * gravity_vec * dt
        vel_next = g_inert * (vel_cur + dv_gravity + dv_learned) + g_col * (dx_col / dt)
        pos_next = pos_cur + g_inert * (vel_cur * dt + dv_gravity * dt + dx_learned) + g_col * dx_col
        return pos_next, vel_next

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
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        sdf: tuple | None = None,
        collider_loading: torch.Tensor | None = None,
        active_k: int | None = None,
        return_vel: bool = False,
        return_stress_contact: bool = False,
    ):
        """Step particles via semi-implicit Euler predictor + local-to-global Transolver residual correction."""
        pos_cur = history[:, -1]
        vel_cur = (history[:, -1] - history[:, -2]) / dt

        # 1. Reconstruct predictor state, SDF collider pairs, and full-bed packing + compaction rate BEFORE selection
        n_dyn = max(1, int(kinds.shape[0]) - 5) if kinds.shape[0] > 6 else 1
        pos_pred, vel_pred, nodes, colliders, influence = features_particle_nerd_local(
            history,
            poses,
            velocities,
            kinds,
            sizes,
            friction,
            material,
            dt=dt,
            support=support,
            far_support_scale=self.config["far_support_scale"],
            far_gap_weight=self.config["far_gap_weight"],
            far_influence_weight=self.config["far_influence_weight"] / float(n_dyn),
            gravity=gravity,
            sdf=sdf,
        )
        fluid_basis, obs_basis, packing = self.local_agg.compute_bases(
            pos_pred,
            vel_pred,
            colliders,
            influence,
        )

        n_pts = pos_cur.shape[0]
        if active_k is not None and self.config.get("update_mode", "gated") == "ballistic":
            raise NotImplementedError("Active-particle selection is not defined for the ballistic update")
        if active_k is not None and 0 < int(active_k) < n_pts:
            topk = self.select_contact_and_packing_aware_particles(
                pos_pred, vel_pred, colliders, influence, packing, int(active_k)
            )
            raw_sub, head_sub = self._encode_With_local_skip(
                pos_pred[topk],
                nodes[topk],
                colliders[topk],
                influence[topk],
                fluid_basis[topk],
                obs_basis[topk],
                packing[topk],
                collider_loading=collider_loading,
            )
            raw_out = torch.zeros((n_pts, raw_sub.shape[-1]), dtype=torch.float32, device=pos_cur.device)
            raw_out[:, -2:] = -10.0
            raw_out[topk] = raw_sub
            head_features = head_sub
            stress_pos = pos_pred[topk]
            stress_vel = vel_pred[topk]
            stress_col = colliders[topk]
            stress_inf = influence[topk]
            stress_pack = packing[topk]
        else:
            raw_out, head_features = self._encode_With_local_skip(
                pos_pred,
                nodes,
                colliders,
                influence,
                fluid_basis,
                obs_basis,
                packing,
                collider_loading=collider_loading,
            )
            stress_pos = pos_pred
            stress_vel = vel_pred
            stress_col = colliders
            stress_inf = influence
            stress_pack = packing

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
        v_rel = colliders[:, :, 4:7]
        v_surf = v_rel + vel_cur.unsqueeze(1)
        if influence.shape[-1] > 6:
            dyn_inf = influence[:, :-5]
            dyn_scale = (dyn_inf.sum(dim=-1, keepdim=True) / 1.25).clamp_min(1.0)
            inf_col = torch.cat((dyn_inf / dyn_scale, influence[:, -5:]), dim=-1)
        else:
            dyn_inf = influence[:, :1]
            inf_col = influence
        dx_col = (inf_col.unsqueeze(-1) * v_surf).sum(dim=1) * dt
        if self.config.get("update_mode", "gated") == "ballistic":
            pos_next, vel_next = self._ballistic_update(raw_out, pos_cur, vel_cur, dx_col, dt, gravity)
        else:
            pos_next = pos_cur + g_inert * (dx_inert + corr) + g_col * dx_col
            vel_next = g_inert * (vel_cur + (scale * 10.0) * dv_corr) + g_col * (dx_col / dt)

        if return_stress_contact:
            p_radius = float(0.02 * material[11].item())
            pressure, deviatoric, wrenches = self.stress_head(
                head_features,
                stress_pos,
                stress_vel,
                stress_col,
                stress_inf,
                stress_pack,
                poses,
                particle_radius=p_radius,
                support=support,
            )
            return pos_next, vel_next, dict(pressure=pressure, deviatoric=deviatoric, wrenches=wrenches)
        if return_vel:
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
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        sdf: tuple | None = None,
        multiplicity: torch.Tensor | None = None,
        return_vel: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict next particle positions [E, M, 3] across E parallel environments in one batched pass."""
        pos_cur = history_env[:, :, -1]
        vel_cur = (history_env[:, :, -1] - history_env[:, :, -2]) / dt
        n_dyn = max(1, int(kinds.shape[0]) - 5) if kinds.shape[0] > 6 else 1
        pos_pred, vel_pred, nodes, colliders, influence = features_particle_nerd_local_batched(
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
            far_gap_weight=self.config.get("far_gap_weight", 0.3),
            far_influence_weight=self.config.get("far_influence_weight", 0.15) / float(n_dyn),
            gravity=gravity,
            sdf=sdf,
        )
        fluid_basis, obs_basis, packing = self.local_agg.compute_bases_batched(
            pos_pred,
            vel_pred,
            colliders,
            influence,
        )
        raw_out, _ = self._encode_With_local_skip(
            pos_pred,
            nodes,
            colliders,
            influence,
            fluid_basis,
            obs_basis,
            packing,
            collider_loading=None,
            multiplicity=multiplicity,
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
        v_rel = colliders[:, :, :, 4:7]
        v_surf = v_rel + vel_cur.unsqueeze(2)
        if influence.shape[-1] > 6:
            dyn_inf = influence[:, :, :-5]
            dyn_scale = (dyn_inf.sum(dim=-1, keepdim=True) / 1.25).clamp_min(1.0)
            inf_col = torch.cat((dyn_inf / dyn_scale, influence[:, :, -5:]), dim=-1)
        else:
            dyn_inf = influence[:, :, :1]
            inf_col = influence
        dx_col = (inf_col.unsqueeze(-1) * v_surf).sum(dim=2) * dt
        if self.config.get("update_mode", "gated") == "ballistic":
            pos_next, vel_next = self._ballistic_update(raw_out, pos_cur, vel_cur, dx_col, dt, gravity)
        else:
            pos_next = pos_cur + g_inert * (dx_inert + corr) + g_col * dx_col
            vel_next = g_inert * (vel_cur + (scale * 10.0) * dv_corr) + g_col * (dx_col / dt)
        if return_vel:
            return pos_next, vel_next
        return pos_next
