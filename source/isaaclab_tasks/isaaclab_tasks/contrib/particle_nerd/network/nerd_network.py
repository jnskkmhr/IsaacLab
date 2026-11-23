# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Particle NeRD network: one learned time step of a particle bed that interacts with rigid colliders.

Inputs of a step
    - ``history`` [N, 3, 3]: the last three particle positions [m]. Only their differences (two velocities) enter
      the network, so the model is translation invariant.
    - ``poses`` [C, 7] and ``velocities`` [C, 6]: collider pose (xyz, quaternion xyzw) and twist (linear, angular),
      with ``kinds`` [C], ``sizes`` [C, 3] and ``friction`` [C]. The last five colliders are the tray boxes.
    - ``material`` [13]: see :func:`~.feature_builder.material_features`.

Pipeline
    1. Predictor: advance every particle with gravity (:func:`~.feature_builder.integrate_pos_vel`). The
       predicted state is used only to build the inputs below.
    2. Inputs: particle features, particle-collider pair features and influence weights
       (:func:`~.feature_builder.compute_features`), and neighbourhood bases plus a packing state
       (:class:`~..modules.cnn.LocalNeighborhoodAggregator`).
    3. Network: local encoders, Transolver blocks (:class:`~..modules.attention.TransolverBlock`), and an output
       head with a skip from the local features.

Output and update (:meth:`ParticleNeRDNetwork._update`)
    Per particle the head returns a position correction ``dx`` (scale 0.02 m), a velocity correction ``dv``
    (scale 0.2 m/s) and two gates ``a`` (keep own motion) and ``b`` (follow the collider), both in (0, 1). With
    ``w`` the influence-weighted velocity of the nearby collider surfaces::

        v_next = a * (v + dv) + b * w
        x_next = x + dt * v_next + a * dx

    There is no explicit gravity term: the corrections carry it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..modules.attention import TransolverBlock
from ..modules.cnn import LocalNeighborhoodAggregator
from ..modules.normalization import RMSNorm
from .feature_builder import compute_features, compute_features_batched

ARCHITECTURE = "particle_nerd_local_v1"
"""Architecture tag stored in checkpoints."""

_REMOVED_MODULES = ("stress_head.", "loading_encoder.")
"""Parameter prefixes of modules that older checkpoints contain and this network no longer has."""

_REMOVED_OPTIONS = {
    "velhead": True,
    "translation_invariant": True,
    "spatial_slice_assign": True,
    "use_rms_swiglu": True,
    "use_cloud_context": True,
    "update_mode": "gated",
}
"""Former constructor options and the only value of each that this network implements."""


class ParticleNeRDNetwork(nn.Module):
    """Translation-invariant particle update: local neighbourhood encoder, Transolver blocks, gated output head."""

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
        far_support_scale: float = 5.0,
        far_gap_weight: float = 0.3,
        far_influence_weight: float = 0.15,
    ):
        super().__init__()
        if hidden <= 0 or heads <= 0 or hidden % heads != 0 or min(slices, layers) < 1 or history < 2:
            raise ValueError("Positive dimensions, hidden divisible by heads, and history >= 2 required")
        self.config = dict(
            hidden=hidden,
            cconv_dim=cconv_dim,
            heads=heads,
            slices=slices,
            layers=layers,
            history=history,
            radius=radius,
            displacement_scale=displacement_scale,
            far_support_scale=float(far_support_scale),
            far_gap_weight=float(far_gap_weight),
            far_influence_weight=float(far_influence_weight),
        )
        node_dim = 16 + 3 * (history - 1)
        self.local_agg = LocalNeighborhoodAggregator(
            fluid_in_channels=4,
            obs_in_channels=9,
            filters=cconv_dim,
            radius=radius,
        )
        # Particle features (22D: velocities, material, dt, gravity) + packing state (6D)
        self.particle_encoder = nn.Sequential(
            nn.Linear(node_dim + 6, 2 * cconv_dim),
            nn.GELU(),
            nn.Linear(2 * cconv_dim, cconv_dim),
        )
        # Particle-collider pair features (8D: gap, normal(3), relative surface velocity(3), friction)
        self.collider_encoder = nn.Sequential(
            nn.Linear(8, cconv_dim),
            nn.GELU(),
            nn.Linear(cconv_dim, cconv_dim),
        )
        # Fuse [collider basis, particle basis, particle + collider encodings] -> hidden
        self.in_proj = nn.Sequential(
            RMSNorm(3 * cconv_dim),
            nn.Linear(3 * cconv_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList([TransolverBlock(hidden, heads, slices) for _ in range(layers)])

        # The output head sees the attended token and, as a skip, the local features, the influence-weighted
        # collider geometry (8D) and the packing state (6D).
        head_in = hidden + 3 * cconv_dim + 8 + 6
        out_dim = 8  # position correction (3), velocity correction (3), two gate logits
        self.motion_head = nn.Sequential(
            RMSNorm(head_in),
            nn.Linear(head_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)
        with torch.no_grad():
            self.motion_head[-1].bias[out_dim - 2 : out_dim] = -3.0

        # Cloud context: statistics of the whole particle cloud (22D), added to the tokens and, through a second
        # head that works in cylindrical coordinates about the cloud axis, to the output.
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

    @classmethod
    def from_config(cls, config: dict) -> ParticleNeRDNetwork:
        """Build the network from a stored model configuration, including configurations of older checkpoints.

        Older configurations carry options that were removed. They are accepted if they hold the value this
        network implements, and rejected otherwise.
        """
        config = dict(config)
        for key, supported in _REMOVED_OPTIONS.items():
            if key not in config:
                continue
            value = config.pop(key)
            if value != supported:
                raise ValueError(
                    f"The model configuration has {key}={value!r}. That option was removed from Particle NeRD, "
                    f"which only implements {key}={supported!r}; the checkpoint cannot be loaded."
                )
        return cls(**config)

    def load_checkpoint_state(self, state: dict) -> None:
        """Load trained weights, ignoring the weights of removed modules and failing on any other mismatch."""
        kept = {k: v for k, v in state.items() if not k.startswith(_REMOVED_MODULES)}
        self.load_state_dict(kept, strict=True)

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> tuple[ParticleNeRDNetwork, dict]:
        """Load a trained Particle NeRD checkpoint.

        Args:
            path: Checkpoint file written by the benchmark trainer.
            device: Device of the returned network.

        Returns:
            The network in evaluation mode and the checkpoint dictionary (``config`` holds ``dt`` and the feature
            settings; ``step`` and ``validation`` describe the training state).

        Raises:
            ValueError: If the checkpoint is of another architecture, or was trained with an option that was removed.
        """
        saved = torch.load(Path(path).expanduser(), map_location=str(device), weights_only=False)
        config = saved.get("config", {})
        arch = saved.get("architecture") or config.get("architecture")
        if arch != ARCHITECTURE:
            raise ValueError(f"Expected a {ARCHITECTURE} checkpoint, got {arch!r}")
        model_config = saved.get("model_config") or config["model"]
        network = cls.from_config(model_config).to(str(device)).eval()
        network.load_checkpoint_state(saved["model"])
        return network, saved

    @staticmethod
    def _cloud_context(
        pos: torch.Tensor,
        nodes: torch.Tensor,
        colliders: torch.Tensor,
        multiplicity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """22D translation-invariant cloud statistics per particle, and the radial and azimuthal unit vectors."""
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

        # The context is applied with full weight everywhere (no hand-written wall or aspect-ratio gate).
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

    def encode(
        self,
        pos_pred: torch.Tensor,
        nodes: torch.Tensor,
        colliders: torch.Tensor,
        influence: torch.Tensor,
        fluid_basis: torch.Tensor,
        obs_basis: torch.Tensor,
        packing: torch.Tensor,
        multiplicity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map the inputs of a step to the raw output [..., 8]: corrections ``dx``, ``dv`` and two gate logits.

        Also returns the input of the output head. Nothing reads it, but the solver compiles this method with
        :func:`torch.compile`, and without this second output the compiled code rounds differently: the Box task
        then differs from the reference rollout by 2.7 cm after 120 steps. It is kept so that results reproduce.
        """
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
        col_weighted = (col_emb * inf_eff.to(dt).unsqueeze(-1)).sum(dim=-2)

        local_cat = torch.cat([conv_obs, conv_fluid, dense_p + col_weighted], dim=-1)
        x = self.in_proj(local_cat)
        cloud_ctx_f, w_open_f, e_r_f, e_theta_f = self._cloud_context(
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
        # Local geometry skip: influence-weighted collider geometry
        nearest_geom = (colliders.to(dt) * inf_eff.to(dt).unsqueeze(-1)).sum(dim=-2)
        head_features = torch.cat([x, local_cat, nearest_geom, packing.to(dt)], dim=-1)
        raw_motion = self.motion_head(head_features).float()
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
        cart_skip = torch.cat((dx_xy, dx_z, dv_xy, dv_z, g_skip), dim=-1)
        raw_motion = raw_motion + w_open.float() * cart_skip
        return raw_motion, head_features

    def _update(
        self,
        raw_out: torch.Tensor,
        pos_cur: torch.Tensor,
        vel_cur: torch.Tensor,
        colliders: torch.Tensor,
        influence: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gated update: ``v_next = a (v + dv) + b w`` and ``x_next = x + dt v_next + a dx`` (see the module text)."""
        scale = self.config["displacement_scale"]
        gates = torch.sigmoid(raw_out[..., -2:])
        g_inert = gates[..., 0:1]
        g_col = gates[..., 1:2]
        dx_corr = raw_out[..., 0:3]
        dv_corr = raw_out[..., 3:6]
        corr = scale * dx_corr + (dt * scale * 10.0) * dv_corr

        dx_inert = vel_cur * dt
        v_rel = colliders[..., 4:7]
        v_surf = v_rel + vel_cur.unsqueeze(-2)
        if influence.shape[-1] > 6:
            dyn_inf = influence[..., :-5]
            dyn_scale = (dyn_inf.sum(dim=-1, keepdim=True) / 1.25).clamp_min(1.0)
            inf_col = torch.cat((dyn_inf / dyn_scale, influence[..., -5:]), dim=-1)
        else:
            inf_col = influence
        dx_col = (inf_col.unsqueeze(-1) * v_surf).sum(dim=-2) * dt
        pos_next = pos_cur + g_inert * (dx_inert + corr) + g_col * dx_col
        vel_next = g_inert * (vel_cur + (scale * 10.0) * dv_corr) + g_col * (dx_col / dt)
        return pos_next, vel_next

    def _far_influence_weight(self, kinds: torch.Tensor) -> float:
        """Far-field influence weight per moving collider (the last five colliders are the tray boxes)."""
        n_dyn = max(1, int(kinds.shape[0]) - 5) if kinds.shape[0] > 6 else 1
        return self.config["far_influence_weight"] / float(n_dyn)

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
        return_vel: bool = False,
    ):
        """One step of one environment; differentiable with respect to ``history`` (used for training).

        Args:
            history: Particle positions [N, H, 3] [m], oldest first.
            poses: Collider poses [C, 7] (xyz, quaternion xyzw).
            velocities: Collider twists [C, 6] (linear, angular) [m/s, rad/s].
            kinds: Collider kinds [C].
            sizes: Collider sizes [C, 3] [m].
            friction: Collider friction [C].
            material: Material features [13].
            dt: Time step [s].
            support: Length that normalizes the particle-collider gap [m].
            gravity: Gravity vector [m/s^2].
            return_vel: Also return the next velocity.

        Returns:
            The next positions [N, 3], or (next positions, next velocities) if ``return_vel``.
        """
        pos_cur = history[:, -1]
        vel_cur = (history[:, -1] - history[:, -2]) / dt
        pos_pred, vel_pred, nodes, colliders, influence = compute_features(
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
            far_influence_weight=self._far_influence_weight(kinds),
            gravity=gravity,
        )
        fluid_basis, obs_basis, packing = self.local_agg.compute_bases(pos_pred, vel_pred, colliders, influence)
        raw_out, _ = self.encode(pos_pred, nodes, colliders, influence, fluid_basis, obs_basis, packing)
        pos_next, vel_next = self._update(raw_out, pos_cur, vel_cur, colliders, influence, dt)
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
        multiplicity: torch.Tensor | None = None,
        return_vel: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """One step of E environments in one pass (inference only, used by the solver).

        Arguments are as in :meth:`step_from_history` with a leading environment axis on ``history_env``
        [E, M, H, 3], ``poses_env`` [E, C, 7] and ``velocities_env`` [E, C, 6]; ``sizes``, ``friction`` and
        ``material`` may be shared or per environment. ``multiplicity`` [E, M] weights the tokens in the slice
        attention (active particles 1, background anchors the number of particles each one stands for).
        """
        pos_cur = history_env[:, :, -1]
        vel_cur = (history_env[:, :, -1] - history_env[:, :, -2]) / dt
        pos_pred, vel_pred, nodes, colliders, influence = compute_features_batched(
            history_env,
            poses_env,
            velocities_env,
            kinds,
            sizes,
            friction,
            material,
            dt=dt,
            support=support,
            far_support_scale=self.config["far_support_scale"],
            far_gap_weight=self.config["far_gap_weight"],
            far_influence_weight=self._far_influence_weight(kinds),
            gravity=gravity,
        )
        fluid_basis, obs_basis, packing = self.local_agg.compute_bases_batched(pos_pred, vel_pred, colliders, influence)
        raw_out, _ = self.encode(
            pos_pred, nodes, colliders, influence, fluid_basis, obs_basis, packing, multiplicity=multiplicity
        )
        pos_next, vel_next = self._update(raw_out, pos_cur, vel_cur, colliders, influence, dt)
        if return_vel:
            return pos_next, vel_next
        return pos_next
