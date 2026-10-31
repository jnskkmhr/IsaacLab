# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded-memory physics-rate G1 records and collider geometry transforms."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from particle_nerd.transolver.features import features, material_features


def rotate(q, v):
    cross = 2 * torch.linalg.cross(q[..., :3], v, dim=-1)
    return v + q[..., 3:] * cross + torch.linalg.cross(q[..., :3], cross, dim=-1)


def multiply(a, b):
    return torch.cat(
        (
            a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + torch.linalg.cross(a[..., :3], b[..., :3], dim=-1),
            a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True),
        ),
        -1,
    )


def collider_state(body_pose, body_velocity, geometry):
    """Compose actual collision boxes; convert COM velocity to shape-origin velocity."""
    ids = geometry["body_ids"]
    b, velocity = body_pose[ids], body_velocity[ids]
    local = geometry["local_pose"]
    offset = rotate(b[:, 3:], local[:, :3])
    position = b[:, :3] + offset
    rotation = multiply(b[:, 3:], local[:, 3:])
    com_offset = rotate(b[:, 3:], geometry["com"][ids])
    linear = velocity[:, :3] + torch.linalg.cross(velocity[:, 3:], offset - com_offset, dim=-1)
    return torch.cat((position, rotation), -1), torch.cat((linear, velocity[:, 3:]), -1)


def infer_collider_slots(sizes: torch.Tensor, foot_slots: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """Identify the 2 foot collider slots and the horizontal bed_floor collider slot."""
    device = sizes.device
    if foot_slots is None or foot_slots.numel() == 0:
        volumes = sizes.prod(dim=-1)
        foot_slots = torch.topk(volumes, k=min(2, len(sizes)), largest=False, sorted=True).indices.long()
    else:
        foot_slots = foot_slots.to(device=device, dtype=torch.long)
    areas = sizes[:, 0] * sizes[:, 1]
    mask = torch.ones(len(sizes), dtype=torch.bool, device=device)
    mask[foot_slots] = False
    floor_Areas = torch.where(mask, areas, torch.full_like(areas, -1.0))
    floor_slot = int(floor_Areas.argmax().item())
    return foot_slots, floor_slot


def resolve_active_k(active_k, num_particles: int, material: torch.Tensor) -> int | None:
    """Resolve 'auto', 'none', None, or integer active_k against particle count and material."""
    if active_k is None:
        return None
    if isinstance(active_k, str):
        val = active_k.strip().lower()
        if val in ("none", "off", "all", "full", "0"):
            return None
        if val == "auto":
            mu_val = float(material[3].item())
            stiff_val = float(material[1].item())
            frac = min(0.15, 0.032 * (1.0 + 1.5 * mu_val + 0.15 * stiff_val))
            k = max(1024, min(num_particles, int(round(num_particles * frac / 64.0)) * 64))
            return None if k >= num_particles else k
        active_k = int(val)
    k = int(active_k)
    return None if k <= 0 or k >= num_particles else k


def unclamped_foot_box_kinematics(
    x_cur: torch.Tensor,
    x_prev: torch.Tensor,
    foot_poses: torch.Tensor,
    foot_velocities: torch.Tensor,
    foot_sizes: torch.Tensor,
    p_radius: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact unclamped signed distance, outward box normal, and approach speed to nearest foot box.

    Supports both unbatched `x_cur` `[N, 3]` (`foot_poses` `[F, 7]`) and batched `[B, N, 3]` (`[B, F, 7]`).
    """
    center = foot_poses[..., None, :, :3]  # [..., 1, F, 3]
    q = foot_poses[..., None, :, 3:7]  # [..., 1, F, 4]
    q_inv = torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)
    rel = x_cur.unsqueeze(-2) - center  # [..., N, F, 3]
    local = rotate(q_inv, rel)  # [..., N, F, 3]
    sizes_exp = foot_sizes.unsqueeze(0) if x_cur.ndim == 2 else foot_sizes.unsqueeze(0).unsqueeze(0)
    q_box = local.abs() - sizes_exp  # [..., N, F, 3]
    outside = q_box.clamp_min(0.0)
    out_len = outside.norm(dim=-1)
    inside = q_box.amax(dim=-1).clamp_max(0.0)
    dist = out_len + inside - p_radius  # [..., N, F], unclamped!

    sign = torch.where(local < 0.0, -1.0, 1.0)
    axis = q_box.argmax(dim=-1, keepdim=True)
    one_hot = torch.zeros_like(q_box).scatter_(-1, axis, 1.0)
    n_local = torch.where(out_len.unsqueeze(-1) > 1e-8, sign * outside / out_len.unsqueeze(-1).clamp_min(1e-8), sign * one_hot)
    n_world = F.normalize(rotate(q, n_local), dim=-1, eps=1e-8)  # [..., N, F, 3]

    surface = x_cur.unsqueeze(-2) - (out_len + inside).unsqueeze(-1) * n_world
    v_lin = foot_velocities[..., None, :, :3]
    v_ang = foot_velocities[..., None, :, 3:6]
    v_surf = v_lin + torch.linalg.cross(v_ang, surface - center, dim=-1)
    v_part = ((x_cur - x_prev) / dt).unsqueeze(-2)
    approach_all = (n_world * (v_surf - v_part)).sum(dim=-1).clamp_min(0.0)  # [..., N, F]

    gap_min, nearest = dist.min(dim=-1)  # [..., N]
    idx_n = nearest.unsqueeze(-1).unsqueeze(-1).expand(*nearest.shape, 1, 3)
    n_nearest = torch.gather(n_world, -2, idx_n).squeeze(-2)  # [..., N, 3]
    approach = approach_all.amax(dim=-1)  # [..., N]
    return gap_min, n_nearest, approach


def _packing_step(
    x_cur_k: torch.Tensor,
    dx_work: torch.Tensor,
    tau_k: torch.Tensor | None,
    min_spacing: float,
    stiffness: float,
    max_step_corr: float,
) -> torch.Tensor:
    if x_cur_k.shape[0] > 1:
        return torch.stack(
            [
                _packing_step(
                    x_cur_k[b : b + 1],
                    dx_work[b : b + 1],
                    None if tau_k is None else tau_k[b : b + 1],
                    min_spacing,
                    stiffness,
                    max_step_corr,
                )[0]
                for b in range(x_cur_k.shape[0])
            ],
            dim=0,
        )
    x_next = x_cur_k + dx_work  # [1, K, 3]
    dist = torch.cdist(x_next, x_next)  # [1, K, K]
    mask = (dist < min_spacing) & (dist > 1e-6)
    overlap = (min_spacing - dist).clamp_min(0.0) * mask.float()
    weight = overlap / dist.clamp_min(1e-6)
    z_col = x_next[:, :, 2:3]
    up_weight = torch.where(z_col >= z_col.transpose(1, 2), 1.45, 0.55)
    if tau_k is not None:
        up_weight = up_weight * (1.0 + 1.5 * (1.0 - tau_k[:, None, :]))
    w_total = weight * up_weight  # [1, K, K]
    corr = stiffness * (w_total.sum(dim=2, keepdim=True) * x_next - torch.bmm(w_total, x_next))  # [1, K, 3]
    if tau_k is not None:
        corr = corr * tau_k.unsqueeze(-1)
    corr_norm = corr.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    corr = corr * torch.minimum(torch.ones_like(corr_norm), max_step_corr / corr_norm)
    dx_next = dx_work + corr
    z_next = x_cur_k[..., 2:3] + dx_next[..., 2:3]
    floor_pen = (-0.29 - z_next).clamp_min(0.0)
    return torch.cat((dx_next[..., :2], dx_next[..., 2:3] + floor_pen), dim=-1)


def enforce_packing_incompressibility(
    x_cur_k: torch.Tensor,
    dx_k: torch.Tensor,
    tau_k: torch.Tensor | None = None,
    min_spacing: float = 0.0165,
    stiffness: float = 0.40,
    max_step_corr: float = 0.0025,
    iterations: int = 2,
) -> torch.Tensor:
    """Enforce granular maximum packing fraction + solid rigid-floor non-penetration.

    1. Prevents particles from coalescing closer than `min_spacing` (~1.65 cm) via smooth quadratic B-spline
       pair repulsion, with upward bearing support from deeper neighbors (`z_j < z_i`) and against static
       outer-shell boundary particles (`tau_j < 0.25`).
    2. Enforces unilateral non-penetration with the rigid bed floor at `z_floor = -0.30 + r_p = -0.29 m`.
    """
    single = x_cur_k.ndim == 2
    if single:
        x_cur_k = x_cur_k.unsqueeze(0)
        dx_k = dx_k.unsqueeze(0)
        if tau_k is not None:
            tau_k = tau_k.unsqueeze(0)
    dx_work = dx_k
    for _ in range(iterations):
        if torch.is_grad_enabled():
            dx_work = checkpoint(
                _packing_step, x_cur_k, dx_work, tau_k, min_spacing, stiffness, max_step_corr, use_reentrant=False
            )
        else:
            dx_work = _packing_step(x_cur_k, dx_work, tau_k, min_spacing, stiffness, max_step_corr)
    return dx_work[0] if single else dx_work


def project_tray_bounds(
    x_cur: torch.Tensor,
    dx: torch.Tensor,
    poses: torch.Tensor,
    sizes: torch.Tensor,
    floor_slot: int,
    p_radius: float,
) -> torch.Tensor:
    """Enforce unilateral non-penetration with the rigid tray floor and retaining walls."""
    z_floor = float((poses[floor_slot, 2] + sizes[floor_slot, 2]).item())
    z_min = torch.maximum(
        torch.full_like(x_cur[..., 2], z_floor),
        torch.minimum(x_cur[..., 2], torch.full_like(x_cur[..., 2], z_floor + p_radius)),
    )
    wall_face = float(sizes[floor_slot, 0].item()) - 0.10
    if wall_face <= 0.5:
        wall_face = 1.50
    xy_inner = wall_face - p_radius
    x_min = torch.minimum(
        torch.full_like(x_cur[..., 0], -wall_face),
        torch.minimum(x_cur[..., 0], torch.full_like(x_cur[..., 0], -xy_inner)),
    )
    x_max = torch.maximum(
        torch.full_like(x_cur[..., 0], wall_face),
        torch.maximum(x_cur[..., 0], torch.full_like(x_cur[..., 0], xy_inner)),
    )
    y_min = torch.minimum(
        torch.full_like(x_cur[..., 1], -wall_face),
        torch.minimum(x_cur[..., 1], torch.full_like(x_cur[..., 1], -xy_inner)),
    )
    y_max = torch.maximum(
        torch.full_like(x_cur[..., 1], wall_face),
        torch.maximum(x_cur[..., 1], torch.full_like(x_cur[..., 1], xy_inner)),
    )
    x_next = x_cur + dx
    x_clamped = torch.stack(
        (
            torch.clamp(x_next[..., 0], min=x_min, max=x_max),
            torch.clamp(x_next[..., 1], min=y_min, max=y_max),
            torch.maximum(x_next[..., 2], z_min),
        ),
        dim=-1,
    )
    return x_clamped - x_cur


def apply_free_surface_ballistics(
    x_cur: torch.Tensor,
    x_prev: torch.Tensor,
    dx: torch.Tensor,
    gap_foot: torch.Tensor,
    z_surf: torch.Tensor | float,
    p_radius: float,
    dt: float,
) -> torch.Tensor:
    """Ballistic free-fall (sigma = 0) above the bed surface and inelastic landing outside foot depressions."""
    h_air = (x_cur[..., 2] - z_surf).clamp_min(0.0)
    tau_air = (h_air / (2.0 * p_radius)).clamp(0.0, 1.0)
    unconfined = torch.maximum(
        ((gap_foot - 0.08) / 0.04).clamp(0.0, 1.0),
        ((h_air - 0.035) / 0.02).clamp(0.0, 1.0),
    )
    beta = tau_air * tau_air * (3.0 - 2.0 * tau_air) * unconfined

    v_prev = x_cur - x_prev
    g_vec = torch.zeros_like(dx)
    g_vec[..., 2] = -9.81 * dt * dt
    dx_ballistic = 0.96 * v_prev + g_vec
    dx_out = (1.0 - beta.unsqueeze(-1)) * dx + beta.unsqueeze(-1) * dx_ballistic

    outside_foot = gap_foot > 0.09
    was_airborne = (h_air > 0.0) & outside_foot
    z_next = x_cur[..., 2] + dx_out[..., 2]
    z_land = z_surf - 0.5 * p_radius
    z_clamped = torch.where(was_airborne, torch.maximum(z_next, z_land), z_next)
    landed = was_airborne & (z_next <= z_land)
    xy_scale = torch.where(landed, torch.full_like(z_next, 0.5), torch.ones_like(z_next))
    return torch.stack(
        (
            dx_out[..., 0] * xy_scale,
            dx_out[..., 1] * xy_scale,
            z_clamped - x_cur[..., 2],
        ),
        dim=-1,
    )


def compact_step(
    model,
    history: torch.Tensor,
    poses: torch.Tensor,
    velocities: torch.Tensor,
    geometry: dict,
    material: torch.Tensor,
    dt: float,
    support: float = 0.07,
    active_k: int | str | None = 4096,
    anchor_m: int = 512,
    checkpoint_model: bool = False,
    return_active_count: bool = False,
    enforce_packing: bool = False,
):
    """Differentiable active-particle compaction with weighted background anchors and confining cup wall.

    1. Selects top-K active particles around the feet using unclamped 3D signed box distance + material reach.
    2. Selects M strided background anchors weighted by (N - K) / M_valid so Transolver slice sums stay in-distribution.
    3. Injects the stationary confining cup boundary into the `floor_slot` collider channel for active particles
       and enforces a smooth unilateral non-penetration boundary envelope against the frozen sand cup and rigid tray.
    """
    n_pts, h_len, _ = history.shape
    device = history.device
    foot_slots, floor_slot = infer_collider_slots(geometry["sizes"], geometry.get("foot_slots"))
    p_radius = 0.02 * float(material[11].item())
    k_act = resolve_active_k(active_k, n_pts, material)
    if k_act is None:
        inputs = features(
            history,
            poses,
            velocities,
            geometry["kinds"],
            geometry["sizes"],
            geometry["friction"],
            material,
            dt,
            support=support,
        )
        disp = (
            checkpoint(model, *inputs, use_reentrant=False)
            if checkpoint_model and torch.is_grad_enabled()
            else model(*inputs)
        )
        z_surf_full = torch.quantile(history[::64, -1, 2], 0.98) + 1.5 * p_radius
        gap_full, _, _ = unclamped_foot_box_kinematics(
            history[:, -1],
            history[:, -2],
            poses[foot_slots],
            velocities[foot_slots],
            geometry["sizes"][foot_slots],
            p_radius,
            dt,
        )
        disp = apply_free_surface_ballistics(
            history[:, -1], history[:, -2], disp, gap_full, z_surf_full, p_radius, dt
        )
        disp = project_tray_bounds(history[:, -1], disp, poses, geometry["sizes"], floor_slot, p_radius)
        return (disp, n_pts) if return_active_count else disp

    with torch.no_grad():
        x_cur = history[:, -1]
        x_prev = history[:, -2]
        gap_min, _n_foot, approach = unclamped_foot_box_kinematics(
            x_cur,
            x_prev,
            poses[foot_slots],
            velocities[foot_slots],
            geometry["sizes"][foot_slots],
            p_radius,
            dt,
        )
        mu_mat = float(material[3].item())
        m_bg = min(int(anchor_m), n_pts - k_act)
        bg_idx = torch.linspace(0, n_pts - 1, m_bg, device=device).long()
        surf_idx = (bg_idx + 17).clamp_max(n_pts - 1)
        z_surf = torch.quantile(x_cur[surf_idx, 2], 0.98) + 1.5 * p_radius
        h_air = (x_cur[:, 2] - z_surf).clamp_min(0.0)
        speed = (x_cur - x_prev).norm(dim=-1) / dt
        v_thresh = 0.015 * (1.0 + 0.5 * mu_mat)
        v_excess = (speed - v_thresh).clamp_min(0.0)
        xi = torch.maximum(h_air / p_radius, v_excess / v_thresh).clamp(0.0, 1.0)
        eval_score = (1.0 - xi) * (gap_min - 2.0 * dt * approach) - 0.8 * v_excess - 2.0 * h_air

        top_vals, topk = torch.topk(eval_score, k=k_act, dim=0, largest=False, sorted=False)
        thresh = top_vals.max()
        bg_valid = (eval_score[bg_idx] > thresh).float()
        if float(bg_valid.sum().item()) < 1.0:
            bg_valid = torch.ones_like(bg_valid)
        bg_mult = bg_valid * (float(n_pts - k_act) / bg_valid.sum().clamp_min(1.0))

        idx = torch.cat((topk, bg_idx), dim=0)
        mult = torch.cat((torch.ones(k_act, dtype=torch.float32, device=device), bg_mult), dim=0)

        # Smooth C1 partition-of-unity taper over the outer 5 cm rim of the K-particle active set
        tau = ((thresh - top_vals) / 0.05).clamp(0.0, 1.0)
        gamma = tau * tau * (3.0 - 2.0 * tau)
        active_count = int((tau > 0.0).sum().item())

    # Differentiable Warp feature extraction on only the (K + M) selected particles
    history_sub = history[idx].contiguous()
    nodes_sub, pairs_sub, inf_sub = features(
        history_sub,
        poses,
        velocities,
        geometry["kinds"],
        geometry["sizes"],
        geometry["friction"],
        material,
        dt,
        support=support,
    )

    disp_sub = (
        checkpoint(model, nodes_sub, pairs_sub, inf_sub, mult, use_reentrant=False)
        if checkpoint_model and torch.is_grad_enabled()
        else model(nodes_sub, pairs_sub, inf_sub, mult)
    )
    dx_k = gamma.unsqueeze(-1) * disp_sub[:k_act]
    dx_k = apply_free_surface_ballistics(
        history[topk, -1], history[topk, -2], dx_k, gap_min[topk], z_surf, p_radius, dt
    )
    if enforce_packing:
        dx_k = enforce_packing_incompressibility(history[topk, -1], dx_k, tau_k=gamma)
    dx_k = project_tray_bounds(history[topk, -1], dx_k, poses, geometry["sizes"], floor_slot, p_radius)
    dx_bg = project_tray_bounds(history[bg_idx, -1], disp_sub[k_act:], poses, geometry["sizes"], floor_slot, p_radius)
    displacement = torch.zeros_like(history[:, -1])
    displacement = displacement.index_put((bg_idx,), dx_bg)
    displacement = displacement.index_put((topk,), dx_k)
    return (displacement, active_count) if return_active_count else displacement


def compact_step_batched(
    model,
    history_flat: torch.Tensor,
    poses_list: list[torch.Tensor],
    velocities_list: list[torch.Tensor],
    geometries: list[dict],
    material: torch.Tensor,
    dt: float,
    support: float = 0.07,
    active_k: int | str | None = 4096,
    anchor_m: int = 512,
    enforce_packing: bool = False,
) -> tuple[torch.Tensor, list[int]]:
    """Batched multi-environment active compaction + quadrature background anchors."""
    num_envs = len(geometries)
    n_pts = history_flat.shape[0] // num_envs
    h_len = history_flat.shape[1]
    device = history_flat.device
    hist_env = history_flat.view(num_envs, n_pts, h_len, 3)

    foot_slots_0, floor_slot_0 = infer_collider_slots(geometries[0]["sizes"], geometries[0].get("foot_slots"))
    p_radius = 0.02 * float(material[11].item())
    k_act = resolve_active_k(active_k, n_pts, material)
    if k_act is None:
        disp_l = []
        for e in range(num_envs):
            g_e = geometries[e]
            n_e, p_e, i_e = features(
                hist_env[e].contiguous(),
                poses_list[e],
                velocities_list[e],
                g_e["kinds"],
                g_e["sizes"],
                g_e["friction"],
                material,
                dt,
                support=support,
            )
            d_e = model(n_e, p_e, i_e)
            d_e = project_tray_bounds(hist_env[e, :, -1], d_e, poses_list[e], g_e["sizes"], floor_slot_0, p_radius)
            disp_l.append(d_e)
        return torch.cat(disp_l, dim=0), [n_pts] * num_envs

    foot_poses = torch.stack([poses_list[e][foot_slots_0] for e in range(num_envs)], dim=0)
    foot_vels = torch.stack([velocities_list[e][foot_slots_0] for e in range(num_envs)], dim=0)
    x_cur = hist_env[:, :, -1]
    x_prev = hist_env[:, :, -2]
    gap_min, _n_foot, approach = unclamped_foot_box_kinematics(
        x_cur, x_prev, foot_poses, foot_vels, geometries[0]["sizes"][foot_slots_0], p_radius, dt
    )
    mu_mat = float(material[3].item())
    m_bg = min(int(anchor_m), n_pts - k_act)
    bg_1d = torch.linspace(0, n_pts - 1, m_bg, device=device).long()
    bg_exp = bg_1d.unsqueeze(0).expand(num_envs, -1)
    surf_exp = (bg_1d + 17).clamp_max(n_pts - 1).unsqueeze(0).expand(num_envs, -1)
    z_bg = torch.gather(x_cur[..., 2], 1, surf_exp)
    z_surf = torch.quantile(z_bg, 0.98, dim=-1, keepdim=True) + 1.5 * p_radius
    h_air = (x_cur[..., 2] - z_surf).clamp_min(0.0)
    speed = (x_cur - x_prev).norm(dim=-1) / dt
    v_thresh = 0.015 * (1.0 + 0.5 * mu_mat)
    v_excess = (speed - v_thresh).clamp_min(0.0)
    xi = torch.maximum(h_air / p_radius, v_excess / v_thresh).clamp(0.0, 1.0)
    eval_score = (1.0 - xi) * (gap_min - 2.0 * dt * approach) - 0.8 * v_excess - 2.0 * h_air

    top_vals, topk = torch.topk(eval_score, k=k_act, dim=-1, largest=False, sorted=False)
    thresh = top_vals.amax(dim=-1, keepdim=True)
    bg_scores = torch.gather(eval_score, 1, bg_exp)
    bg_valid = (bg_scores > thresh).float()
    bg_valid = torch.where(bg_valid.sum(dim=-1, keepdim=True) < 1.0, torch.ones_like(bg_valid), bg_valid)
    bg_mult = bg_valid * (float(n_pts - k_act) / bg_valid.sum(dim=-1, keepdim=True).clamp_min(1.0))

    idx = torch.cat((topk, bg_exp), dim=1)
    mult = torch.cat((torch.ones((num_envs, k_act), dtype=torch.float32, device=device), bg_mult), dim=1)

    tau = ((thresh - top_vals) / 0.05).clamp(0.0, 1.0)
    gamma = tau * tau * (3.0 - 2.0 * tau)
    active_counts = (tau > 0.0).sum(dim=-1).int().tolist()

    nodes_l, pairs_l, inf_l = [], [], []
    for e in range(num_envs):
        g_e = geometries[e]
        h_sub_e = hist_env[e, idx[e]].contiguous()
        n_e, p_e, i_e = features(
            h_sub_e,
            poses_list[e],
            velocities_list[e],
            g_e["kinds"],
            g_e["sizes"],
            g_e["friction"],
            material,
            dt,
            support=support,
        )
        nodes_l.append(n_e)
        pairs_l.append(p_e)
        inf_l.append(i_e)

    disp_sub = model(torch.stack(nodes_l, 0), torch.stack(pairs_l, 0), torch.stack(inf_l, 0), mult)
    dx_k = gamma.unsqueeze(-1) * disp_sub[:, :k_act].float()
    dx_bg = disp_sub[:, k_act:].float()
    x_cur_k = torch.gather(x_cur, 1, topk.unsqueeze(-1).expand(-1, -1, 3))
    x_prev_k = torch.gather(x_prev, 1, topk.unsqueeze(-1).expand(-1, -1, 3))
    gap_k = torch.gather(gap_min, 1, topk)
    dx_k = apply_free_surface_ballistics(x_cur_k, x_prev_k, dx_k, gap_k, z_surf, p_radius, dt)
    x_cur_bg = torch.gather(x_cur, 1, bg_exp.unsqueeze(-1).expand(-1, -1, 3))
    if enforce_packing:
        dx_k = enforce_packing_incompressibility(x_cur_k, dx_k, tau_k=gamma)
    dx_k = project_tray_bounds(x_cur_k, dx_k, poses_list[0], geometries[0]["sizes"], floor_slot_0, p_radius)
    dx_bg = project_tray_bounds(x_cur_bg, dx_bg, poses_list[0], geometries[0]["sizes"], floor_slot_0, p_radius)
    disp_env = torch.zeros((num_envs, n_pts, 3), dtype=torch.float32, device=device)
    disp_env.scatter_(1, bg_exp.unsqueeze(-1).expand(-1, -1, 3), dx_bg)
    disp_env.scatter_(1, topk.unsqueeze(-1).expand(-1, -1, 3), dx_k)
    return disp_env.reshape(-1, 3), active_counts


def read_geometry(f, device):
    def tensor(a):
        return torch.as_tensor(a, device=device, dtype=torch.float32)

    shapes = f["collider_body"][:]
    sizes = tensor(f["collider_half_size"][:])
    feet = f["foot_body_ids"][:] if "foot_body_ids" in f else np.array([], dtype=np.int64)
    raw_slots = [int(np.flatnonzero(shapes == body).item()) for body in feet if np.any(shapes == body)]
    foot_slots_in = torch.tensor(raw_slots, device=device, dtype=torch.long) if raw_slots else None
    foot_slots, floor_slot = infer_collider_slots(sizes, foot_slots_in)
    return dict(
        body_ids=torch.as_tensor(shapes, device=device, dtype=torch.long),
        local_pose=tensor(f["collider_local_pose"][:]),
        sizes=sizes,
        friction=tensor(f["collider_friction"][:]),
        com=tensor(f["body_com_local"][:]),
        kinds=torch.ones(len(shapes), dtype=torch.int32, device=device),
        foot_slots=foot_slots,
        floor_slot=floor_slot,
    )


class Episodes:
    def __init__(self, root, split, history=3, stride=1, settled_frame=150):
        self.history, self.stride, self.settled_frame = history, stride, int(settled_frame)
        self.records = []
        for path in sorted(Path(root).rglob("episode-*.hdf5")):
            if ".partial." in path.name:
                continue
            with h5py.File(path, "r") as f:
                m = json.loads(f.attrs["manifest"])
                if m["split"] == split:
                    times = f["time_s"][:]
                    if len(times) > history and not np.allclose(np.diff(times), 0.005, atol=1e-8):
                        raise ValueError(f"Nonuniform physics records: {path}")
                    self.records.append(dict(path=path, index=m["trial"]["trial_id"], frames=len(times), manifest=m))
        ids = [r["index"] for r in self.records]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicated {split} episode IDs")
        self.dt = 0.005 * stride

    @property
    def first_frame(self) -> int:
        return max(self.settled_frame, (self.history - 1) * self.stride)

    def sequence(self, rng, steps, device, episode=None, frame=None):
        first = self.first_frame
        eligible = [r for r in self.records if r["frames"] > first + steps * self.stride]
        if not eligible:
            raise ValueError("No recorded trajectories can supply the requested recurrent horizon")
        r = eligible[int(rng.integers(len(eligible))) if episode is None else episode % len(eligible)]
        end = r["frames"] - steps * self.stride
        start = int(rng.integers(first, end)) if frame is None else frame
        if not first <= start < end:
            raise ValueError(f"Invalid sequence frame {start} for {r['path']} (first={first}, end={end})")

        def tensor(a):
            return torch.as_tensor(np.asarray(a), device=device, dtype=torch.float32)

        with h5py.File(r["path"], "r") as f:
            history = np.stack(
                [f["particle_position"][start - k * self.stride] for k in reversed(range(self.history))], axis=1
            )
            targets = np.stack([f["particle_position"][start + (k + 1) * self.stride] for k in range(steps)])
            indices = start + np.arange(steps) * self.stride
            return dict(
                history=tensor(history),
                targets=tensor(targets),
                poses=tensor(f["body_pose"][indices]),
                velocities=tensor(f["body_velocity"][indices]),
                geometry=read_geometry(f, device),
                material=tensor(
                    material_features(
                        r["manifest"], float(f["particle_mass"][0]), float(f["particle_radius"][0]), self.dt
                    )
                ),
                trial_id=r["index"],
                frame=start,
            )
