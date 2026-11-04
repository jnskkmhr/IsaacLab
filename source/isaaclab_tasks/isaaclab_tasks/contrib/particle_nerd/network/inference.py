# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded-memory physics-rate G1 records and collider geometry transforms."""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .features import features, features_batched


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
            k = max(1024, min(7104, num_particles, int(round(num_particles * frac / 64.0)) * 64))
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
    n_local = torch.where(
        out_len.unsqueeze(-1) > 1e-8, sign * outside / out_len.unsqueeze(-1).clamp_min(1e-8), sign * one_hot
    )
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


@torch.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
def _compiled_foot_gap_only(
    x_cur: torch.Tensor,
    foot_poses: torch.Tensor,
    foot_sizes: torch.Tensor,
    p_radius: float,
) -> torch.Tensor:
    """Fused signed box distance to nearest foot collider without normals or approach velocity."""
    center = foot_poses[..., None, :, :3]
    q = foot_poses[..., None, :, 3:7]
    q_inv = torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)
    rel = x_cur.unsqueeze(-2) - center
    local = rotate(q_inv, rel)
    sizes_exp = foot_sizes.unsqueeze(0) if x_cur.ndim == 2 else foot_sizes.unsqueeze(0).unsqueeze(0)
    q_box = local.abs() - sizes_exp
    outside = q_box.clamp_min(0.0)
    out_len = outside.norm(dim=-1)
    inside = q_box.amax(dim=-1).clamp_max(0.0)
    dist = out_len + inside - p_radius
    return dist.amin(dim=-1)


@torch.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
def _compiled_select_score(
    x_cur: torch.Tensor,
    x_prev: torch.Tensor,
    foot_poses: torch.Tensor,
    foot_velocities: torch.Tensor,
    foot_sizes: torch.Tensor,
    z_surf: torch.Tensor,
    p_radius: float,
    mu_mat: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused full-bed active-particle score evaluation over N particles."""
    center = foot_poses[None, :, :3]
    q = foot_poses[None, :, 3:7]
    q_inv = torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)
    rel = x_cur.unsqueeze(-2) - center
    local = rotate(q_inv, rel)
    q_box = local.abs() - foot_sizes.unsqueeze(0)
    outside = q_box.clamp_min(0.0)
    out_len = outside.norm(dim=-1)
    inside = q_box.amax(dim=-1).clamp_max(0.0)
    dist = out_len + inside - p_radius

    sign = torch.where(local < 0.0, -1.0, 1.0)
    axis = q_box.argmax(dim=-1, keepdim=True)
    one_hot = torch.zeros_like(q_box).scatter_(-1, axis, 1.0)
    n_local = torch.where(
        out_len.unsqueeze(-1) > 1e-8, sign * outside / out_len.unsqueeze(-1).clamp_min(1e-8), sign * one_hot
    )
    n_world = F.normalize(rotate(q, n_local), dim=-1, eps=1e-8)

    surface = x_cur.unsqueeze(-2) - (out_len + inside).unsqueeze(-1) * n_world
    v_lin = foot_velocities[None, :, :3]
    v_ang = foot_velocities[None, :, 3:6]
    v_surf = v_lin + torch.linalg.cross(v_ang, surface - center, dim=-1)
    dx = x_cur - x_prev
    v_part = (dx / dt).unsqueeze(-2)
    approach = (n_world * (v_surf - v_part)).sum(dim=-1).clamp_min(0.0).amax(dim=-1)
    gap_min = dist.amin(dim=-1)

    h_air = (x_cur[:, 2] - z_surf).clamp_min(0.0)
    speed = dx.norm(dim=-1) / dt
    v_thresh = 0.015 * (1.0 + 0.5 * mu_mat)
    v_excess = (speed - v_thresh).clamp_min(0.0)
    xi = torch.maximum(h_air / p_radius, v_excess / v_thresh).clamp(0.0, 1.0)
    eval_score = (1.0 - xi) * (gap_min - 2.0 * dt * approach) - 0.8 * v_excess - 2.0 * h_air
    return eval_score, gap_min, xi


@torch.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
def _compiled_sliding_window_gather(
    history: torch.Tensor,
    idx: torch.Tensor,
    poses: torch.Tensor,
    can_poses: torch.Tensor,
    foot_slots: torch.Tensor,
    k_act: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused gather and canonical sliding-window shift for (K + M) particles."""
    history_sub = history[idx]
    foot_mid = poses[foot_slots, :2].mean(dim=0)
    ox = (foot_mid[0].abs() - 0.75).clamp_min(0.0) * foot_mid[0].sign()
    oy = (foot_mid[1].abs() - 0.75).clamp_min(0.0) * foot_mid[1].sign()
    shift_vec = torch.stack((ox, oy, torch.zeros_like(ox)))
    h_act = history_sub[:k_act] - shift_vec[None, None, :]
    h_bg = history_sub[k_act:] - shift_vec[None, None, :]
    bg_cur_xy = h_bg[:, -1, :2]
    bg_wrapped_xy = torch.remainder(bg_cur_xy + 1.45, 2.90) - 1.45
    bg_delta_xy = bg_wrapped_xy - bg_cur_xy
    h_bg = torch.cat((h_bg[..., :2] + bg_delta_xy[:, None, :], h_bg[..., 2:3]), dim=-1)
    history_out = torch.cat((h_act, h_bg), dim=0).contiguous()
    poses_feat = can_poses.clone()
    f_poses = poses[foot_slots].clone()
    f_poses[:, 0] -= ox
    f_poses[:, 1] -= oy
    poses_feat[foot_slots] = f_poses
    return history_out, poses_feat


@torch.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
def _compiled_postprocess_sparse(
    history: torch.Tensor,
    topk: torch.Tensor,
    bg_idx: torch.Tensor,
    disp_sub: torch.Tensor,
    gamma: torch.Tensor,
    gap_k: torch.Tensor,
    z_surf: torch.Tensor,
    foot_engage: torch.Tensor,
    poses: torch.Tensor,
    sizes: torch.Tensor,
    floor_slot: int,
    k_act: int,
    p_radius: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused free-surface ballistics and tray-bounds projection on (K + M) particles."""
    x_cur_k = history[topk, -1]
    x_prev_k = history[topk, -2]
    dx_k = gamma.unsqueeze(-1) * disp_sub[:k_act]
    dx_k = apply_free_surface_ballistics(x_cur_k, x_prev_k, dx_k, gap_k, z_surf, p_radius, dt)
    dx_k = project_tray_bounds(x_cur_k, dx_k, poses, sizes, floor_slot, p_radius)
    dx_bg = project_tray_bounds(history[bg_idx, -1], foot_engage * disp_sub[k_act:], poses, sizes, floor_slot, p_radius)
    dx_sub = torch.cat((dx_k, dx_bg), dim=0)
    return dx_k, dx_bg, dx_sub


@torch.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
def _compiled_select_score_batched(
    x_cur: torch.Tensor,
    x_prev: torch.Tensor,
    foot_poses: torch.Tensor,
    foot_velocities: torch.Tensor,
    foot_sizes: torch.Tensor,
    z_surf: torch.Tensor,
    p_radius: float,
    mu_mat: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused multi-world active-particle score evaluation over [E, N, 3] particles."""
    center = foot_poses[:, None, :, :3]
    q = foot_poses[:, None, :, 3:7]
    q_inv = torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)
    rel = x_cur.unsqueeze(-2) - center
    local = rotate(q_inv, rel)
    q_box = local.abs() - foot_sizes[None, None, :, :]
    outside = q_box.clamp_min(0.0)
    out_len = outside.norm(dim=-1)
    inside = q_box.amax(dim=-1).clamp_max(0.0)
    dist = out_len + inside - p_radius

    sign = torch.where(local < 0.0, -1.0, 1.0)
    axis = q_box.argmax(dim=-1, keepdim=True)
    one_hot = torch.zeros_like(q_box).scatter_(-1, axis, 1.0)
    n_local = torch.where(
        out_len.unsqueeze(-1) > 1e-8, sign * outside / out_len.unsqueeze(-1).clamp_min(1e-8), sign * one_hot
    )
    n_world = F.normalize(rotate(q, n_local), dim=-1, eps=1e-8)

    surface = x_cur.unsqueeze(-2) - (out_len + inside).unsqueeze(-1) * n_world
    v_lin = foot_velocities[:, None, :, :3]
    v_ang = foot_velocities[:, None, :, 3:6]
    v_surf = v_lin + torch.linalg.cross(v_ang, surface - center, dim=-1)
    dx = x_cur - x_prev
    v_part = (dx / dt).unsqueeze(-2)
    approach = (n_world * (v_surf - v_part)).sum(dim=-1).clamp_min(0.0).amax(dim=-1)
    gap_min = dist.amin(dim=-1)

    h_air = (x_cur[..., 2] - z_surf).clamp_min(0.0)
    speed = dx.norm(dim=-1) / dt
    v_thresh = 0.015 * (1.0 + 0.5 * mu_mat)
    v_excess = (speed - v_thresh).clamp_min(0.0)
    xi = torch.maximum(h_air / p_radius, v_excess / v_thresh).clamp(0.0, 1.0)
    eval_score = (1.0 - xi) * (gap_min - 2.0 * dt * approach) - 0.8 * v_excess - 2.0 * h_air
    return eval_score, gap_min, xi


@torch.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
def _compiled_sliding_window_shift_batched(
    history_sub: torch.Tensor,
    poses: torch.Tensor,
    can_poses: torch.Tensor,
    k_act: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused canonical sliding-window shift for [E, K + M, 3, 3] selected particles."""
    foot_mid = poses[:, :2, :2].mean(dim=1)
    ox = (foot_mid[:, 0].abs() - 0.75).clamp_min(0.0) * foot_mid[:, 0].sign()
    oy = (foot_mid[:, 1].abs() - 0.75).clamp_min(0.0) * foot_mid[:, 1].sign()
    shift_vec = torch.stack((ox, oy, torch.zeros_like(ox)), dim=-1)
    h_act = history_sub[:, :k_act] - shift_vec[:, None, None, :]
    h_bg = history_sub[:, k_act:] - shift_vec[:, None, None, :]
    bg_cur_xy = h_bg[:, :, -1, :2]
    bg_wrapped_xy = torch.remainder(bg_cur_xy + 1.45, 2.90) - 1.45
    bg_delta_xy = bg_wrapped_xy - bg_cur_xy
    h_bg = torch.cat((h_bg[..., :2] + bg_delta_xy[:, :, None, :], h_bg[..., 2:3]), dim=-1)
    history_out = torch.cat((h_act, h_bg), dim=1).contiguous()
    poses_feat = can_poses.unsqueeze(0).expand(poses.shape[0], -1, -1).clone()
    f_poses = poses[:, :2].clone()
    f_poses[:, :, 0] -= ox.unsqueeze(-1)
    f_poses[:, :, 1] -= oy.unsqueeze(-1)
    poses_feat[:, :2] = f_poses
    return history_out, poses_feat


@torch.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
def _compiled_postprocess_sparse_batched(
    history_sub: torch.Tensor,
    disp_sub: torch.Tensor,
    gamma: torch.Tensor,
    gap_k: torch.Tensor,
    z_surf: torch.Tensor,
    foot_engage: torch.Tensor,
    poses_0: torch.Tensor,
    sizes: torch.Tensor,
    floor_slot: int,
    k_act: int,
    p_radius: float,
    dt: float,
) -> torch.Tensor:
    """Fused batched free-surface ballistics and tray-bounds projection on [E, K + M, 3]."""
    x_cur_k = history_sub[:, :k_act, -1]
    x_prev_k = history_sub[:, :k_act, -2]
    x_cur_bg = history_sub[:, k_act:, -1]
    dx_k = gamma.unsqueeze(-1) * disp_sub[:, :k_act].float()
    dx_k = apply_free_surface_ballistics(x_cur_k, x_prev_k, dx_k, gap_k, z_surf, p_radius, dt)
    dx_k = project_tray_bounds(x_cur_k, dx_k, poses_0, sizes, floor_slot, p_radius)
    dx_bg = project_tray_bounds(
        x_cur_bg, foot_engage.unsqueeze(-1) * disp_sub[:, k_act:].float(), poses_0, sizes, floor_slot, p_radius
    )
    return torch.cat((dx_k, dx_bg), dim=1)


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
    p_radius: float | torch.Tensor,
) -> torch.Tensor:
    """Enforce unilateral non-penetration with the rigid tray floor and retaining walls."""
    z_floor = poses[floor_slot, 2] + sizes[floor_slot, 2]
    z_min = torch.maximum(
        z_floor.expand_as(x_cur[..., 2]),
        torch.minimum(x_cur[..., 2], (z_floor + p_radius).expand_as(x_cur[..., 2])),
    )
    raw_wx = sizes[floor_slot, 0] - 0.10
    wall_face_x = torch.where(raw_wx > 0.5, raw_wx, torch.full_like(raw_wx, 1.50))
    raw_wy = sizes[floor_slot, 1] - 0.10
    wall_face_y = torch.where(raw_wy > 0.5, raw_wy, torch.full_like(raw_wy, 1.50))
    x_inner = wall_face_x - p_radius
    y_inner = wall_face_y - p_radius
    x_min = torch.minimum(
        (-wall_face_x).expand_as(x_cur[..., 0]),
        torch.minimum(x_cur[..., 0], (-x_inner).expand_as(x_cur[..., 0])),
    )
    x_max = torch.maximum(
        wall_face_x.expand_as(x_cur[..., 0]),
        torch.maximum(x_cur[..., 0], x_inner.expand_as(x_cur[..., 0])),
    )
    y_min = torch.minimum(
        (-wall_face_y).expand_as(x_cur[..., 1]),
        torch.minimum(x_cur[..., 1], (-y_inner).expand_as(x_cur[..., 1])),
    )
    y_max = torch.maximum(
        wall_face_y.expand_as(x_cur[..., 1]),
        torch.maximum(x_cur[..., 1], y_inner.expand_as(x_cur[..., 1])),
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


class ActiveParticleSelection(NamedTuple):
    """Per-world selection and blend weights retained through network inference."""

    topk: torch.Tensor
    background: torch.Tensor
    blend: torch.Tensor
    gap: torch.Tensor
    surface: torch.Tensor
    engagement: torch.Tensor
    count: int


@dataclass
class PreparedParticleStep:
    """Independent world inputs and state needed to reconstruct particle displacement."""

    inputs: tuple[torch.Tensor, ...]
    history: torch.Tensor
    poses: torch.Tensor
    velocities: torch.Tensor
    geometry: dict
    dt: float
    active_k: int | None
    radius: float
    floor_slot: int
    selection: ActiveParticleSelection | None = None


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
    reuse_selection: bool = False,
):
    """Predict one world's particle displacement with independent selection and boundaries."""
    prepared = prepare_compact_step(
        history,
        poses,
        velocities,
        geometry,
        material,
        dt,
        support,
        active_k,
        anchor_m,
        return_active_count=return_active_count,
        reuse_selection=reuse_selection,
    )
    displacement = (
        checkpoint(model, *prepared.inputs, use_reentrant=False)
        if checkpoint_model and torch.is_grad_enabled()
        else model(*prepared.inputs)
    )
    return finish_compact_step(
        prepared,
        displacement,
        return_active_count=return_active_count,
        enforce_packing=enforce_packing,
    )


def prepare_compact_step(
    history: torch.Tensor,
    poses: torch.Tensor,
    velocities: torch.Tensor,
    geometry: dict,
    material: torch.Tensor,
    dt: float,
    support: float = 0.07,
    active_k: int | str | None = 4096,
    anchor_m: int = 512,
    return_active_count: bool = False,
    reuse_selection: bool = False,
) -> PreparedParticleStep:
    """Select particles and build features without sharing geometry or materials across worlds."""
    n_pts, h_len, _ = history.shape
    device = history.device
    foot_slots = geometry["foot_slots"] if "foot_slots" in geometry else infer_collider_slots(geometry["sizes"])[0]
    floor_slot = geometry["floor_slot"] if "floor_slot" in geometry else infer_collider_slots(geometry["sizes"])[1]
    p_radius = geometry.get("p_radius", 0.02 * material[11])
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
        return PreparedParticleStep(inputs, history, poses, velocities, geometry, dt, k_act, p_radius, floor_slot)

    with torch.no_grad():
        mu_mat = geometry.get("mu_mat", material[3])
        m_bg = min(int(anchor_m), n_pts - k_act)
        bg_idx = geometry.get("bg_idx")
        if bg_idx is None or len(bg_idx) != m_bg:
            bg_idx = torch.linspace(0, n_pts - 1, m_bg, device=device).long()
        if reuse_selection and "cached_topk" in geometry:
            topk = geometry["cached_topk"]
            idx = geometry["cached_idx"]
            mult = geometry["cached_mult"]
            gamma = geometry["cached_gamma"]
            z_surf = geometry["cached_z_surf"]
            foot_engage = geometry["cached_foot_engage"]
            x_cur_k = history[topk, -1]
            x_prev_k = history[topk, -2]
            if history.is_cuda and not torch.is_grad_enabled():
                gap_k = _compiled_foot_gap_only(
                    x_cur_k, poses[foot_slots], geometry["sizes"][foot_slots], float(p_radius)
                )
            else:
                gap_k, _, _ = unclamped_foot_box_kinematics(
                    x_cur_k,
                    x_prev_k,
                    poses[foot_slots],
                    velocities[foot_slots],
                    geometry["sizes"][foot_slots],
                    p_radius,
                    dt,
                )
            active_count = k_act
        else:
            x_cur = history[:, -1]
            x_prev = history[:, -2]
            surf_idx = (bg_idx + 17).clamp_max(n_pts - 1)
            z_surf = torch.quantile(x_cur[surf_idx, 2], 0.98) + 1.5 * p_radius
            if history.is_cuda and not torch.is_grad_enabled():
                eval_score, gap_min, xi = _compiled_select_score(
                    x_cur,
                    x_prev,
                    poses[foot_slots],
                    velocities[foot_slots],
                    geometry["sizes"][foot_slots],
                    z_surf,
                    float(p_radius),
                    float(mu_mat),
                    float(dt),
                )
            else:
                gap_min, _n_foot, approach = unclamped_foot_box_kinematics(
                    x_cur,
                    x_prev,
                    poses[foot_slots],
                    velocities[foot_slots],
                    geometry["sizes"][foot_slots],
                    p_radius,
                    dt,
                )
                h_air = (x_cur[:, 2] - z_surf).clamp_min(0.0)
                speed = (x_cur - x_prev).norm(dim=-1) / dt
                v_thresh = 0.015 * (1.0 + 0.5 * mu_mat)
                v_excess = (speed - v_thresh).clamp_min(0.0)
                xi = torch.maximum(h_air / p_radius, v_excess / v_thresh).clamp(0.0, 1.0)
                eval_score = (1.0 - xi) * (gap_min - 2.0 * dt * approach) - 0.8 * v_excess - 2.0 * h_air

            top_vals, topk = torch.topk(eval_score, k=k_act, dim=0, largest=False, sorted=False)
            thresh = top_vals.max()
            bg_valid = (eval_score[bg_idx] > thresh).float()
            bg_valid = torch.where(bg_valid.sum() < 1.0, torch.ones_like(bg_valid), bg_valid)
            bg_mult = bg_valid * (float(n_pts - k_act) / bg_valid.sum().clamp_min(1.0))

            idx = torch.cat((topk, bg_idx), dim=0)
            mult = torch.cat((torch.ones(k_act, dtype=torch.float32, device=device), bg_mult), dim=0)

            # Smooth C1 partition-of-unity taper over the outer 5 cm rim of the K-particle active set
            tau = ((thresh - top_vals) / 0.05).clamp(0.0, 1.0)
            gap_k = gap_min[topk]
            min_gap = gap_k.min()
            foot_engage = ((0.14 - min_gap) / 0.04).clamp(0.0, 1.0)
            gamma = tau * tau * (3.0 - 2.0 * tau) * torch.maximum(foot_engage, xi[topk])
            if "cached_topk" not in geometry:
                geometry["cached_topk"] = topk.clone()
                geometry["cached_idx"] = idx.clone()
                geometry["cached_mult"] = mult.clone()
                geometry["cached_gamma"] = gamma.clone()
                geometry["cached_z_surf"] = z_surf.clone()
                geometry["cached_foot_engage"] = foot_engage.clone()
            else:
                geometry["cached_topk"].copy_(topk)
                geometry["cached_idx"].copy_(idx)
                geometry["cached_mult"].copy_(mult)
                geometry["cached_gamma"].copy_(gamma)
                geometry["cached_z_surf"].copy_(z_surf)
                geometry["cached_foot_engage"].copy_(foot_engage)
            active_count = (
                int((gamma > 0.0).sum().item())
                if (return_active_count and not torch.cuda.is_current_stream_capturing())
                else k_act
            )

    # Differentiable Warp feature extraction on only the (K + M) selected particles
    sizes_feat = geometry["sizes"]
    if geometry.get("sliding_window", False):
        if history.is_cuda and not torch.is_grad_enabled():
            history_sub, poses_feat = _compiled_sliding_window_gather(
                history, idx, poses, geometry.get("canonical_poses", poses), foot_slots, k_act
            )
            sizes_feat = geometry.get("canonical_sizes", geometry["sizes"])
        else:
            history_sub = history[idx].contiguous()
            foot_mid = poses[foot_slots, :2].mean(dim=0)
            ox = (foot_mid[0].abs() - 0.75).clamp_min(0.0) * foot_mid[0].sign()
            oy = (foot_mid[1].abs() - 0.75).clamp_min(0.0) * foot_mid[1].sign()
            shift_vec = torch.stack((ox, oy, torch.zeros_like(ox)))
            h_act = history_sub[:k_act] - shift_vec[None, None, :]
            h_bg = history_sub[k_act:] - shift_vec[None, None, :]
            bg_cur_xy = h_bg[:, -1, :2]
            bg_wrapped_xy = torch.remainder(bg_cur_xy + 1.45, 2.90) - 1.45
            bg_delta_xy = bg_wrapped_xy - bg_cur_xy
            h_bg = torch.cat((h_bg[..., :2] + bg_delta_xy[:, None, :], h_bg[..., 2:3]), dim=-1)
            history_sub = torch.cat((h_act, h_bg), dim=0).contiguous()
            poses_feat = geometry.get("canonical_poses", poses).clone()
            poses_feat[foot_slots] = poses[foot_slots]
            poses_feat[foot_slots, 0] -= ox
            poses_feat[foot_slots, 1] -= oy
            sizes_feat = geometry.get("canonical_sizes", geometry["sizes"])
    else:
        history_sub = history[idx].contiguous()
        poses_feat = poses

    nodes_sub, pairs_sub, inf_sub = features(
        history_sub,
        poses_feat,
        velocities,
        geometry["kinds"],
        sizes_feat,
        geometry["friction"],
        material,
        dt,
        support=support,
    )

    selection = ActiveParticleSelection(topk, bg_idx, gamma, gap_k, z_surf, foot_engage, active_count)
    return PreparedParticleStep(
        (nodes_sub, pairs_sub, inf_sub, mult),
        history,
        poses,
        velocities,
        geometry,
        dt,
        k_act,
        p_radius,
        floor_slot,
        selection,
    )


def finish_compact_step(
    prepared: PreparedParticleStep,
    disp_sub: torch.Tensor,
    return_active_count: bool = False,
    enforce_packing: bool = False,
):
    """Apply a world's blend, free-surface motion and boundaries to its network output."""
    history, poses, velocities, geometry = prepared.history, prepared.poses, prepared.velocities, prepared.geometry
    dt, k_act, p_radius, floor_slot = prepared.dt, prepared.active_k, prepared.radius, prepared.floor_slot
    n_pts = len(history)
    device = history.device
    if k_act is None:
        disp = disp_sub
        foot_slots = geometry["foot_slots"] if "foot_slots" in geometry else infer_collider_slots(geometry["sizes"])[0]
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
        disp = apply_free_surface_ballistics(history[:, -1], history[:, -2], disp, gap_full, z_surf_full, p_radius, dt)
        disp = project_tray_bounds(history[:, -1], disp, poses, geometry["sizes"], floor_slot, p_radius)
        return (disp, n_pts) if return_active_count else disp

    topk, bg_idx, gamma, gap_k, z_surf, foot_engage, active_count = prepared.selection
    if history.is_cuda and not torch.is_grad_enabled() and not enforce_packing:
        dx_k, dx_bg, dx_sub = _compiled_postprocess_sparse(
            history,
            topk,
            bg_idx,
            disp_sub,
            gamma,
            gap_k,
            z_surf,
            foot_engage,
            poses,
            geometry["sizes"],
            floor_slot,
            k_act,
            float(p_radius),
            float(dt),
        )
        geometry["cached_dx_sub"] = dx_sub
        if geometry.get("return_sparse_disp", False):
            return (dx_sub, active_count) if return_active_count else dx_sub
    else:
        dx_k = gamma.unsqueeze(-1) * disp_sub[:k_act]
        dx_k = apply_free_surface_ballistics(history[topk, -1], history[topk, -2], dx_k, gap_k, z_surf, p_radius, dt)
        if enforce_packing:
            dx_k = enforce_packing_incompressibility(history[topk, -1], dx_k, tau_k=gamma)
        dx_k = project_tray_bounds(history[topk, -1], dx_k, poses, geometry["sizes"], floor_slot, p_radius)
        dx_bg = project_tray_bounds(
            history[bg_idx, -1], foot_engage * disp_sub[k_act:], poses, geometry["sizes"], floor_slot, p_radius
        )
        if not torch.is_grad_enabled():
            geometry["cached_dx_sub"] = torch.cat((dx_k, dx_bg), dim=0)
            if geometry.get("return_sparse_disp", False):
                return (geometry["cached_dx_sub"], active_count) if return_active_count else geometry["cached_dx_sub"]
    if not torch.is_grad_enabled():
        disp_buf = geometry.get("disp_buf")
        if disp_buf is None or disp_buf.shape != (n_pts, 3):
            disp_buf = torch.zeros((n_pts, 3), dtype=history.dtype, device=device)
            geometry["disp_buf"] = disp_buf
        else:
            disp_buf.zero_()
        disp_buf.index_put_((bg_idx,), dx_bg)
        disp_buf.index_put_((topk,), dx_k)
        return (disp_buf, active_count) if return_active_count else disp_buf
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
    reuse_selection: bool = False,
    return_active_count: bool = False,
) -> tuple[torch.Tensor, list[int]]:
    """Batched multi-environment active compaction + quadrature background anchors."""
    num_envs = len(geometries)
    n_pts = history_flat.shape[0] // num_envs
    h_len = history_flat.shape[1]
    device = history_flat.device
    hist_env = history_flat.view(num_envs, n_pts, h_len, 3)

    foot_slots_0 = geometries[0]["foot_slots"]
    floor_slot_0 = geometries[0]["floor_slot"]
    p_radius = geometries[0].get("p_radius", 0.02 * material[11])
    k_act = resolve_active_k(active_k, n_pts, material)
    poses_env = torch.stack(poses_list, dim=0)
    vels_env = torch.stack(velocities_list, dim=0)
    if k_act is None:
        n_all, p_all, i_all = features_batched(
            hist_env,
            poses_env,
            vels_env,
            geometries[0]["kinds"],
            geometries[0]["sizes"],
            geometries[0]["friction"],
            material,
            dt=dt,
            support=support,
        )
        d_all = model(n_all, p_all, i_all).float()
        d_all = project_tray_bounds(
            hist_env[:, :, -1], d_all, poses_list[0], geometries[0]["sizes"], floor_slot_0, p_radius
        )
        return d_all.reshape(-1, 3), [n_pts] * num_envs

    foot_poses = poses_env[:, foot_slots_0]
    foot_vels = vels_env[:, foot_slots_0]
    x_cur = hist_env[:, :, -1]
    x_prev = hist_env[:, :, -2]
    mu_mat = geometries[0].get("mu_mat", material[3])
    m_bg = min(int(anchor_m), n_pts - k_act)
    bg_1d = geometries[0].get("bg_idx")
    if bg_1d is None or len(bg_1d) != m_bg:
        bg_1d = torch.linspace(0, n_pts - 1, m_bg, device=device).long()
    bg_exp = bg_1d.unsqueeze(0).expand(num_envs, -1)

    if reuse_selection and "cached_topk_batched" in geometries[0]:
        topk = geometries[0]["cached_topk_batched"]
        idx = geometries[0]["cached_idx_batched"]
        mult = geometries[0]["cached_mult_batched"]
        gamma = geometries[0]["cached_gamma_batched"]
        z_surf = geometries[0]["cached_z_surf_batched"]
        x_cur_k = torch.gather(x_cur, 1, topk.unsqueeze(-1).expand(-1, -1, 3))
        x_prev_k = torch.gather(x_prev, 1, topk.unsqueeze(-1).expand(-1, -1, 3))
        gap_k, _, _ = unclamped_foot_box_kinematics(
            x_cur_k, x_prev_k, foot_poses, foot_vels, geometries[0]["sizes"][foot_slots_0], p_radius, dt
        )
        active_counts = [k_act] * num_envs
    else:
        gap_min, _n_foot, approach = unclamped_foot_box_kinematics(
            x_cur, x_prev, foot_poses, foot_vels, geometries[0]["sizes"][foot_slots_0], p_radius, dt
        )
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
        gap_k = torch.gather(gap_min, 1, topk)
        x_cur_k = torch.gather(x_cur, 1, topk.unsqueeze(-1).expand(-1, -1, 3))
        x_prev_k = torch.gather(x_prev, 1, topk.unsqueeze(-1).expand(-1, -1, 3))
        if "cached_topk_batched" not in geometries[0]:
            geometries[0]["cached_topk_batched"] = topk.clone()
            geometries[0]["cached_idx_batched"] = idx.clone()
            geometries[0]["cached_mult_batched"] = mult.clone()
            geometries[0]["cached_gamma_batched"] = gamma.clone()
            geometries[0]["cached_z_surf_batched"] = z_surf.clone()
        else:
            geometries[0]["cached_topk_batched"].copy_(topk)
            geometries[0]["cached_idx_batched"].copy_(idx)
            geometries[0]["cached_mult_batched"].copy_(mult)
            geometries[0]["cached_gamma_batched"].copy_(gamma)
            geometries[0]["cached_z_surf_batched"].copy_(z_surf)
        active_counts = (
            (tau > 0.0).sum(dim=-1).int().tolist()
            if (return_active_count and not torch.cuda.is_current_stream_capturing())
            else [k_act] * num_envs
        )

    idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_len, 3)
    h_sub = torch.gather(hist_env, 1, idx_exp)
    n_sub, p_sub, i_sub = features_batched(
        h_sub,
        poses_env,
        vels_env,
        geometries[0]["kinds"],
        geometries[0]["sizes"],
        geometries[0]["friction"],
        material,
        dt=dt,
        support=support,
    )
    disp_sub = model(n_sub, p_sub, i_sub, mult)
    dx_k = gamma.unsqueeze(-1) * disp_sub[:, :k_act].float()
    dx_bg = disp_sub[:, k_act:].float()
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
