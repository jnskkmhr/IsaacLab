# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replace the native material step with Transolver and analytical foot reactions.

The native coupled solver still advances the actuated robot. No MPM step or
constitutive solve is called after installation (or optional initial settling warmup).
Supports flat trays, sloped ramps, sinusoidal dunes, and dynamic tilting trays via
intrinsic terrain-chart coordinates.
"""

import math

import newton
import numpy as np
import torch
import warp as wp
from torch.nn import functional as F

from particle_nerd.g1.data import (
    collider_state,
    compact_step,
    compact_step_batched,
    infer_collider_slots,
    multiply,
    resolve_active_k,
    rotate,
)
from particle_nerd.g1.forces import G1ProxyEstimator
from particle_nerd.transolver.features import material_features
from particle_nerd.transolver.model import ParticleTransolver


@wp.kernel
def scatter_wrench(
    local_feet: wp.array(dtype=int),
    mapping: wp.array(dtype=int),
    source: wp.array(dtype=wp.spatial_vector),
    target: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    target[mapping[local_feet[i]]] = source[i]


@torch.compile(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
def _compiled_tilting_bed_to_world(
    x_bed: torch.Tensor,
    x_world_prev: torch.Tensor,
    p_tray: torch.Tensor,
    q_tray: torch.Tensor,
    v_tray: torch.Tensor,
    w_tray: torch.Tensor,
    bed_depth: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused rigid-tray world transform and particle velocity for tilting beds."""
    p_next = p_tray + dt * v_tray
    half_w = 0.5 * dt * w_tray
    q_delta = torch.cat((half_w, torch.ones(1, device=x_bed.device, dtype=half_w.dtype)))
    q_next = F.normalize(multiply(q_delta, q_tray), dim=-1)
    shifted = x_bed + torch.tensor([0.0, 0.0, bed_depth], device=x_bed.device, dtype=x_bed.dtype)
    q_exp = q_next.unsqueeze(0).expand(x_bed.shape[0], 4)
    pred_world = p_next + rotate(q_exp, shifted)
    v_world = (pred_world - x_world_prev) / dt
    return pred_world, v_world


def _canonical_boundary_boxes(lower, upper, device):
    """Build 5 boundary box descriptors (1 floor + 4 walls) for intrinsic bed bounds [lower, upper]."""
    x_lo, y_lo, z_lo = float(lower[0]), float(lower[1]), float(lower[2])
    x_hi, y_hi, _z_hi = float(upper[0]), float(upper[1]), float(upper[2])
    lx, ly = x_hi - x_lo, y_hi - y_lo
    cx, cy = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    wall = 0.1
    wall_h = 0.4
    floor_h = 0.1
    poses = torch.tensor(
        [
            [cx, cy, z_lo - 0.5 * floor_h, 0.0, 0.0, 0.0, 1.0],
            [x_hi + 0.5 * wall, cy, z_lo + 0.5 * wall_h, 0.0, 0.0, 0.0, 1.0],
            [x_lo - 0.5 * wall, cy, z_lo + 0.5 * wall_h, 0.0, 0.0, 0.0, 1.0],
            [cx, y_hi + 0.5 * wall, z_lo + 0.5 * wall_h, 0.0, 0.0, 0.0, 1.0],
            [cx, y_lo - 0.5 * wall, z_lo + 0.5 * wall_h, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    sizes = torch.tensor(
        [
            [0.5 * lx + wall, 0.5 * ly + wall, 0.5 * floor_h],
            [0.5 * wall, 0.5 * ly + wall, 0.5 * wall_h],
            [0.5 * wall, 0.5 * ly + wall, 0.5 * wall_h],
            [0.5 * lx + wall, 0.5 * wall, 0.5 * wall_h],
            [0.5 * lx + wall, 0.5 * wall, 0.5 * wall_h],
        ],
        dtype=torch.float32,
        device=device,
    )
    friction = torch.full((5,), 0.85, dtype=torch.float32, device=device)
    return poses, sizes, friction


class LearnedMaterial:
    def __init__(
        self,
        raw,
        checkpoint,
        *,
        iterations=50,
        effective_mass_scale=10.0,
        contact_relaxation=0.1,
        contact_solver="jacobi",
        active_k=None,
        warmup_steps=0,
        tangent_mass_scale=None,
        compile_model=True,
    ):
        from isaaclab_newton.physics import NewtonManager

        self.num_envs = raw.num_envs
        self.raw, self.manager = raw, NewtonManager
        self.coupled = NewtonManager._solver
        self.entry = self.coupled._entries["sand"]
        self.mpm = self.entry.solver
        self.master = NewtonManager.get_model()
        self.device = str(self.master.device)
        self.warmup_steps = int(warmup_steps)
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if saved["config"].get("schema") != "g1_particle_nerd_v1":
            raise ValueError("Requires the G1-trained checkpoint")
        self.dt = saved["config"]["dt"]
        if not np.isclose(self.dt, 0.005):
            raise ValueError("This adapter requires a 200 Hz particle model")
        self.architecture = saved.get("architecture")
        if self.architecture != "particle_transolver_v1":
            raise ValueError(f"Expected particle_transolver_v1, got {self.architecture}")
        self.active_k = active_k
        self.model_config = saved["config"]["model"]
        self.model = ParticleTransolver(**self.model_config).to(self.device).eval()
        self.model.load_state_dict(saved["model"])
        self.compiled = bool(compile_model)
        if self.compiled:
            self.model = torch.compile(self.model, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        self.support = saved["config"]["features"]["support"]
        pw, bw = self.master.particle_world.numpy(), self.master.body_world.numpy()
        self.pids = [
            torch.as_tensor(np.flatnonzero(pw == e), device=self.device, dtype=torch.long) for e in range(self.num_envs)
        ]
        self.particles_per_env = len(self.pids[0])
        if any(len(p) != self.particles_per_env for p in self.pids):
            raise ValueError("All G1 environments must have equal particle counts")
        global_feet_all = raw._foot_body_ids.cpu().numpy()
        mapping = self.entry.body_local_to_global.numpy()
        local_feet_all = []
        for e in range(self.num_envs):
            for gf in global_feet_all[e]:
                local_feet_all.append(int(np.flatnonzero(mapping == gf).item()))
        self.estimator = G1ProxyEstimator(
            self.mpm,
            local_feet_all,
            iterations=iterations,
            effective_mass_scale=effective_mass_scale,
            relaxation=contact_relaxation,
            contact_solver=contact_solver,
            tangent_mass_scale=tangent_mass_scale,
            particles_per_env=self.particles_per_env,
        )
        self.local_feet = wp.array(local_feet_all, dtype=int, device=self.master.device)
        self.local_foot_tensor = torch.tensor(local_feet_all, device=self.device, dtype=torch.long).view(
            self.num_envs, -1
        )
        self.global_foot_tensor = torch.tensor(global_feet_all, device=self.device, dtype=torch.long)
        self.wrench = self.estimator.wrench
        self.metadata = self.estimator.metadata

        sand_spawn = raw.cfg.scene.sand.spawn
        self.bed_lower = tuple(float(v) for v in getattr(sand_spawn, "lower", (-1.5, -1.5, -0.3)))
        self.bed_upper = tuple(float(v) for v in getattr(sand_spawn, "upper", (1.5, 1.5, 0.0)))
        self.bed_depth = float(self.bed_upper[2] - self.bed_lower[2])
        self.terrain_kind = "flat"
        self.terrain_slope = 0.0
        self.dune_height = 0.0
        self.dune_waves = 1
        self.tray_body_ids = []
        if getattr(raw.cfg.scene, "testbed", None) is not None:
            self.terrain_kind = "tilting"
            for e in range(self.num_envs):
                matches = [
                    i
                    for i, lbl in enumerate(self.master.body_label)
                    if f"env_{e}/" in lbl and lbl.endswith("/Testbed/tray")
                ]
                if len(matches) != 1:
                    raise ValueError(f"Env {e}: Expected one /Testbed/tray body, got {matches}")
                self.tray_body_ids.append(matches[0])
        elif getattr(sand_spawn, "terrain_generator", None) is not None:
            sub = next(iter(sand_spawn.terrain_generator.sub_terrains.values()))
            if hasattr(sub, "slope_range"):
                self.terrain_kind = "slope"
                self.terrain_slope = float(sub.slope_range[0])
            elif hasattr(sub, "height_range") and hasattr(sub, "num_waves"):
                self.terrain_kind = "dune"
                self.dune_height = float(sub.height_range[0])
                self.dune_waves = int(sub.num_waves)

        sb, st = self.master.shape_body.numpy(), self.master.shape_type.numpy()

        def tensor(a):
            return torch.as_tensor(a, dtype=torch.float32, device=self.device)

        can_b_poses, can_b_sizes, can_b_fric = _canonical_boundary_boxes(
            (-1.5, -1.5, -self.bed_depth), (1.5, 1.5, 0.0), self.device
        )
        bed_b_poses, bed_b_sizes, bed_b_fric = _canonical_boundary_boxes(self.bed_lower, self.bed_upper, self.device)

        self.geometries = []
        self.foot_slots_list = []
        for e in range(self.num_envs):
            bids_e = set(np.flatnonzero(bw == e))
            gfeet_e = global_feet_all[e]
            eligible = set(gfeet_e) | {i for i in bids_e if "/MPM" in self.master.body_label[i]}
            ids = np.array([i for i, b in enumerate(sb) if b in eligible and st[i] == newton.GeoType.BOX])
            if len(ids) == 7 and self.terrain_kind == "flat":
                sizes_e = tensor(self.master.shape_scale.numpy()[ids])
                fslots_e = torch.tensor(
                    [int(np.flatnonzero(sb[ids] == foot).item()) for foot in gfeet_e],
                    device=self.device,
                    dtype=torch.long,
                )
                fslots_e, floor_slot_e = infer_collider_slots(sizes_e, fslots_e)
                local_pose_e = tensor(self.master.shape_transform.numpy()[ids])
                fric_e = tensor(self.master.shape_material_mu.numpy()[ids])
                body_ids_e = torch.as_tensor(sb[ids], device=self.device, dtype=torch.long)
                synthetic_mask = torch.zeros(7, dtype=torch.bool, device=self.device)
                has_synthetic = False
                non_foot = [i for i in range(7) if i not in fslots_e.tolist()]
                can_poses_e = local_pose_e.clone()
                can_sizes_e = sizes_e.clone()
                can_poses_e[floor_slot_e] = can_b_poses[0]
                can_sizes_e[floor_slot_e] = can_b_sizes[0]
                wall_slots = [i for i in non_foot if i != floor_slot_e]
                for w_idx, slot in enumerate(wall_slots[:4]):
                    can_poses_e[slot] = can_b_poses[1 + w_idx]
                    can_sizes_e[slot] = can_b_sizes[1 + w_idx]
            else:
                foot_shape_ids = np.array(
                    [int(np.flatnonzero((sb == foot) & (st == newton.GeoType.BOX)).item()) for foot in gfeet_e]
                )
                foot_sizes = tensor(self.master.shape_scale.numpy()[foot_shape_ids])
                foot_local = tensor(self.master.shape_transform.numpy()[foot_shape_ids])
                foot_fric = tensor(self.master.shape_material_mu.numpy()[foot_shape_ids])
                sizes_e = torch.cat((foot_sizes, bed_b_sizes), dim=0)
                local_pose_e = torch.cat((foot_local, bed_b_poses), dim=0)
                fric_e = torch.cat((foot_fric, bed_b_fric), dim=0)
                body_ids_e = torch.cat(
                    (
                        torch.as_tensor(gfeet_e, device=self.device, dtype=torch.long),
                        torch.zeros(5, device=self.device, dtype=torch.long),
                    ),
                    dim=0,
                )
                fslots_e = torch.tensor([0, 1], device=self.device, dtype=torch.long)
                floor_slot_e = 2
                synthetic_mask = torch.tensor([False, False, True, True, True, True, True], device=self.device)
                has_synthetic = True
                can_poses_e = torch.cat((foot_local, can_b_poses), dim=0)
                can_sizes_e = torch.cat((foot_sizes, can_b_sizes), dim=0)

            self.geometries.append(
                dict(
                    body_ids=body_ids_e,
                    local_pose=local_pose_e,
                    sizes=sizes_e,
                    friction=fric_e,
                    com=tensor(self.master.body_com.numpy()),
                    kinds=torch.ones(7, dtype=torch.int32, device=self.device),
                    foot_slots=fslots_e,
                    floor_slot=floor_slot_e,
                    synthetic_mask=synthetic_mask,
                    has_synthetic=has_synthetic,
                    canonical_poses=can_poses_e,
                    canonical_sizes=can_sizes_e,
                )
            )
            self.foot_slots_list.append(fslots_e)
        self.geometry = self.geometries[0]
        self.foot_slots = self.foot_slots_list[0]
        from particle_nerd.g1 import MATERIAL_KEYS

        material = {k: getattr(raw.cfg.scene.sand.spawn.material, k) for k in MATERIAL_KEYS}
        self.material = tensor(
            material_features(
                dict(material=material, grid_spacing_m=0.04),
                float(self.mpm.model.particle_mass.numpy()[0]),
                float(self.mpm.model.particle_radius.numpy()[0]),
                self.dt,
            )
        )
        self.origins = raw.scene.env_origins.clone()
        self.origin = self.origins[0]
        self.body_world_tensor = torch.as_tensor(bw, device=self.device, dtype=torch.long).clamp_min(0)
        self.particle_world_tensor = torch.as_tensor(pw, device=self.device, dtype=torch.long).clamp_min(0)
        self.body_origins = self.origins[self.body_world_tensor]
        self.particle_origins = self.origins[self.particle_world_tensor]
        self.zero_origins = bool((self.origins.abs().max() == 0).item())
        raw_active_k = active_k if active_k is not None else saved["config"].get("active_k")
        self.active_k = resolve_active_k(raw_active_k, self.particles_per_env, self.material)
        self.anchor_m = int(saved["config"].get("anchor_m", 512) or 512)
        p_rad_float = 0.02 * float(self.material[11].item())
        mu_mat_float = float(self.material[3].item())
        needs_sliding = max(self.bed_upper[0] - self.bed_lower[0], self.bed_upper[1] - self.bed_lower[1]) > 3.1
        m_bg = min(self.anchor_m, self.particles_per_env - (self.active_k or 0))
        bg_idx_cached = torch.linspace(0, self.particles_per_env - 1, max(1, m_bg), device=self.device).long()
        for g_e in self.geometries:
            g_e["p_radius"] = p_rad_float
            g_e["mu_mat"] = mu_mat_float
            g_e["sliding_window"] = needs_sliding
            g_e["bg_idx"] = bg_idx_cached
            g_e["return_sparse_disp"] = False
        self.last_active_counts = [self.active_k or self.particles_per_env] * self.num_envs
        state_0 = NewtonManager.get_state_0()
        self.body_pose = wp.to_torch(state_0.body_q).clone()
        self.body_pose[:, :3] -= self.body_origins
        self.body_velocity = wp.to_torch(state_0.body_qd).clone()
        self.history = None
        self.steps = 0
        self.total_steps = 0
        self._substep = 0
        self._graph_scheduled = False
        self.original_coupled_step = self.coupled.step
        self.original_material_step = self.mpm.step
        self.original_harvest = self.mpm.coupling_harvest_proxy_wrenches
        self.coupled.step = self.step
        self.mpm.step = self.particle_step
        self.mpm.coupling_harvest_proxy_wrenches = self.harvest
        self.manager._graph = None
        self.manager._graph_capture_pending = False

    def _terrain_z_and_slope(self, x_coord: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_lo = self.bed_lower[0]
        lx = max(1e-6, self.bed_upper[0] - x_lo)
        u = (x_coord - x_lo).clamp(0.0, lx)
        if self.terrain_kind == "slope":
            s = self.terrain_slope
            return s * u, torch.full_like(x_coord, s)
        if self.terrain_kind == "dune":
            omega = 2.0 * math.pi * self.dune_waves / lx
            phi = omega * u
            eta = 0.5 * self.dune_height * (1.0 - torch.cos(phi))
            slope = 0.5 * self.dune_height * omega * torch.sin(phi)
            return eta, slope
        return torch.zeros_like(x_coord), torch.zeros_like(x_coord)

    def _world_to_bed(self, x_world: torch.Tensor, env_idx: int = 0) -> torch.Tensor:
        if self.terrain_kind in ("slope", "dune"):
            eta, _ = self._terrain_z_and_slope(x_world[..., 0])
            return torch.stack((x_world[..., 0], x_world[..., 1], x_world[..., 2] - eta), dim=-1)
        if self.terrain_kind == "tilting":
            tid = self.tray_body_ids[env_idx]
            p_tray = self.body_pose[tid, :3]
            q_tray = self.body_pose[tid, 3:7]
            q_inv = torch.cat((-q_tray[:3], q_tray[3:4]), dim=-1)
            rel = x_world - p_tray
            local = rotate(
                q_inv.expand_as(rel) if rel.ndim == 1 else q_inv.unsqueeze(0).expand(*rel.shape[:-1], 4), rel
            )
            local = local.clone()
            local[..., 2] -= self.bed_depth
            return local
        return x_world

    def _bed_to_world(self, x_bed: torch.Tensor, dt: float, env_idx: int = 0) -> torch.Tensor:
        if self.terrain_kind in ("slope", "dune"):
            eta, _ = self._terrain_z_and_slope(x_bed[..., 0])
            return torch.stack((x_bed[..., 0], x_bed[..., 1], x_bed[..., 2] + eta), dim=-1)
        if self.terrain_kind == "tilting":
            tid = self.tray_body_ids[env_idx]
            p_tray = self.body_pose[tid, :3]
            q_tray = self.body_pose[tid, 3:7]
            v_tray = self.body_velocity[tid, :3]
            w_tray = self.body_velocity[tid, 3:6]
            p_next = p_tray + dt * v_tray
            half_w = 0.5 * dt * w_tray
            q_delta = torch.cat((half_w, torch.ones(1, device=self.device, dtype=half_w.dtype)))
            q_next = F.normalize(multiply(q_delta, q_tray), dim=-1)
            shifted = x_bed.clone()
            shifted[..., 2] += self.bed_depth
            q_exp = q_next.unsqueeze(0).expand(*shifted.shape[:-1], 4)
            return p_next + rotate(q_exp, shifted)
        return x_bed

    def _collider_state_in_bed(self, geom: dict, env_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        poses, velocities = collider_state(self.body_pose, self.body_velocity, geom)
        if geom.get("has_synthetic", False):
            syn = geom["synthetic_mask"]
            poses = torch.where(syn.unsqueeze(-1), geom["local_pose"], poses)
            velocities = torch.where(syn.unsqueeze(-1), torch.zeros_like(velocities), velocities)
        fslots = geom["foot_slots"]
        if self.terrain_kind in ("slope", "dune"):
            poses = poses.clone()
            velocities = velocities.clone()
            eta_f, s_f = self._terrain_z_and_slope(poses[fslots, 0])
            theta_f = torch.atan(s_f)
            half_t = 0.5 * theta_f
            zeros = torch.zeros_like(half_t)
            q_slope = torch.stack((zeros, torch.sin(half_t), zeros, torch.cos(half_t)), dim=-1)
            poses[fslots, 2] -= eta_f
            poses[fslots, 3:7] = F.normalize(multiply(q_slope, poses[fslots, 3:7]), dim=-1)
            velocities[fslots, 2] -= s_f * velocities[fslots, 0]
        elif self.terrain_kind == "tilting":
            poses = poses.clone()
            velocities = velocities.clone()
            tid = self.tray_body_ids[env_idx]
            p_tray = self.body_pose[tid, :3]
            q_tray = self.body_pose[tid, 3:7]
            v_tray = self.body_velocity[tid, :3]
            w_tray = self.body_velocity[tid, 3:6]
            q_inv = torch.cat((-q_tray[:3], q_tray[3:4]), dim=-1)
            q_inv_f = q_inv.unsqueeze(0).expand(len(fslots), 4)
            rel_p = poses[fslots, :3] - p_tray.unsqueeze(0)
            local_p = rotate(q_inv_f, rel_p)
            local_p[:, 2] -= self.bed_depth
            poses[fslots, :3] = local_p
            poses[fslots, 3:7] = F.normalize(multiply(q_inv_f, poses[fslots, 3:7]), dim=-1)
            v_tray_at_f = v_tray.unsqueeze(0) + torch.linalg.cross(w_tray.unsqueeze(0).expand_as(rel_p), rel_p, dim=-1)
            velocities[fslots, :3] = rotate(q_inv_f, velocities[fslots, :3] - v_tray_at_f)
            velocities[fslots, 3:6] = rotate(q_inv_f, velocities[fslots, 3:6] - w_tray.unsqueeze(0))
        return poses, velocities

    @torch.inference_mode()
    def step(self, state_in, state_out, control, contacts, dt):
        warp_stream = wp.get_stream(self.device)
        torch_stream = wp.stream_to_torch(warp_stream)
        with torch.cuda.stream(torch_stream):
            self.body_pose.copy_(wp.to_torch(state_in.body_q))
            if not self.zero_origins:
                self.body_pose[:, :3].sub_(self.body_origins)
            self.body_velocity.copy_(wp.to_torch(state_in.body_qd))
        return self.original_coupled_step(state_in, state_out, control, contacts, dt)

    @torch.inference_mode()
    def particle_step(self, state_in, state_out, control, contacts, dt):
        warp_stream = wp.get_stream(self.device)
        torch_stream = wp.stream_to_torch(warp_stream)
        with torch.cuda.stream(torch_stream):
            is_capturing = torch.cuda.is_current_stream_capturing()
        if self.total_steps < self.warmup_steps and not is_capturing:
            self.total_steps += 1
            return self.original_material_step(state_in, state_out, control, contacts, dt)
        del control, contacts
        with torch.cuda.stream(torch_stream):
            alias_io = state_in.particle_q.ptr == state_out.particle_q.ptr
            need_copy_p = self.active_k is None or alias_io
            self.estimator.capture(state_in, dt, copy_particles=need_copy_p)
            q_in_t = wp.to_torch(state_in.particle_q)
            q_out_t = wp.to_torch(state_out.particle_q)
            qd_out_t = wp.to_torch(state_out.particle_qd)
            if self.history is None:
                x_world_init = q_in_t if self.zero_origins else (q_in_t - self.particle_origins)
                v_world = wp.to_torch(state_in.particle_qd)
                frames = []
                for k in reversed(range(self.model_config["history"])):
                    xk_world = x_world_init - k * dt * v_world
                    if self.num_envs == 1:
                        frames.append(self._world_to_bed(xk_world, 0))
                    else:
                        frames.append(
                            torch.cat(
                                [
                                    self._world_to_bed(
                                        xk_world[e * self.particles_per_env : (e + 1) * self.particles_per_env], e
                                    )
                                    for e in range(self.num_envs)
                                ],
                                dim=0,
                            )
                        )
                self.history = torch.stack(frames, dim=1).contiguous()
                if not alias_io:
                    q_out_t.copy_(q_in_t)
                qd_out_t.zero_()
            if self.num_envs == 1:
                poses, velocities = self._collider_state_in_bed(self.geometry, 0)
                reuse_sel = (self._substep % 4 != 0) and ("cached_topk" in self.geometry)
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    disp_bed = compact_step(
                        self.model,
                        self.history,
                        poses,
                        velocities,
                        self.geometry,
                        self.material,
                        dt,
                        support=self.support,
                        active_k=self.active_k,
                        anchor_m=self.anchor_m,
                        return_active_count=False,
                        reuse_selection=reuse_sel,
                    )
                active_idx_eval = self.geometry.get("cached_topk") if self.active_k is not None else None
                if self.terrain_kind == "tilting":
                    tid = self.tray_body_ids[0]
                    x_world_prev = q_in_t if self.zero_origins else (q_in_t - self.particle_origins)
                    next_bed = self.history[:, 2] + disp_bed
                    pred_w, v_w = _compiled_tilting_bed_to_world(
                        next_bed,
                        x_world_prev,
                        self.body_pose[tid, :3],
                        self.body_pose[tid, 3:7],
                        self.body_velocity[tid, :3],
                        self.body_velocity[tid, 3:6],
                        self.bed_depth,
                        float(dt),
                    )
                    q_out_t.copy_(pred_w if self.zero_origins else (pred_w + self.particle_origins))
                    qd_out_t.copy_(v_w)
                    self.history[:, 0].copy_(self.history[:, 1])
                    self.history[:, 1].copy_(self.history[:, 2])
                    self.history[:, 2].copy_(next_bed)
                else:
                    x_world = q_in_t if self.zero_origins else (q_in_t - self.particle_origins)
                    next_bed = self.history[:, -1] + disp_bed
                    predicted_world = self._bed_to_world(next_bed, dt, 0)
                    world_displacement = predicted_world - x_world
                    q_out_t.copy_(predicted_world if self.zero_origins else (predicted_world + self.particle_origins))
                    qd_out_t.copy_(world_displacement / dt)
                    self.history[:, 0].copy_(self.history[:, 1])
                    self.history[:, 1].copy_(self.history[:, 2])
                    self.history[:, 2].copy_(next_bed)
            else:
                poses_list, vels_list = [], []
                for e in range(self.num_envs):
                    p_e, v_e = self._collider_state_in_bed(self.geometries[e], e)
                    poses_list.append(p_e)
                    vels_list.append(v_e)
                reuse_sel = (self._substep % 4 != 0) and ("cached_topk_batched" in self.geometries[0])
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    disp_bed, self.last_active_counts = compact_step_batched(
                        self.model,
                        self.history,
                        poses_list,
                        vels_list,
                        self.geometries,
                        self.material,
                        dt,
                        support=self.support,
                        active_k=self.active_k,
                        anchor_m=self.anchor_m,
                        reuse_selection=reuse_sel,
                        return_active_count=False,
                    )
                x_world = q_in_t if self.zero_origins else (q_in_t - self.particle_origins)
                next_bed = self.history[:, -1] + disp_bed
                predicted_world = (
                    next_bed
                    if self.terrain_kind == "flat"
                    else torch.cat(
                        [
                            self._bed_to_world(
                                next_bed[e * self.particles_per_env : (e + 1) * self.particles_per_env], dt, e
                            )
                            for e in range(self.num_envs)
                        ],
                        dim=0,
                    )
                )
                active_idx_eval = self.geometries[0].get("cached_topk_batched") if self.active_k is not None else None
                world_displacement = predicted_world - x_world
                q_out_t.copy_(predicted_world if self.zero_origins else (predicted_world + self.particle_origins))
                qd_out_t.copy_(world_displacement / dt)
                self.history[:, 0].copy_(self.history[:, 1])
                self.history[:, 1].copy_(self.history[:, 2])
                self.history[:, 2].copy_(next_bed)
            self.estimator.evaluate(state_out, active_idx=active_idx_eval)
        self._substep = (self._substep + 1) % 4
        if not is_capturing:
            self.steps += 1
            self.total_steps += 1
            if not self._graph_scheduled and self.history is not None and self.steps >= 4:
                self._substep = 0
                self.manager._invalidate_graph()
                self._graph_scheduled = True

    def harvest(self, body_local_to_proxy_global, out_body_f, **kwargs):
        if self.total_steps <= self.warmup_steps and self.steps == 0:
            return self.original_harvest(body_local_to_proxy_global, out_body_f, **kwargs)
        del kwargs
        out_body_f.zero_()
        wp.launch(
            scatter_wrench,
            len(self.local_feet),
            [self.local_feet, body_local_to_proxy_global, self.wrench, out_body_f],
            device=self.master.device,
        )

    def close(self):
        self.coupled.step = self.original_coupled_step
        self.mpm.step = self.original_material_step
        self.mpm.coupling_harvest_proxy_wrenches = self.original_harvest
