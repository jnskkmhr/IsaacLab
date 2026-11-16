# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Newton solver interface for ParticleNeRD-Local."""

import re
from pathlib import Path

import newton
import numpy as np
import torch
import warp as wp
from newton.solvers import SolverBase
from newton.solvers.experimental.coupled import CouplingInterface

from ..particle_nerd.kernels import gather_collider_state, reset_history, rewind_proxy, scatter_wrench
from ..particle_nerd.network.geometry import boundary_boxes
from ..particle_nerd.network.inference import resolve_active_k
from ..particle_world.contact import WorldParticleProxyEstimator
from .features import (
    active_particle_score_batched_kernel,
    advance_stationary_history_local_kernel,
    build_collider_sdf_grid,
    extract_sub_history_local_kernel,
    material_features,
    relax_amorphous_bed_kernel,
    scatter_sub_step_local_kernel,
)
from .model import ParticleNeRDLocalNetwork


class ParticleNeRDLocalSolver(SolverBase, CouplingInterface):
    """Step Lagrangian particles with ParticleNeRDLocalNetwork and reconstruct contact wrenches."""

    def __init__(self, model, cfg):  # noqa: C901
        super().__init__(model)
        if not model.device.is_cuda:
            raise ValueError("ParticleNeRDLocalSolver requires a CUDA device")
        self.cfg = cfg
        path = Path(cfg.checkpoint).expanduser()
        if not cfg.checkpoint or not path.is_file():
            raise FileNotFoundError("Set ParticleNeRDLocalSolverCfg.checkpoint to an existing trained checkpoint")
        saved = torch.load(path, map_location=str(model.device), weights_only=False)
        config = saved["config"]
        arch = saved.get("architecture") or config.get("architecture")
        if arch != "particle_nerd_local_v1":
            raise ValueError(f"Expected particle_nerd_local_v1 checkpoint, got {arch!r}")
        self.dt = float(config["dt"])
        self.support = float(config.get("features", {}).get("support", 0.07))
        model_cfg = dict(config["model"])
        for k in ("far_support_scale", "far_gap_weight", "far_influence_weight"):
            if getattr(cfg, k, None) is not None and k not in model_cfg:
                model_cfg[k] = float(getattr(cfg, k))
        if "spatial_slice_assign" not in model_cfg:
            model_cfg["spatial_slice_assign"] = False
            model_cfg["use_rms_swiglu"] = False
        self.network = ParticleNeRDLocalNetwork(**model_cfg).to(str(model.device)).eval()
        self.network.load_state_dict(saved["model"])

        self.worlds = model.particle_world.numpy()
        self.world_count = model.world_count
        counts = np.bincount(self.worlds, minlength=self.world_count)
        self.particle_world_ids = np.flatnonzero(counts)
        if len(counts) != self.world_count or not len(self.particle_world_ids):
            raise ValueError("ParticleNeRDLocalSolver requires particles assigned to simulation worlds")
        self.capacity = int(counts[self.particle_world_ids[0]])
        self.history_wp = wp.zeros(model.particle_count, dtype=wp.mat33, device=model.device)
        self.history = wp.to_torch(self.history_wp)
        self.neural_state = newton.State()
        self.neural_state.body_q = wp.empty_like(model.body_q)
        self.neural_state.body_qd = wp.empty_like(model.body_qd)
        self._has_neural_state = False
        self.selection_phase = 0
        self.capture_revision = 0
        self.reset(model.state())

        bw = model.body_world.numpy()
        compiled_patterns = [re.compile(pattern) for pattern in cfg.collider_body_patterns]
        world_pattern_matches: dict[tuple[int, int], list[int]] = {}
        particle_world_set = {int(w) for w in self.particle_world_ids}
        for i, label in enumerate(model.body_label):
            w = int(bw[i])
            if w not in particle_world_set:
                continue
            for p_idx, regex in enumerate(compiled_patterns):
                if regex.fullmatch(label):
                    world_pattern_matches.setdefault((w, p_idx), []).append(i)
        ids = []
        for world in self.particle_world_ids:
            w = int(world)
            selected = []
            for p_idx, pattern in enumerate(cfg.collider_body_patterns):
                found = world_pattern_matches.get((w, p_idx), [])
                if not found:
                    raise ValueError(
                        f"Expected at least one collider matching {pattern!r} in world {world}; found {found}"
                    )
                selected += found
            ids += selected
        self.num_colliders_per_world = len(ids) // len(self.particle_world_ids)
        self.collider_ids = torch.tensor(ids, device=str(model.device), dtype=torch.long).view(
            -1, self.num_colliders_per_world
        )
        self.collider_ids_wp = wp.array(ids, dtype=int, device=model.device)
        self.estimator = WorldParticleProxyEstimator(
            model,
            ids,
            colliders_per_world=self.num_colliders_per_world,
            iterations=cfg.contact_iterations,
            effective_mass_scale=cfg.effective_mass_scale,
            tangent_mass_scale=cfg.tangent_mass_scale,
            horiz_mass_scale=cfg.horizontal_mass_scale,
            up_mass_scale=cfg.upward_mass_scale,
            normal_power=cfg.normal_power,
            relaxation=cfg.contact_relaxation,
            gradient_radius=cfg.gradient_radius,
            gradient_contact_band=cfg.gradient_contact_band,
            stabilization=cfg.contact_stabilization,
            contact_offset=cfg.contact_offset,
            particles_per_env=self.capacity,
        )

        sb = model.shape_body.numpy()
        st = model.shape_type.numpy()
        sscale_np = model.shape_scale.numpy()
        sflags_np = model.shape_flags.numpy()
        colliding_mask = (sflags_np & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0
        geo_map = {
            int(newton.GeoType.BOX): 1,
            int(newton.GeoType.SPHERE): 2,
            int(newton.GeoType.CYLINDER): 4,
            int(newton.GeoType.CAPSULE): 5,
        }
        shapes = []
        for body in ids:
            cand = [s for s in np.flatnonzero((sb == body) & colliding_mask) if int(st[s]) in geo_map]
            if not cand:
                cand = [s for s in np.flatnonzero(sb == body) if int(st[s]) in geo_map]
            if not cand:
                cand = [int(np.flatnonzero(sb == body)[0])]
            shapes.append(int(max(cand, key=lambda s: float(np.max(sscale_np[s])))))

        def tensor(a):
            return torch.as_tensor(a, device=str(model.device), dtype=torch.float32)

        n_worlds = len(self.particle_world_ids)
        sw = model.shape_world.numpy()
        per_world_static = [
            [
                s
                for s in np.flatnonzero((sb < 0) & (sw == int(w)) & colliding_mask)
                if int(st[s]) == int(newton.GeoType.BOX)
            ][:5]
            for w in self.particle_world_ids
        ]
        if all(len(c) == 5 for c in per_world_static):
            flat_static = [s for c in per_world_static for s in c]
            st_tf = wp.to_torch(model.shape_transform)[flat_static].view(n_worlds, 5, 7).clone()
            st_sz = wp.to_torch(model.shape_scale)[per_world_static[0]].clone()
            st_mu = wp.to_torch(model.shape_material_mu)[per_world_static[0]].clone()
            self.boundary_poses = st_tf
            boundary_size, boundary_mu = st_sz, st_mu
        else:
            boundary_pose, boundary_size, boundary_mu = boundary_boxes(
                (-1.0, -0.6, -cfg.bed_depth), (1.0, 0.6, 0.0), str(model.device)
            )
            q_init = wp.to_torch(model.particle_q).view(n_worlds, self.capacity, 3)
            bed_xy = 0.5 * (q_init[:, :, :2].amin(dim=1) + q_init[:, :, :2].amax(dim=1))
            self.boundary_poses = boundary_pose.unsqueeze(0).repeat(n_worlds, 1, 1)
            self.boundary_poses[:, :, :2] += bed_xy.unsqueeze(1)
        self.boundary_pose = self.boundary_poses[0]
        q_init_env = wp.to_torch(model.particle_q).view(n_worlds, self.capacity, 3)
        q_min = q_init_env.amin(dim=1)
        q_max = q_init_env.amax(dim=1)
        bp_env = self.boundary_poses
        bsz = boundary_size
        z_fl = bp_env[:, 0, 2] + bsz[0, 2]
        wx_0 = torch.minimum(bp_env[:, 1, 0], bp_env[:, 2, 0]) + bsz[1, 0]
        wx_1 = torch.maximum(bp_env[:, 1, 0], bp_env[:, 2, 0]) - bsz[2, 0]
        wy_0 = torch.minimum(bp_env[:, 3, 1], bp_env[:, 4, 1]) + bsz[3, 1]
        wy_1 = torch.maximum(bp_env[:, 3, 1], bp_env[:, 4, 1]) - bsz[4, 1]
        self.tray_lower = torch.stack((wx_0 + 0.008, wy_0 + 0.008, z_fl + 0.004), dim=-1).contiguous()
        self.tray_upper = torch.stack(
            (wx_1 - 0.008, wy_1 - 0.008, torch.maximum(q_max[:, 2], z_fl) + 0.60), dim=-1
        ).contiguous()
        all_shape_tf = wp.to_torch(model.shape_transform)[shapes].view(
            len(self.particle_world_ids), self.num_colliders_per_world, 7
        )
        all_body_com = wp.to_torch(model.body_com)[ids].view(
            len(self.particle_world_ids), self.num_colliders_per_world, 3
        )
        self.shape_local_tf = all_shape_tf
        self.body_com_local = all_body_com
        self.geometry = []
        self.sdfs = []
        all_scales = wp.to_torch(model.shape_scale)
        all_margins = wp.to_torch(model.shape_margin)
        for env_idx in range(len(self.particle_world_ids)):
            env_shapes = shapes[env_idx * self.num_colliders_per_world : (env_idx + 1) * self.num_colliders_per_world]
            env_kinds = [geo_map.get(int(st[s]), 1) for s in env_shapes]
            norm_scales = []
            for s, k_raw in zip(env_shapes, env_kinds, strict=True):
                sc_t = all_scales[s].cpu().tolist()
                m_s = float(all_margins[s].item())
                if k_raw == 2:  # SPHERE -> (r, r, r)
                    r_eff = float(sc_t[0]) + m_s
                    norm_scales.append([r_eff, r_eff, r_eff])
                elif k_raw == 4:  # CYLINDER -> (r, r, half_height)
                    hh = float(sc_t[1]) if float(sc_t[1]) > 0.0 else float(sc_t[2])
                    norm_scales.append([float(sc_t[0]) + m_s, float(sc_t[0]) + m_s, hh + m_s])
                elif k_raw == 5:  # CAPSULE -> (r, half_height, r)
                    hh = float(sc_t[1]) if float(sc_t[1]) > 0.0 else float(sc_t[2])
                    norm_scales.append([float(sc_t[0]) + m_s, hh + m_s, float(sc_t[0]) + m_s])
                else:
                    norm_scales.append([float(sc_t[0]) + m_s, float(sc_t[1]) + m_s, float(sc_t[2]) + m_s])
            env_scales_t = torch.tensor(norm_scales, device=str(model.device), dtype=torch.float32)
            if cfg.use_unified_sdf_colliders and len(env_shapes) == 1 and env_kinds[0] in (1, 2, 4):
                size_tuple = tuple(norm_scales[0])
                sdf_grid = build_collider_sdf_grid(env_kinds[0], size_tuple, device=str(model.device))
                env_kinds = [3]
            else:
                sdf_grid = None
            kinds = torch.cat(
                (
                    torch.tensor(env_kinds, device=str(model.device), dtype=torch.int32),
                    torch.ones(5, device=str(model.device), dtype=torch.int32),
                )
            )
            sizes = torch.cat((env_scales_t, boundary_size))
            mu = torch.ones(len(env_shapes) + 5, device=str(model.device), dtype=torch.float32)
            self.geometry.append((kinds, sizes, mu))
            self.sdfs.append(sdf_grid)

        self.materials = []
        mass = wp.to_torch(model.particle_mass)[:: self.capacity].cpu().numpy()
        radius = wp.to_torch(model.particle_radius)[:: self.capacity].cpu().numpy()
        scene_radius = float(radius[0])
        if scene_radius > 0.0:
            self.network.local_agg.radius = 4.5 * scene_radius
            self.support = max(0.035, 4.375 * scene_radius)
        self.estimator.gradient_radius = float(self.network.local_agg.radius)
        material_keys = (
            "density",
            "young_modulus",
            "poisson_ratio",
            "friction",
            "yield_pressure",
            "tensile_yield_ratio",
            "yield_stress",
            "hardening",
            "dilatancy",
            "viscosity",
        )
        raw = {
            name: wp.to_torch(getattr(getattr(model, "mpm", None), name))[:: self.capacity].cpu().numpy()
            if hasattr(getattr(model, "mpm", None), name)
            else np.full(len(self.particle_world_ids), default, dtype=np.float32)
            for name, default in zip(
                material_keys, (1600.0, 1.0e15, 0.3, 0.48, 1.0e15, 0.0, 0.0, 0.0, 0.0, 0.0), strict=True
            )
        }
        for env in range(len(self.particle_world_ids)):
            values = {name: float(raw[name][env]) for name in material_keys}
            self.materials.append(tensor(material_features({"material": values}, mass[env], radius[env], self.dt)))

        device_str = str(model.device)
        settle_steps = int(getattr(cfg, "settle_steps", 0))
        if settle_steps > 0:
            self._settle_and_cache_particles(
                model,
                q_init_env,
                q_min,
                q_max,
                settle_steps,
                str(getattr(cfg, "settle_cache_dir", "/tmp/particle_nerd_local_settled_cache")),
                device_str,
            )

        if getattr(cfg, "compile_model", True):
            self.network._encode_With_local_skip = torch.compile(
                self.network._encode_With_local_skip,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
        self.zero_wall_vel = torch.zeros((n_worlds, 5, 6), device=device_str, dtype=torch.float32)
        self.mat_env = torch.stack(self.materials, dim=0).contiguous()
        self.dyn_sizes_env = torch.stack(
            [g[1][: self.num_colliders_per_world] for g in self.geometry], dim=0
        ).contiguous()
        resolved_k = resolve_active_k(getattr(cfg, "active_k", None), self.capacity, self.materials[0])
        if (
            isinstance(getattr(cfg, "active_k", None), str)
            and str(cfg.active_k).strip().lower() == "auto"
            and resolved_k is not None
        ):
            resolved_k = min(self.capacity, max(1536, int(resolved_k)))
        if resolved_k is not None and 0 < int(resolved_k) < self.capacity:
            self.active_k = int(resolved_k)
            self.m_bg = min(256, self.capacity - self.active_k)
            k_tot = self.active_k + self.m_bg
            self.estimator._batch_grid = self.network.local_agg._get_grid(
                device_str, n_worlds * self.active_k, with_groups=True
            )
            self._bg_1d = torch.linspace(0, self.capacity - 1, self.m_bg, device=device_str).long()
            self._bg_2d = self._bg_1d.unsqueeze(0).expand(n_worlds, -1)
            self._scores_buf = torch.empty((n_worlds, self.capacity), dtype=torch.float32, device=device_str)
            self._min_gaps_buf = torch.empty((n_worlds, self.capacity), dtype=torch.float32, device=device_str)
            self._hist_sub_buf = torch.empty((n_worlds, k_tot, 3, 3), dtype=torch.float32, device=device_str)
            self._cached_topk = torch.zeros((n_worlds, self.active_k), dtype=torch.int64, device=device_str)
            self._cached_idx = torch.zeros((n_worlds, k_tot), dtype=torch.int64, device=device_str)
            self._cached_mult = torch.ones((n_worlds, k_tot), dtype=torch.float32, device=device_str)
            self._cached_gamma = torch.ones((n_worlds, self.active_k), dtype=torch.float32, device=device_str)
        else:
            self.active_k = None
            self.m_bg = 0
            self.estimator._batch_grid = self.network.local_agg._get_grid(
                device_str, n_worlds * self.capacity, with_groups=True
            )
            self._cached_topk = None
            self._cached_idx = None
            self._cached_mult = None
            self._cached_gamma = None
        self.selection_phase = 0

    @torch.inference_mode()
    def _settle_and_cache_particles(
        self,
        model,
        q_init_env: torch.Tensor,
        q_min: torch.Tensor,
        q_max: torch.Tensor,
        settle_steps: int,
        cache_dir_str: str,
        device_str: str,
    ) -> None:
        """Settle one particle bed with full-bed ParticleNeRDLocalNetwork and cache it for all worlds & resets."""
        n_worlds = len(self.particle_world_ids)
        bed_center_xy = 0.5 * (q_min[:, :2] + q_max[:, :2])
        span = (q_max[0] - q_min[0]).cpu().tolist()
        cache_dir = Path(cache_dir_str)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = f"settled_v2_n{self.capacity}_s{span[0]:.3f}_{span[1]:.3f}_{span[2]:.3f}_steps{settle_steps}.pt"
        cache_path = cache_dir / cache_key

        if cache_path.is_file():
            q_local = torch.load(cache_path, map_location=device_str, weights_only=True)
        else:
            q0 = q_init_env[0].clone()
            # Apply strong amorphous jitter + 12-pass minimum-distance + gravity relaxation to break all lattice lines
            g_rng = torch.Generator(device=device_str)
            g_rng.manual_seed(2026)
            p_radius = float(wp.to_torch(model.particle_radius)[0].item())
            spacing_est = 2.0 * p_radius
            jitter_amp = 0.45 * spacing_est
            q0 = q0 + (2.0 * torch.rand(q0.shape, generator=g_rng, device=device_str) - 1.0) * jitter_amp
            lo_vec = wp.vec3(float(q_min[0, 0].item()), float(q_min[0, 1].item()), float(q_min[0, 2].item()))
            hi_vec = wp.vec3(float(q_max[0, 0].item()), float(q_max[0, 1].item()), float(q_max[0, 2].item()))
            q0 = torch.max(torch.min(q0, q_max[0].unsqueeze(0)), q_min[0].unsqueeze(0)).contiguous()
            q_tmp = torch.empty_like(q0)
            relax_grid = wp.HashGrid(32, 32, 16, device=device_str)
            relax_grid.reserve(self.capacity)
            min_dist = 0.88 * spacing_est
            for _ in range(12):
                q0_wp = wp.from_torch(q0, dtype=wp.vec3, requires_grad=False)
                q_tmp_wp = wp.from_torch(q_tmp, dtype=wp.vec3, requires_grad=False)
                relax_grid.build(q0_wp, min_dist)
                wp.launch(
                    relax_amorphous_bed_kernel,
                    self.capacity,
                    [relax_grid.id, q0_wp, min_dist, lo_vec, hi_vec, q_tmp_wp],
                    device=device_str,
                )
                q0.copy_(q_tmp)

            # Now run full-bed ParticleNeRDLocalNetwork settling steps with initial downward settling velocity
            v_init_settle = torch.zeros_like(q0)
            v_init_settle[:, 2] = -0.08
            hist0 = torch.stack(
                (q0 - 2.0 * self.dt * v_init_settle, q0 - self.dt * v_init_settle, q0), dim=1
            ).contiguous()
            kinds0, sizes0, mu0 = self.geometry[0]
            dyn_parked = torch.zeros((self.num_colliders_per_world, 7), device=device_str, dtype=torch.float32)
            dyn_parked[:, 2] = 10.0
            dyn_parked[:, 6] = 1.0
            poses0 = torch.cat((dyn_parked, self.boundary_poses[0]), dim=0).contiguous()
            vels0 = torch.zeros((poses0.shape[0], 6), device=device_str, dtype=torch.float32)
            lo0 = self.tray_lower[0].unsqueeze(0)
            hi0 = q_max[0].unsqueeze(0)
            for _ in range(settle_steps):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    next_p, next_v = self.network.step_from_history(
                        hist0,
                        poses0,
                        vels0,
                        kinds0,
                        sizes0,
                        mu0,
                        self.materials[0],
                        dt=self.dt,
                        support=self.support,
                        sdf=self.sdfs[0],
                        active_k=None,
                        return_vel=True,
                    )
                next_p = torch.max(torch.min(next_p, hi0), lo0)
                prev_p = torch.max(torch.min(next_p - self.dt * next_v, hi0), lo0)
                hist0[:, 0] = hist0[:, 1]
                hist0[:, 1] = prev_p
                hist0[:, 2] = next_p
            q_local = hist0[:, 2].clone()
            q_local[:, :2] -= bed_center_xy[0].unsqueeze(0)
            torch.save(q_local.cpu(), cache_path)
            q_local = q_local.to(device_str)

        q_all = q_local.unsqueeze(0).repeat(n_worlds, 1, 1)
        q_all[:, :, :2] += bed_center_xy.unsqueeze(1)
        wp.to_torch(model.particle_q).copy_(q_all.reshape(-1, 3))
        wp.to_torch(model.particle_qd).zero_()
        self.settled_particle_q = wp.clone(model.particle_q)
        self.reset(model.state())

    def notify_model_changed(self, flags):
        if flags & newton.ModelFlags.BODY_INERTIAL_PROPERTIES:
            self.estimator.refresh_body_inertia()

    def capture_collider_state(self, state, body_local_to_global):
        wp.launch(
            gather_collider_state,
            self.model.body_count,
            [body_local_to_global, state.body_q, state.body_qd, self.neural_state.body_q, self.neural_state.body_qd],
            device=self.model.device,
        )
        self._has_neural_state = True

    def coupling_rewind_proxy_body(self, mapping, state, forces, body_gravity_acceleration, dt):
        del body_gravity_acceleration
        wp.launch(
            rewind_proxy,
            len(mapping),
            [mapping, forces, state.body_q, self.model.body_inv_mass, self.model.body_inv_inertia, state.body_qd, dt],
            device=self.model.device,
        )

    def coupling_harvest_proxy_wrenches(self, mapping, out_body_f, **kwargs):
        del kwargs
        out_body_f.zero_()
        wp.launch(
            scatter_wrench,
            len(self.collider_ids_wp),
            [self.collider_ids_wp, mapping, self.estimator.wrench, out_body_f],
            device=self.model.device,
        )

    def reset(self, state, world_mask=None, flags=None):
        self._normalize_reset_world_mask(world_mask)
        if flags is not None and not (flags & newton.StateFlags.PARTICLE):
            return
        if hasattr(self, "settled_particle_q") and world_mask is None:
            wp.copy(state.particle_q, self.settled_particle_q)
            state.particle_qd.zero_()
        mask = (
            world_mask
            if world_mask is not None
            else wp.zeros(self.world_count + 1, dtype=wp.bool, device=self.model.device)
        )
        wp.launch(
            reset_history,
            self.model.particle_count,
            [
                state.particle_q,
                state.particle_qd,
                self.model.particle_world,
                mask,
                world_mask is None,
                self.dt,
                self.history_wp,
            ],
            device=self.model.device,
        )
        self.selection_phase = 0

    @torch.inference_mode()
    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        del control
        self.validate_observables(observables, contacts)
        if self.model.particle_count == 0:
            return
        stream = wp.stream_to_torch(wp.get_stream(self.model.device))
        with torch.cuda.stream(stream):
            self.estimator.capture(state_in, dt, copy_particles=(state_in.particle_q.ptr == state_out.particle_q.ptr))
            collider_source = self.neural_state if self._has_neural_state else state_in
            self._has_neural_state = False
            body_q = wp.to_torch(collider_source.body_q)
            body_qd = wp.to_torch(collider_source.body_qd)
            pos_in = wp.to_torch(state_in.particle_q)
            pos_out = wp.to_torch(state_out.particle_q)
            vel_out = wp.to_torch(state_out.particle_qd)

            num_envs = len(self.particle_world_ids)
            bp = body_q[self.collider_ids]
            bv = body_qd[self.collider_ids]
            ltf = self.shape_local_tf
            bcom = self.body_com_local
            q_xyz, q_w = bp[..., 3:6], bp[..., 6:7]
            t_rot = 2.0 * torch.linalg.cross(q_xyz, ltf[..., :3], dim=-1)
            w_pos = bp[..., :3] + ltf[..., :3] + q_w * t_rot + torch.linalg.cross(q_xyz, t_rot, dim=-1)
            x1, y1, z1, w1 = bp[..., 3], bp[..., 4], bp[..., 5], bp[..., 6]
            x2, y2, z2, w2 = ltf[..., 3], ltf[..., 4], ltf[..., 5], ltf[..., 6]
            w_quat = torch.stack(
                (
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                ),
                dim=-1,
            )
            t_com = 2.0 * torch.linalg.cross(q_xyz, bcom, dim=-1)
            p_com = bp[..., :3] + bcom + q_w * t_com + torch.linalg.cross(q_xyz, t_com, dim=-1)
            v_lin = bv[..., :3] + torch.linalg.cross(bv[..., 3:6], w_pos - p_com, dim=-1)
            dyn_pose = torch.cat((w_pos, w_quat), dim=-1).contiguous()
            dyn_vel = torch.cat((v_lin, bv[..., 3:6]), dim=-1).contiguous()
            poses_env = torch.cat((dyn_pose, self.boundary_poses), dim=1).contiguous()
            vels_env = torch.cat((dyn_vel, self.zero_wall_vel), dim=1).contiguous()
            kinds, sizes, mu = self.geometry[0]

            if self.active_k is not None:
                k_act = self.active_k
                k_tot = k_act + self.m_bg
                if state_in.particle_q.ptr != state_out.particle_q.ptr:
                    pos_out.copy_(pos_in)
                vel_out.zero_()

                if self.selection_phase == 0:
                    wp.launch(
                        active_particle_score_batched_kernel,
                        num_envs * self.capacity,
                        [
                            state_in.particle_q,
                            self.history_wp,
                            wp.from_torch(dyn_pose, requires_grad=False),
                            wp.from_torch(self.dyn_sizes_env, requires_grad=False),
                            self.capacity,
                            self.num_colliders_per_world,
                            float(dt),
                            wp.from_torch(self._scores_buf.view(-1), requires_grad=False),
                            wp.from_torch(self._min_gaps_buf.view(-1), requires_grad=False),
                        ],
                        device=self.model.device,
                    )
                    top_vals, topk = torch.topk(self._scores_buf, k=k_act, dim=1, largest=False, sorted=False)
                    thresh = top_vals.amax(dim=1, keepdim=True)
                    bg_scores = torch.gather(self._scores_buf, 1, self._bg_2d)
                    bg_valid = (bg_scores > thresh).float()
                    bg_valid = torch.where(bg_valid.sum(dim=1, keepdim=True) < 1.0, torch.ones_like(bg_valid), bg_valid)
                    bg_mult = bg_valid * (
                        float(self.capacity - k_act) / bg_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                    )
                    tau = ((thresh - top_vals) / 0.015).clamp(0.0, 1.0)
                    gamma = tau * tau * (3.0 - 2.0 * tau)
                    self._cached_topk.copy_(topk)
                    self._cached_idx[:, :k_act].copy_(topk)
                    self._cached_idx[:, k_act:].copy_(self._bg_2d)
                    self._cached_mult[:, :k_act].fill_(1.0)
                    self._cached_mult[:, k_act:].copy_(bg_mult)
                    self._cached_gamma.copy_(gamma)

                wp.launch(
                    extract_sub_history_local_kernel,
                    (num_envs, k_tot),
                    [
                        wp.from_torch(self._cached_idx, dtype=wp.int64, requires_grad=False),
                        state_in.particle_q,
                        self.history_wp,
                        self.capacity,
                        wp.from_torch(self._hist_sub_buf, requires_grad=False),
                    ],
                    device=self.model.device,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    next_pos_sub, next_vel_sub = self.network.step_from_history_batched(
                        self._hist_sub_buf,
                        poses_env,
                        vels_env,
                        kinds,
                        sizes,
                        mu,
                        self.mat_env,
                        dt=dt,
                        support=self.support,
                        sdf=self.sdfs[0],
                        multiplicity=self._cached_mult,
                        return_vel=True,
                    )
                wp.launch(
                    advance_stationary_history_local_kernel,
                    num_envs * self.capacity,
                    [self.estimator.captured_q, self.history_wp],
                    device=self.model.device,
                )
                wp.launch(
                    scatter_sub_step_local_kernel,
                    (num_envs, k_act),
                    [
                        wp.from_torch(self._cached_topk, dtype=wp.int64, requires_grad=False),
                        wp.from_torch(next_pos_sub[:, :k_act].contiguous(), requires_grad=False),
                        wp.from_torch(next_vel_sub[:, :k_act].contiguous(), requires_grad=False),
                        wp.from_torch(self._cached_gamma, requires_grad=False),
                        wp.from_torch(self.tray_lower, requires_grad=False),
                        wp.from_torch(self.tray_upper, requires_grad=False),
                        self.capacity,
                        float(dt),
                        self.estimator.captured_q,
                        state_out.particle_q,
                        state_out.particle_qd,
                        self.history_wp,
                    ],
                    device=self.model.device,
                )
                self.estimator.evaluate(state_out, active_idx=self._cached_topk)
            else:
                hist_env = self.history.view(num_envs, self.capacity, 3, 3).clone()
                hist_env[:, :, -1] = pos_in.view(num_envs, self.capacity, 3)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    next_pos_env, next_vel_env = self.network.step_from_history_batched(
                        hist_env,
                        poses_env,
                        vels_env,
                        kinds,
                        sizes,
                        mu,
                        self.mat_env,
                        dt=dt,
                        support=self.support,
                        sdf=self.sdfs[0],
                        return_vel=True,
                    )
                next_pos_env = torch.max(
                    torch.min(next_pos_env, self.tray_upper.unsqueeze(1)),
                    self.tray_lower.unsqueeze(1),
                )
                next_pos_flat = next_pos_env.reshape(-1, 3)
                vel_out.copy_((next_pos_flat - pos_in) / dt)
                pos_out.copy_(next_pos_flat)
                self.history.view(num_envs, self.capacity, 3, 3)[:, :, 0] = hist_env[:, :, 1]
                self.history.view(num_envs, self.capacity, 3, 3)[:, :, 1] = next_pos_env - dt * next_vel_env
                self.history.view(num_envs, self.capacity, 3, 3)[:, :, 2] = next_pos_env
                self.estimator.evaluate(state_out)
        self.selection_phase = (self.selection_phase + 1) % max(1, int(getattr(self.cfg, "selection_interval", 1)))
