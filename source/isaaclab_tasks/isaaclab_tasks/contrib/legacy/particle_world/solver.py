# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Newton solver interface for WorldParticle Lagrangian particle dynamics."""

import re
from pathlib import Path

import newton
import numpy as np
import torch
import warp as wp
from newton.solvers import SolverBase
from newton.solvers.experimental.coupled import CouplingInterface

from ..particle_nerd.kernels import (
    gather_collider_state,
    reset_history,
    rewind_proxy,
    scatter_wrench,
)
from ..particle_nerd.network.features import material_features
from ..particle_nerd.network.geometry import boundary_boxes
from .contact import WorldParticleProxyEstimator
from .network.model import WorldParticleNetwork


class ParticleWorldSolver(SolverBase, CouplingInterface):
    """Step Lagrangian particles with WorldParticleNetwork and reconstruct proxy contact reactions."""

    def __init__(self, model, cfg):
        super().__init__(model)
        if not model.device.is_cuda:
            raise ValueError("ParticleWorldSolver requires a CUDA device")
        self.cfg = cfg
        path = Path(cfg.checkpoint).expanduser()
        if not cfg.checkpoint or not path.is_file():
            raise FileNotFoundError("Set ParticleWorldSolverCfg.checkpoint to an existing trained checkpoint")
        saved = torch.load(path, map_location=str(model.device), weights_only=False)
        config = saved["config"]
        arch = saved.get("architecture") or config.get("architecture")
        if arch != "world_particle_v1":
            raise ValueError(f"Expected world_particle_v1 checkpoint, got {arch!r}")
        self.dt = float(config["dt"])
        self.support = float(config.get("features", {}).get("support", 0.07))
        model_cfg = dict(config["model"])
        for k in ("far_support_scale", "far_gap_weight", "far_influence_weight", "use_actual_velocity"):
            if getattr(cfg, k, None) is not None and k not in model_cfg:
                model_cfg[k] = getattr(cfg, k)
        self.network = WorldParticleNetwork(**model_cfg).to(str(model.device)).eval()
        self.network.load_state_dict(saved["model"])

        self.worlds = model.particle_world.numpy()
        self.world_count = model.world_count
        counts = np.bincount(self.worlds, minlength=self.world_count)
        self.particle_world_ids = np.flatnonzero(counts)
        if len(counts) != self.world_count or not len(self.particle_world_ids):
            raise ValueError("ParticleWorldSolver requires particles assigned to simulation worlds")
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
        all_shape_tf = wp.to_torch(model.shape_transform)[shapes].view(
            len(self.particle_world_ids), self.num_colliders_per_world, 7
        )
        all_body_com = wp.to_torch(model.body_com)[ids].view(
            len(self.particle_world_ids), self.num_colliders_per_world, 3
        )
        self.shape_local_tf = all_shape_tf
        self.body_com_local = all_body_com
        self.geometry = []
        all_scales = wp.to_torch(model.shape_scale)
        all_mu = wp.to_torch(model.shape_material_mu)
        for env_idx in range(len(self.particle_world_ids)):
            env_shapes = shapes[env_idx * self.num_colliders_per_world : (env_idx + 1) * self.num_colliders_per_world]
            env_kinds = [geo_map.get(int(st[s]), 1) for s in env_shapes]
            kinds = torch.cat(
                (
                    torch.tensor(env_kinds, device=str(model.device), dtype=torch.int32),
                    torch.ones(5, device=str(model.device), dtype=torch.int32),
                )
            )
            sizes = torch.cat((all_scales[env_shapes], boundary_size))
            mu = torch.cat((all_mu[env_shapes], boundary_mu))
            self.geometry.append((kinds, sizes, mu))

        self.materials = []
        mass = wp.to_torch(model.particle_mass)[:: self.capacity].cpu().numpy()
        radius = wp.to_torch(model.particle_radius)[:: self.capacity].cpu().numpy()
        # Scale ContinuousConv neighborhood radius with the actual scene particle radius
        # (matching upstream WorldParticle `apply_radius_override`: 3 * radius_scale * r_p = 4.5 * r_p).
        scene_radius = float(radius[0])
        if scene_radius > 0.0:
            self.network.tokenizer.cconv.radius = 4.5 * scene_radius
            self.support = max(0.035, 4.375 * scene_radius)
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

    def notify_model_changed(self, flags):
        if flags & newton.ModelFlags.BODY_INERTIAL_PROPERTIES:
            self.estimator.refresh_body_inertia()
            self.capture_revision += 1

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
        self.selection_phase = 0
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

    @torch.inference_mode()
    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        del control
        self.validate_observables(observables, contacts)
        if self.model.particle_count == 0:
            return
        stream = wp.stream_to_torch(wp.get_stream(self.model.device))
        with torch.cuda.stream(stream):
            self.estimator.capture(state_in, dt)
            collider_source = self.neural_state if self._has_neural_state else state_in
            self._has_neural_state = False
            body_q = wp.to_torch(collider_source.body_q)
            body_qd = wp.to_torch(collider_source.body_qd)
            pos_in = wp.to_torch(state_in.particle_q)
            pos_out = wp.to_torch(state_out.particle_q)
            vel_out = wp.to_torch(state_out.particle_qd)

            num_envs = len(self.particle_world_ids)
            use_vel_state = bool(self.network.config.get("translation_invariant", False))
            if self.cfg.batch_inference and num_envs > 1:
                poses_list = []
                vels_list = []
                zero_wall_vel = torch.zeros((5, 6), device=body_qd.device, dtype=body_qd.dtype)
                for env_idx in range(num_envs):
                    cid = self.collider_ids[env_idx]
                    bp = body_q[cid]
                    bv = body_qd[cid]
                    ltf = self.shape_local_tf[env_idx]
                    bcom = self.body_com_local[env_idx]
                    q_xyz, q_w = bp[:, 3:6], bp[:, 6:7]
                    t_rot = 2.0 * torch.linalg.cross(q_xyz, ltf[:, :3], dim=-1)
                    w_pos = bp[:, :3] + ltf[:, :3] + q_w * t_rot + torch.linalg.cross(q_xyz, t_rot, dim=-1)
                    x1, y1, z1, w1 = bp[:, 3], bp[:, 4], bp[:, 5], bp[:, 6]
                    x2, y2, z2, w2 = ltf[:, 3], ltf[:, 4], ltf[:, 5], ltf[:, 6]
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
                    p_com = bp[:, :3] + bcom + q_w * t_com + torch.linalg.cross(q_xyz, t_com, dim=-1)
                    v_lin = bv[:, :3] + torch.linalg.cross(bv[:, 3:6], w_pos - p_com, dim=-1)
                    dyn_pose = torch.cat((w_pos, w_quat), dim=-1)
                    dyn_vel = torch.cat((v_lin, bv[:, 3:6]), dim=-1)
                    poses_list.append(torch.cat((dyn_pose, self.boundary_poses[env_idx]), dim=0))
                    vels_list.append(torch.cat((dyn_vel, zero_wall_vel), dim=0))
                poses_env = torch.stack(poses_list, dim=0).contiguous()
                vels_env = torch.stack(vels_list, dim=0).contiguous()
                kinds, sizes, mu = self.geometry[0]
                mat_env = torch.stack(self.materials, dim=0).contiguous()
                hist_env = self.history.view(num_envs, self.capacity, 3, 3).clone()
                hist_env[:, :, -1] = pos_in.view(num_envs, self.capacity, 3)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if use_vel_state:
                        next_pos_env, next_vel_env = self.network.step_from_history_batched(
                            hist_env,
                            poses_env,
                            vels_env,
                            kinds,
                            sizes,
                            mu,
                            mat_env,
                            dt=dt,
                            support=self.support,
                            return_vel=True,
                        )
                    else:
                        next_pos_env = self.network.step_from_history_batched(
                            hist_env, poses_env, vels_env, kinds, sizes, mu, mat_env, dt=dt, support=self.support
                        )
                next_pos_flat = next_pos_env.reshape(-1, 3)
                vel_out.copy_((next_pos_flat - pos_in) / dt)
                pos_out.copy_(next_pos_flat)
                self.history.view(num_envs, self.capacity, 3, 3)[:, :, 0] = hist_env[:, :, 1]
                if use_vel_state:
                    self.history.view(num_envs, self.capacity, 3, 3)[:, :, 1] = next_pos_env - dt * next_vel_env
                else:
                    self.history.view(num_envs, self.capacity, 3, 3)[:, :, 1] = hist_env[:, :, 2]
                self.history.view(num_envs, self.capacity, 3, 3)[:, :, 2] = next_pos_env
            else:
                for env_idx in range(num_envs):
                    start = env_idx * self.capacity
                    end = start + self.capacity
                    cid = self.collider_ids[env_idx]
                    bp = body_q[cid]
                    bv = body_qd[cid]
                    ltf = self.shape_local_tf[env_idx]
                    bcom = self.body_com_local[env_idx]
                    q_xyz, q_w = bp[:, 3:6], bp[:, 6:7]
                    t_rot = 2.0 * torch.linalg.cross(q_xyz, ltf[:, :3], dim=-1)
                    w_pos = bp[:, :3] + ltf[:, :3] + q_w * t_rot + torch.linalg.cross(q_xyz, t_rot, dim=-1)
                    x1, y1, z1, w1 = bp[:, 3], bp[:, 4], bp[:, 5], bp[:, 6]
                    x2, y2, z2, w2 = ltf[:, 3], ltf[:, 4], ltf[:, 5], ltf[:, 6]
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
                    p_com = bp[:, :3] + bcom + q_w * t_com + torch.linalg.cross(q_xyz, t_com, dim=-1)
                    v_lin = bv[:, :3] + torch.linalg.cross(bv[:, 3:6], w_pos - p_com, dim=-1)
                    dyn_pose = torch.cat((w_pos, w_quat), dim=-1)
                    dyn_vel = torch.cat((v_lin, bv[:, 3:6]), dim=-1)
                    poses = torch.cat((dyn_pose, self.boundary_poses[env_idx]), dim=0).contiguous()
                    vels = torch.cat(
                        (dyn_vel, torch.zeros((5, 6), device=body_qd.device, dtype=body_qd.dtype)), dim=0
                    ).contiguous()
                    kinds, sizes, mu = self.geometry[env_idx]
                    hist = self.history[start:end].clone()
                    hist[:, -1] = pos_in[start:end]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if use_vel_state:
                            next_pos, next_vel = self.network.step_from_history(
                                hist,
                                poses,
                                vels,
                                kinds,
                                sizes,
                                mu,
                                self.materials[env_idx],
                                dt=dt,
                                support=self.support,
                                return_vel=True,
                            )
                        else:
                            next_pos = self.network.step_from_history(
                                hist,
                                poses,
                                vels,
                                kinds,
                                sizes,
                                mu,
                                self.materials[env_idx],
                                dt=dt,
                                support=self.support,
                            )
                    vel_out[start:end] = (next_pos - pos_in[start:end]) / dt
                    pos_out[start:end] = next_pos
                    self.history[start:end, 0] = hist[:, 1]
                    self.history[start:end, 1] = (next_pos - dt * next_vel) if use_vel_state else hist[:, 2]
                    self.history[start:end, 2] = next_pos

            active_ids = [torch.arange(self.capacity, device=pos_in.device, dtype=torch.long) for _ in range(num_envs)]
            self.estimator.evaluate(state_out, active_idx=active_ids)
        self.selection_phase = (self.selection_phase + 1) % self.cfg.selection_interval
