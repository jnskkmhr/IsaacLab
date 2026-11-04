# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Native Newton solver interface for learned particle motion; no MPM solver instance."""

import re
from pathlib import Path

import newton
import numpy as np
import torch
import warp as wp
from newton.solvers import SolverBase
from newton.solvers.experimental.coupled import CouplingInterface

from .contact.proxy import ProxyEstimator
from .kernels import (
    advance_history,
    extract_chart_cur_prev,
    extract_chart_history_sub,
    gather_collider_state,
    reset_history,
    rewind_proxy,
    scatter_wrench,
    terrain_chart,
    terrain_chart_heights,
)
from .network.features import features_batched, material_features
from .network.geometry import boundary_boxes
from .network.inference import (
    _compiled_foot_gap_only,
    _compiled_postprocess_sparse_batched,
    _compiled_select_score_batched,
    _compiled_sliding_window_shift_batched,
    collider_state,
    compact_step,
    finish_compact_step,
    multiply,
    prepare_compact_step,
    resolve_active_k,
    rotate,
)
from .network.model import ParticleTransolver


class ParticleNeRDSolver(SolverBase, CouplingInterface):
    """Predict particle positions and expose reconstructed reactions to the proxy coupler.

    Particle storage is world-major with equal capacity per particle-bearing world.
    Rigid-only worlds have no particle storage or neural inference. Recycling keeps
    slot bindings fixed. Neural history uses world coordinates for the current
    patch; terrain charts are temporary model inputs.
    """

    def __init__(self, model, cfg):
        super().__init__(model)
        if not model.device.is_cuda:
            raise ValueError("Particle NeRD currently requires a CUDA device")
        self.cfg = cfg
        path = Path(cfg.checkpoint).expanduser()
        if not cfg.checkpoint or not path.is_file():
            raise FileNotFoundError("Set ParticleNeRDSolverCfg.checkpoint to an existing trained checkpoint")
        for name in (
            "effective_mass_scale",
            "tangent_mass_scale",
            "horizontal_mass_scale",
            "upward_mass_scale",
            "normal_power",
            "gradient_radius",
            "gradient_contact_band",
            "bed_depth",
        ):
            if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(cfg.selection_interval, bool)
            or not isinstance(cfg.selection_interval, int)
            or cfg.selection_interval < 1
        ):
            raise ValueError("selection_interval must be a positive integer")
        if cfg.anchor_particles < 1:
            raise ValueError("anchor_particles must be positive")
        saved = torch.load(path, map_location=str(model.device), weights_only=False)
        config = saved["config"]
        if saved.get("architecture") != "particle_transolver_v1" or config.get("schema") != "g1_particle_nerd_v1":
            raise ValueError("Expected particle_transolver_v1 with g1_particle_nerd_v1 schema")
        if config["model"]["history"] != 3:
            raise ValueError("Particle NeRD supports three-frame checkpoints")
        self.dt = float(config["dt"])
        self.support = float(config["features"]["support"])
        self.network = ParticleTransolver(**config["model"]).to(str(model.device)).eval()
        self.network.load_state_dict(saved["model"])
        if cfg.compile_model:
            self.network = torch.compile(
                self.network, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
            )
        self.worlds = model.particle_world.numpy()
        self.world_count = model.world_count
        counts = np.bincount(self.worlds, minlength=self.world_count)
        self.particle_world_ids = np.flatnonzero(counts)
        if len(counts) != self.world_count or not len(self.particle_world_ids):
            raise ValueError("Particle NeRD requires particles assigned to simulation worlds")
        self.capacity = int(counts[self.particle_world_ids[0]])
        if np.any(counts[self.particle_world_ids] != self.capacity):
            raise ValueError("Particle NeRD requires equal capacities in particle-bearing worlds")
        if not np.array_equal(self.worlds, np.repeat(self.particle_world_ids, self.capacity)):
            raise ValueError("Particle NeRD requires contiguous world-major particle storage")
        self.history_wp = wp.zeros(model.particle_count, dtype=wp.mat33, device=model.device)
        self.history = wp.to_torch(self.history_wp)
        self.neural_state = newton.State()
        self.neural_state.body_q = wp.empty_like(model.body_q)
        self.neural_state.body_qd = wp.empty_like(model.body_qd)
        self._has_neural_state = False
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
                if len(found) != 1:
                    raise ValueError(f"Expected one collider matching {pattern!r} in world {world}; found {found}")
                selected += found
            if len(selected) != 2:
                raise ValueError("This checkpoint schema requires exactly two box collider bodies per world")
            ids += selected
        self.collider_ids = torch.tensor(ids, device=str(model.device), dtype=torch.long).view(-1, 2)
        self.collider_ids_wp = wp.array(ids, dtype=int, device=model.device)
        self.estimator = ProxyEstimator(
            model,
            ids,
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
            contact_solver="rft_body_velocity",
            particles_per_env=self.capacity,
            batch_contacts=cfg.batch_contacts,
        )
        self.geometry = []
        sb = model.shape_body.numpy()
        st = model.shape_type.numpy()
        shapes = [int(np.flatnonzero((sb == body) & (st == int(newton.GeoType.BOX))).item()) for body in ids]

        def tensor(a):
            return torch.as_tensor(a, device=str(model.device), dtype=torch.float32)

        boundary_pose, boundary_size, boundary_mu = boundary_boxes(
            (-1.5, -1.5, -cfg.bed_depth), (1.5, 1.5, 0), str(model.device)
        )
        self.boundary_pose = boundary_pose
        self.depth = cfg.bed_depth
        self.materials = []
        mass = wp.to_torch(model.particle_mass)[:: self.capacity].cpu().numpy()
        radius = wp.to_torch(model.particle_radius)[:: self.capacity].cpu().numpy()
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
        mpm_world_arrays = {
            key: wp.to_torch(getattr(model.mpm, key))[:: self.capacity].cpu().numpy()
            for key in material_keys
            if key != "density"
        }
        shape_scale_np = model.shape_scale.numpy()
        shape_transform_np = model.shape_transform.numpy()
        shape_mu_np = model.shape_material_mu.numpy()
        body_com_t = wp.to_torch(model.body_com)
        kinds_t = torch.ones(7, device=str(model.device), dtype=torch.int32)
        foot_slots_t = torch.tensor([0, 1], device=str(model.device))
        for world in range(len(self.particle_world_ids)):
            shape_ids = shapes[2 * world : 2 * world + 2]
            material = {key: float(arr[world]) for key, arr in mpm_world_arrays.items()}
            particle_mass, particle_radius = float(mass[world]), float(radius[world])
            material["density"] = particle_mass / (8 * particle_radius**3)
            self.materials.append(
                tensor(material_features({"material": material}, particle_mass, particle_radius, self.dt))
            )
            sizes = torch.cat((tensor(shape_scale_np[shape_ids]), boundary_size))
            local = tensor(shape_transform_np[shape_ids])
            self.geometry.append(
                dict(
                    body_ids=self.collider_ids[world],
                    local_pose=local,
                    com=body_com_t,
                    sizes=sizes,
                    kinds=kinds_t,
                    friction=torch.cat((tensor(shape_mu_np[shape_ids]), boundary_mu)),
                    foot_slots=foot_slots_t,
                    floor_slot=2,
                    p_radius=0.02 * float(self.materials[-1][11]),
                    mu_mat=material["friction"],
                    sliding_window=True,
                    canonical_poses=torch.cat((local, boundary_pose)),
                    canonical_sizes=sizes,
                )
            )
        self.fixed_bed = None
        self.terrain = None
        self.capture_revision = 0
        self.slots = []
        self.refresh_particle_slots()

    def bind_terrain(self, terrain, depth):
        """Bind the immutable terrain chart; the solver does not search the scene singleton."""
        self.terrain = terrain
        self.depth = float(depth)
        self.capture_revision += 1

    def bind_fixed_bed(self, cfg, origins=None):
        """Bind fixed-bed coordinate transforms outside the solver configuration."""
        from .fixed_bed import FixedBed

        self.fixed_bed = FixedBed(self.model, cfg, self.geometry)
        if origins is not None:
            self.fixed_bed.bind_origins(origins)
        self.depth = self.fixed_bed.depth
        self._batch_solver_signature = None
        self.capture_revision += 1

    def notify_model_changed(self, flags):
        """Consume virtual proxy masses assigned after the neural solver is constructed."""
        if flags & newton.ModelFlags.BODY_INERTIAL_PROPERTIES:
            self.estimator.refresh_body_inertia()
            # Scalar masses are arguments baked into the captured contact kernels.
            self.capture_revision += 1

    def refresh_particle_slots(self):
        """Bind visible slots outside CUDA capture; recycling preserves these bindings."""
        flags_t = wp.to_torch(self.model.particle_flags)
        visible_2d = ((flags_t & 1) != 0).view(len(self.particle_world_ids), self.capacity)
        device = str(self.model.device)
        if bool(visible_2d.all().item()):
            all_slots = torch.arange(self.model.particle_count, device=device, dtype=torch.long).view(
                len(self.particle_world_ids), self.capacity
            )
            self.slots = [all_slots[w] for w in range(len(self.particle_world_ids))]
        else:
            self.slots = [
                torch.nonzero(visible_2d[w], as_tuple=False).squeeze(1) + w * self.capacity
                for w in range(len(self.particle_world_ids))
            ]
        for geometry in getattr(self, "geometry", []):
            for key in list(geometry):
                if key.startswith("cached_") or key == "bg_idx":
                    del geometry[key]
        self.active_counts = [
            resolve_active_k(self.cfg.active_k, len(slots), self.materials[e]) for e, slots in enumerate(self.slots)
        ]
        self._batch_solver_signature = None
        self._cached_topk = None
        self.capture_revision += 1

    def _prepare_batched_solver(self) -> bool:
        """Prepare pre-allocated multi-world buffers when all sand worlds share uniform active_k."""
        if not self.cfg.batch_inference or self.fixed_bed is not None or not self.slots:
            return False
        slot_lengths = {len(s) for s in self.slots}
        if len(slot_lengths) != 1 or next(iter(slot_lengths)) == 0:
            return False
        if len(set(self.active_counts)) != 1 or self.active_counts[0] is None:
            return False
        num_envs = len(self.slots)
        n_pts = len(self.slots[0])
        k_act = int(self.active_counts[0])
        m_bg = min(int(self.cfg.anchor_particles), n_pts - k_act)
        sig = (num_envs, n_pts, k_act, m_bg)
        if getattr(self, "_batch_solver_signature", None) == sig:
            return True
        device = self.slots[0].device
        self._slots_2d = torch.stack(self.slots, dim=0).contiguous()
        self._slots_flat = self._slots_2d.view(-1)
        self._slots_offset = (torch.arange(num_envs, device=device, dtype=torch.long) * self.capacity).unsqueeze(1)
        self._body_ids_2d = torch.stack([g["body_ids"] for g in self.geometry], dim=0).contiguous()
        self._local_pose_2d = torch.stack([g["local_pose"] for g in self.geometry], dim=0).contiguous()
        self._sizes_3d = torch.stack([g.get("canonical_sizes", g["sizes"]) for g in self.geometry], dim=0).contiguous()
        self._friction_2d = torch.stack([g["friction"] for g in self.geometry], dim=0).contiguous()
        self._materials_2d = torch.stack(self.materials, dim=0).contiguous()
        self._bg_1d = torch.linspace(0, n_pts - 1, m_bg, device=device).long()
        self._bg_2d = self._bg_1d.unsqueeze(0).expand(num_envs, -1)
        self._surf_2d = (self._bg_1d + 17).clamp_max(n_pts - 1).unsqueeze(0).expand(num_envs, -1)
        self._x_cur_buf = torch.empty((num_envs, n_pts, 3), dtype=torch.float32, device=device)
        self._x_prev_buf = torch.empty((num_envs, n_pts, 3), dtype=torch.float32, device=device)
        self._hist_sub_wp = wp.empty(num_envs * (k_act + m_bg), dtype=wp.mat33, device=self.model.device)
        self._hist_sub_buf = wp.to_torch(self._hist_sub_wp).view(num_envs, k_act + m_bg, 3, 3)
        self._cached_topk = None
        self._batch_solver_signature = sig
        return True

    def _step_batched(self, state_in, state_out, collider_input, src, dst, velocity, dt):
        num_envs, n_pts = self._slots_2d.shape
        k_act = int(self.active_counts[0])
        m_bg = int(self._bg_1d.shape[0])
        geom0 = self.geometry[0]
        p_radius = float(geom0["p_radius"])
        mu_mat = float(geom0["mu_mat"])
        floor_slot = int(geom0["floor_slot"])
        foot_sizes = geom0["sizes"][:2]

        body_pose = wp.to_torch(collider_input.body_q)
        body_vel = wp.to_torch(collider_input.body_qd)
        b = body_pose[self._body_ids_2d]
        vel = body_vel[self._body_ids_2d]
        offset = rotate(b[..., 3:], self._local_pose_2d[..., :3])
        position = b[..., :3] + offset
        rotation = multiply(b[..., 3:], self._local_pose_2d[..., 3:])
        com_offset = rotate(b[..., 3:], geom0["com"][self._body_ids_2d])
        linear = vel[..., :3] + torch.linalg.cross(vel[..., 3:], offset - com_offset, dim=-1)
        foot_poses = torch.cat((position, rotation), dim=-1)
        foot_vels = torch.cat((linear, vel[..., 3:]), dim=-1)

        heights_f, normals_f = self._chart(foot_poses[..., :3].reshape(-1, 3))
        heights_f = heights_f.view(num_envs, 2)
        normals_f = normals_f.view(num_envs, 2, 3)
        foot_poses = torch.cat(
            (foot_poses[..., :2], (foot_poses[..., 2] - heights_f).unsqueeze(-1), foot_poses[..., 3:]), dim=-1
        )
        z_up = torch.zeros_like(normals_f)
        z_up[..., 2] = 1.0
        quat = torch.cat((torch.linalg.cross(normals_f, z_up, dim=-1), 1.0 + normals_f[..., 2:3]), dim=-1)
        quat = torch.nn.functional.normalize(quat, dim=-1)
        foot_poses = torch.cat((foot_poses[..., :3], multiply(quat, foot_poses[..., 3:])), dim=-1)
        gradient = -normals_f[..., :2] / normals_f[..., 2:3].clamp_min(1e-5)
        vz_adj = foot_vels[..., 2] - (gradient * foot_vels[..., :2]).sum(dim=-1)
        foot_vels = torch.cat((foot_vels[..., :2], vz_adj.unsqueeze(-1), foot_vels[..., 3:]), dim=-1)
        poses = torch.cat((foot_poses, self.boundary_pose.unsqueeze(0).expand(num_envs, -1, -1)), dim=1)
        velocities = torch.cat(
            (foot_vels, torch.zeros((num_envs, 5, 6), dtype=foot_vels.dtype, device=foot_vels.device)), dim=1
        )

        has_terrain = self.terrain is not None
        mesh_id = self.terrain.mesh.id if has_terrain else wp.uint64(0)
        ray_z = float(self.terrain.height_query_start_z) if has_terrain else 0.0
        ray_len = float(self.terrain.height_query_max_distance) if has_terrain else 0.0
        reuse = self.selection_phase != 0 and self._cached_topk is not None
        if not reuse:
            wp.launch(
                extract_chart_cur_prev,
                num_envs * n_pts,
                [
                    wp.from_torch(self._slots_flat, dtype=wp.int64),
                    self.history_wp,
                    state_in.particle_q,
                    has_terrain,
                    mesh_id,
                    ray_z,
                    ray_len,
                    wp.from_torch(self._x_cur_buf.view(-1, 3), dtype=wp.vec3),
                    wp.from_torch(self._x_prev_buf.view(-1, 3), dtype=wp.vec3),
                ],
                device=self.model.device,
            )
            z_bg = torch.gather(self._x_cur_buf[..., 2], 1, self._surf_2d)
            z_surf = torch.quantile(z_bg, 0.98, dim=-1, keepdim=True) + 1.5 * p_radius
            eval_score, gap_min, xi = _compiled_select_score_batched(
                self._x_cur_buf,
                self._x_prev_buf,
                foot_poses,
                foot_vels,
                foot_sizes,
                z_surf,
                p_radius,
                mu_mat,
                float(dt),
            )
            top_vals, topk = torch.topk(eval_score, k=k_act, dim=-1, largest=False, sorted=False)
            thresh = top_vals.amax(dim=-1, keepdim=True)
            bg_scores = torch.gather(eval_score, 1, self._bg_2d)
            bg_valid = (bg_scores > thresh).float()
            bg_valid = torch.where(bg_valid.sum(dim=-1, keepdim=True) < 1.0, torch.ones_like(bg_valid), bg_valid)
            bg_mult = bg_valid * (float(n_pts - k_act) / bg_valid.sum(dim=-1, keepdim=True).clamp_min(1.0))
            idx = torch.cat((topk, self._bg_2d), dim=1)
            mult = torch.cat((torch.ones((num_envs, k_act), dtype=torch.float32, device=src.device), bg_mult), dim=1)
            tau = ((thresh - top_vals) / 0.05).clamp(0.0, 1.0)
            gap_k = torch.gather(gap_min, 1, topk)
            min_gap = gap_k.amin(dim=-1, keepdim=True)
            foot_engage = ((0.14 - min_gap) / 0.04).clamp(0.0, 1.0)
            xi_k = torch.gather(xi, 1, topk)
            gamma = tau * tau * (3.0 - 2.0 * tau) * torch.maximum(foot_engage, xi_k)
            if self._cached_topk is None:
                self._cached_topk = topk.clone()
                self._cached_idx = idx.clone()
                self._cached_mult = mult.clone()
                self._cached_gamma = gamma.clone()
                self._cached_z_surf = z_surf.clone()
                self._cached_foot_engage = foot_engage.clone()
                for w, geom in enumerate(self.geometry):
                    geom["cached_topk"] = self._cached_topk[w]
            else:
                self._cached_topk.copy_(topk)
                self._cached_idx.copy_(idx)
                self._cached_mult.copy_(mult)
                self._cached_gamma.copy_(gamma)
                self._cached_z_surf.copy_(z_surf)
                self._cached_foot_engage.copy_(foot_engage)
        else:
            topk = self._cached_topk
            idx = self._cached_idx
            mult = self._cached_mult
            gamma = self._cached_gamma
            z_surf = self._cached_z_surf
            foot_engage = self._cached_foot_engage

        sub_slots = torch.gather(self._slots_2d, 1, idx).contiguous()
        wp.launch(
            extract_chart_history_sub,
            num_envs * (k_act + m_bg),
            [
                wp.from_torch(sub_slots.view(-1), dtype=wp.int64),
                self.history_wp,
                state_in.particle_q,
                has_terrain,
                mesh_id,
                ray_z,
                ray_len,
                self._hist_sub_wp,
            ],
            device=self.model.device,
        )
        hist_sub = self._hist_sub_buf
        if reuse:
            gap_k = _compiled_foot_gap_only(hist_sub[:, :k_act, -1], foot_poses, foot_sizes, p_radius)

        hist_sub_feat, poses_feat = _compiled_sliding_window_shift_batched(
            hist_sub, poses, geom0["canonical_poses"], k_act
        )
        nodes_sub, pairs_sub, inf_sub = features_batched(
            hist_sub_feat,
            poses_feat,
            velocities,
            geom0["kinds"],
            self._sizes_3d,
            self._friction_2d,
            self._materials_2d,
            dt=dt,
            support=self.support,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.model.device.is_cuda):
            if num_envs == 1:
                disp_sub = self.network(nodes_sub[0], pairs_sub[0], inf_sub[0], mult[0]).unsqueeze(0)
            else:
                disp_sub = self.network(nodes_sub, pairs_sub, inf_sub, mult)

        dx_sub = _compiled_postprocess_sparse_batched(
            hist_sub,
            disp_sub,
            gamma,
            gap_k,
            z_surf,
            foot_engage,
            poses[0],
            geom0["sizes"],
            floor_slot,
            k_act,
            p_radius,
            float(dt),
        )
        predicted_sub = hist_sub[:, :, -1] + dx_sub
        if has_terrain:
            pts_sub = predicted_sub.reshape(-1, 3).contiguous()
            heights_sub = torch.empty(pts_sub.shape[0], dtype=pts_sub.dtype, device=pts_sub.device)
            wp.launch(
                terrain_chart_heights,
                pts_sub.shape[0],
                [
                    wp.from_torch(pts_sub, dtype=wp.vec3),
                    mesh_id,
                    ray_z,
                    ray_len,
                    wp.from_torch(heights_sub),
                ],
                device=self.model.device,
            )
            predicted_sub[..., 2] += heights_sub.view(num_envs, k_act + m_bg)

        src_sub = src[sub_slots]
        dynamic_sub = wp.to_torch(self.model.particle_mass)[sub_slots] > 0
        predicted_sub = torch.where(dynamic_sub.unsqueeze(-1), predicted_sub, src_sub)
        vel_sub = (predicted_sub - src_sub) / dt
        bg_slots_flat = sub_slots[:, k_act:].reshape(-1)
        act_slots_flat = sub_slots[:, :k_act].reshape(-1)
        dst[bg_slots_flat] = predicted_sub[:, k_act:].reshape(-1, 3)
        dst[act_slots_flat] = predicted_sub[:, :k_act].reshape(-1, 3)
        velocity[bg_slots_flat] = vel_sub[:, k_act:].reshape(-1, 3)
        velocity[act_slots_flat] = vel_sub[:, :k_act].reshape(-1, 3)
        wp.launch(
            advance_history,
            num_envs * n_pts,
            [
                wp.from_torch(self._slots_flat, dtype=wp.int64),
                state_in.particle_q,
                state_out.particle_q,
                self.history_wp,
            ],
            device=self.model.device,
        )
        active_ids = sub_slots[:, :k_act] - self._slots_offset
        self.estimator.evaluate(state_out, active_idx=active_ids)

    def _chart(self, points):
        if self.terrain is None:
            return torch.zeros_like(points[:, 2]), torch.tensor([0.0, 0.0, 1.0], device=points.device).expand_as(points)
        points = points.contiguous()
        heights = torch.empty_like(points[:, 2])
        normals = torch.empty_like(points)
        wp.launch(
            terrain_chart,
            len(points),
            [
                wp.from_torch(points, dtype=wp.vec3),
                self.terrain.mesh.id,
                self.terrain.height_query_start_z,
                self.terrain.height_query_max_distance,
                wp.from_torch(heights),
                wp.from_torch(normals, dtype=wp.vec3),
            ],
            device=self.model.device,
        )
        return heights, normals

    def capture_collider_state(self, state, body_local_to_global):
        """Snapshot network inputs before rigid integration changes proxy velocities.

        Contact reconstruction separately reads the rewound virtual proxy state.
        The checkpoint's collider features use the beginning-of-step rigid state,
        as in the G1 example adapter.
        """
        wp.launch(
            gather_collider_state,
            self.model.body_count,
            [body_local_to_global, state.body_q, state.body_qd, self.neural_state.body_q, self.neural_state.body_qd],
            device=self.model.device,
        )
        self._has_neural_state = True

    @torch.inference_mode()
    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        self.validate_observables(observables, contacts)
        if not np.isclose(dt, self.dt):
            raise ValueError(f"Checkpoint timestep is {self.dt}s, but coupler requested {dt}s")
        stream = wp.stream_to_torch(wp.get_stream(self.model.device))
        with torch.cuda.stream(stream):
            self.estimator.capture(state_in, dt)
            collider_input = self.neural_state if self._has_neural_state else state_in
            src, dst = wp.to_torch(state_in.particle_q), wp.to_torch(state_out.particle_q)
            velocity = wp.to_torch(state_out.particle_qd)
            if src.data_ptr() != dst.data_ptr():
                dst.copy_(src)
            velocity.zero_()
            if self._prepare_batched_solver():
                self._step_batched(state_in, state_out, collider_input, src, dst, velocity, dt)
                self.selection_phase = (self.selection_phase + 1) % self.cfg.selection_interval
                return
            active_ids = []
            prepared_steps = []
            world_histories = []
            displacements = {}
            for world, slots in enumerate(self.slots):
                if len(slots) == 0:
                    raise ValueError("Each particle world must contain visible particles before stepping")
                geom = self.geometry[world]
                hist_world = self.history[slots].clone()
                # Incoming state is authoritative after resets and boundary pinning.
                hist_world[:, -1] = src[slots]
                if self.fixed_bed is not None:
                    hist, poses, velocities = self.fixed_bed.inputs(hist_world, collider_input, world, geom)
                else:
                    hist = hist_world.clone()
                    heights, _ = self._chart(hist.reshape(-1, 3))
                    hist[:, :, 2] -= heights.view(-1, 3)
                    poses, velocities = collider_state(
                        wp.to_torch(collider_input.body_q), wp.to_torch(collider_input.body_qd), geom
                    )
                    heights, normals = self._chart(poses[:, :3])
                    poses[:, 2] -= heights
                    # Local normal-to-Z rotation matches the slope transform used by the adapter.
                    z = torch.zeros_like(normals)
                    z[:, 2] = 1
                    quat = torch.cat((torch.linalg.cross(normals, z), 1 + normals[:, 2:3]), dim=-1)
                    quat = torch.nn.functional.normalize(quat, dim=-1)
                    poses[:, 3:] = multiply(quat, poses[:, 3:])
                    gradient = -normals[:, :2] / normals[:, 2:3].clamp_min(1e-5)
                    velocities[:, 2] -= (gradient * velocities[:, :2]).sum(-1)
                    poses = torch.cat((poses, self.boundary_pose))
                    velocities = torch.cat((velocities, torch.zeros((5, 6), device=velocities.device)))
                k = self.active_counts[world]
                world_histories.append((hist_world, hist))
                if self.cfg.batch_inference:
                    prepared_steps.append(
                        prepare_compact_step(
                            hist,
                            poses,
                            velocities,
                            geom,
                            self.materials[world],
                            dt,
                            support=self.support,
                            active_k=k,
                            anchor_m=self.cfg.anchor_particles,
                            reuse_selection=self.selection_phase != 0 and "cached_topk" in geom,
                        )
                    )
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.model.device.is_cuda):
                        displacement = compact_step(
                            self.network,
                            hist,
                            poses,
                            velocities,
                            geom,
                            self.materials[world],
                            dt,
                            support=self.support,
                            active_k=k,
                            anchor_m=self.cfg.anchor_particles,
                            reuse_selection=self.selection_phase != 0 and "cached_topk" in geom,
                        )
                    displacements[world] = displacement

            if self.cfg.batch_inference:
                groups = {}
                for world, prepared in enumerate(prepared_steps):
                    signature = tuple(value.shape for value in prepared.inputs)
                    groups.setdefault(signature, []).append(world)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.model.device.is_cuda):
                    for worlds in groups.values():
                        if len(worlds) == 1:
                            world = worlds[0]
                            prepared = prepared_steps[world]
                            displacements[world] = finish_compact_step(prepared, self.network(*prepared.inputs))
                        else:
                            inputs = tuple(
                                torch.stack(values)
                                for values in zip(
                                    *(prepared_steps[world].inputs for world in worlds),
                                    strict=True,
                                )
                            )
                            outputs = self.network(*inputs)
                            for batch, world in enumerate(worlds):
                                displacements[world] = finish_compact_step(prepared_steps[world], outputs[batch])

            for world, slots in enumerate(self.slots):
                hist_world, hist = world_histories[world]
                geom, k = self.geometry[world], self.active_counts[world]
                displacement = displacements[world]
                predicted = hist[:, -1] + displacement
                if self.fixed_bed is not None:
                    predicted = self.fixed_bed.to_world(predicted, collider_input, world, dt)
                else:
                    heights, _ = self._chart(predicted)
                    predicted[:, 2] += heights
                dynamic = wp.to_torch(self.model.particle_mass)[slots] > 0
                predicted = torch.where(dynamic[:, None], predicted, src[slots])
                velocity[slots] = (predicted - src[slots]) / dt
                dst[slots] = predicted
                self.history[slots] = torch.cat((hist_world[:, 1:], predicted[:, None]), dim=1)
                selected = geom["cached_topk"] if k is not None else torch.arange(len(slots), device=slots.device)
                active_ids.append(slots[selected] - world * self.capacity)
            self.estimator.evaluate(state_out, active_idx=active_ids)
        self.selection_phase = (self.selection_phase + 1) % self.cfg.selection_interval

    def coupling_rewind_proxy_body(self, mapping, state, forces, body_gravity_acceleration, dt):
        wp.launch(
            rewind_proxy,
            len(mapping),
            [mapping, forces, state.body_q, self.model.body_inv_mass, self.model.body_inv_inertia, state.body_qd, dt],
            device=self.model.device,
        )

    def coupling_harvest_proxy_wrenches(self, mapping, out_body_f, **kwargs):
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
