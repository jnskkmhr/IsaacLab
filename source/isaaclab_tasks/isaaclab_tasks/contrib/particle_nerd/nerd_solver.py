# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton solver that steps particles with :class:`~.network.nerd_network.ParticleNeRDNetwork`.

One step:
    1. Read the collider poses and twists from the rigid-body state.
    2. Step the particles of all environments with the network, either all of them or an active subset.
    3. Clamp the result to the tray, put slow particles to sleep, and write position, velocity and history.
    4. Solve for the wrench of the particles on every collider
       (:class:`~.nerd_wrench_solver.ContactWrenchSolver`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import newton
import numpy as np
import torch
import warp as wp
from newton.solvers import SolverBase
from newton.solvers.experimental.coupled import CouplingInterface

from .nerd_kernels import (
    active_particle_score,
    advance_stationary_history,
    extract_active_history,
    gather_collider_state,
    reset_history,
    rewind_proxy,
    scatter_active_step,
    scatter_wrench,
)
from .nerd_wrench_solver import ContactWrenchSolver
from .network.feature_builder import material_features
from .network.nerd_network import ParticleNeRDNetwork

if TYPE_CHECKING:
    from .nerd_solver_cfg import ParticleNeRDSolverCfg

_NETWORK_KINDS = {
    int(newton.GeoType.BOX): 1,
    int(newton.GeoType.SPHERE): 2,
    int(newton.GeoType.CYLINDER): 4,
    int(newton.GeoType.CAPSULE): 5,
}
"""Collider kinds of the network inputs (see :mod:`~.nerd_kernels`)."""

_MATERIAL_DEFAULTS = {
    "density": 1600.0,
    "young_modulus": 1.0e15,
    "poisson_ratio": 0.3,
    "friction": 0.48,
    "yield_pressure": 1.0e15,
    "tensile_yield_ratio": 0.0,
    "yield_stress": 0.0,
    "hardening": 0.0,
    "dilatancy": 0.0,
    "viscosity": 0.0,
}
"""Material parameters of the network input, used for those the model does not define."""

_NUM_BACKGROUND = 256
"""Maximum number of background particles added to the active set as network input."""


def tray_boxes(
    lower: tuple[float, float, float], upper: tuple[float, float, float], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Poses [5, 7] and half extents [5, 3] of the floor and the four walls around the bed ``[lower, upper]``."""
    x_lo, y_lo, z_lo = float(lower[0]), float(lower[1]), float(lower[2])
    x_hi, y_hi = float(upper[0]), float(upper[1])
    lx, ly = x_hi - x_lo, y_hi - y_lo
    cx, cy = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    wall, wall_h, floor_h = 0.1, 0.4, 0.1
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
    return poses, sizes


def resolve_active_k(active_k: int | str | None, num_particles: int, material: torch.Tensor) -> int | None:
    """Number of particles the network steps per environment, or None to step all of them.

    Args:
        active_k: A number, ``"auto"``, or None / ``"none"`` for all particles.
        num_particles: Number of particles per environment.
        material: Material features [13]; ``"auto"`` grows with friction and stiffness.
    """
    if active_k is None:
        return None
    if isinstance(active_k, str):
        value = active_k.strip().lower()
        if value in ("none", "off", "all", "full", "0"):
            return None
        if value == "auto":
            friction, stiffness = float(material[3].item()), float(material[1].item())
            fraction = min(0.15, 0.032 * (1.0 + 1.5 * friction + 0.15 * stiffness))
            k = max(1536, min(7104, num_particles, int(round(num_particles * fraction / 64.0)) * 64))
            return None if k >= num_particles else k
        active_k = int(value)
    k = int(active_k)
    return None if k <= 0 or k >= num_particles else k


class ParticleNeRDSolver(SolverBase, CouplingInterface):
    """Step the particles with a trained network and return the contact wrenches on the colliders."""

    def __init__(self, model: newton.Model, cfg: ParticleNeRDSolverCfg):
        """Initialize the solver.

        Args:
            model: Newton model. Every environment must hold the same number of particles.
            cfg: Solver configuration (:class:`~.nerd_solver_cfg.ParticleNeRDSolverCfg`).
        """
        super().__init__(model)
        if not model.device.is_cuda:
            raise ValueError("ParticleNeRDSolver requires a CUDA device")
        self.cfg = cfg
        device = str(model.device)
        if not cfg.checkpoint or not Path(cfg.checkpoint).expanduser().is_file():
            raise FileNotFoundError("Set ParticleNeRDSolverCfg.checkpoint to an existing trained checkpoint")
        self.network, saved = ParticleNeRDNetwork.load_checkpoint(cfg.checkpoint, device)
        self.dt = float(saved["config"]["dt"])
        self.support = float(saved["config"].get("features", {}).get("support", 0.07))

        # Particles: the same count in every world that has particles
        counts = np.bincount(model.particle_world.numpy(), minlength=model.world_count)
        self.world_count = model.world_count
        self.particle_world_ids = np.flatnonzero(counts)
        if len(counts) != self.world_count or not len(self.particle_world_ids):
            raise ValueError("ParticleNeRDSolver requires particles assigned to simulation worlds")
        self.capacity = int(counts[self.particle_world_ids[0]])
        num_envs = len(self.particle_world_ids)
        self.history_wp = wp.zeros(model.particle_count, dtype=wp.mat33, device=model.device)
        self.history = wp.to_torch(self.history_wp)
        """Last three positions of every particle [N, 3, 3] [m], oldest first."""
        self.neural_state = newton.State()
        self.neural_state.body_q = wp.empty_like(model.body_q)
        self.neural_state.body_qd = wp.empty_like(model.body_qd)
        self._has_neural_state = False
        self.selection_phase = 0
        self.capture_revision = 0
        self.reset(model.state())

        # Colliders: the bodies of every world that match the configured patterns
        body_world = model.body_world.numpy()
        patterns = [re.compile(pattern) for pattern in cfg.collider_body_patterns]
        particle_worlds = {int(w) for w in self.particle_world_ids}
        matches: dict[tuple[int, int], list[int]] = {}
        for body, label in enumerate(model.body_label):
            world = int(body_world[body])
            if world not in particle_worlds:
                continue
            for index, pattern in enumerate(patterns):
                if pattern.fullmatch(label):
                    matches.setdefault((world, index), []).append(body)
        ids = []
        for world in self.particle_world_ids:
            for index, pattern in enumerate(cfg.collider_body_patterns):
                found = matches.get((int(world), index), [])
                if not found:
                    raise ValueError(f"Expected at least one collider matching {pattern!r} in world {world}")
                ids += found
        self.num_colliders_per_world = len(ids) // num_envs
        self.collider_ids = torch.tensor(ids, device=device, dtype=torch.long).view(-1, self.num_colliders_per_world)
        self.collider_ids_wp = wp.array(ids, dtype=int, device=model.device)

        # One primitive shape per collider: the largest one, preferring shapes that collide with particles
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()
        shape_scale = model.shape_scale.numpy()
        collides = (model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0
        shapes = []
        for body in ids:
            found = [s for s in np.flatnonzero((shape_body == body) & collides) if int(shape_type[s]) in _NETWORK_KINDS]
            if not found:
                found = [s for s in np.flatnonzero(shape_body == body) if int(shape_type[s]) in _NETWORK_KINDS]
            if not found:
                found = [int(np.flatnonzero(shape_body == body)[0])]
            shapes.append(int(max(found, key=lambda s: float(np.max(shape_scale[s])))))
        self.shape_local_tf = wp.to_torch(model.shape_transform)[shapes].view(num_envs, -1, 7)
        self.body_com_local = wp.to_torch(model.body_com)[ids].view(num_envs, -1, 3)

        # Tray: the static boxes of the scene if every world has five of them, else boxes around the bed
        shape_world = model.shape_world.numpy()
        static_boxes = [
            [
                s
                for s in np.flatnonzero((shape_body < 0) & (shape_world == int(w)) & collides)
                if int(shape_type[s]) == int(newton.GeoType.BOX)
            ][:5]
            for w in self.particle_world_ids
        ]
        q_init = wp.to_torch(model.particle_q).view(num_envs, self.capacity, 3)
        if all(len(boxes) == 5 for boxes in static_boxes):
            flat = [s for boxes in static_boxes for s in boxes]
            self.boundary_poses = wp.to_torch(model.shape_transform)[flat].view(num_envs, 5, 7).clone()
            boundary_size = wp.to_torch(model.shape_scale)[static_boxes[0]].clone()
        else:
            pose, boundary_size = tray_boxes((-1.0, -0.6, -cfg.bed_depth), (1.0, 0.6, 0.0), device)
            bed_xy = 0.5 * (q_init[:, :, :2].amin(dim=1) + q_init[:, :, :2].amax(dim=1))
            self.boundary_poses = pose.unsqueeze(0).repeat(num_envs, 1, 1)
            self.boundary_poses[:, :, :2] += bed_xy.unsqueeze(1)
        # Particles are kept inside the tray, a little away from the floor and the walls
        bp, bs = self.boundary_poses, boundary_size
        z_floor = bp[:, 0, 2] + bs[0, 2]
        x_lo = torch.minimum(bp[:, 1, 0], bp[:, 2, 0]) + bs[1, 0]
        x_hi = torch.maximum(bp[:, 1, 0], bp[:, 2, 0]) - bs[2, 0]
        y_lo = torch.minimum(bp[:, 3, 1], bp[:, 4, 1]) + bs[3, 1]
        y_hi = torch.maximum(bp[:, 3, 1], bp[:, 4, 1]) - bs[4, 1]
        z_top = torch.maximum(q_init.amax(dim=1)[:, 2], z_floor) + 0.60
        self.tray_lower = torch.stack((x_lo + 0.008, y_lo + 0.008, z_floor + 0.004), dim=-1).contiguous()
        self.tray_upper = torch.stack((x_hi - 0.008, y_hi - 0.008, z_top), dim=-1).contiguous()

        # Collider geometry of the network inputs (kinds and sizes as in nerd_kernels)
        scales = wp.to_torch(model.shape_scale)
        margins = wp.to_torch(model.shape_margin)
        sizes_env = []
        for env in range(num_envs):
            sizes = []
            for shape in shapes[env * self.num_colliders_per_world : (env + 1) * self.num_colliders_per_world]:
                kind = _NETWORK_KINDS.get(int(shape_type[shape]), 1)
                scale = scales[shape].cpu().tolist()
                margin = float(margins[shape].item())
                radius = float(scale[0]) + margin
                half_height = (float(scale[1]) if float(scale[1]) > 0.0 else float(scale[2])) + margin
                if kind == 2:
                    sizes.append([radius, radius, radius])
                elif kind == 4:
                    sizes.append([radius, radius, half_height])
                elif kind == 5:
                    sizes.append([radius, half_height, radius])
                else:
                    sizes.append([radius, float(scale[1]) + margin, float(scale[2]) + margin])
            sizes_env.append(torch.tensor(sizes, device=device, dtype=torch.float32))
        self.dyn_sizes_env = torch.stack(sizes_env, dim=0).contiguous()
        # The network reads the collider kinds and sizes of the first environment for all environments.
        kinds = [_NETWORK_KINDS.get(int(shape_type[s]), 1) for s in shapes[: self.num_colliders_per_world]]
        self.collider_kinds = torch.cat(
            (torch.tensor(kinds, device=device, dtype=torch.int32), torch.ones(5, device=device, dtype=torch.int32))
        )
        self.collider_sizes = torch.cat((sizes_env[0], boundary_size))
        self.collider_friction = torch.ones(self.num_colliders_per_world + 5, device=device, dtype=torch.float32)
        self.zero_wall_vel = torch.zeros((num_envs, 5, 6), device=device, dtype=torch.float32)

        # Material features, and the neighbourhood scales that follow the particle radius of the scene
        mass = wp.to_torch(model.particle_mass)[:: self.capacity].tolist()
        radius = wp.to_torch(model.particle_radius)[:: self.capacity].tolist()
        if radius[0] > 0.0:
            self.network.local_agg.radius = 4.5 * radius[0]
            self.support = max(0.035, 4.375 * radius[0])
        mpm = getattr(model, "mpm", None)
        raw = {
            name: wp.to_torch(getattr(mpm, name))[:: self.capacity].tolist()
            if hasattr(mpm, name)
            else [default] * num_envs
            for name, default in _MATERIAL_DEFAULTS.items()
        }
        materials = [
            material_features(
                {"material": {name: raw[name][env] for name in _MATERIAL_DEFAULTS}}, mass[env], radius[env], self.dt
            )
            for env in range(num_envs)
        ]
        self.mat_env = torch.stack(materials, dim=0).to(device).contiguous()

        if cfg.compile_model:
            self.network.encode = torch.compile(
                self.network.encode, dynamic=False, options={"triton.cudagraphs": False}
            )

        # Active set: K stepped particles and up to 256 background particles per environment
        self.active_k = resolve_active_k(cfg.active_k, self.capacity, materials[0])
        self.sleep_speed = float(cfg.sleep_speed)
        if self.active_k is not None:
            self.num_background = min(_NUM_BACKGROUND, self.capacity - self.active_k)
            total = self.active_k + self.num_background
            background = torch.linspace(0, self.capacity - 1, self.num_background, device=device).long()
            self._background = background.unsqueeze(0).expand(num_envs, -1)
            self._scores = torch.empty((num_envs, self.capacity), dtype=torch.float32, device=device)
            self._min_gaps = torch.empty((num_envs, self.capacity), dtype=torch.float32, device=device)
            self._active_history = torch.empty((num_envs, total, 3, 3), dtype=torch.float32, device=device)
            self._active_idx = torch.zeros((num_envs, self.active_k), dtype=torch.int64, device=device)
            self._input_idx = torch.zeros((num_envs, total), dtype=torch.int64, device=device)
            self._multiplicity = torch.ones((num_envs, total), dtype=torch.float32, device=device)
            self._gamma = torch.ones((num_envs, self.active_k), dtype=torch.float32, device=device)
        else:
            self.num_background = 0
        evaluated = self.capacity if self.active_k is None else self.active_k
        self.wrench_solver = ContactWrenchSolver(
            model,
            ids,
            colliders_per_world=self.num_colliders_per_world,
            particles_per_env=self.capacity,
            grid=self.network.local_agg.get_grid(device, num_envs * evaluated, with_groups=True),
            iterations=cfg.contact_iterations,
            effective_mass_scale=cfg.effective_mass_scale,
            tangent_mass_scale=cfg.tangent_mass_scale,
            horizontal_mass_scale=cfg.horizontal_mass_scale,
            upward_mass_scale=cfg.upward_mass_scale,
            normal_power=cfg.normal_power,
            relaxation=cfg.contact_relaxation,
            gradient_radius=float(self.network.local_agg.radius),
            gradient_contact_band=cfg.gradient_contact_band,
            stabilization=cfg.contact_stabilization,
            contact_offset=cfg.contact_offset,
        )

    def notify_model_changed(self, flags: newton.ModelFlags | int) -> None:
        if flags & newton.ModelFlags.BODY_INERTIAL_PROPERTIES:
            self.wrench_solver.refresh_body_inertia()

    def capture_collider_state(self, state: newton.State, body_local_to_global: wp.array) -> None:
        """Copy the rigid-body state that the next step reads the collider poses and twists from."""
        wp.launch(
            gather_collider_state,
            self.model.body_count,
            [body_local_to_global, state.body_q, state.body_qd, self.neural_state.body_q, self.neural_state.body_qd],
            device=self.model.device,
        )
        self._has_neural_state = True

    def coupling_rewind_proxy_body(
        self,
        mapping: wp.array,
        state: newton.State,
        forces: wp.array,
        body_gravity_acceleration: wp.array,
        dt: float,
    ) -> None:
        del body_gravity_acceleration
        wp.launch(
            rewind_proxy,
            len(mapping),
            [mapping, forces, state.body_q, self.model.body_inv_mass, self.model.body_inv_inertia, state.body_qd, dt],
            device=self.model.device,
        )

    def coupling_harvest_proxy_wrenches(self, mapping: wp.array, out_body_f: wp.array, **kwargs: object) -> None:
        del kwargs
        out_body_f.zero_()
        wp.launch(
            scatter_wrench,
            len(self.collider_ids_wp),
            [self.collider_ids_wp, mapping, self.wrench_solver.wrench, out_body_f],
            device=self.model.device,
        )

    def reset(
        self,
        state: newton.State,
        world_mask: wp.array | None = None,
        flags: newton.StateFlags | int | None = None,
    ) -> None:
        self._normalize_reset_world_mask(world_mask)
        if flags is not None and not (flags & newton.StateFlags.PARTICLE):
            return
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

    def _collider_state(self, state: newton.State) -> tuple[torch.Tensor, torch.Tensor]:
        """Poses [E, C + 5, 7] and twists [E, C + 5, 6] of the collider shapes and the tray boxes."""
        body_q = wp.to_torch(state.body_q)[self.collider_ids]
        body_qd = wp.to_torch(state.body_qd)[self.collider_ids]
        shape_tf, com = self.shape_local_tf, self.body_com_local
        q_xyz, q_w = body_q[..., 3:6], body_q[..., 6:7]
        # Shape pose in the world: body pose composed with the shape transform
        t_rot = 2.0 * torch.linalg.cross(q_xyz, shape_tf[..., :3], dim=-1)
        pos = body_q[..., :3] + shape_tf[..., :3] + q_w * t_rot + torch.linalg.cross(q_xyz, t_rot, dim=-1)
        x1, y1, z1, w1 = body_q[..., 3], body_q[..., 4], body_q[..., 5], body_q[..., 6]
        x2, y2, z2, w2 = shape_tf[..., 3], shape_tf[..., 4], shape_tf[..., 5], shape_tf[..., 6]
        quat = torch.stack(
            (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ),
            dim=-1,
        )
        # Linear velocity at the shape origin; the body velocity is given at the centre of mass
        t_com = 2.0 * torch.linalg.cross(q_xyz, com, dim=-1)
        p_com = body_q[..., :3] + com + q_w * t_com + torch.linalg.cross(q_xyz, t_com, dim=-1)
        v_lin = body_qd[..., :3] + torch.linalg.cross(body_qd[..., 3:6], pos - p_com, dim=-1)
        dyn_pose = torch.cat((pos, quat), dim=-1).contiguous()
        dyn_vel = torch.cat((v_lin, body_qd[..., 3:6]), dim=-1).contiguous()
        poses = torch.cat((dyn_pose, self.boundary_poses), dim=1).contiguous()
        velocities = torch.cat((dyn_vel, self.zero_wall_vel), dim=1).contiguous()
        return poses, velocities

    def _select_active(self, state_in: newton.State, poses: torch.Tensor, dt: float) -> None:
        """Choose the K stepped particles and the background particles of every environment.

        The score of a particle is its gap to the nearest collider minus a speed term, so particles near a
        collider and moving particles are chosen. Background particles are evenly spaced particle indices; the ones
        outside the active set share the weight of all particles that are not stepped.
        """
        k = self.active_k
        dyn_pose = poses[:, : self.num_colliders_per_world].contiguous()
        wp.launch(
            active_particle_score,
            len(self.particle_world_ids) * self.capacity,
            [
                state_in.particle_q,
                self.history_wp,
                wp.from_torch(dyn_pose, requires_grad=False),
                wp.from_torch(self.dyn_sizes_env, requires_grad=False),
                self.capacity,
                self.num_colliders_per_world,
                float(dt),
                wp.from_torch(self._scores.view(-1), requires_grad=False),
                wp.from_torch(self._min_gaps.view(-1), requires_grad=False),
            ],
            device=self.model.device,
        )
        top_scores, active = torch.topk(self._scores, k=k, dim=1, largest=False, sorted=False)
        threshold = top_scores.amax(dim=1, keepdim=True)
        valid = (torch.gather(self._scores, 1, self._background) > threshold).float()
        valid = torch.where(valid.sum(dim=1, keepdim=True) < 1.0, torch.ones_like(valid), valid)
        weight = valid * (float(self.capacity - k) / valid.sum(dim=1, keepdim=True).clamp_min(1.0))
        # Particles that barely made the active set are blended in, so that the set changes smoothly
        tau = ((threshold - top_scores) / 0.015).clamp(0.0, 1.0)
        self._active_idx.copy_(active)
        self._input_idx[:, :k].copy_(active)
        self._input_idx[:, k:].copy_(self._background)
        self._multiplicity[:, :k].fill_(1.0)
        self._multiplicity[:, k:].copy_(weight)
        self._gamma.copy_(tau * tau * (3.0 - 2.0 * tau))

    @torch.inference_mode()
    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        del control
        self.validate_observables(observables, contacts)
        if self.model.particle_count == 0:
            return
        stream = wp.stream_to_torch(wp.get_stream(self.model.device))
        with torch.cuda.stream(stream):
            in_place = state_in.particle_q.ptr == state_out.particle_q.ptr
            self.wrench_solver.capture(state_in, dt, copy_particles=in_place)
            collider_source = self.neural_state if self._has_neural_state else state_in
            self._has_neural_state = False
            pos_in = wp.to_torch(state_in.particle_q)
            pos_out = wp.to_torch(state_out.particle_q)
            vel_out = wp.to_torch(state_out.particle_qd)
            num_envs = len(self.particle_world_ids)
            poses, velocities = self._collider_state(collider_source)

            if self.active_k is not None:
                k = self.active_k
                if not in_place:
                    pos_out.copy_(pos_in)
                vel_out.zero_()
                if self.selection_phase == 0:
                    self._select_active(state_in, poses, dt)
                wp.launch(
                    extract_active_history,
                    (num_envs, k + self.num_background),
                    [
                        wp.from_torch(self._input_idx, dtype=wp.int64, requires_grad=False),
                        state_in.particle_q,
                        self.history_wp,
                        self.capacity,
                        wp.from_torch(self._active_history, requires_grad=False),
                    ],
                    device=self.model.device,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    next_pos, next_vel = self.network.step_from_history_batched(
                        self._active_history,
                        poses,
                        velocities,
                        self.collider_kinds,
                        self.collider_sizes,
                        self.collider_friction,
                        self.mat_env,
                        dt=dt,
                        support=self.support,
                        multiplicity=self._multiplicity,
                        return_vel=True,
                    )
                wp.launch(
                    advance_stationary_history,
                    num_envs * self.capacity,
                    [self.wrench_solver.captured_q, self.history_wp],
                    device=self.model.device,
                )
                wp.launch(
                    scatter_active_step,
                    (num_envs, k),
                    [
                        wp.from_torch(self._active_idx, dtype=wp.int64, requires_grad=False),
                        wp.from_torch(next_pos[:, :k].contiguous(), requires_grad=False),
                        wp.from_torch(next_vel[:, :k].contiguous(), requires_grad=False),
                        wp.from_torch(self._gamma, requires_grad=False),
                        wp.from_torch(self.tray_lower, requires_grad=False),
                        wp.from_torch(self.tray_upper, requires_grad=False),
                        self.capacity,
                        float(dt),
                        self.sleep_speed * float(dt),
                        self.wrench_solver.captured_q,
                        state_out.particle_q,
                        state_out.particle_qd,
                        self.history_wp,
                    ],
                    device=self.model.device,
                )
                self.wrench_solver.evaluate(state_out, active_idx=self._active_idx)
            else:
                history = self.history.view(num_envs, self.capacity, 3, 3)
                history_in = history.clone()
                history_in[:, :, -1] = pos_in.view(num_envs, self.capacity, 3)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    next_pos, next_vel = self.network.step_from_history_batched(
                        history_in,
                        poses,
                        velocities,
                        self.collider_kinds,
                        self.collider_sizes,
                        self.collider_friction,
                        self.mat_env,
                        dt=dt,
                        support=self.support,
                        return_vel=True,
                    )
                next_pos = torch.max(torch.min(next_pos, self.tray_upper.unsqueeze(1)), self.tray_lower.unsqueeze(1))
                if self.sleep_speed > 0.0:
                    pos_cur = history_in[:, :, -1]
                    asleep = (next_pos - pos_cur).norm(dim=-1, keepdim=True) < self.sleep_speed * dt
                    next_pos = torch.where(asleep, pos_cur, next_pos)
                    next_vel = torch.where(asleep, torch.zeros_like(next_vel), next_vel)
                next_pos_flat = next_pos.reshape(-1, 3)
                vel_out.copy_((next_pos_flat - pos_in) / dt)
                pos_out.copy_(next_pos_flat)
                history[:, :, 0] = history_in[:, :, 1]
                history[:, :, 1] = next_pos - dt * next_vel
                history[:, :, 2] = next_pos
                self.wrench_solver.evaluate(state_out)
        self.selection_phase = (self.selection_phase + 1) % max(1, int(self.cfg.selection_interval))
