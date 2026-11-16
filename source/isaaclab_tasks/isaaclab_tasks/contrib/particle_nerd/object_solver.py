# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The current G1 compact NeRD method adapted to primitive object scenes."""

import re
from pathlib import Path

import newton
import numpy as np
import torch
import warp as wp
from newton.solvers import SolverBase
from newton.solvers.experimental.coupled import CouplingInterface

from .contact.proxy import ProxyEstimator
from .kernels import gather_collider_state, reset_history, rewind_proxy, scatter_wrench
from .network.features import material_features
from .network.model import ParticleTransolver
from .network.object_inference import PIPELINE, ObjectParticleStepper, validate_checkpoint


class ParticleNeRDObjectSolver(SolverBase, CouplingInterface):
    """Predict particle displacement and return an estimated wrench to each rigid object.

    Top-K selection, background particles, displacement stabilization and tray
    projection share the G1 implementation. Neural inputs use pre-rigid body
    state; force reconstruction separately uses the rewound proxy free velocity.
    """

    def __init__(self, model, cfg):
        super().__init__(model)
        if not model.device.is_cuda:
            raise ValueError("ParticleNeRDObjectSolver requires a CUDA device")
        self.cfg = cfg
        path = Path(cfg.checkpoint).expanduser()
        saved = torch.load(path, map_location=str(model.device), weights_only=False)
        validate_checkpoint(saved)
        for key in ("active_k", "anchor_particles", "selection_interval"):
            if getattr(cfg, key) != PIPELINE[key]:
                raise ValueError(f"{key} must match the retrained checkpoint pipeline")
        self.dt = float(saved["config"]["dt"])
        self.support = float(saved["config"]["features"]["support"])
        self.network = ParticleTransolver(**saved["model_config"]).to(str(model.device)).eval()
        self.network.load_state_dict(saved["model"])
        if self.network.config["history"] != 3:
            raise ValueError("Object examples require a three-frame particle history")

        worlds = model.particle_world.numpy()
        self.world_ids, counts = np.unique(worlds, return_counts=True)
        if not len(counts) or np.any(self.world_ids < 0) or np.any(counts != counts[0]):
            raise ValueError("Object examples require equal particle counts in assigned simulation worlds")
        self.capacity = int(counts[0])
        self.num_envs = len(counts)
        if not np.array_equal(worlds, np.repeat(self.world_ids, self.capacity)):
            raise ValueError("Particles must be contiguous and ordered by simulation world")

        body_world = model.body_world.numpy()
        ids = []
        for world in self.world_ids:
            for pattern in cfg.collider_body_patterns:
                found = [
                    i
                    for i, label in enumerate(model.body_label)
                    if body_world[i] == world and re.fullmatch(pattern, label)
                ]
                if len(found) != 1:
                    raise ValueError(f"Expected one body matching {pattern!r} in world {world}, found {found}")
                ids.extend(found)
        if len(cfg.collider_body_patterns) != 1:
            raise ValueError("Object examples require one interacting body per environment")
        self.collider_ids = torch.tensor(ids, dtype=torch.long, device=str(model.device))
        self.collider_ids_wp = wp.array(ids, dtype=int, device=model.device)
        self._init_geometry(ids)
        self._init_materials()
        self.estimator = ProxyEstimator(
            model,
            ids,
            particles_per_env=self.capacity,
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
            batch_contacts=False,
            primitive_colliders=True,
            colliders_per_world=1,
        )
        self.history_wp = wp.zeros(model.particle_count, dtype=wp.mat33, device=model.device)
        self.history = wp.to_torch(self.history_wp).view(self.num_envs, self.capacity, 3, 3)
        self.steppers = [
            ObjectParticleStepper(
                self.kinds, self.sizes[e], self.friction[e], self.material[e], self.dt, floor_slot=self.floor_slots[e]
            )
            for e in range(self.num_envs)
        ]
        self.neural_state = model.state()
        self._has_neural_state = False
        self.reset(model.state())

    def _init_geometry(self, body_ids):
        """Read the real collider and tray shapes, including per-world translations."""
        model = self.model
        bodies = model.shape_body.numpy()
        worlds = model.shape_world.numpy()
        types = model.shape_type.numpy()
        transforms = model.shape_transform.numpy()
        scales = model.shape_scale.numpy()
        friction = model.shape_material_mu.numpy()
        collide = (model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0
        com = model.body_com.numpy()
        kinds, sizes, mus, poses, origins = [], [], [], [], []
        self.floor_slots = []
        for world, body in zip(self.world_ids, body_ids, strict=True):
            moving = np.flatnonzero((bodies == body) & collide)
            walls = np.flatnonzero((worlds == world) & (bodies == -1) & collide)
            floors = [i for i in walls if "/Floor/" in model.shape_label[i]]
            if len(moving) != 1 or len(walls) != 5 or len(floors) != 1:
                raise ValueError("Expected one object shape and five static tray shapes per environment")
            if not np.allclose(com[body], 0) or not np.allclose(transforms[moving[0]], (0, 0, 0, 0, 0, 0, 1)):
                raise ValueError("Object contact reconstruction requires a centered collider and center of mass")
            shape_ids = np.concatenate((moving, walls))
            geo_to_kind = {int(newton.GeoType.BOX): 1, int(newton.GeoType.SPHERE): 2, int(newton.GeoType.CYLINDER): 4}
            try:
                kinds.append([geo_to_kind[int(types[i])] for i in shape_ids])
            except KeyError as exc:
                raise ValueError("Object examples support box, sphere and cylinder colliders") from exc
            local_sizes = scales[shape_ids].copy()
            if kinds[-1][0] == 2:
                local_sizes[0] = local_sizes[0, 0]
            elif kinds[-1][0] == 4:
                local_sizes[0] = (local_sizes[0, 0], local_sizes[0, 0], local_sizes[0, 1])
            sizes.append(local_sizes)
            self.floor_slots.append(int(np.flatnonzero(shape_ids == floors[0])[0]))
            mus.append(friction[shape_ids])
            poses.append(transforms[shape_ids])
            floor = floors[0]
            origin = transforms[floor, :3].copy()
            origin[2] += scales[floor, 2] + self.cfg.bed_depth
            origins.append(origin)
        if not np.all(np.asarray(kinds) == kinds[0]):
            raise ValueError("Batched worlds must share collider types and ordering")
        device = str(model.device)
        self.kinds = torch.tensor(kinds[0], dtype=torch.int32, device=device)
        self.sizes = torch.tensor(np.asarray(sizes), device=device)
        self.friction = torch.tensor(np.asarray(mus), device=device)
        self.poses = torch.tensor(np.asarray(poses), device=device)
        self.origins = torch.tensor(np.asarray(origins), device=device)
        self.poses[:, :, :3] -= self.origins[:, None]
        gravity = model.gravity.numpy()
        if not np.allclose(gravity, gravity[0]):
            raise ValueError("Object examples require identical gravity in every world")
        self.gravity = tuple(float(v) for v in gravity[0])

    def _init_materials(self):
        """Encode the scene material, without replacing snow with the training sand."""
        names = (
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
        raw = {name: getattr(self.model.mpm, name).numpy().reshape(self.num_envs, self.capacity) for name in names}
        mass = self.model.particle_mass.numpy().reshape(self.num_envs, self.capacity)
        radius = self.model.particle_radius.numpy().reshape(self.num_envs, self.capacity)
        material = []
        for env in range(self.num_envs):
            if any(not np.allclose(values[env], values[env, 0]) for values in (*raw.values(), mass, radius)):
                raise ValueError("Object examples require uniform material, particle mass and radius within each world")
            values = {name: float(data[env, 0]) for name, data in raw.items()}
            values["density"] = float(mass[env, 0] / (8 * radius[env, 0] ** 3))
            material.append(material_features({"material": values}, mass[env, 0], radius[env, 0], self.dt))
        self.material = torch.tensor(np.asarray(material), device=str(self.model.device))

    def notify_model_changed(self, flags):
        if flags & newton.ModelFlags.BODY_INERTIAL_PROPERTIES:
            self.estimator.refresh_body_inertia()

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
        if flags is not None and not flags & newton.StateFlags.PARTICLE:
            return
        for stepper in self.steppers:
            stepper.reset()
        mask = (
            world_mask
            if world_mask is not None
            else wp.zeros(self.model.world_count + 1, dtype=wp.bool, device=self.model.device)
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

    def capture_collider_state(self, state, body_local_to_global):
        """Snapshot neural collider inputs before the rigid solver integrates velocities."""
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
        if abs(dt - self.dt) > 1e-8:
            raise ValueError(f"Object checkpoint requires dt={self.dt}, received {dt}")
        with torch.cuda.stream(wp.stream_to_torch(wp.get_stream(self.model.device))):
            if not self._has_neural_state:
                raise RuntimeError("Capture the pre-rigid collider state before object NeRD stepping")
            self.estimator.capture(state_in, dt)
            position = wp.to_torch(state_in.particle_q).view(self.num_envs, self.capacity, 3)
            history = self.history.clone()
            history[:, :, -1] = position
            history -= self.origins[:, None, None]
            poses = self.poses.clone()
            poses[:, 0] = wp.to_torch(self.neural_state.body_q)[self.collider_ids]
            poses[:, 0, :3] -= self.origins
            velocities = torch.zeros((self.num_envs, len(self.kinds), 6), device=position.device)
            velocities[:, 0] = wp.to_torch(self.neural_state.body_qd)[self.collider_ids]
            batch_size = self.num_envs if self.cfg.batch_inference else 1
            for start in range(0, self.num_envs, batch_size):
                selected = slice(start, start + batch_size)
                prepared = [
                    self.steppers[e].prepare(history[e], poses[e], velocities[e])
                    for e in range(start, min(start + batch_size, self.num_envs))
                ]
                inputs = [torch.stack(tensors) for tensors in zip(*(p.inputs for p in prepared), strict=True)]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw = self.network(*inputs)
                displacement = torch.stack([self.steppers[start + i].finish(p, raw[i]) for i, p in enumerate(prepared)])
                next_position = position[selected] + displacement
                wp.to_torch(state_out.particle_qd).view_as(position)[selected] = displacement / dt
                wp.to_torch(state_out.particle_q).view_as(position)[selected] = next_position
                self.history[selected, :, :2] = history[selected, :, 1:] + self.origins[selected, None, None]
                self.history[selected, :, 2] = next_position
            self.estimator.evaluate(state_out, active_idx=[s.geometry["cached_topk"] for s in self.steppers])
            self._has_neural_state = False
