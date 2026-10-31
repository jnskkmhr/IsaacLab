# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One batched GPU step: features, Transolver, particles, contacts and MJWarp."""

import numpy as np
import torch
import warp as wp

from . import io, sdf, state
from .features import FeatureBuffers
from .force.estimator import ContactEstimator
from .force.kernels import apply_wrenches
from .model import ParticleTransolver
from .rigid import RigidBatch, rigid_scene_description


class Simulation:
    """Independent worlds, each with particles, one dynamic intruder and fixed ground.

    States and inputs use world-local coordinates. Particle arrays are [E,N,3];
    rigid arrays are [E,7] xyzw poses and [E,6] linear/angular velocities.
    All worlds share N, collider geometry, rigid mass/inertia and NN weights.
    step() performs no CPU state transfer or Python loop over environments.
    """

    @classmethod
    def from_files(cls, checkpoint, episode, environments=1, **options):
        saved, manifest, initial = io.load(checkpoint, episode, environments)
        return cls(saved, manifest, **initial, **options)

    def __init__(
        self,
        saved,
        manifest,
        history,
        material,
        particle_mass,
        body_pose,
        body_velocity,
        device="cuda:0",
        rigid_dt=0.005,
        compile_model=False,
        capture=True,
        effective_mass_scale=None,
        relaxation=None,
        max_iterations=2000,
        active_particles=None,
        anchor_particles=64,
        force_model="auto",
    ):
        wp.init()
        arch = saved.get("architecture")
        if arch != "particle_transolver_v1":
            raise ValueError(f"Expected particle_transolver_v1 checkpoint, got {arch}")
        self.architecture = arch
        self.config, self.manifest = saved["config"], manifest
        self.device, self.dt, self.rigid_dt = device, float(self.config["dt"]), rigid_dt
        self.environments, self.particles, h, xyz = history.shape
        if min(self.environments, self.particles) < 1 or h != self.config["model"]["history"] or xyz != 3:
            raise ValueError("history must be [E,N,checkpoint_history,3]")
        if np.shape(body_pose) != (self.environments, 7) or np.shape(body_velocity) != (self.environments, 6):
            raise ValueError("Rigid state shapes must be [E,7] and [E,6]")
        if rigid_dt <= 0:
            raise ValueError("Rigid timestep must be positive")
        self.substeps = round(self.dt / rigid_dt)
        if self.substeps < 1 or not np.isclose(self.substeps * rigid_dt, self.dt):
            raise ValueError("Particle timestep must be an integer number of rigid substeps")
        if np.shape(material) != (self.environments, 13) or np.shape(particle_mass) != (
            self.environments,
            self.particles,
        ):
            raise ValueError("Material/mass shapes must be [E,13] and [E,N]")

        if force_model in ("auto", "analytical"):
            self.use_rft = False
        elif force_model == "rft":
            self.use_rft = True
        else:
            raise ValueError(f"Unknown force_model: {force_model}")

        self.capture_enabled, self.compiled = capture, compile_model
        self.stream = wp.Stream(device)
        self.torch_stream = wp.stream_to_torch(self.stream)
        self.torch_stream.wait_stream(torch.cuda.current_stream(device))
        self.parity, self.steps = 0, 0
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream), torch.inference_mode():
            self.history = torch.as_tensor(history, dtype=torch.float32, device=device).clone()
            self.positions = self.history[:, :, -1].contiguous()
            self.velocities = ((self.history[:, :, -1] - self.history[:, :, -2]) / self.dt).contiguous()
            self.material = torch.as_tensor(material, dtype=torch.float32, device=device).contiguous()
            self.x = wp.from_torch(self.positions.view(-1, 3), dtype=wp.vec3)
            self.v = wp.from_torch(self.velocities.view(-1, 3), dtype=wp.vec3)
            self.hist = wp.from_torch(self.history.view(-1, h, 3))
            self.old_position = wp.empty(self.x.shape, dtype=wp.vec3, device=device)
            scene = rigid_scene_description(manifest, body_pose, body_velocity, rigid_dt)
            self.rigid = RigidBatch(scene, device)
            self.external = wp.zeros(self.environments, dtype=wp.spatial_vector, device=device)
            self.external_wrench = wp.to_torch(self.external)

            self.model = ParticleTransolver(**self.config["model"]).to(device).eval()
            self.model.load_state_dict(saved["model"])

            if compile_model:
                self.model = torch.compile(
                    self.model, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
                )

            shape = manifest.get("geometry", "box")
            if effective_mass_scale is None:
                effective_mass_scale = (220.0 if self.use_rft else 180.0) if shape == "shovel" else 10.0
            if relaxation is None:
                relaxation = 0.075 if shape == "shovel" else 0.1
            kind = {"box": 1, "sphere": 2, "shovel": 3}.get(shape)
            if kind is None:
                raise ValueError("This checkpoint feature encoder supports box, sphere and shovel colliders")
            kinds = torch.tensor([kind, 0], dtype=torch.int32, device=device)
            sizes = torch.as_tensor(
                np.array([np.asarray(manifest["body_size_m"]) / 2, [0, 0, 0]], dtype=np.float32), device=device
            )
            friction = float(manifest.get("body_contact_friction", 0.6))
            frictions = torch.tensor([friction, 0.6 if shape == "shovel" else friction], device=device)
            self.sdf = sdf.tensors(device, shape)
            self.features = FeatureBuffers(
                self.history,
                self.material,
                kinds,
                sizes,
                frictions,
                self.sdf,
                self.dt,
                active_particles=active_particles,
                anchor_particles=anchor_particles,
                **self.config.get("features", {}),
            )
            self.estimator = ContactEstimator(
                particle_mass,
                scene["body_mass"][0],
                scene["body_inertia"][0],
                scene["body_size"][0],
                shape,
                self.sdf,
                self.dt,
                device=device,
                effective_mass_scale=effective_mass_scale,
                relaxation=relaxation,
                max_iterations=max_iterations,
                friction=friction,
                rft=self.use_rft,
            )
            self.wrench = self.estimator.wrench
            self.stats = self.estimator.stats
            self.contact_wrench = wp.to_torch(self.estimator.wrench)

            self._initial_history = wp.clone(self.hist)
            initial = self.rigid.states[0]
            self._initial_rigid = tuple(
                wp.clone(a) for a in (initial.joint_q, initial.joint_qd, initial.body_q, initial.body_qd)
            )
            self._reset_mask = wp.ones(self.environments + 1, dtype=wp.bool, device=device)
            self.graphs = []
            self._graph_outputs = []
            self._advance()
            self.reset()
            if capture:
                for parity in range(2 if self.substeps % 2 else 1):
                    self.parity = parity
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self.torch_stream):
                        with wp.ScopedCapture(device=device, stream=self.stream, external=True):
                            self._advance()
                    self.graphs.append(graph)
                    self._graph_outputs.append(self._displacement)
                self.parity = 0
                self.reset()
        torch.cuda.current_stream(device).wait_stream(self.torch_stream)

    @property
    def body_pose(self):
        return wp.to_torch(self.rigid.states[self.parity].body_q)

    @property
    def body_velocity(self):
        return wp.to_torch(self.rigid.states[self.parity].body_qd)

    def _advance(self):
        current = self.rigid.states[self.parity]
        wp.copy(self.old_position, self.x)
        self.features.build(current.body_q, current.body_qd)
        nodes, pairs, inf = self.features.inputs
        k_act = self.features.active_k
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            tokens = self.model.encode(nodes, pairs, inf, multiplicity=self.features.multiplicity)
            tokens_active = tokens[:, :k_act] if self.features.compact else tokens
            displacement = (
                self.model.config["displacement_scale"] * self.model.decoder(tokens_active).float()
            ).contiguous()

        if self.features.compact:
            wp.launch(
                state.shift_history_zero_vel,
                len(self.x),
                [self.hist, self.x, self.v],
                device=self.device,
            )
            wp.launch(
                state.apply_active_displacement,
                self.environments * k_act,
                [
                    self.hist,
                    self.x,
                    self.v,
                    wp.from_torch(self.features.active_global.view(-1), dtype=wp.int32),
                    wp.from_torch(self.features.active_mask.view(-1), dtype=wp.float32),
                    wp.from_torch(displacement.view(-1, 3)),
                    self.dt,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                state.update,
                len(self.x),
                [self.hist, self.x, self.v, wp.from_torch(displacement.view(-1, 3)), self.dt],
                device=self.device,
            )

        self.estimator.estimate(self.old_position, self.v, current.body_q, current.body_qd, self.external)
        for _ in range(self.substeps):
            current, target = self.rigid.states[self.parity], self.rigid.states[1 - self.parity]
            wp.launch(
                apply_wrenches,
                self.environments,
                [self.estimator.wrench, self.external, current.body_f],
                device=self.device,
            )
            self.rigid.step(current, target, self.rigid_dt)
            self.parity = 1 - self.parity
        self._displacement = displacement

    @torch.inference_mode()
    def step(self, external_wrench=None):
        """Advance one particle interval. Optional CUDA [E,6] force/COM torque."""
        caller = torch.cuda.current_stream(self.device)
        self.torch_stream.wait_stream(caller)
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            if external_wrench is None:
                self.external.zero_()
            else:
                if external_wrench.shape != (self.environments, 6) or external_wrench.device != self.positions.device:
                    raise ValueError("external_wrench must be CUDA [E,6] on the simulation device")
                self.external_wrench.copy_(external_wrench)
            if self.capture_enabled:
                self.graphs[self.parity if self.substeps % 2 else 0].replay()
                self.parity = (self.parity + self.substeps) % 2
            else:
                self._advance()
        caller.wait_stream(self.torch_stream)
        self.steps += 1

    @torch.inference_mode()
    def reset(self, mask=None):
        """Restore initial states for a CUDA boolean [E] mask, or every world."""
        caller = torch.cuda.current_stream(self.device)
        self.torch_stream.wait_stream(caller)
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            if mask is None:
                self._reset_mask.fill_(True)
            else:
                if (
                    mask.shape != (self.environments,)
                    or mask.dtype != torch.bool
                    or mask.device != self.positions.device
                ):
                    raise ValueError("reset mask must be CUDA bool [E]")
                wp.to_torch(self._reset_mask)[: self.environments].copy_(mask)
                wp.to_torch(self._reset_mask)[self.environments :] = False
            wp.launch(
                state.reset_particles,
                len(self.x),
                [self._reset_mask, self.particles, self._initial_history, self.hist, self.x, self.v, self.dt],
                device=self.device,
            )
            for rigid_state in self.rigid.states:
                wp.launch(
                    state.reset_rigid,
                    self.environments,
                    [
                        self._reset_mask,
                        *self._initial_rigid,
                        rigid_state.joint_q,
                        rigid_state.joint_qd,
                        rigid_state.body_q,
                        rigid_state.body_qd,
                        rigid_state.body_f,
                    ],
                    device=self.device,
                )
            self.rigid.solver.reset(self.rigid.states[self.parity], world_mask=self._reset_mask, flags=0)
            wp.launch(
                state.reset_loads,
                self.environments,
                [self._reset_mask, self.external, self.wrench, self.stats],
                device=self.device,
            )
        caller.wait_stream(self.torch_stream)

    def diagnostics(self):
        """Explicit host synchronization for validation/logging; not part of step()."""
        wp.synchronize_stream(self.stream)
        stats = self.stats.numpy().reshape(self.environments, 3)
        tol = self.estimator.tolerance
        return dict(
            finite=bool(
                torch.isfinite(self.positions).all()
                and torch.isfinite(self.velocities).all()
                and torch.isfinite(self.body_pose).all()
                and torch.isfinite(self.body_velocity).all()
                and torch.isfinite(self.contact_wrench).all()
            ),
            converged=bool(np.all(stats[:, 2] < tol)),
            contacts=stats[:, 0].tolist(),
            iterations=stats[:, 1].tolist(),
            residual=stats[:, 2].tolist(),
            rigid_overflow=bool(self.rigid.solver.mjw_data.overflow.numpy().any()),
            force_model="rft" if self.use_rft else "analytical",
            active_particles=self.features.active_k,
            heuristic_active_counts=self.features.active_count.cpu().tolist(),
        )
