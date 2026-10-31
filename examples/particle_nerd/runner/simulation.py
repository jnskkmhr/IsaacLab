# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled particle predictor + analytical material force estimator + MJWarp."""

import numpy as np
import torch
import warp as wp

from particle_nerd.force_estimation import kernels
from particle_nerd.force_estimation.estimator import ContactEstimator
from particle_nerd.runner.rigid import NewtonRigidStepper, rigid_scene_description


class CoupledSimulation:
    """Explicit coupling for the open-ground dynamic box or shovel simulation.

    At t, predict particle state at t+dt from the current rigid state. Reconstruct
    the material wrench using the analytical ContactEstimator and advance the rigid
    body with this wrench and known external wrench over MJWarp substeps.
    """

    def __init__(
        self,
        predictor,
        manifest,
        particle_mass,
        body_pose,
        body_velocity,
        rigid_dt=0.005,
        max_iterations=2000,
        effective_mass_scale=10.0,
        fixed_material_velocity=False,
        jacobi_relaxation=0.1,
        contact_warm_start=False,
        contact_solver="jacobi",
        use_cuda_graph=None,
        force_model="auto",
    ):
        del force_model
        size = np.asarray(manifest["body_size_m"], dtype=np.float32)
        if manifest.get("geometry", "box") != "shovel" and not np.allclose(size, [0.12, 0.12, 0.08]):
            raise ValueError("The current box collider expects a 0.12 x 0.12 x 0.08 m box")
        self.predictor, self.dt = predictor, predictor.dt
        self.use_cuda_graph = (
            getattr(predictor, "use_cuda_graph", False) if use_cuda_graph is None else bool(use_cuda_graph)
        )
        self.estimator_graphs = {}
        self.substeps = round(self.dt / rigid_dt)
        if self.substeps < 1 or not np.isclose(self.substeps * rigid_dt, self.dt):
            raise ValueError("The particle interval must be a whole number of rigid steps")
        mass = float(manifest["body_mass_kg"])
        friction = float(manifest.get("body_contact_friction", 0.6))
        inertia = (
            np.asarray(manifest["body_inertia"])
            if "body_inertia" in manifest
            else np.diag(mass / 12 * (np.sum(size**2) - size**2))
        )
        self.device = predictor.device
        self.stream = wp.Stream(self.device)
        self.torch_stream = wp.stream_to_torch(self.stream)
        self.torch_stream.wait_stream(torch.cuda.current_stream(predictor.x.device))
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            episode = rigid_scene_description(manifest, body_pose, body_velocity, rigid_dt)
            self.rigid = NewtonRigidStepper(episode, device=self.device)
            self.estimator = ContactEstimator(
                particle_mass,
                mass,
                inertia,
                size,
                self.dt,
                device=self.device,
                shape=manifest.get("geometry", "box"),
                max_iterations=max_iterations,
                effective_mass_scale=effective_mass_scale,
                fixed_material_velocity=fixed_material_velocity,
                jacobi_relaxation=jacobi_relaxation,
                warm_start=contact_warm_start,
                contact_solver=contact_solver,
                friction=friction,
            )
            self.old_position = wp.empty(len(predictor.x), dtype=wp.vec3, device=self.device)
            self.external = wp.zeros(1, dtype=wp.spatial_vector, device=self.device)
            self.steps = 0
            initial_q = self.rigid.state.joint_q.numpy().copy()
            initial_qd = self.rigid.state.joint_qd.numpy().copy()
            self.rigid.pipeline.collide(self.rigid.state, self.rigid.contacts)
            self.rigid.solver.step(
                self.rigid.state, self.rigid.next_state, self.rigid.control, self.rigid.contacts, rigid_dt
            )
            self.graphs = []
            for source, target in [
                (self.rigid.state, self.rigid.next_state),
                (self.rigid.next_state, self.rigid.state),
            ]:
                with wp.ScopedCapture(device=self.device, stream=self.stream) as capture:
                    wp.launch(kernels.apply_wrenches, 1, [self.estimator.wrench, self.external, source.body_f])
                    self.rigid.pipeline.collide(source, self.rigid.contacts)
                    self.rigid.solver.step(source, target, self.rigid.control, self.rigid.contacts, rigid_dt)
                self.graphs.append(capture.graph)
            for state in [self.rigid.state, self.rigid.next_state]:
                state.joint_q.assign(initial_q)
                state.joint_qd.assign(initial_qd)
                state.body_q.assign(np.asarray(body_pose)[None])
                state.body_qd.assign(np.asarray(body_velocity)[None])
            self.parity = 0

    @torch.no_grad()
    def step(self, external_wrench=None, substep_controller=None):
        """Advance one particle interval without host state transfers."""
        if external_wrench is not None and substep_controller is not None:
            raise ValueError("Choose an external wrench or a substep controller")
        self.torch_stream.wait_stream(torch.cuda.current_stream(self.predictor.x.device))
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            if substep_controller is not None:
                substep_controller(self.rigid.state, 0, self.external)
            elif external_wrench is None:
                self.external.zero_()
            else:
                wp.copy(self.external, external_wrench)
            wp.copy(self.old_position, wp.from_torch(self.predictor.x, dtype=wp.vec3))
            self.predictor.step(wp.to_torch(self.rigid.state.body_q)[0], wp.to_torch(self.rigid.state.body_qd)[0])
            estimate_args = (
                self.old_position,
                wp.from_torch(self.predictor.v, dtype=wp.vec3),
                self.rigid.state.body_q,
                self.rigid.state.body_qd,
                self.external,
            )
            if self.use_cuda_graph:
                key = tuple(array.ptr for array in estimate_args)
                if key not in self.estimator_graphs:
                    self.estimator.estimate(*estimate_args)
                    with wp.ScopedCapture(device=self.device, stream=self.stream) as capture:
                        self.estimator.estimate(*estimate_args)
                    self.estimator_graphs[key] = capture.graph
                else:
                    wp.capture_launch(self.estimator_graphs[key])
            else:
                self.estimator.estimate(*estimate_args)
            for substep in range(self.substeps):
                if substep > 0 and substep_controller is not None:
                    substep_controller(self.rigid.state, substep, self.external)
                wp.capture_launch(self.graphs[self.parity])
                self.rigid.state, self.rigid.next_state = self.rigid.next_state, self.rigid.state
                self.parity = 1 - self.parity
        self.steps += 1

    def diagnostics(self):
        wp.synchronize_stream(self.stream)
        particle = self.predictor.diagnostics()
        particle["velocity_finite"] = bool(torch.isfinite(self.predictor.v).all())
        return dict(
            step=self.steps,
            particle=particle,
            contact=self.estimator.diagnostics(),
            body_pose=self.rigid.state.body_q.numpy()[0].tolist(),
            body_velocity=self.rigid.state.body_qd.numpy()[0].tolist(),
            rigid_contact=dict(
                generated=int(self.rigid.contacts.rigid_contact_count.numpy()[0]),
                accepted=int(self.rigid.solver.mjw_data.nacon.numpy()[0]),
                capacity=self.rigid.solver.mjw_data.naconmax,
                overflow=bool(self.rigid.solver.mjw_data.overflow.numpy().any()),
            ),
        )
