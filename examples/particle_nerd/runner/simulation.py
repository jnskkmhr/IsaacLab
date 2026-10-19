# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Current particle GNN + analytical material forces + dynamic Newton rigid body."""

import numpy as np
import torch
import warp as wp

from particle_nerd.force_estimation import kernels
from particle_nerd.force_estimation.estimator import ContactEstimator
from particle_nerd.runner.rigid import NewtonRigidStepper


class CoupledSimulation:
    """Explicit coupling for the current open-ground, single-box dataset.

    At t, predict particle state at t+dt from the current rigid state. Reconstruct
    the material wrench from that predicted velocity. Advance the rigid body with
    this wrench and the known external wrench, using smaller MJWarp substeps.
    GNN particle velocities already include interaction: do not apply a second
    contact impulse to them. This integration has not been accuracy-calibrated.
    """

    def __init__(
        self, predictor, manifest, particle_mass, body_pose, body_velocity, rigid_dt=0.005, max_iterations=2000
    ):
        size = np.asarray(manifest["body_size_m"], dtype=np.float32)
        if not np.allclose(size, [0.12, 0.12, 0.08]):
            raise ValueError("The current GNN encodes a 0.12 x 0.12 x 0.08 m box")
        self.predictor, self.dt = predictor, predictor.dt
        self.substeps = round(self.dt / rigid_dt)
        if self.substeps < 1 or not np.isclose(self.substeps * rigid_dt, self.dt):
            raise ValueError("The GNN interval must be a whole number of rigid steps")
        mass = float(manifest["body_mass_kg"])
        friction = float(manifest.get("body_contact_friction", 0.6))
        inertia = np.diag(mass / 12 * (np.sum(size**2) - size**2))
        self.device = predictor.device
        self.stream = wp.Stream(self.device)
        self.torch_stream = wp.stream_to_torch(self.stream)
        self.torch_stream.wait_stream(torch.cuda.current_stream(predictor.x.device))
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            episode = dict(
                body_pose=np.asarray(body_pose)[None, None],
                body_velocity=np.asarray(body_velocity)[None, None],
                body_size=size[None],
                body_mass=np.array([mass]),
                body_inertia=inertia[None],
                body_com=np.zeros((1, 3)),
                body_shapes=["box"],
                body_friction=np.array([friction]),
                body_color=np.array([[0.32, 0.38, 0.42]]),
                gravity=[0, 0, -9.81],
                dt=rigid_dt,
                ground_plane=True,
                ground_friction=friction,
            )
            self.rigid = NewtonRigidStepper(episode, device=self.device)
            self.estimator = ContactEstimator(
                particle_mass,
                mass,
                inertia,
                size,
                self.dt,
                device=self.device,
                max_iterations=max_iterations,
                friction=friction,
            )
            self.old_position = wp.empty(len(predictor.x), dtype=wp.vec3, device=self.device)
            self.external = wp.zeros(1, dtype=wp.spatial_vector, device=self.device)
            self.steps = 0
            # Warm MJWarp, then restore the initial state before capturing both
            # ping-pong directions. Its convergence loop stays on the GPU.
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
    def step(self, external_wrench=None):
        """Advance one GNN interval without host state transfers.

        Optional external_wrench is a device Warp [1] spatial_vector array, world
        force then torque about the box COM. Omission means no external force.
        """
        self.torch_stream.wait_stream(torch.cuda.current_stream(self.predictor.x.device))
        with torch.cuda.stream(self.torch_stream), wp.ScopedStream(self.stream):
            if external_wrench is None:
                self.external.zero_()
            else:
                wp.copy(self.external, external_wrench)
            wp.copy(self.old_position, wp.from_torch(self.predictor.x, dtype=wp.vec3))
            self.predictor.step(wp.to_torch(self.rigid.state.body_q)[0], wp.to_torch(self.rigid.state.body_qd)[0])
            self.estimator.estimate(
                self.old_position,
                wp.from_torch(self.predictor.v, dtype=wp.vec3),
                self.rigid.state.body_q,
                self.rigid.state.body_qd,
                self.external,
            )
            for _ in range(self.substeps):
                wp.capture_launch(self.graphs[self.parity])
                self.rigid.state, self.rigid.next_state = self.rigid.next_state, self.rigid.state
                self.parity = 1 - self.parity
        self.steps += 1

    def diagnostics(self):
        wp.synchronize_stream(self.stream)
        return dict(
            step=self.steps,
            particle=self.predictor.diagnostics(),
            contact=self.estimator.diagnostics(),
            body_pose=self.rigid.state.body_q.numpy()[0].tolist(),
        )
