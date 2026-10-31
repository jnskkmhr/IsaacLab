# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytical GPU contact estimation, independent of the particle predictor."""

import numpy as np
import warp as wp

from particle_nerd.force_estimation import kernels
from particle_nerd.force_estimation.body_velocity import initialize_body_velocity_solve


class ContactEstimator:
    """Estimate material-on-body world force and COM torque from next particle velocity.

    Geometry and mass are fixed at construction. All step arguments are Warp GPU
    arrays. Positions are at t; particle velocities are the prediction at t+dt.
    The external wrench contains known forces, excluding material contact and gravity.
    The effective particle mass scale and stabilization are approximation parameters,
    not identified material laws. Warm-starting only changes solver initialization;
    the contact law and convergence tolerance are unchanged. With fixed_material_velocity=True, contact
    impulses do not change the supplied material velocity; the effective mass
    scale is ignored. No MPM state or future rigid state is used.
    """

    def __init__(
        self,
        particle_mass,
        body_mass,
        body_inertia,
        body_size,
        dt,
        shape="box",
        friction=0.48,
        gravity=(0, 0, -9.81),
        effective_mass_scale=10.0,
        stabilization=0.01,
        gradient_radius=0.065,
        max_iterations=2000,
        tolerance=1e-9,
        device="cuda:0",
        fixed_material_velocity=False,
        jacobi_relaxation=0.1,
        warm_start=False,
        contact_solver="jacobi",
        horiz_mass_scale=None,
        up_mass_scale=None,
        tangent_mass_scale=None,
        normal_power=None,
    ):
        if dt <= 0 or body_mass <= 0 or max_iterations < 1:
            raise ValueError("dt, body mass, and iteration limit must be positive")
        if effective_mass_scale <= 0 or gradient_radius <= 0 or tolerance <= 0 or friction < 0:
            raise ValueError("Invalid force-estimator parameters")
        if len(particle_mass) == 0 or np.any(np.asarray(particle_mass) <= 0):
            raise ValueError("Particle masses must be positive")
        if not 0 < jacobi_relaxation <= 1:
            raise ValueError("Jacobi relaxation must be in (0, 1]")
        if contact_solver not in ("jacobi", "body_velocity", "rft_body_velocity"):
            raise ValueError("contact_solver must be jacobi, body_velocity, or rft_body_velocity")
        self.use_body_velocity = contact_solver in ("body_velocity", "rft_body_velocity") and not fixed_material_velocity
        if self.use_body_velocity and warm_start:
            raise ValueError("Choose body-velocity initialization or contact warm start")
        self.contact_solver = contact_solver if self.use_body_velocity else "jacobi"
        self.warm_start = bool(warm_start)
        self.jacobi_relaxation = float(jacobi_relaxation)
        self.n, self.device = len(particle_mass), device
        self.mass, self.dt = float(body_mass), float(dt)
        # Asset inertia may contain float32 accumulation asymmetry. Both contact
        # solvers require the same symmetric physical inertia; otherwise the
        # tangent solve invents a correction to a converged body-velocity state.
        inertia = np.asarray(body_inertia, dtype=np.float64)
        self.inertia = wp.mat33d(*((inertia + inertia.T) * 0.5).flatten())
        self.gravity = wp.vec3(*gravity)
        self.fixed_material_velocity = bool(fixed_material_velocity)
        self.scale, self.stabilization = float(effective_mass_scale), float(stabilization)
        rft_default = contact_solver == "rft_body_velocity"
        self.horiz_scale = float(horiz_mass_scale if horiz_mass_scale is not None else (8.0 if rft_default else self.scale))
        self.up_scale = float(up_mass_scale if up_mass_scale is not None else (2.0 if rft_default else self.scale))
        self.tangent_scale = float(tangent_mass_scale if tangent_mass_scale is not None else (4.0 if rft_default else self.scale))
        self.normal_power = float(normal_power if normal_power is not None else (3.0 if rft_default else 1.0))
        self.radius, self.friction = float(gradient_radius), float(friction)
        self.max_iterations, self.tolerance = max_iterations, tolerance
        self.particle_mass = wp.array(particle_mass, dtype=float, device=device)
        self.sizes = wp.array([body_size], dtype=wp.vec3, device=device)
        self.shapes = wp.array([{"box": 0, "sphere": 1, "cylinder": 2, "shovel": 3}[shape]], dtype=int, device=device)

        from particle_nerd.shovel.sdf import tensors

        sdf, origin, spacing = tensors(str(device), "shovel" if shape == "shovel" else "none")
        self.sdf = (wp.from_torch(sdf), wp.vec3(*origin), spacing)

        def zeros(n, dtype):
            return wp.zeros(n, dtype=dtype, device=device)

        self.grid = wp.HashGrid(64, 64, 64, device=device)
        self.grid.reserve(self.n)
        self.gradients = zeros(self.n, wp.mat33d)
        self.free, self.inverse = zeros(2, wp.vec3d), zeros(1, wp.mat33d)
        self.flags, self.offsets, self.ids = [zeros(self.n, int) for _ in range(3)]
        self.frames, self.angular, self.response, self.compliance = [zeros(self.n, wp.mat33d) for _ in range(4)]
        self.contact_velocity, self.impulses = [zeros(self.n, wp.vec3d) for _ in range(2)]
        self.gap, self.stats = zeros(self.n, wp.float64), zeros(3, wp.float64)
        self.wrench = zeros(1, wp.spatial_vector)
        self.cached_world_impulses = zeros(self.n, wp.vec3d)
        self.initial_velocity = zeros(2, wp.vec3d)
        self.body_velocity_stats = zeros(3, wp.float64)

    def reset(self):
        """Clear contact history after an episode reset or discontinuous state change.

        Enqueue on the same stream used for estimate(). Particle ordering and the
        estimator's rigid-body identity must remain fixed between resets.
        """
        self.cached_world_impulses.zero_()
        self.impulses.zero_()

    def estimate(self, position, next_velocity, body_pose, body_velocity, external_wrench):
        """Enqueue the solve on the active Warp stream; return a device wrench array.

        Shapes: particle arrays [N] vec3, body arrays [1] transform/spatial_vector.
        The caller selects the stream (use the Torch stream for Torch/Warp sharing).
        This method does not update particles or rigid bodies and never reads GPU data.
        """
        self.grid.build(position, self.radius)
        wp.launch(
            kernels.fit_gradients,
            self.n,
            [
                self.grid.id,
                position,
                next_velocity,
                self.radius,
                self.gradients,
                body_pose,
                self.sizes,
                self.shapes,
                0,
                -1.0,
            ],
            device=self.device,
        )
        wp.launch(
            kernels.prepare_body,
            1,
            [
                body_pose,
                body_velocity,
                external_wrench,
                0,
                self.mass,
                self.inertia,
                self.gravity,
                self.dt,
                self.free,
                self.inverse,
            ],
            device=self.device,
        )
        wp.launch(
            kernels.prepare_contacts,
            self.n,
            [
                position,
                next_velocity,
                self.gradients,
                body_pose,
                self.sizes,
                self.shapes,
                0,
                self.free,
                self.inverse,
                self.mass,
                self.particle_mass,
                self.dt,
                self.scale,
                self.horiz_scale,
                self.up_scale,
                self.tangent_scale,
                self.normal_power,
                self.stabilization,
                self.flags,
                self.frames,
                self.angular,
                self.response,
                self.compliance,
                self.contact_velocity,
                self.gap,
                self.fixed_material_velocity,
                *self.sdf,
            ],
            device=self.device,
        )
        wp.utils.array_scan(self.flags, self.offsets, inclusive=True)
        wp.launch(kernels.compact_contacts, self.n, [self.flags, self.offsets, self.ids], device=self.device)
        initial_velocity = self.free
        if self.use_body_velocity:
            wp.launch(
                initialize_body_velocity_solve,
                128,
                [
                    self.offsets,
                    self.ids,
                    self.frames,
                    self.angular,
                    self.response,
                    self.contact_velocity,
                    self.gap,
                    self.particle_mass,
                    self.free,
                    self.mass,
                    self.scale,
                    self.horiz_scale,
                    self.up_scale,
                    self.tangent_scale,
                    self.normal_power,
                    self.friction,
                    self.impulses,
                    self.initial_velocity,
                    self.body_velocity_stats,
                ],
                block_dim=128,
                device=self.device,
            )
            initial_velocity = self.initial_velocity
        elif self.warm_start:
            wp.launch(
                kernels.initialize_contact_warmstart,
                128,
                [
                    self.offsets,
                    self.ids,
                    self.frames,
                    self.response,
                    self.cached_world_impulses,
                    self.free,
                    self.mass,
                    self.friction,
                    self.impulses,
                    self.initial_velocity,
                ],
                block_dim=128,
                device=self.device,
            )
            initial_velocity = self.initial_velocity
        else:
            self.impulses.zero_()
        self.wrench.zero_()
        wp.launch(
            kernels.solve_contacts_parallel,
            128,
            [
                self.offsets,
                self.ids,
                self.frames,
                self.angular,
                self.response,
                self.compliance,
                self.contact_velocity,
                self.gap,
                self.particle_mass,
                initial_velocity,
                self.mass,
                self.scale,
                self.horiz_scale,
                self.up_scale,
                self.tangent_scale,
                self.normal_power,
                self.friction,
                self.dt,
                self.max_iterations,
                wp.float64(self.tolerance),
                self.impulses,
                self.wrench,
                0,
                self.stats,
                self.fixed_material_velocity,
                self.jacobi_relaxation,
            ],
            block_dim=128,
            device=self.device,
        )
        if self.warm_start:
            wp.launch(
                kernels.cache_contact_impulses,
                self.n,
                [
                    self.flags,
                    self.frames,
                    self.impulses,
                    self.stats,
                    wp.float64(self.tolerance),
                    self.cached_world_impulses,
                ],
                device=self.device,
            )
        return self.wrench

    def diagnostics(self):
        """Explicit host read for logging, outside the simulation step."""
        stats, wrench = self.stats.numpy(), self.wrench.numpy()[0]
        initializer = self.body_velocity_stats.numpy()
        return dict(
            solver=self.contact_solver,
            body_velocity_iterations=int(initializer[0]),
            body_velocity_residual=float(initializer[1]),
            body_velocity_fallback=bool(initializer[2]),
            impulse_iterations=int(stats[1]),
            contacts=int(stats[0]),
            iterations=int(stats[1] + initializer[0]),
            residual=float(stats[2]),
            converged=bool(stats[2] <= self.tolerance),
            finite=bool(np.isfinite(wrench).all()),
            wrench=wrench.tolist(),
        )
