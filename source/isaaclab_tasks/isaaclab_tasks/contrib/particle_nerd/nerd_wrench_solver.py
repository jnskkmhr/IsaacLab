# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Body wrench solver: the force and torque that the particles of a neural step apply to every rigid collider.

The network moves the particles but returns no force. :class:`ContactWrenchSolver` reconstructs the wrench on
every collider from the particle step. It solves for the particle-body contact impulses (normal, then friction)
that keep the particles from entering the collider, treating every contact particle as a point mass with a
direction-dependent effective mass. The sum of the impulses, divided by the time step, is the wrench.

This file does not detect contacts. Which particles are in contact with which collider is decided inside the
kernels it launches: :func:`~.nerd_kernels.prepare_contacts` flags the candidate contacts (particles near a
collider surface that are closing in on it) and :func:`~.nerd_kernels.compact_contacts` packs them.
"""

from __future__ import annotations

import newton
import numpy as np
import torch
import warp as wp

from .nerd_kernels import (
    ContactProblem,
    compact_contacts,
    gather_active_particles,
    initialize_contacts,
    prepare_contact_bodies,
    prepare_contacts,
    solve_contacts,
)

_SHAPE_CODES = {
    int(newton.GeoType.BOX): 0,
    int(newton.GeoType.SPHERE): 1,
    int(newton.GeoType.CYLINDER): 2,
    int(newton.GeoType.CAPSULE): 4,
}
"""Shape codes of :func:`~.nerd_kernels.collider_surface`."""


class ContactWrenchSolver:
    """Solve the particle-body contact impulses of one step and return the wrench on every collider.

    One instance handles all colliders of all environments. Call :meth:`capture` before the particle step and
    :meth:`evaluate` after it. Candidate-contact detection is not done here but in the kernels that
    :meth:`evaluate` launches.
    """

    def __init__(
        self,
        model: newton.Model,
        body_ids: list[int],
        *,
        colliders_per_world: int,
        particles_per_env: int,
        grid: wp.HashGrid,
        iterations: int = 200,
        effective_mass_scale: float = 10.0,
        tangent_mass_scale: float = 10.0,
        horizontal_mass_scale: float = 1.0,
        upward_mass_scale: float = 1.0,
        normal_power: float = 1.0,
        relaxation: float = 0.02,
        gradient_radius: float = 0.065,
        gradient_contact_band: float = 0.08,
        stabilization: float = 0.01,
        contact_offset: float = 0.0,
    ):
        """Initialize the solver.

        Args:
            model: Newton model that holds the particles and the collider bodies.
            body_ids: Collider body indices, environment by environment.
            colliders_per_world: Number of colliders in every environment.
            particles_per_env: Number of particles in every environment.
            grid: Hash grid (with groups) used for the neighbour search among the evaluated particles.
            iterations: Maximum number of Jacobi iterations.
            effective_mass_scale: Effective particle mass, as a multiple of the particle mass, for a contact normal
                that points down.
            tangent_mass_scale: Effective particle mass multiple for friction.
            horizontal_mass_scale: Effective particle mass multiple for a horizontal contact normal.
            upward_mass_scale: Effective particle mass multiple for a contact normal that points up.
            normal_power: Exponent of the blend between the normal-direction multiples.
            relaxation: Jacobi relaxation factor.
            gradient_radius: Neighbour radius of the least-squares velocity gradient [m].
            gradient_contact_band: Distance from the collider surface within which a particle is a contact
                candidate [m].
            stabilization: Fraction of the penetration that is removed per step.
            contact_offset: Extra margin added to the collider surface [m].
        """
        self.model = model
        self.device = model.device
        self.colliders_per_world = int(colliders_per_world)
        self.particles_per_env = int(particles_per_env)
        self.grid = grid
        self.iterations = int(iterations)
        self.effective_mass_scale = float(effective_mass_scale)
        self.horizontal_mass_scale = float(horizontal_mass_scale)
        self.upward_mass_scale = float(upward_mass_scale)
        self.tangent_mass_scale = float(tangent_mass_scale)
        self.normal_power = float(normal_power)
        self.relaxation = float(relaxation)
        self.gradient_radius = float(gradient_radius)
        self.contact_band = float(gradient_contact_band)
        self.stabilization = float(stabilization)
        self.dt = 0.005
        self._captured = False
        self._problem_size = None

        shape_body = model.shape_body.numpy()
        types = model.shape_type.numpy()
        flags = model.shape_flags.numpy()
        scales = model.shape_scale.numpy()
        margins = model.shape_margin.numpy()
        shape_friction = model.shape_material_mu.numpy()
        masses = model.body_mass.numpy()
        collides = (flags & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0

        with wp.ScopedDevice(self.device):
            self._particle_q = wp.empty(model.particle_count, dtype=wp.vec3)
            self._body_q = wp.empty(model.body_count, dtype=wp.transform)
            self._body_qd = wp.empty(model.body_count, dtype=wp.spatial_vector)
            self.wrench = wp.zeros(len(body_ids), dtype=wp.spatial_vector)
            """Wrench on every collider [N, N m], in the order of ``body_ids``."""
            self.captured_q = self._particle_q
            """Particle positions before the step."""

            self.colliders = []
            for body in (int(b) for b in body_ids):
                # The largest primitive shape of the body, preferring shapes that collide with particles
                selected = [s for s in np.flatnonzero((shape_body == body) & collides) if int(types[s]) in _SHAPE_CODES]
                if not selected:
                    selected = [s for s in np.flatnonzero(shape_body == body) if int(types[s]) in _SHAPE_CODES]
                if not selected:
                    raise ValueError(f"Collider body {body} has no box, sphere, cylinder or capsule shape")
                shape = int(max(selected, key=lambda s: float(np.max(scales[s]))))
                code = _SHAPE_CODES[int(types[shape])]
                margin = float(margins[shape]) + float(contact_offset)
                scale = scales[shape]
                radius = float(scale[0])
                half_height = float(scale[1]) if float(scale[1]) > 0.0 else float(scale[2])
                if code == 1:
                    half = np.array([radius, radius, radius], dtype=np.float32)
                elif code == 2:
                    half = np.array([radius, radius, half_height], dtype=np.float32)
                elif code == 4:
                    half = np.array([radius, half_height, radius], dtype=np.float32)
                else:
                    half = np.asarray(scale, dtype=np.float32)
                self.colliders.append(
                    {
                        "body": body,
                        "shape": shape,
                        "code": code,
                        "mass": float(masses[body]) if float(masses[body]) > 0 else float("inf"),
                        "friction": float(shape_friction[shape]),
                        "surface_pose": wp.empty(1, dtype=wp.transform),
                        "sizes": wp.array([half * 2.0 + 2.0 * margin], dtype=wp.vec3),
                        "free": wp.empty(2, dtype=wp.vec3d),
                        "initial": wp.empty(2, dtype=wp.vec3d),
                        "inverse": wp.empty(1, dtype=wp.mat33d),
                        "stats": wp.zeros(3, dtype=wp.float64),
                    }
                )

    def refresh_body_inertia(self) -> None:
        """Read the collider masses again after the model changed."""
        masses = self.model.body_mass.numpy()
        for collider in self.colliders:
            mass = float(masses[collider["body"]])
            collider["mass"] = mass if mass > 0 else float("inf")
        self._problem_size = None

    def _allocate(self, count: int) -> None:
        """Allocate the contact problems for ``count`` evaluated particles per environment."""
        if self._problem_size == count:
            return
        num_colliders = len(self.colliders)
        num_envs = max(1, num_colliders // self.colliders_per_world)
        total_particles = num_envs * count
        total = num_colliders * count
        with wp.ScopedDevice(self.device):
            self._x = wp.empty(total_particles, dtype=wp.vec3)
            self._v = wp.empty(total_particles, dtype=wp.vec3)
            self._mass = wp.empty(total_particles, dtype=float)
            self._groups = wp.array(np.repeat(np.arange(num_envs, dtype=np.int32), count), dtype=wp.int32)
            self._flags = wp.empty(total, dtype=int)
            self._scan = wp.empty(total, dtype=int)
            offsets = wp.empty(total, dtype=int)
            ids = wp.empty(total, dtype=int)
            frames = wp.empty(total, dtype=wp.mat33d)
            angular = wp.empty(total, dtype=wp.mat33d)
            response = wp.empty(total, dtype=wp.mat33d)
            compliance = wp.empty(total, dtype=wp.mat33d)
            slack = wp.empty(total, dtype=wp.vec2d)
            velocity = wp.empty(total, dtype=wp.vec3d)
            gap = wp.empty(total, dtype=wp.float64)
            impulses = wp.empty(total, dtype=wp.vec3d)
            self.grid.reserve(total_particles, with_groups=True)

            problems = []
            for index, collider in enumerate(self.colliders):
                env = index // self.colliders_per_world
                s, e = index * count, (index + 1) * count
                p = ContactProblem()
                p.body = collider["body"]
                p.shape = collider["shape"]
                p.env_idx = env
                p.count = count
                p.shape_code = collider["code"]
                p.mass = collider["mass"]
                p.friction = collider["friction"]
                p.surface_pose = collider["surface_pose"]
                p.sizes = collider["sizes"]
                p.free = collider["free"]
                p.initial = collider["initial"]
                p.inverse = collider["inverse"]
                p.offsets = offsets[s:e]
                p.ids = ids[s:e]
                p.frames = frames[s:e]
                p.angular = angular[s:e]
                p.response = response[s:e]
                p.compliance = compliance[s:e]
                p.particle_slack = slack[s:e]
                p.contact_velocity = velocity[s:e]
                p.gap = gap[s:e]
                p.particle_mass = self._mass[env * count : (env + 1) * count]
                p.impulses = impulses[s:e]
                p.stats = collider["stats"]
                problems.append(p)
            self._problems = wp.array(problems, dtype=ContactProblem)
            # The problems hold views of these arrays but do not keep them alive, so the solver must.
            self._buffers = (offsets, ids, frames, angular, response, compliance, slack, velocity, gap, impulses)
        self._problem_size = count

    def capture(self, state: newton.State, dt: float, copy_particles: bool = True) -> None:
        """Record the particle positions and the body state before the particle step.

        Args:
            state: State before the step.
            dt: Time step [s].
            copy_particles: Copy the particle positions. Needed when the step writes into the same array.
        """
        self.dt = float(dt)
        if copy_particles:
            wp.copy(self._particle_q, state.particle_q)
            self.captured_q = self._particle_q
        else:
            self.captured_q = state.particle_q
        wp.copy(self._body_q, state.body_q)
        wp.copy(self._body_qd, state.body_qd)
        self._captured = True

    def evaluate(self, state_out: newton.State, active_idx: torch.Tensor | None = None) -> wp.array:
        """Solve for the wrenches of the step that produced ``state_out``.

        Args:
            state_out: State after the particle step; its particle velocity is the velocity of the step.
            active_idx: Indices [E, K] of the particles that moved, within their environment. All particles
                are evaluated if None.

        Returns:
            :attr:`wrench`.
        """
        if not self._captured:
            raise RuntimeError("capture must precede evaluate for every step")
        num_colliders = len(self.colliders)
        num_envs = max(1, num_colliders // self.colliders_per_world)
        count = self.particles_per_env if active_idx is None else int(active_idx.shape[1])
        self._allocate(count)

        if active_idx is not None:
            wp.launch(
                gather_active_particles,
                (num_envs, count),
                [
                    wp.from_torch(active_idx.int().contiguous().view(-1), dtype=wp.int32),
                    count,
                    self.particles_per_env,
                    self.captured_q,
                    state_out.particle_qd,
                    self.model.particle_mass,
                    self._x,
                    self._v,
                    self._mass,
                ],
                device=self.device,
            )
            x_all, v_all = self._x, self._v
        else:
            total = num_envs * count
            x_all = self.captured_q[:total]
            v_all = state_out.particle_qd[:total]
            wp.copy(self._mass, self.model.particle_mass, count=total)

        self.grid.build(x_all, self.gradient_radius, groups=self._groups)
        wp.launch(
            prepare_contact_bodies,
            num_colliders,
            [self._problems, self._body_q, self._body_qd, self.model.body_inertia, self.model.shape_transform],
            device=self.device,
        )
        wp.launch(
            prepare_contacts,
            (num_colliders, count),
            [
                self._problems,
                self.grid.id,
                x_all,
                v_all,
                self.gradient_radius,
                self.dt,
                self.contact_band,
                self.effective_mass_scale,
                self.horizontal_mass_scale,
                self.upward_mass_scale,
                self.tangent_mass_scale,
                self.normal_power,
                self.stabilization,
                self._flags,
            ],
            device=self.device,
        )
        wp.utils.array_scan(self._flags, self._scan, inclusive=True)
        wp.launch(
            compact_contacts, (num_colliders, count), [self._problems, self._flags, self._scan], device=self.device
        )
        wp.launch(initialize_contacts, (num_colliders, 128), [self._problems], device=self.device, block_dim=128)
        wp.launch(
            solve_contacts,
            (num_colliders, 128),
            [self._problems, self.dt, self.iterations, wp.float64(1.0e-8), self.relaxation, self.wrench],
            device=self.device,
            block_dim=128,
        )
        self._captured = False
        return self.wrench
