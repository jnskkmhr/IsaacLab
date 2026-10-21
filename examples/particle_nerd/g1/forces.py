# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU contact reconstruction matching Newton's independent virtual foot proxies.

The robot solver has already integrated actuation, gravity and joint dynamics.
Capture the MPM input AFTER coupling_rewind_proxy_body. Its body velocity u_free
is the virtual proxy's contact-free starting velocity; do not integrate it again.
For each particle/contact, J maps COM linear/angular velocity to contact velocity:

    u = u_free + M_proxy_inverse @ sum(J.T @ impulse)
    relative_velocity = J @ u - frame @ v_particle_next + impulse / effective_mass
    effective_mass = effective_mass_scale * particle_mass

Normal impulses are unilateral and tangent impulses satisfy Coulomb friction.
Force is sum(world_impulse)/dt; torque is sum((point-COM) cross impulse)/dt.
The local affine fit extrapolates next particle velocities to the foot surface.
The effective particle mass is a fixed approximation, not a constitutive law.

This reconstructs contact loads from particle NeRD next-step particle motion.
It is NOT the full articulated J M^-1 J.T solve and does not predict particle motion.
All per-step computation stays on the GPU; diagnostics() explicitly copies to CPU.
"""

import newton
import numpy as np
import warp as wp

from particle_nerd.force_estimation import kernels


@wp.kernel
def _prepare_proxy(
    poses: wp.array(dtype=wp.transform),
    velocities: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inertia: wp.array(dtype=wp.mat33),
    shape_transform: wp.array(dtype=wp.transform),
    body: int,
    shape: int,
    surface_pose: wp.array(dtype=wp.transform),
    center: wp.array(dtype=wp.vec3d),
    free: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.mat33d),
):
    surface_pose[0] = poses[body] * shape_transform[shape]
    center[0] = wp.vec3d(wp.transform_point(poses[body], body_com[body]))
    rotation = wp.quat_to_matrix(wp.normalize(wp.quatd(wp.transform_get_rotation(poses[body]))))
    inverse[0] = wp.inverse(rotation * wp.mat33d(body_inertia[body]) * wp.transpose(rotation))
    free[0] = wp.vec3d(wp.spatial_top(velocities[body]))
    free[1] = wp.vec3d(wp.spatial_bottom(velocities[body]))


@wp.kernel
def _prepare_contacts(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    gradients: wp.array(dtype=wp.mat33d),
    surface_pose: wp.array(dtype=wp.transform),
    sizes: wp.array(dtype=wp.vec3),
    center: wp.array(dtype=wp.vec3d),
    free: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.mat33d),
    mass: float,
    particle_mass: wp.array(dtype=float),
    dt: float,
    effective_scale: float,
    stabilization: float,
    margin: float,
    flags: wp.array(dtype=int),
    frames: wp.array(dtype=wp.mat33d),
    angular_jacobian: wp.array(dtype=wp.mat33d),
    response: wp.array(dtype=wp.mat33d),
    compliance: wp.array(dtype=wp.mat33d),
    material: wp.array(dtype=wp.vec3d),
    gap_speed: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    distance, normal, point = kernels.geometry(wp.vec3d(x[i]), surface_pose[0], sizes[0], 0)
    distance -= wp.float64(margin)
    # Newton subtracts margin from distance but keeps the physical closest point.
    velocity = wp.vec3d(v[i]) + wp.transpose(gradients[i]) * (point - wp.vec3d(x[i]))
    arm = point - center[0]
    surface = free[0] + wp.cross(free[1], arm)
    approach = wp.dot(normal, surface - velocity)
    flags[i] = 0
    if distance <= wp.max(approach, wp.float64(0.0)) * wp.float64(dt):
        flags[i] = 1
        n = -normal
        axis = int(0)
        if wp.abs(n[1]) < wp.abs(n[axis]):
            axis = 1
        if wp.abs(n[2]) < wp.abs(n[axis]):
            axis = 2
        basis = wp.vec3d(0.0)
        basis[axis] = wp.float64(1.0)
        tangent = wp.normalize(wp.cross(n, basis))
        tangent2 = wp.cross(n, tangent)
        frame = wp.mat33d(n[0], n[1], n[2], tangent[0], tangent[1], tangent[2], tangent2[0], tangent2[1], tangent2[2])
        a0 = wp.cross(arm, n)
        a1 = wp.cross(arm, tangent)
        a2 = wp.cross(arm, tangent2)
        angular = wp.mat33d(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2])
        respond = inverse[0] * wp.transpose(angular)
        k = frame * wp.transpose(frame) / wp.float64(mass) + angular * respond
        slack = wp.float64(1.0) / (wp.float64(particle_mass[i]) * wp.float64(effective_scale))
        k += wp.mat33d(slack, 0.0, 0.0, 0.0, slack, 0.0, 0.0, 0.0, slack)
        frames[i] = frame
        angular_jacobian[i] = angular
        response[i] = respond
        compliance[i] = k
        material[i] = frame * velocity
        gap_speed[i] = (
            wp.max(distance, wp.float64(0.0)) + wp.float64(stabilization) * wp.min(distance, wp.float64(0.0))
        ) / wp.float64(dt)


class G1ProxyEstimator:
    """Reconstruct material-on-foot COM wrenches without modifying simulation state.

    Only one particle-colliding box per selected foot is currently supported.
    Unsupported geometry fails explicitly; visual meshes are never substituted.
    """

    def __init__(
        self,
        mpm_solver,
        proxy_body_ids,
        *,
        iterations=50,
        effective_mass_scale=10.0,
        friction=None,
        gradient_radius=0.065,
        gradient_contact_band=0.08,
        stabilization=0.01,
        contact_offset=0.0,
    ):
        if iterations < 1 or effective_mass_scale <= 0 or gradient_radius <= 0 or gradient_contact_band <= 0:
            raise ValueError("Iterations, effective mass and gradient distances must be positive")
        if contact_offset < 0 or not np.isfinite(contact_offset):
            raise ValueError("Contact offset must be finite and nonnegative")
        self.contact_offset = float(contact_offset)
        self.model = mpm_solver.model
        self.body_ids = [int(body) for body in proxy_body_ids]
        if not self.body_ids or len(set(self.body_ids)) != len(self.body_ids):
            raise ValueError("Provide distinct proxy body indices")
        self.iteration_limit = int(iterations)
        self.effective_scale = float(effective_mass_scale)
        self.gradient_radius = float(gradient_radius)
        self.contact_band = float(gradient_contact_band)
        self.stabilization = float(stabilization)
        self.n = self.model.particle_count
        self.device = self.model.device
        self.dt = None
        self.captured = False
        shape_body = self.model.shape_body.numpy()
        flags = self.model.shape_flags.numpy()
        types = self.model.shape_type.numpy()
        scales = self.model.shape_scale.numpy()
        masses = self.model.body_mass.numpy()
        particle_masses = self.model.particle_mass.numpy()
        if self.n == 0 or np.any(particle_masses <= 0):
            raise ValueError("Estimator requires positive particle masses")
        shape_friction = self.model.shape_material_mu.numpy()
        margins = self.model.shape_margin.numpy()
        self.metadata = {
            "method": "independent_virtual_foot_proxy",
            "iterations": self.iteration_limit,
            "effective_mass_scale": self.effective_scale,
            "gradient_radius_m": self.gradient_radius,
            "gradient_contact_band_m": self.contact_band,
            "stabilization": self.stabilization,
            "additional_contact_offset_m": self.contact_offset,
            "surface_velocity_approximation": "instantaneous COM twist; MPM uses forward finite rotation",
            "input_velocity": "MPM proxy input after lagged-wrench rewind; no additional gravity or actuation",
            "particle_velocity": "particle NeRD predicted next velocity",
            "feet": [],
        }
        with wp.ScopedDevice(self.device):
            self.x = wp.empty(self.n, dtype=wp.vec3)
            self.input_pose = wp.empty(self.model.body_count, dtype=wp.transform)
            self.input_velocity = wp.empty(self.model.body_count, dtype=wp.spatial_vector)
            self.wrench = wp.zeros(len(self.body_ids), dtype=wp.spatial_vector)
            self.grid = wp.HashGrid(128, 128, 128)
            self.grid.reserve(self.n)
            self.feet = []
            for body in self.body_ids:
                selected = np.flatnonzero(
                    (shape_body == body) & ((flags & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0)
                )
                if len(selected) != 1 or types[selected[0]] != int(newton.GeoType.BOX):
                    raise ValueError(f"Foot body {body}: expected one collision box; got shapes {selected.tolist()}")
                shape = int(selected[0])
                if masses[body] <= 0:
                    raise ValueError(f"Foot body {body} has nonpositive virtual mass")
                mu = float(shape_friction[shape] if friction is None else friction)
                if mu < 0 or not np.isfinite(mu):
                    raise ValueError("Contact friction must be finite and nonnegative")
                foot = {
                    "body": body,
                    "shape": shape,
                    "mass": float(masses[body]),
                    "friction": mu,
                    "margin": float(margins[shape]) + self.contact_offset,
                }
                foot["surface_pose"] = wp.empty(1, dtype=wp.transform)
                foot["sizes"] = wp.array([scales[shape] * 2], dtype=wp.vec3)
                foot["shapes"] = wp.zeros(1, dtype=int)
                foot["center"] = wp.empty(1, dtype=wp.vec3d)
                foot["free"] = wp.empty(2, dtype=wp.vec3d)
                foot["inverse"] = wp.empty(1, dtype=wp.mat33d)
                for name in ("gradients", "frames", "angular", "response", "compliance"):
                    foot[name] = wp.empty(self.n, dtype=wp.mat33d)
                for name in ("flags", "offsets", "ids"):
                    foot[name] = wp.empty(self.n, dtype=int)
                for name in ("material", "impulses"):
                    foot[name] = wp.empty(self.n, dtype=wp.vec3d)
                foot["gap"] = wp.empty(self.n, dtype=wp.float64)
                foot["stats"] = wp.zeros(3, dtype=wp.float64)
                self.feet.append(foot)
                self.metadata["feet"].append(
                    {
                        "body_id": body,
                        "shape_id": shape,
                        "virtual_mass_kg": float(masses[body]),
                        "virtual_inertia_local": self.model.body_inertia.numpy()[body].tolist(),
                        "body_com_local_m": self.model.body_com.numpy()[body].tolist(),
                        "shape_half_extents_m": scales[shape].tolist(),
                        "shape_local_transform": self.model.shape_transform.numpy()[shape].tolist(),
                        "shape_margin_m": float(margins[shape]),
                        "contact_friction": mu,
                    }
                )

    def capture(self, state, dt):
        """Copy the input immediately before the MPM step, after proxy rewind."""
        if dt <= 0:
            raise ValueError("Timestep must be positive")
        self.dt = float(dt)
        wp.copy(self.x, state.particle_q)
        wp.copy(self.input_pose, state.body_q)
        wp.copy(self.input_velocity, state.body_qd)
        self.captured = True

    def evaluate(self, state_out):
        """Use predicted next particle velocities; compute forces on the GPU."""
        if not self.captured:
            raise RuntimeError("capture must precede evaluate for every step")
        self.grid.build(self.x, self.gradient_radius)
        self.wrench.zero_()
        for index, f in enumerate(self.feet):
            wp.launch(
                _prepare_proxy,
                1,
                [
                    self.input_pose,
                    self.input_velocity,
                    self.model.body_com,
                    self.model.body_inertia,
                    self.model.shape_transform,
                    f["body"],
                    f["shape"],
                    f["surface_pose"],
                    f["center"],
                    f["free"],
                    f["inverse"],
                ],
                device=self.device,
            )
            wp.launch(
                kernels.fit_gradients,
                self.n,
                [
                    self.grid.id,
                    self.x,
                    state_out.particle_qd,
                    self.gradient_radius,
                    f["gradients"],
                    f["surface_pose"],
                    f["sizes"],
                    f["shapes"],
                    0,
                    self.contact_band,
                ],
                device=self.device,
            )
            wp.launch(
                _prepare_contacts,
                self.n,
                [
                    self.x,
                    state_out.particle_qd,
                    f["gradients"],
                    f["surface_pose"],
                    f["sizes"],
                    f["center"],
                    f["free"],
                    f["inverse"],
                    f["mass"],
                    self.model.particle_mass,
                    self.dt,
                    self.effective_scale,
                    self.stabilization,
                    f["margin"],
                    f["flags"],
                    f["frames"],
                    f["angular"],
                    f["response"],
                    f["compliance"],
                    f["material"],
                    f["gap"],
                ],
                device=self.device,
            )
            wp.utils.array_scan(f["flags"], f["offsets"], inclusive=True)
            wp.launch(kernels.compact_contacts, self.n, [f["flags"], f["offsets"], f["ids"]], device=self.device)
            f["impulses"].zero_()
            wp.launch(
                kernels.solve_contacts_parallel,
                128,
                [
                    f["offsets"],
                    f["ids"],
                    f["frames"],
                    f["angular"],
                    f["response"],
                    f["compliance"],
                    f["material"],
                    f["gap"],
                    self.model.particle_mass,
                    f["free"],
                    f["mass"],
                    self.effective_scale,
                    f["friction"],
                    self.dt,
                    self.iteration_limit,
                    wp.float64(1.0e-9),
                    f["impulses"],
                    self.wrench,
                    index,
                    f["stats"],
                    False,
                    0.1,
                ],
                device=self.device,
                block_dim=128,
            )
        self.captured = False
        return self.wrench

    def diagnostics(self):
        """Return host arrays for logging, outside the estimator compute path."""
        stats = np.stack([f["stats"].numpy() for f in self.feet])
        wrench = self.wrench.numpy()
        if not np.isfinite(wrench).all() or not np.isfinite(stats).all():
            raise FloatingPointError("Nonfinite reconstructed foot wrench or solver statistics")
        return {
            "body_wrench": wrench,
            "free_body_pose": self.input_pose.numpy()[self.body_ids],
            "free_body_velocity": self.input_velocity.numpy()[self.body_ids],
            "contact_count": stats[:, 0].astype(np.int32),
            "iterations": stats[:, 1].astype(np.int32),
            "residual": stats[:, 2],
        }
