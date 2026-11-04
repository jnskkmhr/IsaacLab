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
"""

import newton
import numpy as np
import warp as wp

from particle_nerd.force_estimation import kernels
from particle_nerd.force_estimation.body_velocity import (
    _contact_response,
    _linear_solve,
    _rft_mobilities,
    initialize_body_velocity_solve,
    mat36d,
    mat63d,
    mat66d,
    vec6d,
)

wp.set_module_options({"enable_backward": False, "max_unroll": 0})


@wp.kernel
def _gather_active_particles(
    idx: wp.array(dtype=int),
    p_start: int,
    x_all: wp.array(dtype=wp.vec3),
    v_all: wp.array(dtype=wp.vec3),
    m_all: wp.array(dtype=float),
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
    m_out: wp.array(dtype=float),
):
    k = wp.tid()
    g = p_start + idx[k]
    x_out[k] = x_all[g]
    v_out[k] = v_all[g]
    m_out[k] = m_all[g]


@wp.kernel
def _prepare_proxy(
    poses: wp.array(dtype=wp.transform),
    velocities: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inertia: wp.array(dtype=wp.mat33),
    shape_transform: wp.array(dtype=wp.transform),
    body: int,
    shape: int,
    count_slot: int,
    offsets: wp.array(dtype=int),
    surface_pose: wp.array(dtype=wp.transform),
    center: wp.array(dtype=wp.vec3d),
    free: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.mat33d),
):
    offsets[count_slot] = 0
    surface_pose[0] = poses[body] * shape_transform[shape]
    center[0] = wp.vec3d(wp.transform_point(poses[body], body_com[body]))
    rotation = wp.quat_to_matrix(wp.normalize(wp.quatd(wp.transform_get_rotation(poses[body]))))
    inverse[0] = wp.inverse(rotation * wp.mat33d(body_inertia[body]) * wp.transpose(rotation))
    free[0] = wp.vec3d(wp.spatial_top(velocities[body]))
    free[1] = wp.vec3d(wp.spatial_bottom(velocities[body]))


@wp.kernel
def _fit_gradients_culled(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: float,
    output: wp.array(dtype=wp.mat33d),
    poses: wp.array(dtype=wp.transform),
    sizes: wp.array(dtype=wp.vec3),
    center: wp.array(dtype=wp.vec3d),
    free: wp.array(dtype=wp.vec3d),
    margin: float,
    dt: float,
    contact_band: float,
):
    i = wp.tid()
    raw_dist, normal, point = kernels.geometry(wp.vec3d(x[i]), poses[0], sizes[0], 0)
    if contact_band > 0.0 and raw_dist > wp.float64(contact_band):
        output[i] = wp.mat33d(0.0)
        return
    distance = raw_dist - wp.float64(margin)
    arm = point - center[0]
    surface = free[0] + wp.cross(free[1], arm)
    app0 = wp.dot(normal, surface - wp.vec3d(v[i]))
    if distance > (wp.max(app0, wp.float64(0.0)) + wp.float64(4.0)) * wp.float64(dt):
        output[i] = wp.mat33d(0.0)
        return
    lhs = wp.mat33(0.0)
    rhs = wp.mat33(0.0)
    xi = x[i]
    vi = v[i]
    query = wp.hash_grid_query(grid, xi, radius + 1.0e-6)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        dx = wp.vec3(x[j]) - wp.vec3(xi)
        dv = wp.vec3(v[j]) - wp.vec3(vi)
        q = wp.length(dx) / float(radius)
        if q < float(1.0):
            a = float(1.0) - q
            weight = a * a * a * a * (float(1.0) + float(4.0) * q)
            lhs += weight * wp.outer(dx, dx)
            rhs += weight * wp.outer(dx, dv)
    ridge = float(0.01) * wp.trace(lhs) / float(3.0) + float(1.0e-12)
    lhs += wp.mat33(ridge, 0.0, 0.0, 0.0, ridge, 0.0, 0.0, 0.0, ridge)
    output[i] = wp.mat33d(wp.inverse(lhs) * rhs)


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
    horiz_scale: float,
    up_scale: float,
    tangent_scale: float,
    normal_power: float,
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
        nz = n[2]
        up_part = wp.pow(wp.max(nz, wp.float64(0.0)), wp.float64(normal_power))
        down_part = wp.pow(wp.max(-nz, wp.float64(0.0)), wp.float64(normal_power))
        horiz_part = wp.max(wp.float64(0.0), wp.float64(1.0) - up_part - down_part)
        eff_n = wp.max(
            wp.float64(effective_scale) * up_part
            + wp.float64(horiz_scale) * horiz_part
            + wp.float64(up_scale) * down_part,
            wp.float64(0.5),
        )
        eff_t = wp.max(wp.float64(tangent_scale), wp.float64(0.5))
        pm = wp.float64(particle_mass[i])
        slack_n = wp.float64(1.0) / (pm * eff_n)
        slack_t = wp.float64(1.0) / (pm * eff_t)
        k += wp.mat33d(slack_n, 0.0, 0.0, 0.0, slack_t, 0.0, 0.0, 0.0, slack_t)
        frames[i] = frame
        angular_jacobian[i] = angular
        response[i] = respond
        compliance[i] = k
        material[i] = frame * velocity
        gap_speed[i] = (
            wp.max(distance, wp.float64(0.0)) + wp.float64(stabilization) * wp.min(distance, wp.float64(0.0))
        ) / wp.float64(dt)


@wp.kernel
def _solve_rft_body_velocity_wrench(
    offsets: wp.array(dtype=int),
    ids: wp.array(dtype=int),
    frames: wp.array(dtype=wp.mat33d),
    angular: wp.array(dtype=wp.mat33d),
    response: wp.array(dtype=wp.mat33d),
    material: wp.array(dtype=wp.vec3d),
    gap: wp.array(dtype=wp.float64),
    particle_mass: wp.array(dtype=float),
    free: wp.array(dtype=wp.vec3d),
    mass: float,
    effective_scale: float,
    horiz_scale: float,
    up_scale: float,
    tangent_scale: float,
    normal_power: float,
    friction: float,
    dt: float,
    impulses: wp.array(dtype=wp.vec3d),
    wrench: wp.array(dtype=wp.spatial_vector),
    body: int,
    stats: wp.array(dtype=wp.float64),
):
    lane = wp.tid()
    count = offsets[offsets.shape[0] - 1]
    if count <= 0:
        if lane == 0:
            wrench[body] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
            stats[0] = wp.float64(0.0)
            stats[1] = wp.float64(0.0)
            stats[2] = wp.float64(0.0)
        return
    origin = vec6d(free[0][0], free[0][1], free[0][2], free[1][0], free[1][1], free[1][2])
    q = origin
    iterations = int(0)
    error = wp.float64(0.0)
    for it in range(15):
        local_force = vec6d(0.0)
        local_jacobian = mat66d(0.0)
        k = lane
        while k < count:
            i = ids[k]
            mob_n, mob_t = _rft_mobilities(
                frames[i], particle_mass[i], effective_scale, horiz_scale, up_scale, tangent_scale, normal_power
            )
            impulse, derivative = _contact_response(
                q, frames[i], angular[i], material[i], gap[i], mob_n, mob_t, wp.float64(friction)
            )
            j = mat36d(0.0)
            b = mat63d(0.0)
            for r in range(3):
                for c in range(3):
                    j[r, c] = frames[i][r, c]
                    j[r, c + 3] = angular[i][r, c]
                    b[c, r] = frames[i][r, c] / wp.float64(mass)
                    b[c + 3, r] = response[i][c, r]
            local_force += b * impulse
            local_jacobian += b * derivative * j
            k += 32
        tf = wp.tile_sum(wp.tile(local_force), axis=1)
        tj = wp.tile_sum(wp.tile(local_jacobian), axis=2)
        force = vec6d(0.0)
        jacobian = mat66d(0.0)
        for r in range(6):
            force[r] = wp.tile_extract(tf, r)
            for c in range(6):
                jacobian[r, c] = -wp.tile_extract(tj, r, c)
            jacobian[r, r] += wp.float64(1.0)
        residual = q - origin - force
        error = wp.length(residual)
        iterations = it + 1
        if error < wp.float64(1e-10):
            break
        delta = _linear_solve(jacobian, residual)
        alpha = wp.float64(1.0)
        accepted = bool(False)
        for _trial in range(10):
            candidate = q - alpha * delta
            local_candidate = vec6d(0.0)
            k = lane
            while k < count:
                i = ids[k]
                mob_n, mob_t = _rft_mobilities(
                    frames[i], particle_mass[i], effective_scale, horiz_scale, up_scale, tangent_scale, normal_power
                )
                impulse, _unused = _contact_response(
                    candidate, frames[i], angular[i], material[i], gap[i], mob_n, mob_t, wp.float64(friction)
                )
                lin = wp.transpose(frames[i]) * impulse / wp.float64(mass)
                sp = response[i] * impulse
                local_candidate += vec6d(lin[0], lin[1], lin[2], sp[0], sp[1], sp[2])
                k += 32
            tc = wp.tile_sum(wp.tile(local_candidate), axis=1)
            candidate_force = vec6d(0.0)
            for r in range(6):
                candidate_force[r] = wp.tile_extract(tc, r)
            new_error = wp.length(candidate - origin - candidate_force)
            if new_error < error * (wp.float64(1.0) - wp.float64(1e-4) * alpha):
                q = candidate
                accepted = True
                break
            alpha *= wp.float64(0.5)
        if not accepted:
            break

    local_total = wp.vec3d(0.0)
    local_torque = wp.vec3d(0.0)
    k = lane
    while k < count:
        i = ids[k]
        mob_n, mob_t = _rft_mobilities(
            frames[i], particle_mass[i], effective_scale, horiz_scale, up_scale, tangent_scale, normal_power
        )
        impulse, _unused = _contact_response(
            q, frames[i], angular[i], material[i], gap[i], mob_n, mob_t, wp.float64(friction)
        )
        impulses[i] = impulse
        local_total += wp.transpose(frames[i]) * impulse
        local_torque += wp.transpose(angular[i]) * impulse
        k += 32
    ttot = wp.tile_sum(wp.tile(local_total), axis=1)
    ttor = wp.tile_sum(wp.tile(local_torque), axis=1)
    total = wp.vec3d(wp.tile_extract(ttot, 0), wp.tile_extract(ttot, 1), wp.tile_extract(ttot, 2))
    torque = wp.vec3d(wp.tile_extract(ttor, 0), wp.tile_extract(ttor, 1), wp.tile_extract(ttor, 2))
    if lane == 0:
        wrench[body] = wp.spatial_vector(wp.vec3(total / wp.float64(dt)), wp.vec3(torque / wp.float64(dt)))
        stats[0] = wp.float64(count)
        stats[1] = wp.float64(iterations)
        stats[2] = error


class G1ProxyEstimator:
    """Reconstruct material-on-foot COM wrenches without modifying simulation state."""

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
        relaxation=0.1,
        contact_solver="jacobi",
        horiz_mass_scale=None,
        up_mass_scale=None,
        tangent_mass_scale=None,
        normal_power=None,
        particles_per_env=None,
    ):
        if iterations < 1 or effective_mass_scale <= 0 or gradient_radius <= 0 or gradient_contact_band <= 0:
            raise ValueError("Iterations, effective mass and gradient distances must be positive")
        if contact_offset < 0 or not np.isfinite(contact_offset):
            raise ValueError("Contact offset must be finite and nonnegative")
        if not np.isfinite(relaxation) or not 0 < relaxation <= 1:
            raise ValueError("Contact relaxation must be finite and in (0, 1]")
        if contact_solver not in ("jacobi", "body_velocity", "rft_body_velocity"):
            raise ValueError("contact_solver must be jacobi, body_velocity, or rft_body_velocity")
        self.contact_solver = contact_solver
        self.use_body_velocity = contact_solver in ("body_velocity", "rft_body_velocity")
        self.relaxation = float(relaxation)
        self.contact_offset = float(contact_offset)
        self.model = mpm_solver.model
        self.body_ids = [int(body) for body in proxy_body_ids]
        if not self.body_ids or len(set(self.body_ids)) != len(self.body_ids):
            raise ValueError("Provide distinct proxy body indices")
        self.iteration_limit = int(iterations)
        self.effective_scale = float(effective_mass_scale)
        rft_default = contact_solver == "rft_body_velocity"
        self.horiz_scale = float(
            horiz_mass_scale if horiz_mass_scale is not None else (8.0 if rft_default else self.effective_scale)
        )
        self.up_scale = float(
            up_mass_scale if up_mass_scale is not None else (2.0 if rft_default else self.effective_scale)
        )
        self.tangent_scale = float(
            tangent_mass_scale if tangent_mass_scale is not None else (40.0 if rft_default else self.effective_scale)
        )
        self.normal_power = float(normal_power if normal_power is not None else (3.0 if rft_default else 1.0))
        self.gradient_radius = float(gradient_radius)
        self.contact_band = float(gradient_contact_band)
        self.stabilization = float(stabilization)
        self.n = self.model.particle_count
        self.particles_per_env = int(particles_per_env) if particles_per_env is not None else self.n
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
            "jacobi_relaxation": self.relaxation,
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
            self.x_active = wp.empty(self.particles_per_env, dtype=wp.vec3)
            self.v_active = wp.empty(self.particles_per_env, dtype=wp.vec3)
            self.m_active = wp.empty(self.particles_per_env, dtype=float)
            self.input_pose = wp.empty(self.model.body_count, dtype=wp.transform)
            self.input_velocity = wp.empty(self.model.body_count, dtype=wp.spatial_vector)
            self.wrench = wp.zeros(len(self.body_ids), dtype=wp.spatial_vector)
            self.grid = wp.HashGrid(128, 128, 128)
            self.grid.reserve(self.particles_per_env)
            shared_bufs = {}
            for name in ("gradients", "frames", "angular", "response", "compliance"):
                shared_bufs[name] = wp.empty(self.particles_per_env, dtype=wp.mat33d)
            for name in ("flags", "offsets", "ids"):
                shared_bufs[name] = wp.empty(self.particles_per_env, dtype=int)
            for name in ("material", "impulses"):
                shared_bufs[name] = wp.empty(self.particles_per_env, dtype=wp.vec3d)
            shared_bufs["gap"] = wp.empty(self.particles_per_env, dtype=wp.float64)
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
                foot.update(shared_bufs)
                foot["stats"] = wp.zeros(3, dtype=wp.float64)
                foot["initial_velocity"] = wp.empty(2, dtype=wp.vec3d)
                foot["bv_stats"] = wp.zeros(3, dtype=wp.float64)
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

    def capture(self, state, dt, copy_particles=True):
        """Copy the input immediately before the MPM step, after proxy rewind."""
        if dt <= 0:
            raise ValueError("Timestep must be positive")
        self.dt = float(dt)
        if copy_particles:
            wp.copy(self.x, state.particle_q)
            self.captured_q = self.x
        else:
            self.captured_q = state.particle_q
        wp.copy(self.input_pose, state.body_q)
        wp.copy(self.input_velocity, state.body_qd)
        self.captured = True

    def evaluate(self, state_out, active_idx=None):
        """Use predicted next particle velocities; compute forces on the GPU."""
        if not self.captured:
            raise RuntimeError("capture must precede evaluate for every step")
        q_src = getattr(self, "captured_q", self.x)
        for index, f in enumerate(self.feet):
            env_idx = index // 2
            p_start = env_idx * self.particles_per_env
            p_end = p_start + self.particles_per_env
            if active_idx is not None:
                idx_e = active_idx if active_idx.ndim == 1 else active_idx[env_idx]
                k_eval = int(idx_e.shape[0])
                if index % 2 == 0:
                    wp_idx = wp.from_torch(idx_e.int())
                    wp.launch(
                        _gather_active_particles,
                        k_eval,
                        [
                            wp_idx,
                            p_start,
                            q_src,
                            state_out.particle_qd,
                            self.model.particle_mass,
                            self.x_active,
                            self.v_active,
                            self.m_active,
                        ],
                        device=self.device,
                    )
                    x_env = self.x_active[:k_eval]
                    v_env = self.v_active[:k_eval]
                    m_env = self.m_active[:k_eval]
                    self.grid.build(x_env, self.gradient_radius)
            else:
                k_eval = self.particles_per_env
                x_env = q_src[p_start:p_end]
                v_env = state_out.particle_qd[p_start:p_end]
                m_env = self.model.particle_mass[p_start:p_end]
                if index % 2 == 0:
                    self.grid.build(x_env, self.gradient_radius)

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
                    k_eval - 1,
                    f["offsets"][:k_eval],
                    f["surface_pose"],
                    f["center"],
                    f["free"],
                    f["inverse"],
                ],
                device=self.device,
            )
            wp.launch(
                _fit_gradients_culled,
                k_eval,
                [
                    self.grid.id,
                    x_env,
                    v_env,
                    self.gradient_radius,
                    f["gradients"][:k_eval],
                    f["surface_pose"],
                    f["sizes"],
                    f["center"],
                    f["free"],
                    f["margin"],
                    self.dt,
                    self.contact_band,
                ],
                device=self.device,
            )
            wp.launch(
                _prepare_contacts,
                k_eval,
                [
                    x_env,
                    v_env,
                    f["gradients"][:k_eval],
                    f["surface_pose"],
                    f["sizes"],
                    f["center"],
                    f["free"],
                    f["inverse"],
                    f["mass"],
                    m_env,
                    self.dt,
                    self.effective_scale,
                    self.horiz_scale,
                    self.up_scale,
                    self.tangent_scale,
                    self.normal_power,
                    self.stabilization,
                    f["margin"],
                    f["flags"][:k_eval],
                    f["frames"][:k_eval],
                    f["angular"][:k_eval],
                    f["response"][:k_eval],
                    f["compliance"][:k_eval],
                    f["material"][:k_eval],
                    f["gap"][:k_eval],
                ],
                device=self.device,
            )
            wp.utils.array_scan(f["flags"][:k_eval], f["offsets"][:k_eval], inclusive=True)
            wp.launch(
                kernels.compact_contacts,
                k_eval,
                [
                    f["flags"][:k_eval],
                    f["offsets"][:k_eval],
                    f["ids"][:k_eval],
                ],
                device=self.device,
            )
            initial_vel = f["free"]
            if self.use_body_velocity:
                wp.launch(
                    initialize_body_velocity_solve,
                    128,
                    [
                        f["offsets"][:k_eval],
                        f["ids"][:k_eval],
                        f["frames"][:k_eval],
                        f["angular"][:k_eval],
                        f["response"][:k_eval],
                        f["material"][:k_eval],
                        f["gap"][:k_eval],
                        m_env,
                        f["free"],
                        f["mass"],
                        self.effective_scale,
                        self.horiz_scale,
                        self.up_scale,
                        self.tangent_scale,
                        self.normal_power,
                        f["friction"],
                        f["impulses"][:k_eval],
                        f["initial_velocity"],
                        f["bv_stats"],
                    ],
                    device=self.device,
                    block_dim=128,
                )
                initial_vel = f["initial_velocity"]
            else:
                f["impulses"][:k_eval].zero_()
            wp.launch(
                kernels.solve_contacts_parallel,
                128,
                [
                    f["offsets"][:k_eval],
                    f["ids"][:k_eval],
                    f["frames"][:k_eval],
                    f["angular"][:k_eval],
                    f["response"][:k_eval],
                    f["compliance"][:k_eval],
                    f["material"][:k_eval],
                    f["gap"][:k_eval],
                    m_env,
                    initial_vel,
                    f["mass"],
                    self.effective_scale,
                    self.horiz_scale,
                    self.up_scale,
                    self.tangent_scale,
                    self.normal_power,
                    f["friction"],
                    self.dt,
                    self.iteration_limit,
                    wp.float64(1.0e-9),
                    f["impulses"][:k_eval],
                    self.wrench,
                    index,
                    f["stats"],
                    False,
                    self.relaxation,
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
