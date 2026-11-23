# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Packing-state and 3D Resistive Force Theory (RFT) proxy contact estimator for `ParticleWorldSolver`."""

import newton
import numpy as np
import warp as wp

from ..particle_nerd.contact import kernels as contact_kernels
from ..particle_nerd.contact.body_velocity import (
    _contact_response,
    _linear_solve,
    mat36d,
    mat63d,
    mat66d,
    vec6d,
)

wp.set_module_options({"enable_backward": False, "max_unroll": 0})


@wp.struct
class RFTContactProblem:
    body: int
    shape: int
    env_idx: int
    count: int
    shape_code: int
    mass: float
    friction: float
    surface_pose: wp.array(dtype=wp.transform)
    sizes: wp.array(dtype=wp.vec3)
    free: wp.array(dtype=wp.vec3d)
    initial: wp.array(dtype=wp.vec3d)
    inverse: wp.array(dtype=wp.mat33d)
    offsets: wp.array(dtype=int)
    ids: wp.array(dtype=int)
    frames: wp.array(dtype=wp.mat33d)
    angular: wp.array(dtype=wp.mat33d)
    response: wp.array(dtype=wp.mat33d)
    compliance: wp.array(dtype=wp.mat33d)
    particle_slack: wp.array(dtype=wp.vec2d)
    contact_velocity: wp.array(dtype=wp.vec3d)
    gap: wp.array(dtype=wp.float64)
    particle_mass: wp.array(dtype=float)
    impulses: wp.array(dtype=wp.vec3d)
    stats: wp.array(dtype=wp.float64)


@wp.kernel
def _gather_active_particles_rft_batched(
    active_idx: wp.array(dtype=wp.int32),
    k_eval: int,
    particles_per_env: int,
    x_src: wp.array(dtype=wp.vec3),
    v_src: wp.array(dtype=wp.vec3),
    m_src: wp.array(dtype=float),
    x_dst: wp.array(dtype=wp.vec3),
    v_dst: wp.array(dtype=wp.vec3),
    m_dst: wp.array(dtype=float),
):
    env_idx, i = wp.tid()
    dst = env_idx * k_eval + i
    src = env_idx * particles_per_env + int(active_idx[dst])
    x_dst[dst] = x_src[src]
    v_dst[dst] = v_src[src]
    m_dst[dst] = m_src[src]


@wp.kernel
def _prepare_world_proxy(
    poses: wp.array(dtype=wp.transform),
    velocities: wp.array(dtype=wp.spatial_vector),
    body_inertia: wp.array(dtype=wp.mat33),
    shape_transform: wp.array(dtype=wp.transform),
    body: int,
    shape: int,
    surface_pose: wp.array(dtype=wp.transform),
    free: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.mat33d),
):
    surface_pose[0] = poses[body] * shape_transform[shape]
    free[0] = wp.vec3d(wp.spatial_top(velocities[body]))
    free[1] = wp.vec3d(wp.spatial_bottom(velocities[body]))
    inertia = wp.mat33d(body_inertia[body])
    trace = inertia[0, 0] + inertia[1, 1] + inertia[2, 2]
    if trace > wp.float64(1.0e-12):
        rotation = wp.quat_to_matrix(wp.normalize(wp.quatd(wp.transform_get_rotation(poses[body]))))
        inverse[0] = wp.inverse(rotation * inertia * wp.transpose(rotation))
    else:
        inverse[0] = wp.mat33d(0.0)


@wp.kernel
def _prepare_world_proxies_batched(
    problems: wp.array(dtype=RFTContactProblem),
    poses: wp.array(dtype=wp.transform),
    velocities: wp.array(dtype=wp.spatial_vector),
    body_inertia: wp.array(dtype=wp.mat33),
    shape_transform: wp.array(dtype=wp.transform),
):
    problem_idx = wp.tid()
    p = problems[problem_idx]
    body = p.body
    shape = p.shape
    p.offsets[p.count - 1] = 0
    p.surface_pose[0] = poses[body] * shape_transform[shape]
    p.free[0] = wp.vec3d(wp.spatial_top(velocities[body]))
    p.free[1] = wp.vec3d(wp.spatial_bottom(velocities[body]))
    inertia = wp.mat33d(body_inertia[body])
    trace = inertia[0, 0] + inertia[1, 1] + inertia[2, 2]
    if trace > wp.float64(1.0e-12):
        rotation = wp.quat_to_matrix(wp.normalize(wp.quatd(wp.transform_get_rotation(poses[body]))))
        p.inverse[0] = wp.inverse(rotation * inertia * wp.transpose(rotation))
    else:
        p.inverse[0] = wp.mat33d(0.0)


@wp.func
def rft_packing_mobilities(
    n: wp.vec3d,
    particle_mass: wp.float64,
    base_scale: wp.float64,
    horiz_scale: wp.float64,
    up_scale: wp.float64,
    tangent_scale: wp.float64,
    normal_power: wp.float64,
) -> wp.vec2d:
    """Compute per-particle normal and tangential compliance (1/m_eff_n, 1/m_eff_t)."""
    nz = n[2]
    up_part = wp.pow(wp.max(nz, wp.float64(0.0)), normal_power)
    down_part = wp.pow(wp.max(-nz, wp.float64(0.0)), normal_power)
    horiz_part = wp.max(wp.float64(0.0), wp.float64(1.0) - up_part - down_part)
    eff_n = wp.max(
        base_scale * up_part + horiz_scale * horiz_part + up_scale * down_part,
        wp.float64(0.5),
    )
    eff_t = wp.max(tangent_scale, wp.float64(0.5))
    slack_n = wp.float64(1.0) / (particle_mass * eff_n)
    slack_t = wp.float64(1.0) / (particle_mass * eff_t)
    return wp.vec2d(slack_n, slack_t)


@wp.kernel
def _fit_and_prepare_contacts_rft_packing_batched(
    problems: wp.array(dtype=RFTContactProblem),
    grid: wp.uint64,
    x_all: wp.array(dtype=wp.vec3),
    v_all: wp.array(dtype=wp.vec3),
    radius: float,
    dt: float,
    contact_band: float,
    base_scale: float,
    horiz_scale: float,
    up_scale: float,
    tangent_scale: float,
    normal_power: float,
    stabilization: float,
    flags_flat: wp.array(dtype=int),
):
    problem_idx, i = wp.tid()
    p = problems[problem_idx]
    k_eval = p.count
    if i >= k_eval:
        return
    p.impulses[i] = wp.vec3d(0.0)
    flat_idx = problem_idx * k_eval + i
    flags_flat[flat_idx] = 0
    pm_f = p.particle_mass[i]
    if pm_f <= 0.0:
        return

    env_offset = p.env_idx * k_eval
    xi = x_all[env_offset + i]
    vi = v_all[env_offset + i]
    pose_b = p.surface_pose[0]
    distance, normal, point = contact_kernels.geometry(wp.vec3d(xi), pose_b, p.sizes[0], p.shape_code)
    if contact_band > 0.0 and distance > wp.float64(contact_band):
        return

    arm = point - wp.vec3d(wp.transform_get_translation(pose_b))
    surface = p.free[0] + wp.cross(p.free[1], arm)
    app0 = wp.dot(normal, surface - wp.vec3d(vi))
    if distance > (wp.max(app0, wp.float64(0.0)) + wp.float64(4.0)) * wp.float64(dt):
        return

    lhs = wp.mat33(0.0)
    rhs = wp.mat33(0.0)
    phi = float(0.0)
    query = wp.hash_grid_query(grid, xi, radius + 1.0e-6, p.env_idx)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        dx = wp.vec3(x_all[j]) - xi
        dv = wp.vec3(v_all[j]) - vi
        dist = wp.length(dx)
        q = dist / float(radius)
        if q < float(1.0):
            a = float(1.0) - q
            weight = a * a * a * a * (float(1.0) + float(4.0) * q)
            lhs += weight * wp.outer(dx, dx)
            rhs += weight * wp.outer(dx, dv)
            if dist > 1.0e-6:
                phi += weight
    ridge = float(0.01) * wp.trace(lhs) / float(3.0) + float(1.0e-12)
    lhs += wp.mat33(ridge, 0.0, 0.0, 0.0, ridge, 0.0, 0.0, 0.0, ridge)
    grad = wp.mat33d(wp.inverse(lhs) * rhs)

    velocity = wp.vec3d(vi) + wp.transpose(grad) * (point - wp.vec3d(xi))
    v_rel_world = surface - velocity
    approach = wp.dot(normal, v_rel_world)
    if distance <= wp.max(approach, wp.float64(0.0)) * wp.float64(dt):
        flags_flat[flat_idx] = 1
        n = -normal
        axis = int(0)
        if wp.abs(n[1]) < wp.abs(n[axis]):
            axis = 1
        if wp.abs(n[2]) < wp.abs(n[axis]):
            axis = 2
        basis = wp.vec3d(0.0)
        basis[axis] = wp.float64(1.0)
        t = wp.normalize(wp.cross(n, basis))
        t2 = wp.cross(n, t)
        frame = wp.mat33d(n[0], n[1], n[2], t[0], t[1], t[2], t2[0], t2[1], t2[2])
        a0 = wp.cross(arm, n)
        a1 = wp.cross(arm, t)
        a2 = wp.cross(arm, t2)
        angular = wp.mat33d(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2])
        respond = p.inverse[0] * wp.transpose(angular)
        k = frame * wp.transpose(frame) / wp.float64(p.mass) + angular * respond

        pm = wp.max(wp.float64(pm_f), wp.float64(1.0e-6))
        slacks = rft_packing_mobilities(
            n,
            pm,
            wp.float64(base_scale),
            wp.float64(horiz_scale),
            wp.float64(up_scale),
            wp.float64(tangent_scale),
            wp.float64(normal_power),
        )
        slack_n = slacks[0]
        slack_t = slacks[1]
        k += wp.mat33d(slack_n, 0.0, 0.0, 0.0, slack_t, 0.0, 0.0, 0.0, slack_t)

        p.frames[i] = frame
        p.angular[i] = angular
        p.response[i] = respond
        p.compliance[i] = k
        p.particle_slack[i] = slacks
        p.contact_velocity[i] = frame * velocity
        p.gap[i] = (
            wp.max(distance, wp.float64(0.0)) + wp.float64(stabilization) * wp.min(distance, wp.float64(0.0))
        ) / wp.float64(dt)


@wp.kernel
def _compact_contacts_rft_batched(
    problems: wp.array(dtype=RFTContactProblem),
    flags_flat: wp.array(dtype=int),
    offsets_flat: wp.array(dtype=int),
):
    problem_idx, i = wp.tid()
    p = problems[problem_idx]
    k_eval = p.count
    if i >= k_eval:
        return
    base = problem_idx * k_eval
    prev_total = int(0)
    if base > 0:
        prev_total = offsets_flat[base - 1]
    local_off = offsets_flat[base + i] - prev_total
    if flags_flat[base + i] != 0:
        p.ids[local_off - 1] = i
    if i == k_eval - 1:
        p.offsets[k_eval - 1] = local_off


@wp.func
def _solve_contacts_rft_packing_block(
    lane: int,
    offsets: wp.array(dtype=int),
    ids: wp.array(dtype=int),
    frames: wp.array(dtype=wp.mat33d),
    angular: wp.array(dtype=wp.mat33d),
    response: wp.array(dtype=wp.mat33d),
    compliance: wp.array(dtype=wp.mat33d),
    particle_slack: wp.array(dtype=wp.vec2d),
    material: wp.array(dtype=wp.vec3d),
    gap: wp.array(dtype=wp.float64),
    free: wp.array(dtype=wp.vec3d),
    mass: float,
    friction: float,
    dt: float,
    max_iterations: int,
    tolerance: wp.float64,
    relaxation_in: float,
    impulses: wp.array(dtype=wp.vec3d),
    wrench: wp.array(dtype=wp.spatial_vector),
    body: int,
    stats: wp.array(dtype=wp.float64),
):
    count = offsets[offsets.shape[0] - 1]
    linear = free[0]
    spin = free[1]
    total = wp.vec3d(0.0)
    torque = wp.vec3d(0.0)
    residual = wp.float64(0.0)
    iterations = int(0)
    if count > 0:
        relaxation = wp.max(wp.float64(relaxation_in), wp.float64(1.0) / wp.float64(count))
        for iteration in range(max_iterations):
            local_delta = wp.vec3d(0.0)
            local_spin = wp.vec3d(0.0)
            local_residual = wp.float64(0.0)
            k = lane
            while k < count:
                i = ids[k]
                frame = frames[i]
                aj = angular[i]
                respond = response[i]
                matrix = compliance[i]
                old = impulses[i]
                slacks = particle_slack[i]
                slack_n = slacks[0]
                slack_t = slacks[1]
                relative = (
                    frame * linear
                    + aj * spin
                    - material[i]
                    + wp.vec3d(old[0] * slack_n, old[1] * slack_t, old[2] * slack_t)
                )
                normal = wp.max(wp.float64(0.0), old[0] - (relative[0] + gap[i]) / matrix[0, 0])
                dn = normal - old[0]
                increment = wp.vec3d(dn, 0.0, 0.0)
                local_linear = linear + wp.transpose(frame) * increment / wp.float64(mass)
                local_angular = spin + respond * increment
                relative = (
                    frame * local_linear
                    + aj * local_angular
                    - material[i]
                    + wp.vec3d(normal * slack_n, old[1] * slack_t, old[2] * slack_t)
                )
                rhs = wp.vec2d(
                    matrix[1, 1] * old[1] + matrix[1, 2] * old[2] - relative[1],
                    matrix[2, 1] * old[1] + matrix[2, 2] * old[2] - relative[2],
                )
                tangent = contact_kernels.friction_disk(
                    matrix[1, 1], matrix[1, 2], matrix[2, 2], rhs, wp.float64(friction) * normal
                )
                delta_t = tangent - wp.vec2d(old[1], old[2])
                change = relaxation * wp.vec3d(dn, delta_t[0], delta_t[1])
                impulses[i] = old + change
                local_delta += wp.transpose(frame) * change
                local_spin += respond * change
                local_residual = wp.max(local_residual, wp.max(wp.abs(dn), wp.length(delta_t)))
                k += 128
            td = wp.tile_sum(wp.tile(local_delta), axis=1)
            ts = wp.tile_sum(wp.tile(local_spin), axis=1)
            delta = wp.vec3d(wp.tile_extract(td, 0), wp.tile_extract(td, 1), wp.tile_extract(td, 2))
            ds = wp.vec3d(wp.tile_extract(ts, 0), wp.tile_extract(ts, 1), wp.tile_extract(ts, 2))
            residual = wp.tile_extract(wp.tile_max(wp.tile(local_residual)), 0)
            linear += delta / wp.float64(mass)
            spin += ds
            iterations = iteration + 1
            if residual < tolerance:
                break
        local_total = wp.vec3d(0.0)
        local_torque = wp.vec3d(0.0)
        k = lane
        while k < count:
            i = ids[k]
            local_total += wp.transpose(frames[i]) * impulses[i]
            local_torque += wp.transpose(angular[i]) * impulses[i]
            k += 128
        ttot = wp.tile_sum(wp.tile(local_total), axis=1)
        ttor = wp.tile_sum(wp.tile(local_torque), axis=1)
        total = wp.vec3d(wp.tile_extract(ttot, 0), wp.tile_extract(ttot, 1), wp.tile_extract(ttot, 2))
        torque = wp.vec3d(wp.tile_extract(ttor, 0), wp.tile_extract(ttor, 1), wp.tile_extract(ttor, 2))
    if lane == 0:
        wrench[body] = wp.spatial_vector(wp.vec3(total / wp.float64(dt)), wp.vec3(torque / wp.float64(dt)))
        stats[0] = wp.float64(count)
        stats[1] = wp.float64(iterations)
        stats[2] = residual


@wp.func
def _initialize_rft_packing_body_velocity_block(
    lane: int,
    offsets: wp.array(dtype=int),
    ids: wp.array(dtype=int),
    frames: wp.array(dtype=wp.mat33d),
    angular: wp.array(dtype=wp.mat33d),
    response: wp.array(dtype=wp.mat33d),
    particle_slack: wp.array(dtype=wp.vec2d),
    material: wp.array(dtype=wp.vec3d),
    gap: wp.array(dtype=wp.float64),
    free: wp.array(dtype=wp.vec3d),
    mass: float,
    friction: float,
    impulses: wp.array(dtype=wp.vec3d),
    initial: wp.array(dtype=wp.vec3d),
):
    count = offsets[offsets.shape[0] - 1]
    origin = vec6d(free[0][0], free[0][1], free[0][2], free[1][0], free[1][1], free[1][2])
    q = origin
    error = wp.float64(0.0)
    if count > 0:
        for _it in range(25):
            local_force = vec6d(0.0)
            local_jacobian = mat66d(0.0)
            k = lane
            while k < count:
                i = ids[k]
                slacks = particle_slack[i]
                impulse, derivative = _contact_response(
                    q,
                    frames[i],
                    angular[i],
                    material[i],
                    gap[i],
                    slacks[0],
                    slacks[1],
                    wp.float64(friction),
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
                k += 128
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
            if error < wp.float64(1e-11):
                break
            delta = _linear_solve(jacobian, residual)
            alpha = wp.float64(1.0)
            accepted = bool(False)
            for _trial in range(12):
                candidate = q - alpha * delta
                local_candidate = vec6d(0.0)
                k = lane
                while k < count:
                    i = ids[k]
                    slacks = particle_slack[i]
                    impulse, _unused = _contact_response(
                        candidate,
                        frames[i],
                        angular[i],
                        material[i],
                        gap[i],
                        slacks[0],
                        slacks[1],
                        wp.float64(friction),
                    )
                    linear = wp.transpose(frames[i]) * impulse / wp.float64(mass)
                    spin = response[i] * impulse
                    local_candidate += vec6d(linear[0], linear[1], linear[2], spin[0], spin[1], spin[2])
                    k += 128
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
    local_force = vec6d(0.0)
    k = lane
    while k < count:
        i = ids[k]
        slacks = particle_slack[i]
        impulse, _unused = _contact_response(
            q,
            frames[i],
            angular[i],
            material[i],
            gap[i],
            slacks[0],
            slacks[1],
            wp.float64(friction),
        )
        impulses[i] = impulse
        linear = wp.transpose(frames[i]) * impulse / wp.float64(mass)
        spin = response[i] * impulse
        local_force += vec6d(linear[0], linear[1], linear[2], spin[0], spin[1], spin[2])
        k += 128
    tf_final = wp.tile_sum(wp.tile(local_force), axis=1)
    force = vec6d(0.0)
    for r in range(6):
        force[r] = wp.tile_extract(tf_final, r)
    final_error = wp.length(q - origin - force)
    fallback = not wp.isfinite(final_error) or final_error > wp.float64(1e-2)
    if fallback:
        k = lane
        while k < count:
            impulses[ids[k]] = wp.vec3d(0.0)
            k += 128
        force = vec6d(0.0)
    if lane == 0:
        initial[0] = free[0] + wp.vec3d(force[0], force[1], force[2])
        initial[1] = free[1] + wp.vec3d(force[3], force[4], force[5])


@wp.kernel
def _initialize_contacts_rft_packing_batched(
    problems: wp.array(dtype=RFTContactProblem),
):
    problem_idx, lane = wp.tid()
    p = problems[problem_idx]
    _initialize_rft_packing_body_velocity_block(
        lane,
        p.offsets,
        p.ids,
        p.frames,
        p.angular,
        p.response,
        p.particle_slack,
        p.contact_velocity,
        p.gap,
        p.free,
        p.mass,
        p.friction,
        p.impulses,
        p.initial,
    )


@wp.kernel
def _solve_contacts_rft_packing_batched(
    problems: wp.array(dtype=RFTContactProblem),
    dt: float,
    max_iterations: int,
    tolerance: wp.float64,
    relaxation_in: float,
    wrench: wp.array(dtype=wp.spatial_vector),
):
    problem_idx, lane = wp.tid()
    p = problems[problem_idx]
    _solve_contacts_rft_packing_block(
        lane,
        p.offsets,
        p.ids,
        p.frames,
        p.angular,
        p.response,
        p.compliance,
        p.particle_slack,
        p.contact_velocity,
        p.gap,
        p.initial,
        p.mass,
        p.friction,
        dt,
        max_iterations,
        tolerance,
        relaxation_in,
        p.impulses,
        wrench,
        problem_idx,
        p.stats,
    )


class WorldParticleProxyEstimator:
    """Reconstruct proxy contact wrenches using local packing state and 3D Resistive Force Theory (RFT)."""

    def __init__(
        self,
        model,
        body_ids,
        *,
        colliders_per_world: int = 1,
        iterations: int = 120,
        effective_mass_scale: float = 10.0,
        tangent_mass_scale: float = 10.0,
        horiz_mass_scale: float = 10.0,
        up_mass_scale: float = 10.0,
        normal_power: float = 1.0,
        relaxation: float = 0.12,
        gradient_radius: float = 0.065,
        gradient_contact_band: float = 0.08,
        stabilization: float = 0.01,
        contact_offset: float = 0.016,
        particles_per_env: int | None = None,
        batch_contacts: bool = True,
    ):
        self.model = model
        self.device = model.device
        self.body_ids = [int(b) for b in body_ids]
        self.colliders_per_world = int(colliders_per_world)
        self.n = int(model.particle_count)
        self.particles_per_env = int(particles_per_env) if particles_per_env is not None else self.n
        self.iteration_limit = int(iterations)
        self.base_scale = float(effective_mass_scale)
        self.horiz_scale = float(horiz_mass_scale)
        self.up_scale = float(up_mass_scale)
        self.tangent_scale = float(tangent_mass_scale)
        self.normal_power = float(normal_power)
        self.relaxation = float(relaxation)
        self.gradient_radius = float(gradient_radius)
        self.contact_band = float(gradient_contact_band)
        self.stabilization = float(stabilization)
        self.contact_offset = float(contact_offset)
        self.batch_contacts = bool(batch_contacts)
        self.captured = False
        self.dt = 0.005

        shape_body = model.shape_body.numpy()
        types = model.shape_type.numpy()
        flags = model.shape_flags.numpy()
        scales = model.shape_scale.numpy()
        margins = model.shape_margin.numpy()
        shape_friction = model.shape_material_mu.numpy()
        masses = model.body_mass.numpy()

        colliding_mask = (flags & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0
        body_to_shapes: dict[int, list[int]] = {}
        for s_idx in np.flatnonzero(colliding_mask):
            body_to_shapes.setdefault(int(shape_body[s_idx]), []).append(int(s_idx))

        geo_to_code = {
            int(newton.GeoType.BOX): 0,
            int(newton.GeoType.SPHERE): 1,
            int(newton.GeoType.CYLINDER): 2,
            int(newton.GeoType.CAPSULE): 4,
        }

        num_colliders = len(self.body_ids)
        with wp.ScopedDevice(self.device):
            self.x = wp.empty(self.n, dtype=wp.vec3)
            self.input_pose = wp.empty(self.model.body_count, dtype=wp.transform)
            self.input_velocity = wp.empty(self.model.body_count, dtype=wp.spatial_vector)
            self.wrench = wp.zeros(num_colliders, dtype=wp.spatial_vector)

            self.colliders = []
            for idx, body in enumerate(self.body_ids):
                selected = [s for s in body_to_shapes.get(int(body), []) if int(types[s]) in geo_to_code]
                if not selected:
                    selected = [s for s in np.flatnonzero(shape_body == body).tolist() if int(types[s]) in geo_to_code]
                if not selected:
                    raise ValueError(
                        f"Collider body {body}: expected box/sphere/cylinder/capsule shape; got {selected}"
                    )
                shape = int(max(selected, key=lambda s: float(np.max(scales[s]))))
                code = geo_to_code[int(types[shape])]
                margin = float(margins[shape]) + self.contact_offset
                sc = scales[shape]
                if code == 1:  # SPHERE
                    r_s = float(sc[0])
                    raw_half = np.array([r_s, r_s, r_s], dtype=np.float32)
                elif code == 2:  # CYLINDER (geometry kernel uses half[0]=radius, half[2]=half_height)
                    r_s = float(sc[0])
                    hh_s = float(sc[1]) if float(sc[1]) > 0.0 else float(sc[2])
                    raw_half = np.array([r_s, r_s, hh_s], dtype=np.float32)
                elif code == 4:  # CAPSULE (geometry kernel uses half[0]=radius, half[1]=half_height)
                    r_s = float(sc[0])
                    hh_s = float(sc[1]) if float(sc[1]) > 0.0 else float(sc[2])
                    raw_half = np.array([r_s, hh_s, r_s], dtype=np.float32)
                else:  # BOX
                    raw_half = np.asarray(sc, dtype=np.float32)
                full_size = raw_half * 2.0 + 2.0 * margin
                mass_val = float(masses[body]) if float(masses[body]) > 0 else float("inf")
                mu_val = float(shape_friction[shape])
                self.colliders.append(
                    {
                        "idx": idx,
                        "body": body,
                        "shape": shape,
                        "code": int(code),
                        "mass": mass_val,
                        "friction": mu_val,
                        "surface_pose": wp.empty(1, dtype=wp.transform),
                        "sizes": wp.array([full_size], dtype=wp.vec3),
                        "shapes": wp.array([code], dtype=int),
                        "free": wp.empty(2, dtype=wp.vec3d),
                        "initial": wp.empty(2, dtype=wp.vec3d),
                        "inverse": wp.empty(1, dtype=wp.mat33d),
                        "stats": wp.zeros(3, dtype=wp.float64),
                    }
                )
        self._batch_signature = None

    def refresh_body_inertia(self):
        masses = self.model.body_mass.numpy()
        for col in self.colliders:
            m = float(masses[col["body"]])
            col["mass"] = m if m > 0 else float("inf")
        self._batch_signature = None

    def _prepare_batch(self, k_eval: int):
        num_colliders = len(self.colliders)
        num_envs = max(1, num_colliders // self.colliders_per_world)
        sig = (num_colliders, num_envs, int(k_eval))
        if getattr(self, "_batch_signature", None) == sig:
            return
        total_active = num_envs * k_eval
        total_col_k = num_colliders * k_eval
        with wp.ScopedDevice(self.device):
            self._batch_x_active = wp.empty(total_active, dtype=wp.vec3)
            self._batch_v_active = wp.empty(total_active, dtype=wp.vec3)
            self._batch_m_active = wp.empty(total_active, dtype=float)
            group_np = np.repeat(np.arange(num_envs, dtype=np.int32), k_eval)
            self._batch_groups = wp.array(group_np, dtype=wp.int32)
            self._batch_flags_flat = wp.empty(total_col_k, dtype=int)
            self._batch_offsets_flat = wp.empty(total_col_k, dtype=int)
            self._batch_scan_offsets_flat = wp.empty(total_col_k, dtype=int)
            self._batch_ids_flat = wp.empty(total_col_k, dtype=int)
            self._batch_frames_flat = wp.empty(total_col_k, dtype=wp.mat33d)
            self._batch_angular_flat = wp.empty(total_col_k, dtype=wp.mat33d)
            self._batch_response_flat = wp.empty(total_col_k, dtype=wp.mat33d)
            self._batch_compliance_flat = wp.empty(total_col_k, dtype=wp.mat33d)
            self._batch_slack_flat = wp.empty(total_col_k, dtype=wp.vec2d)
            self._batch_vel_flat = wp.empty(total_col_k, dtype=wp.vec3d)
            self._batch_gap_flat = wp.empty(total_col_k, dtype=wp.float64)
            self._batch_impulses_flat = wp.empty(total_col_k, dtype=wp.vec3d)
            if getattr(self, "_batch_grid", None) is None:
                self._batch_grid = wp.HashGrid(32, 32, 16)
            self._batch_grid.reserve(total_active, with_groups=True)

            problems = []
            for index, col in enumerate(self.colliders):
                env_idx = index // self.colliders_per_world
                s = index * k_eval
                e = s + k_eval
                p = RFTContactProblem()
                p.body = int(col["body"])
                p.shape = int(col["shape"])
                p.env_idx = int(env_idx)
                p.count = int(k_eval)
                p.shape_code = int(col["code"])
                p.mass = float(col["mass"])
                p.friction = float(col["friction"])
                p.surface_pose = col["surface_pose"]
                p.sizes = col["sizes"]
                p.free = col["free"]
                p.initial = col["initial"]
                p.inverse = col["inverse"]
                p.offsets = self._batch_offsets_flat[s:e]
                p.ids = self._batch_ids_flat[s:e]
                p.frames = self._batch_frames_flat[s:e]
                p.angular = self._batch_angular_flat[s:e]
                p.response = self._batch_response_flat[s:e]
                p.compliance = self._batch_compliance_flat[s:e]
                p.particle_slack = self._batch_slack_flat[s:e]
                p.contact_velocity = self._batch_vel_flat[s:e]
                p.gap = self._batch_gap_flat[s:e]
                p.particle_mass = self._batch_m_active[env_idx * k_eval : (env_idx + 1) * k_eval]
                p.impulses = self._batch_impulses_flat[s:e]
                p.stats = col["stats"]
                problems.append(p)
            self._problems = wp.array(problems, dtype=RFTContactProblem)
        self._batch_signature = sig

    def capture(self, state, dt, copy_particles: bool = True):
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
        if not self.captured:
            raise RuntimeError("capture must precede evaluate for every step")
        q_src = getattr(self, "captured_q", self.x)
        num_colliders = len(self.colliders)
        num_envs = max(1, num_colliders // self.colliders_per_world)

        if active_idx is not None:
            if hasattr(active_idx, "ndim") and active_idx.ndim == 2:
                k_eval = int(active_idx.shape[1])
                idx_flat = active_idx.int().contiguous().view(-1)
            else:
                import torch

                k_eval = int(active_idx[0].shape[0])
                idx_flat = torch.cat([t.int() for t in active_idx], dim=0).contiguous()
        else:
            k_eval = self.particles_per_env
            idx_flat = None

        self._prepare_batch(k_eval)

        if idx_flat is not None:
            wp.launch(
                _gather_active_particles_rft_batched,
                (num_envs, k_eval),
                [
                    wp.from_torch(idx_flat, dtype=wp.int32),
                    k_eval,
                    self.particles_per_env,
                    q_src,
                    state_out.particle_qd,
                    self.model.particle_mass,
                    self._batch_x_active,
                    self._batch_v_active,
                    self._batch_m_active,
                ],
                device=self.device,
            )
            x_all = self._batch_x_active
            v_all = self._batch_v_active
        else:
            total_active = num_envs * k_eval
            x_all = q_src[:total_active]
            v_all = state_out.particle_qd[:total_active]
            wp.copy(self._batch_m_active, self.model.particle_mass, count=total_active)

        self._batch_grid.build(x_all, self.gradient_radius, groups=self._batch_groups)
        wp.launch(
            _prepare_world_proxies_batched,
            num_colliders,
            [
                self._problems,
                self.input_pose,
                self.input_velocity,
                self.model.body_inertia,
                self.model.shape_transform,
            ],
            device=self.device,
        )
        wp.launch(
            _fit_and_prepare_contacts_rft_packing_batched,
            (num_colliders, k_eval),
            [
                self._problems,
                self._batch_grid.id,
                x_all,
                v_all,
                self.gradient_radius,
                self.dt,
                self.contact_band,
                self.base_scale,
                self.horiz_scale,
                self.up_scale,
                self.tangent_scale,
                self.normal_power,
                self.stabilization,
                self._batch_flags_flat,
            ],
            device=self.device,
        )
        wp.utils.array_scan(self._batch_flags_flat, self._batch_scan_offsets_flat, inclusive=True)
        wp.launch(
            _compact_contacts_rft_batched,
            (num_colliders, k_eval),
            [
                self._problems,
                self._batch_flags_flat,
                self._batch_scan_offsets_flat,
            ],
            device=self.device,
        )
        wp.launch(
            _initialize_contacts_rft_packing_batched,
            (num_colliders, 128),
            [self._problems],
            device=self.device,
            block_dim=128,
        )
        wp.launch(
            _solve_contacts_rft_packing_batched,
            (num_colliders, 128),
            [
                self._problems,
                self.dt,
                self.iteration_limit,
                wp.float64(1.0e-8),
                self.relaxation,
                self.wrench,
            ],
            device=self.device,
            block_dim=128,
        )
        self.captured = False
        return self.wrench
