# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched velocity fitting and unilateral Coulomb contact equations."""

import warp as wp

from ..sdf import surface as sdf_surface
from .grid import cell_key

wp.set_module_options({"enable_backward": False, "max_unroll": 0})


@wp.func
def geometry(point: wp.vec3d, pose: wp.transform, size: wp.vec3, shape: int):
    center = wp.vec3d(wp.transform_get_translation(pose))
    rotation = wp.normalize(wp.quatd(wp.transform_get_rotation(pose)))
    local = wp.quat_rotate_inv(rotation, point - center)
    half = wp.vec3d(size) * wp.float64(0.5)
    normal = wp.vec3d(1.0, 0.0, 0.0)
    distance = wp.float64(0.0)
    eps = wp.float64(1.0e-15)
    if shape == 1:
        length = wp.length(local)
        distance = length - half[0]
        if length > eps:
            normal = local / length
    elif shape == 2:
        radial = wp.sqrt(local[0] * local[0] + local[1] * local[1])
        direction = wp.vec2d(1.0, 0.0)
        if radial > eps:
            direction = wp.vec2d(local[0], local[1]) / radial
        q_cylinder = wp.vec2d(radial - half[0], wp.abs(local[2]) - half[2])
        positive_cylinder = wp.vec2d(wp.max(q_cylinder[0], wp.float64(0.0)), wp.max(q_cylinder[1], wp.float64(0.0)))
        length = wp.length(positive_cylinder)
        distance = length + wp.min(wp.max(q_cylinder[0], q_cylinder[1]), wp.float64(0.0))
        weights = wp.vec2d(1.0, 0.0)
        if length > eps:
            weights = positive_cylinder / length
        elif q_cylinder[1] > q_cylinder[0]:
            weights = wp.vec2d(0.0, 1.0)
        sign_z = wp.float64(1.0)
        if local[2] < wp.float64(0.0):
            sign_z = wp.float64(-1.0)
        normal = wp.vec3d(direction[0] * weights[0], direction[1] * weights[0], sign_z * weights[1])
    else:
        q = wp.vec3d(wp.abs(local[0]), wp.abs(local[1]), wp.abs(local[2])) - half
        positive = wp.vec3d(wp.max(q[0], wp.float64(0.0)), wp.max(q[1], wp.float64(0.0)), wp.max(q[2], wp.float64(0.0)))
        length = wp.length(positive)
        distance = length + wp.min(wp.max(q[0], wp.max(q[1], q[2])), wp.float64(0.0))
        sign = wp.vec3d(1.0, 1.0, 1.0)
        for axis in range(3):
            if local[axis] < wp.float64(0.0):
                sign[axis] = wp.float64(-1.0)
        if length > eps:
            normal = wp.cw_mul(sign, positive) / length
        else:
            axis = int(0)
            if q[1] > q[axis]:
                axis = 1
            if q[2] > q[axis]:
                axis = 2
            normal = wp.vec3d(0.0)
            normal[axis] = sign[axis]
    normal = wp.quat_rotate(rotation, normal)
    return distance, normal, point - distance * normal


@wp.kernel
def fit_gradients(
    grid_starts: wp.array(dtype=int),
    grid_ends: wp.array(dtype=int),
    grid_ids: wp.array(dtype=int),
    grid_dim: int,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: float,
    output: wp.array(dtype=wp.mat33d),
    poses: wp.array(dtype=wp.transform),
    sizes: wp.array(dtype=wp.vec3),
    shapes: wp.array(dtype=int),
    particles: int,
    contact_band: float,
):
    # Accumulate in float32; contact equations retain float64.
    i = wp.tid()
    body = i // particles
    # Optional approximation: beyond this body-surface distance, use constant
    # particle velocity in contact selection instead of affine extrapolation.
    if contact_band > 0.0:
        distance, normal, point = geometry(wp.vec3d(x[i]), poses[body], sizes[body], shapes[body])
        if distance > wp.float64(contact_band):
            output[i] = wp.mat33d(0.0)
            return
    lhs = wp.mat33(0.0)
    rhs = wp.mat33(0.0)
    lo = (x[i] - wp.vec3(radius + 1.0e-6)) / radius
    hi = (x[i] + wp.vec3(radius + 1.0e-6)) / radius
    for cz in range(int(wp.floor(lo[2])), int(wp.floor(hi[2])) + 1):
        for cy in range(int(wp.floor(lo[1])), int(wp.floor(hi[1])) + 1):
            for cx in range(int(wp.floor(lo[0])), int(wp.floor(hi[0])) + 1):
                cell = cell_key(cx, cy, cz, body, grid_dim)
                for slot in range(grid_starts[cell], grid_ends[cell]):
                    j = grid_ids[slot]
                    dx = wp.vec3(x[j]) - wp.vec3(x[i])
                    dv = wp.vec3(v[j]) - wp.vec3(v[i])
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
def prepare_body(
    pose: wp.array(dtype=wp.transform),
    velocity: wp.array(dtype=wp.spatial_vector),
    external: wp.array(dtype=wp.spatial_vector),
    mass: float,
    inertia: wp.mat33d,
    gravity: wp.vec3,
    dt: float,
    free: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.mat33d),
):
    body = wp.tid()
    rotation = wp.quat_to_matrix(wp.normalize(wp.quatd(wp.transform_get_rotation(pose[body]))))
    world = rotation * inertia * wp.transpose(rotation)
    inv = wp.inverse(world)
    linear = wp.vec3d(wp.spatial_top(velocity[body]))
    angular = wp.vec3d(wp.spatial_bottom(velocity[body]))
    force = wp.vec3d(wp.spatial_top(external[body]))
    torque = wp.vec3d(wp.spatial_bottom(external[body]))
    free[2 * body] = linear + wp.float64(dt) * (wp.vec3d(gravity) + force / wp.float64(mass))
    free[2 * body + 1] = angular + wp.float64(dt) * (inv * (torque - wp.cross(angular, world * angular)))
    inverse[body] = inv


@wp.kernel
def prepare_contacts(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    gradients: wp.array(dtype=wp.mat33d),
    pose: wp.array(dtype=wp.transform),
    sizes: wp.array(dtype=wp.vec3),
    shapes: wp.array(dtype=int),
    particles: int,
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
    flags: wp.array(dtype=int),
    frames: wp.array(dtype=wp.mat33d),
    angular_jacobian: wp.array(dtype=wp.mat33d),
    response: wp.array(dtype=wp.mat33d),
    compliance: wp.array(dtype=wp.mat33d),
    material: wp.array(dtype=wp.vec3d),
    gap_speed: wp.array(dtype=wp.float64),
    fixed_material_velocity: bool,
    sdf: wp.array3d(dtype=float),
    sdf_origin: wp.vec3,
    sdf_spacing: float,
):
    i = wp.tid()
    body = i // particles
    distance, normal, point = geometry(wp.vec3d(x[i]), pose[body], sizes[body], shapes[body])
    if shapes[body] == 3:
        rotation = wp.transform_get_rotation(pose[body])
        local = wp.transform_point(wp.transform_inverse(pose[body]), x[i])
        sdf_distance, sdf_normal = sdf_surface(local, sdf, sdf_origin, sdf_spacing)
        distance = wp.float64(sdf_distance)
        normal = wp.vec3d(wp.quat_rotate(rotation, sdf_normal))
        point = wp.vec3d(x[i]) - distance * normal
    velocity = wp.vec3d(v[i]) + wp.transpose(gradients[i]) * (point - wp.vec3d(x[i]))
    arm = point - wp.vec3d(wp.transform_get_translation(pose[body]))
    surface = free[2 * body] + wp.cross(free[2 * body + 1], arm)
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
        t = wp.normalize(wp.cross(n, basis))
        t2 = wp.cross(n, t)
        frame = wp.mat33d(n[0], n[1], n[2], t[0], t[1], t[2], t2[0], t2[1], t2[2])
        a0 = wp.cross(arm, n)
        a1 = wp.cross(arm, t)
        a2 = wp.cross(arm, t2)
        angular = wp.mat33d(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2])
        respond = inverse[body] * wp.transpose(angular)
        k = frame * wp.transpose(frame) / wp.float64(mass) + angular * respond
        slack_n = wp.float64(0.0)
        slack_t = wp.float64(0.0)
        if not fixed_material_velocity:
            nz = n[2]
            up_part = wp.max(nz, wp.float64(0.0))
            down_part = wp.max(-nz, wp.float64(0.0))
            horiz_part = wp.max(wp.float64(0.0), wp.float64(1.0) - nz * nz)
            eff_n = wp.max(
                wp.float64(effective_scale) * wp.pow(up_part, wp.float64(normal_power))
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
def compact_contacts(flags: wp.array(dtype=int), offsets: wp.array(dtype=int), ids: wp.array(dtype=int)):
    i = wp.tid()
    if flags[i] != 0:
        ids[offsets[i] - 1] = i


@wp.func
def friction_disk(a: wp.float64, b: wp.float64, d: wp.float64, rhs: wp.vec2d, radius: wp.float64):
    result = wp.vec2d(0.0)
    if radius > wp.float64(0.0):
        det = a * d - b * b
        result = wp.vec2d(d * rhs[0] - b * rhs[1], a * rhs[1] - b * rhs[0]) / det
        if wp.length(result) > radius:
            lower = wp.float64(0.0)
            upper = wp.length(rhs) / radius
            multiplier = wp.float64(0.5) * (lower + upper)
            for k in range(45):
                aa = a + multiplier
                dd = d + multiplier
                det = aa * dd - b * b
                result = wp.vec2d(dd * rhs[0] - b * rhs[1], aa * rhs[1] - b * rhs[0]) / det
                norm = wp.length(result)
                if wp.abs(norm - radius) <= wp.float64(1.0e-12) * radius:
                    break
                if norm > radius:
                    lower = multiplier
                else:
                    upper = multiplier
                inverse_x = wp.vec2d(dd * result[0] - b * result[1], aa * result[1] - b * result[0]) / det
                candidate = multiplier + (norm - radius) * norm / wp.dot(result, inverse_x)
                if candidate <= lower or candidate >= upper:
                    candidate = wp.float64(0.5) * (lower + upper)
                multiplier = candidate
    return result


@wp.kernel
def apply_wrenches(
    material: wp.array(dtype=wp.spatial_vector),
    external: wp.array(dtype=wp.spatial_vector),
    output: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    output[i] = material[i] + external[i]


@wp.kernel
def apply_damped_wrenches(
    material: wp.array(dtype=wp.spatial_vector),
    damping_world: wp.array(dtype=wp.spatial_vector),
    ref_velocity: wp.array(dtype=wp.spatial_vector),
    current_velocity: wp.array(dtype=wp.spatial_vector),
    external: wp.array(dtype=wp.spatial_vector),
    output: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    dv = current_velocity[i] - ref_velocity[i]
    d = damping_world[i]
    damp = wp.spatial_vector(
        d[0] * dv[0],
        d[1] * dv[1],
        d[2] * dv[2],
        d[3] * dv[3],
        d[4] * dv[4],
        d[5] * dv[5],
    )
    output[i] = material[i] - damp + external[i]


@wp.kernel
def solve_contacts_parallel(
    offsets: wp.array(dtype=int),
    ids: wp.array(dtype=int),
    frames: wp.array(dtype=wp.mat33d),
    angular: wp.array(dtype=wp.mat33d),
    response: wp.array(dtype=wp.mat33d),
    compliance: wp.array(dtype=wp.mat33d),
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
    max_iterations: int,
    tolerance: wp.float64,
    impulses: wp.array(dtype=wp.vec3d),
    wrench: wp.array(dtype=wp.spatial_vector),
    particles: int,
    stats: wp.array(dtype=wp.float64),
    fixed_material_velocity: bool,
    jacobi_relaxation: float,
):
    # One cooperative block handles all contacts, with strided work per lane.
    # Relaxed Jacobi solves the same contact fixed point as ordered PGS; the
    # convergence check uses the unrelaxed local impulse correction.
    index = wp.tid()
    body = index // 128
    lane = index % 128
    start = int(0)
    if body > 0:
        start = offsets[body * particles - 1]
    count = offsets[(body + 1) * particles - 1] - start
    linear = free[2 * body]
    spin = free[2 * body + 1]
    total = wp.vec3d(0.0)
    torque = wp.vec3d(0.0)
    residual = wp.float64(0.0)
    iterations = int(0)
    relaxation = wp.float64(jacobi_relaxation)
    if fixed_material_velocity:
        # With no material mobility, every contact acts on the same six body
        # velocities. Limit simultaneous Jacobi corrections as contacts grow.
        relaxation = wp.float64(1.0) / wp.float64(wp.max(count, 10))
    if count > 0:
        for iteration in range(max_iterations):
            local_delta = wp.vec3d(0.0)
            local_spin = wp.vec3d(0.0)
            local_residual = wp.float64(0.0)
            k = lane
            while k < count:
                i = ids[start + k]
                frame = frames[i]
                aj = angular[i]
                respond = response[i]
                matrix = compliance[i]
                old = impulses[i]
                slack_n = wp.float64(0.0)
                slack_t = wp.float64(0.0)
                if not fixed_material_velocity:
                    nz = frame[0, 2]
                    up_part = wp.max(nz, wp.float64(0.0))
                    down_part = wp.max(-nz, wp.float64(0.0))
                    horiz_part = wp.max(wp.float64(0.0), wp.float64(1.0) - nz * nz)
                    eff_n = wp.max(
                        wp.float64(effective_scale) * wp.pow(up_part, wp.float64(normal_power))
                        + wp.float64(horiz_scale) * horiz_part
                        + wp.float64(up_scale) * down_part,
                        wp.float64(0.5),
                    )
                    eff_t = wp.max(wp.float64(tangent_scale), wp.float64(0.5))
                    pm = wp.float64(particle_mass[i])
                    slack_n = wp.float64(1.0) / (eff_n * pm)
                    slack_t = wp.float64(1.0) / (eff_t * pm)
                relative = frame * linear + aj * spin - material[i] + wp.vec3d(old[0] * slack_n, old[1] * slack_t, old[2] * slack_t)
                normal = wp.max(wp.float64(0.0), old[0] - (relative[0] + gap[i]) / matrix[0, 0])
                dn = normal - old[0]
                # Apply the local normal correction only when solving this
                # contact's tangent block; other contacts see the old velocity.
                increment = wp.vec3d(dn, 0.0, 0.0)
                local_linear = linear + wp.transpose(frame) * increment / wp.float64(mass)
                local_angular = spin + respond * increment
                relative = frame * local_linear + aj * local_angular - material[i] + wp.vec3d(normal * slack_n, old[1] * slack_t, old[2] * slack_t)
                rhs = wp.vec2d(
                    matrix[1, 1] * old[1] + matrix[1, 2] * old[2] - relative[1],
                    matrix[2, 1] * old[1] + matrix[2, 2] * old[2] - relative[2],
                )
                tangent = friction_disk(matrix[1, 1], matrix[1, 2], matrix[2, 2], rhs, wp.float64(friction) * normal)
                delta_t = tangent - wp.vec2d(old[1], old[2])
                change = relaxation * wp.vec3d(dn, delta_t[0], delta_t[1])
                impulses[i] = old + change
                local_delta += wp.transpose(frame) * change
                local_spin += respond * change
                local_residual = wp.max(local_residual, wp.max(wp.abs(dn), wp.length(delta_t)))
                k += 128
            delta = wp.vec3d(
                wp.tile_extract(wp.tile_sum(wp.tile(local_delta[0])), 0),
                wp.tile_extract(wp.tile_sum(wp.tile(local_delta[1])), 0),
                wp.tile_extract(wp.tile_sum(wp.tile(local_delta[2])), 0),
            )
            ds = wp.vec3d(
                wp.tile_extract(wp.tile_sum(wp.tile(local_spin[0])), 0),
                wp.tile_extract(wp.tile_sum(wp.tile(local_spin[1])), 0),
                wp.tile_extract(wp.tile_sum(wp.tile(local_spin[2])), 0),
            )
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
            i = ids[start + k]
            local_total += wp.transpose(frames[i]) * impulses[i]
            local_torque += wp.transpose(angular[i]) * impulses[i]
            k += 128
        total = wp.vec3d(
            wp.tile_extract(wp.tile_sum(wp.tile(local_total[0])), 0),
            wp.tile_extract(wp.tile_sum(wp.tile(local_total[1])), 0),
            wp.tile_extract(wp.tile_sum(wp.tile(local_total[2])), 0),
        )
        torque = wp.vec3d(
            wp.tile_extract(wp.tile_sum(wp.tile(local_torque[0])), 0),
            wp.tile_extract(wp.tile_sum(wp.tile(local_torque[1])), 0),
            wp.tile_extract(wp.tile_sum(wp.tile(local_torque[2])), 0),
        )
    if lane == 0:
        wrench[body] = wp.spatial_vector(wp.vec3(total / wp.float64(dt)), wp.vec3(torque / wp.float64(dt)))
        stats[3 * body + 0] = wp.float64(count)
        stats[3 * body + 1] = wp.float64(iterations)
        stats[3 * body + 2] = residual
