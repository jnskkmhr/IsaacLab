# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU velocity fitting and unilateral Coulomb contact impulse equations."""

import warp as wp


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
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    radius: float,
    output: wp.array(dtype=wp.mat33d),
    poses: wp.array(dtype=wp.transform),
    sizes: wp.array(dtype=wp.vec3),
    shapes: wp.array(dtype=int),
    body: int,
    contact_band: float,
):
    # Accumulate in float32; contact equations retain float64.
    i = wp.tid()
    # Optional approximation: beyond this body-surface distance, use constant
    # particle velocity in contact selection instead of affine extrapolation.
    if contact_band > 0.0:
        distance, normal, point = geometry(wp.vec3d(x[i]), poses[body], sizes[body], shapes[body])
        if distance > wp.float64(contact_band):
            output[i] = wp.mat33d(0.0)
            return
    lhs = wp.mat33(0.0)
    rhs = wp.mat33(0.0)
    query = wp.hash_grid_query(grid, x[i], radius + 1.0e-6)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
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
    body: int,
    mass: float,
    inertia: wp.mat33d,
    gravity: wp.vec3,
    dt: float,
    free: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.mat33d),
):
    rotation = wp.quat_to_matrix(wp.normalize(wp.quatd(wp.transform_get_rotation(pose[body]))))
    world = rotation * inertia * wp.transpose(rotation)
    inv = wp.inverse(world)
    linear = wp.vec3d(wp.spatial_top(velocity[body]))
    angular = wp.vec3d(wp.spatial_bottom(velocity[body]))
    force = wp.vec3d(wp.spatial_top(external[body]))
    torque = wp.vec3d(wp.spatial_bottom(external[body]))
    free[0] = linear + wp.float64(dt) * (wp.vec3d(gravity) + force / wp.float64(mass))
    free[1] = angular + wp.float64(dt) * (inv * (torque - wp.cross(angular, world * angular)))
    inverse[0] = inv


@wp.kernel
def prepare_contacts(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    gradients: wp.array(dtype=wp.mat33d),
    pose: wp.array(dtype=wp.transform),
    sizes: wp.array(dtype=wp.vec3),
    shapes: wp.array(dtype=int),
    body: int,
    free: wp.array(dtype=wp.vec3d),
    inverse: wp.array(dtype=wp.mat33d),
    mass: float,
    particle_mass: wp.array(dtype=float),
    dt: float,
    effective_scale: float,
    stabilization: float,
    flags: wp.array(dtype=int),
    frames: wp.array(dtype=wp.mat33d),
    angular_jacobian: wp.array(dtype=wp.mat33d),
    response: wp.array(dtype=wp.mat33d),
    compliance: wp.array(dtype=wp.mat33d),
    material: wp.array(dtype=wp.vec3d),
    gap_speed: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    distance, normal, point = geometry(wp.vec3d(x[i]), pose[body], sizes[body], shapes[body])
    velocity = wp.vec3d(v[i]) + wp.transpose(gradients[i]) * (point - wp.vec3d(x[i]))
    arm = point - wp.vec3d(wp.transform_get_translation(pose[body]))
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
        t = wp.normalize(wp.cross(n, basis))
        t2 = wp.cross(n, t)
        frame = wp.mat33d(n[0], n[1], n[2], t[0], t[1], t[2], t2[0], t2[1], t2[2])
        a0 = wp.cross(arm, n)
        a1 = wp.cross(arm, t)
        a2 = wp.cross(arm, t2)
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
    friction: float,
    dt: float,
    max_iterations: int,
    tolerance: wp.float64,
    impulses: wp.array(dtype=wp.vec3d),
    wrench: wp.array(dtype=wp.spatial_vector),
    body: int,
    stats: wp.array(dtype=wp.float64),
):
    # One cooperative block handles all contacts, with strided work per lane.
    # Relaxed Jacobi solves the same contact fixed point as ordered PGS; the
    # convergence check uses the unrelaxed local impulse correction.
    lane = wp.tid()
    count = offsets[offsets.shape[0] - 1]
    linear = free[0]
    spin = free[1]
    total = wp.vec3d(0.0)
    torque = wp.vec3d(0.0)
    residual = wp.float64(0.0)
    iterations = int(0)
    relaxation = wp.float64(0.1)
    if count > 0:
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
                slack = wp.float64(1.0) / (wp.float64(effective_scale) * wp.float64(particle_mass[i]))
                relative = frame * linear + aj * spin - material[i] + old * slack
                normal = wp.max(wp.float64(0.0), old[0] - (relative[0] + gap[i]) / matrix[0, 0])
                dn = normal - old[0]
                # Apply the local normal correction only when solving this
                # contact's tangent block; other contacts see the old velocity.
                increment = wp.vec3d(dn, 0.0, 0.0)
                local_linear = linear + wp.transpose(frame) * increment / wp.float64(mass)
                local_angular = spin + respond * increment
                normal_state = wp.vec3d(normal, old[1], old[2])
                relative = frame * local_linear + aj * local_angular - material[i] + normal_state * slack
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
            i = ids[k]
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
        stats[0] = wp.float64(count)
        stats[1] = wp.float64(iterations)
        stats[2] = residual


@wp.kernel
def solver_health(
    stats: wp.array(dtype=wp.float64),
    wrench: wp.array(dtype=wp.spatial_vector),
    body: int,
    failures: wp.array(dtype=int),
    error: wp.array(dtype=int),
):
    if stats[2] >= wp.float64(1.0e-9):
        failures[0] += 1
    force = wrench[body]
    for a in range(6):
        if not wp.isfinite(force[a]):
            error[0] = 3
