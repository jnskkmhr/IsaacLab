# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""GPU body-velocity initializer for one rigid body and finite material mobility.

At fixed body velocity each independent effective particle has a closed-form
normal impulse and Coulomb tangential impulse. Solve momentum balance for the
body's six velocities with a piecewise analytic Jacobian and backtracking.
The caller then verifies/polishes with the original contact solver. This changes
numerical solution strategy, not the effective-mass or friction model.

Each GPU block solves one independent environment with one dynamic body.
Contacts within an articulated asset are not supported. Fixed-material-velocity mode must use
its existing solver because the local material mobility is zero.
"""

import warp as wp

wp.set_module_options({"enable_backward": False, "max_unroll": 0})
vec6d = wp.types.vector(length=6, dtype=wp.float64)
mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
mat36d = wp.types.matrix(shape=(3, 6), dtype=wp.float64)
mat63d = wp.types.matrix(shape=(6, 3), dtype=wp.float64)


@wp.func
def _contact_response(
    q: vec6d,
    frame: wp.mat33d,
    angular: wp.mat33d,
    material: wp.vec3d,
    gap: wp.float64,
    mobility_n: wp.float64,
    mobility_t: wp.float64,
    friction: wp.float64,
):
    relative = frame * wp.vec3d(q[0], q[1], q[2]) + angular * wp.vec3d(q[3], q[4], q[5]) - material
    normal = wp.max(wp.float64(0.0), -(relative[0] + gap) / mobility_n)
    impulse = wp.vec3d(normal, 0.0, 0.0)
    derivative = wp.mat33d(0.0)
    if normal > wp.float64(0.0):
        derivative[0, 0] = -wp.float64(1.0) / mobility_n
        tangent = -wp.vec2d(relative[1], relative[2]) / mobility_t
        length = wp.length(tangent)
        radius = friction * normal
        if radius > wp.float64(0.0) and length <= radius:
            impulse[1] = tangent[0]
            impulse[2] = tangent[1]
            derivative[1, 1] = -wp.float64(1.0) / mobility_t
            derivative[2, 2] = -wp.float64(1.0) / mobility_t
        elif radius > wp.float64(0.0):
            direction = tangent / length
            impulse[1] = radius * direction[0]
            impulse[2] = radius * direction[1]
            derivative[1, 0] = -friction * direction[0] / mobility_n
            derivative[2, 0] = -friction * direction[1] / mobility_n
            scale = -radius / (length * mobility_t)
            derivative[1, 1] = scale * (wp.float64(1.0) - direction[0] * direction[0])
            derivative[2, 2] = scale * (wp.float64(1.0) - direction[1] * direction[1])
            derivative[1, 2] = -scale * direction[0] * direction[1]
            derivative[2, 1] = derivative[1, 2]
    return impulse, derivative


@wp.func
def _rft_mobilities(
    frame: wp.mat33d,
    p_mass: float,
    kappa_vert: float,
    kappa_horiz: float,
    kappa_up: float,
    kappa_tan: float,
    power_z: float,
):
    nz = frame[0, 2]
    up_part = wp.max(nz, wp.float64(0.0))
    down_part = wp.max(-nz, wp.float64(0.0))
    horiz_part = wp.max(wp.float64(0.0), wp.float64(1.0) - nz * nz)
    eff_n = wp.max(
        wp.float64(kappa_vert) * wp.pow(up_part, wp.float64(power_z))
        + wp.float64(kappa_horiz) * horiz_part
        + wp.float64(kappa_up) * down_part,
        wp.float64(0.5),
    )
    eff_t = wp.max(wp.float64(kappa_tan), wp.float64(0.5))
    pm = wp.float64(p_mass)
    return wp.float64(1.0) / (eff_n * pm), wp.float64(1.0) / (eff_t * pm)


@wp.func
def _linear_solve(matrix: mat66d, rhs: vec6d):
    a = matrix
    b = rhs
    for k in range(6):
        pivot = k
        for r in range(k + 1, 6):
            if wp.abs(a[r, k]) > wp.abs(a[pivot, k]):
                pivot = r
        for c in range(6):
            tmp = a[k, c]
            a[k, c] = a[pivot, c]
            a[pivot, c] = tmp
        tmp = b[k]
        b[k] = b[pivot]
        b[pivot] = tmp
        if wp.abs(a[k, k]) < wp.float64(1e-20):
            return vec6d(0.0)
        for r in range(k + 1, 6):
            factor = a[r, k] / a[k, k]
            for c in range(k + 1, 6):
                a[r, c] = a[r, c] - factor * a[k, c]
            b[r] = b[r] - factor * b[k]
    x = vec6d(0.0)
    for rev in range(6):
        r = 5 - rev
        v = b[r]
        for c in range(r + 1, 6):
            v -= a[r, c] * x[c]
        x[r] = v / a[r, r]
    return x


@wp.kernel
def initialize_body_velocity_solve(
    particles: int,
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
    impulses: wp.array(dtype=wp.vec3d),
    initial: wp.array(dtype=wp.vec3d),
    stats: wp.array(dtype=wp.float64),
):
    index = wp.tid()
    body = index // 128
    lane = index % 128
    start = int(0)
    if body > 0:
        start = offsets[body * particles - 1]
    count = offsets[(body + 1) * particles - 1] - start
    origin = vec6d(
        free[2 * body][0],
        free[2 * body][1],
        free[2 * body][2],
        free[2 * body + 1][0],
        free[2 * body + 1][1],
        free[2 * body + 1][2],
    )
    q = origin
    iterations = int(0)
    error = wp.float64(0.0)
    if count > 0:
        for it in range(30):
            local_force = vec6d(0.0)
            local_jacobian = mat66d(0.0)
            k = lane
            while k < count:
                i = ids[start + k]
                mob_n, mob_t = _rft_mobilities(
                    frames[i],
                    particle_mass[i],
                    effective_scale,
                    horiz_scale,
                    up_scale,
                    tangent_scale,
                    normal_power,
                )
                impulse, derivative = _contact_response(
                    q,
                    frames[i],
                    angular[i],
                    material[i],
                    gap[i],
                    mob_n,
                    mob_t,
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
            force = vec6d(0.0)
            jacobian = mat66d(0.0)
            for r in range(6):
                force[r] = wp.tile_extract(wp.tile_sum(wp.tile(local_force[r])), 0)
                for c in range(6):
                    jacobian[r, c] = -wp.tile_extract(wp.tile_sum(wp.tile(local_jacobian[r, c])), 0)
                jacobian[r, r] += wp.float64(1.0)
            residual = q - origin - force
            error = wp.length(residual)
            iterations = it + 1
            if error < wp.float64(1e-12):
                break
            delta = _linear_solve(jacobian, residual)
            alpha = wp.float64(1.0)
            accepted = bool(False)
            for trial in range(16):
                candidate = q - alpha * delta
                local_candidate = vec6d(0.0)
                k = lane
                while k < count:
                    i = ids[start + k]
                    mob_n, mob_t = _rft_mobilities(
                        frames[i],
                        particle_mass[i],
                        effective_scale,
                        horiz_scale,
                        up_scale,
                        tangent_scale,
                        normal_power,
                    )
                    impulse, unused = _contact_response(
                        candidate,
                        frames[i],
                        angular[i],
                        material[i],
                        gap[i],
                        mob_n,
                        mob_t,
                        wp.float64(friction),
                    )
                    linear = wp.transpose(frames[i]) * impulse / wp.float64(mass)
                    spin = response[i] * impulse
                    local_candidate += vec6d(linear[0], linear[1], linear[2], spin[0], spin[1], spin[2])
                    k += 128
                candidate_force = vec6d(0.0)
                for r in range(6):
                    candidate_force[r] = wp.tile_extract(wp.tile_sum(wp.tile(local_candidate[r])), 0)
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
        i = ids[start + k]
        mob_n, mob_t = _rft_mobilities(
            frames[i],
            particle_mass[i],
            effective_scale,
            horiz_scale,
            up_scale,
            tangent_scale,
            normal_power,
        )
        impulse, unused = _contact_response(
            q,
            frames[i],
            angular[i],
            material[i],
            gap[i],
            mob_n,
            mob_t,
            wp.float64(friction),
        )
        impulses[i] = impulse
        linear = wp.transpose(frames[i]) * impulse / wp.float64(mass)
        spin = response[i] * impulse
        local_force += vec6d(linear[0], linear[1], linear[2], spin[0], spin[1], spin[2])
        k += 128
    force = vec6d(0.0)
    for r in range(6):
        force[r] = wp.tile_extract(wp.tile_sum(wp.tile(local_force[r])), 0)
    # Keep the original solver as a cold-start fallback if the initializer stalls.
    final_error = wp.length(q - origin - force)
    fallback = not wp.isfinite(final_error) or final_error > wp.float64(1e-8)
    if fallback:
        k = lane
        while k < count:
            impulses[ids[start + k]] = wp.vec3d(0.0)
            k += 128
        force = vec6d(0.0)
    if lane == 0:
        initial[2 * body] = free[2 * body] + wp.vec3d(force[0], force[1], force[2])
        initial[2 * body + 1] = free[2 * body + 1] + wp.vec3d(force[3], force[4], force[5])
        stats[3 * body + 0] = wp.float64(iterations)
        stats[3 * body + 1] = final_error
        stats[3 * body + 2] = wp.float64(fallback)
