# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Independent proxy contact blocks sharing the sequential contact equations."""

import warp as wp

from .body_velocity import initialize_body_velocity_solve_block
from .kernels import geometry, solve_contacts_parallel_block

wp.set_module_options({"enable_backward": False, "max_unroll": 0})


@wp.struct
class ContactProblem:
    body: int
    shape: int
    env_idx: int
    count: int
    margin: float
    surface_pose: wp.array(dtype=wp.transform)
    sizes: wp.array(dtype=wp.vec3)
    center: wp.array(dtype=wp.vec3d)
    inverse: wp.array(dtype=wp.mat33d)
    offsets: wp.array(dtype=int)
    ids: wp.array(dtype=int)
    frames: wp.array(dtype=wp.mat33d)
    angular: wp.array(dtype=wp.mat33d)
    response: wp.array(dtype=wp.mat33d)
    compliance: wp.array(dtype=wp.mat33d)
    material: wp.array(dtype=wp.vec3d)
    gap: wp.array(dtype=wp.float64)
    particle_mass: wp.array(dtype=float)
    free: wp.array(dtype=wp.vec3d)
    mass: float
    friction: float
    impulses: wp.array(dtype=wp.vec3d)
    initial: wp.array(dtype=wp.vec3d)
    stats: wp.array(dtype=wp.float64)
    initializer_stats: wp.array(dtype=wp.float64)


@wp.kernel
def gather_active_particles_batched(
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
def prepare_proxies_batched(
    problems: wp.array(dtype=ContactProblem),
    poses: wp.array(dtype=wp.transform),
    velocities: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inertia: wp.array(dtype=wp.mat33),
    shape_transform: wp.array(dtype=wp.transform),
):
    problem = wp.tid()
    p = problems[problem]
    body = p.body
    shape = p.shape
    p.offsets[p.count - 1] = 0
    p.surface_pose[0] = wp.transform_multiply(poses[body], shape_transform[shape])
    p.center[0] = wp.vec3d(wp.transform_point(poses[body], body_com[body]))
    rotation = wp.quat_to_matrix(wp.normalize(wp.quatd(wp.transform_get_rotation(poses[body]))))
    p.inverse[0] = wp.inverse(rotation * wp.mat33d(body_inertia[body]) * wp.transpose(rotation))
    p.free[0] = wp.vec3d(wp.spatial_top(velocities[body]))
    p.free[1] = wp.vec3d(wp.spatial_bottom(velocities[body]))


@wp.kernel
def fit_and_prepare_contacts_batched(
    problems: wp.array(dtype=ContactProblem),
    grid: wp.uint64,
    x_all: wp.array(dtype=wp.vec3),
    v_all: wp.array(dtype=wp.vec3),
    radius: float,
    dt: float,
    contact_band: float,
    effective_scale: float,
    horiz_scale: float,
    up_scale: float,
    tangent_scale: float,
    normal_power: float,
    stabilization: float,
    clear_impulses: bool,
    flags_flat: wp.array(dtype=int),
):
    problem, i = wp.tid()
    p = problems[problem]
    k_eval = p.count
    if i >= k_eval:
        return
    if clear_impulses:
        p.impulses[i] = wp.vec3d(0.0)
    flat_idx = problem * k_eval + i
    flags_flat[flat_idx] = 0
    pm_f = p.particle_mass[i]
    if pm_f <= 0.0:
        return
    env_offset = p.env_idx * k_eval
    xi = x_all[env_offset + i]
    vi = v_all[env_offset + i]
    raw_dist, normal, point = geometry(wp.vec3d(xi), p.surface_pose[0], p.sizes[0], 0)
    distance = raw_dist - wp.float64(p.margin)
    arm = point - p.center[0]
    surface = p.free[0] + wp.cross(p.free[1], arm)
    app0 = wp.dot(normal, surface - wp.vec3d(vi))
    grad = wp.mat33d(0.0)
    if not (contact_band > 0.0 and raw_dist > wp.float64(contact_band)):
        if distance <= (wp.max(app0, wp.float64(0.0)) + wp.float64(4.0)) * wp.float64(dt):
            lhs = wp.mat33(0.0)
            rhs = wp.mat33(0.0)
            query = wp.hash_grid_query(grid, xi, radius + 1.0e-6, p.env_idx)
            j = int(0)
            while wp.hash_grid_query_next(query, j):
                dx = wp.vec3(x_all[j]) - wp.vec3(xi)
                dv = wp.vec3(v_all[j]) - wp.vec3(vi)
                q = wp.length(dx) / float(radius)
                if q < float(1.0):
                    a = float(1.0) - q
                    weight = a * a * a * a * (float(1.0) + float(4.0) * q)
                    lhs += weight * wp.outer(dx, dx)
                    rhs += weight * wp.outer(dx, dv)
            ridge = float(0.01) * wp.trace(lhs) / float(3.0) + float(1.0e-12)
            lhs += wp.mat33(ridge, 0.0, 0.0, 0.0, ridge, 0.0, 0.0, 0.0, ridge)
            grad = wp.mat33d(wp.inverse(lhs) * rhs)
    velocity = wp.vec3d(vi) + wp.transpose(grad) * (point - wp.vec3d(xi))
    approach = wp.dot(normal, surface - velocity)
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
        tangent = wp.normalize(wp.cross(n, basis))
        tangent2 = wp.cross(n, tangent)
        frame = wp.mat33d(n[0], n[1], n[2], tangent[0], tangent[1], tangent[2], tangent2[0], tangent2[1], tangent2[2])
        a0 = wp.cross(arm, n)
        a1 = wp.cross(arm, tangent)
        a2 = wp.cross(arm, tangent2)
        angular = wp.mat33d(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2])
        respond = p.inverse[0] * wp.transpose(angular)
        k = frame * wp.transpose(frame) / wp.float64(p.mass) + angular * respond
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
        pm = wp.float64(pm_f)
        slack_n = wp.float64(1.0) / (pm * eff_n)
        slack_t = wp.float64(1.0) / (pm * eff_t)
        k += wp.mat33d(slack_n, 0.0, 0.0, 0.0, slack_t, 0.0, 0.0, 0.0, slack_t)
        p.frames[i] = frame
        p.angular[i] = angular
        p.response[i] = respond
        p.compliance[i] = k
        p.material[i] = frame * velocity
        p.gap[i] = (
            wp.max(distance, wp.float64(0.0)) + wp.float64(stabilization) * wp.min(distance, wp.float64(0.0))
        ) / wp.float64(dt)


@wp.kernel
def compact_contacts_batched(
    problems: wp.array(dtype=ContactProblem),
    flags_flat: wp.array(dtype=int),
    offsets_flat: wp.array(dtype=int),
):
    problem, i = wp.tid()
    p = problems[problem]
    k_eval = p.count
    if i >= k_eval:
        return
    base = problem * k_eval
    prev_total = int(0)
    if base > 0:
        prev_total = offsets_flat[base - 1]
    local_off = offsets_flat[base + i] - prev_total
    if flags_flat[base + i] != 0:
        p.ids[local_off - 1] = i
    if i == k_eval - 1:
        p.offsets[k_eval - 1] = local_off


@wp.kernel
def initialize_contacts_batched(
    problems: wp.array(dtype=ContactProblem),
    effective_scale: float,
    horiz_scale: float,
    up_scale: float,
    tangent_scale: float,
    normal_power: float,
):
    problem, lane = wp.tid()
    p = problems[problem]
    initialize_body_velocity_solve_block(
        lane,
        p.offsets,
        p.ids,
        p.frames,
        p.angular,
        p.response,
        p.material,
        p.gap,
        p.particle_mass,
        p.free,
        p.mass,
        effective_scale,
        horiz_scale,
        up_scale,
        tangent_scale,
        normal_power,
        p.friction,
        p.impulses,
        p.initial,
        p.initializer_stats,
    )


@wp.kernel
def solve_contacts_batched(
    problems: wp.array(dtype=ContactProblem),
    effective_scale: float,
    horiz_scale: float,
    up_scale: float,
    tangent_scale: float,
    normal_power: float,
    dt: float,
    max_iterations: int,
    tolerance: wp.float64,
    relaxation: float,
    use_initializer: bool,
    wrench: wp.array(dtype=wp.spatial_vector),
):
    problem, lane = wp.tid()
    p = problems[problem]
    free = p.free
    if use_initializer:
        free = p.initial
    solve_contacts_parallel_block(
        lane,
        p.offsets,
        p.ids,
        p.frames,
        p.angular,
        p.response,
        p.compliance,
        p.material,
        p.gap,
        p.particle_mass,
        free,
        p.mass,
        effective_scale,
        horiz_scale,
        up_scale,
        tangent_scale,
        normal_power,
        p.friction,
        dt,
        max_iterations,
        tolerance,
        p.impulses,
        wrench,
        problem,
        p.stats,
        False,
        relaxation,
    )
