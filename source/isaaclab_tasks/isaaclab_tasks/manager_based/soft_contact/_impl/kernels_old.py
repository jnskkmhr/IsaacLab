import warp as wp

"""
index slicing kernels
"""


@wp.kernel
def update_array_with_index(
    env_ids: wp.array(dtype=wp.int64),  # (n, )
    source_array: wp.array(dtype=wp.float32),  # (n, )
    target_array: wp.array(dtype=wp.float32),  # (num_envs, )
):
    i = wp.tid()
    env_id = env_ids[i]
    target_array[env_id] = source_array[i]


"""
expand float arrays
"""


@wp.kernel
def expand_array_1d_to_3d(
    source_array: wp.array1d(dtype=wp.float32),
    target_array: wp.array3d(dtype=wp.float32),
):
    """
    Equivalent of source_array.unsqueeze(0).unsqueeze(0).repeat(N, M, 1) in torch
    This transforms shape to (D,) to (N, M, D)
    """

    i, j, k = wp.tid()
    target_array[i, j, k] = source_array[k]


@wp.kernel
def expand_array_1d_to_4d(
    source_array: wp.array1d(dtype=wp.float32),
    target_array: wp.array4d(dtype=wp.float32),
):
    """
    Equivalent of source_array.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N, M, K, 1) in torch
    This transforms shape to (D,) to (N, M, K, D)
    """

    i, j, k, d = wp.tid()
    target_array[i, j, k, d] = source_array[d]


@wp.kernel
def expand_array_2d_to_3d(
    source_array: wp.array2d(dtype=wp.float32),
    target_array: wp.array3d(dtype=wp.float32),
):
    """
    Equivalent of source_array.unsqueeze(0).repeat(N, 1, 1) in torch
    This transforms shape to (B, D) to (N, B, D)
    """

    i, j, k = wp.tid()
    target_array[i, j, k] = source_array[j, k]


@wp.kernel
def expand_array_2d_to_4d(
    source_array: wp.array2d(dtype=wp.float32),
    target_array: wp.array4d(dtype=wp.float32),
):
    """
    Equivalent of source_array.unsqueeze(0).unsqueeze(1).repeat(N, M, 1, 1)
    This transforms shape to (B, D) to (N, M, B, D)
    """
    i, j, k, d = wp.tid()
    target_array[i, j, k, d] = source_array[k, d]


"""
expand vector arrays
"""


@wp.kernel
def expand_vec3_to_1d_vec3(
    source_array: wp.vec3f,
    target_array: wp.array1d(dtype=wp.vec3f),
):
    """
    Repeat along dim=0
    This transforms shape to (3,) to (N, 3)
    """

    i = wp.tid()
    target_array[i] = source_array


@wp.kernel
def expand_vec3_to_2d_vec3(
    source_array: wp.vec3f,
    target_array: wp.array2d(dtype=wp.vec3f),
):
    """
    Repeat along dim=0, 1
    This transforms shape to (3,) to (N, M, 3)
    """

    i, j = wp.tid()
    target_array[i, j] = source_array


@wp.kernel
def expand_vec3_to_3d_vec3(
    source_array: wp.vec3f,
    target_array: wp.array3d(dtype=wp.vec3f),
):
    """
    Repeat along dim=0, 1, 2
    This transforms shape to (3,) to (N, M, K, 3)
    """

    i, j, k = wp.tid()
    target_array[i, j, k] = source_array


@wp.kernel
def expand_1d_vec3_to_2d_vec3(
    source_array: wp.array1d(dtype=wp.vec3f),
    target_array: wp.array2d(dtype=wp.vec3f),
):
    """
    Repeat along dim=0
    This transforms shape to (M, 3) to (N, M, 3)
    """

    i, j = wp.tid()
    target_array[i, j] = source_array[j]


@wp.kernel
def expand_1d_vec3_to_3d_vec3(
    source_array: wp.array1d(dtype=wp.vec3f),
    target_array: wp.array3d(dtype=wp.vec3f),
):
    """
    Repeat along dim=0, 1
    This transforms shape to (K, 3) to (N, M, K, 3)
    """

    i, j, k = wp.tid()
    target_array[i, j, k] = source_array[k]


"""
shape kernels
"""


@wp.kernel
def plane_contact_points(
    edge_x: wp.vec2f,           # (x_min, x_max)
    edge_y: wp.vec2f,           # (y_min, y_max)
    z_offset: wp.float32,
    nx: wp.int32,
    ny: wp.int32,
    contact_points: wp.array1d(dtype=wp.vec3f),   # (nx * ny,)
    contact_normals: wp.array1d(dtype=wp.vec3f),   # (nx * ny,)
):
    """
    Generate contact points and outward normals for a single planar face
    on the XY plane at the given z offset. Normal = (0, 0, -1).

    Thread dims: (ny, nx).
    """
    i, j = wp.tid()

    # u = wp.float32(j) / wp.max(wp.float32(nx) - 1.0, 1.0)
    # v = wp.float32(i) / wp.max(wp.float32(ny) - 1.0, 1.0)
    u = wp.float32(j)/wp.float32(nx-1)
    v = wp.float32(i)/wp.float32(ny-1)

    pos = wp.vec3f(
        edge_x[0] + u * (edge_x[1] - edge_x[0]),
        edge_y[0] + v * (edge_y[1] - edge_y[0]),
        z_offset,
    )

    idx = i * nx + j
    contact_points[idx] = pos
    contact_normals[idx] = wp.vec3f(0.0, 0.0, -1.0)


@wp.kernel
def box_contact_points(
    edge_x: wp.vec2f,           # (x_min, x_max)
    edge_y: wp.vec2f,           # (y_min, y_max)
    edge_z: wp.vec2f,           # (z_min, z_max)
    nx: wp.int32,
    ny: wp.int32,
    contact_points: wp.array1d(dtype=wp.vec3f),   # (6 * nx * ny,)
    contact_normals: wp.array1d(dtype=wp.vec3f),   # (6 * nx * ny,)
):
    """
    Generate contact points and outward normals for the 6 faces of an
    axis-aligned box defined by edge bounds.

    Face ordering: -Z, +Z, -Y, +Y, -X, +X.
    Thread dims: (6, ny, nx).
    """
    face_id, i, j = wp.tid()
    pts_per_face = nx * ny

    # Parametric coordinates in [0, 1]
    u = wp.float32(j) / wp.max(wp.float32(nx) - 1.0, 1.0)
    v = wp.float32(i) / wp.max(wp.float32(ny) - 1.0, 1.0)

    x_min = edge_x[0]
    x_max = edge_x[1]
    y_min = edge_y[0]
    y_max = edge_y[1]
    z_min = edge_z[0]
    z_max = edge_z[1]

    idx = face_id * pts_per_face + i * nx + j

    pos = wp.vec3f(0.0, 0.0, 0.0)
    nrm = wp.vec3f(0.0, 0.0, 0.0)

    # face 0: -Z  (bottom)
    if face_id == 0:
        pos = wp.vec3f(
            x_min + u * (x_max - x_min),
            y_min + v * (y_max - y_min),
            z_min,
        )
        nrm = wp.vec3f(0.0, 0.0, -1.0)
    # face 1: +Z  (top)
    elif face_id == 1:
        pos = wp.vec3f(
            x_min + u * (x_max - x_min),
            y_min + v * (y_max - y_min),
            z_max,
        )
        nrm = wp.vec3f(0.0, 0.0, 1.0)
    # face 2: -Y  (front)
    elif face_id == 2:
        pos = wp.vec3f(
            x_min + u * (x_max - x_min),
            y_min,
            z_min + v * (z_max - z_min),
        )
        nrm = wp.vec3f(0.0, -1.0, 0.0)
    # face 3: +Y  (back)
    elif face_id == 3:
        pos = wp.vec3f(
            x_min + u * (x_max - x_min),
            y_max,
            z_min + v * (z_max - z_min),
        )
        nrm = wp.vec3f(0.0, 1.0, 0.0)
    # face 4: -X  (left)
    elif face_id == 4:
        pos = wp.vec3f(
            x_min,
            y_min + u * (y_max - y_min),
            z_min + v * (z_max - z_min),
        )
        nrm = wp.vec3f(-1.0, 0.0, 0.0)
    # face 5: +X  (right)
    elif face_id == 5:
        pos = wp.vec3f(
            x_max,
            y_min + u * (y_max - y_min),
            z_min + v * (z_max - z_min),
        )
        nrm = wp.vec3f(1.0, 0.0, 0.0)

    contact_points[idx] = pos
    contact_normals[idx] = nrm


@wp.kernel
def sphere_contact_points(
    radius: wp.float32,
    center: wp.vec3f,
    n_theta: wp.int32,
    n_phi: wp.int32,
    contact_points: wp.array1d(dtype=wp.vec3f),   # (n_theta * n_phi,)
    contact_normals: wp.array1d(dtype=wp.vec3f),   # (n_theta * n_phi,)
):
    """
    Generate contact points and outward normals for a sphere using
    a regular (theta, phi) spherical grid.

    theta ∈ (0, π)   — polar angle  (excludes poles for uniform-ish coverage)
    phi   ∈ [0, 2π)  — azimuthal angle

    Thread dims: (n_theta, n_phi).
    """
    i, j = wp.tid()

    # Avoid exact poles: offset by half-step
    theta = wp.PI * (wp.float32(i) + 0.5) / wp.float32(n_theta)
    phi = 2.0 * wp.PI * wp.float32(j) / wp.float32(n_phi)

    sin_theta = wp.sin(theta)
    cos_theta = wp.cos(theta)
    sin_phi = wp.sin(phi)
    cos_phi = wp.cos(phi)

    # Outward unit normal
    nrm = wp.vec3f(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)

    idx = i * n_phi + j
    contact_points[idx] = center + radius * nrm
    contact_normals[idx] = nrm


"""
state processing kernels
"""


@wp.kernel
def compute_contact_point_pos_w(
    body_pos: wp.array2d(dtype=wp.vec3f),  # (N, B, 3)
    body_quat: wp.array2d(dtype=wp.quatf),  # (N, B, 4)
    contact_point_local: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    contact_point_pos_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
):
    env_id, body_id, contact_point_id = wp.tid()
    contact_point_pos_w[env_id, body_id, contact_point_id] = body_pos[env_id, body_id] + wp.quat_rotate(
        body_quat[env_id, body_id], contact_point_local[env_id, body_id, contact_point_id]
    )


@wp.kernel
def compute_contact_point_lin_vel_w(
    body_pos: wp.array2d(dtype=wp.vec3f),  # (N, B, 3)
    body_lin_vel: wp.array2d(dtype=wp.vec3f),  # (N, B, 3)
    body_ang_vel: wp.array2d(dtype=wp.vec3f),  # (N, B, 3)
    contact_point_pos_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    contact_point_lin_vel_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
):

    env_id, body_id, contact_point_id = wp.tid()
    contact_point_lin_vel_w[env_id, body_id, contact_point_id] = body_lin_vel[env_id, body_id] + wp.cross(
        body_ang_vel[env_id, body_id],
        contact_point_pos_w[env_id, body_id, contact_point_id] - body_pos[env_id, body_id],
    )


@wp.kernel
def compute_normal_direction_w(
    body_quat: wp.array2d(dtype=wp.quatf),  # (N, B, 4)
    normal_direction_local: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    normal_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
):
    env_id, body_id, contact_point_id = wp.tid()
    normal_direction_w[env_id, body_id, contact_point_id] = wp.quat_rotate(
        body_quat[env_id, body_id], normal_direction_local[env_id, body_id, contact_point_id]
    )


@wp.kernel
def compute_v_direction_w(
    contact_point_lin_vel_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    v_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
):

    env_id, body_id, contact_point_id = wp.tid()
    v = contact_point_lin_vel_w[env_id, body_id, contact_point_id]
    v_norm = wp.length(v)
    v_direction_w[env_id, body_id, contact_point_id] = v / (v_norm + 1e-6)


@wp.kernel
def compute_r_direction_w(
    n_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    z_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    v_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    r_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    thresh: wp.float32,
):

    env_id, body_id, contact_point_id = wp.tid()

    n_element = n_direction_w[env_id, body_id, contact_point_id]
    z_element = z_direction_w[env_id, body_id, contact_point_id]
    v_element = v_direction_w[env_id, body_id, contact_point_id]

    n_rt = n_element - wp.dot(n_element, z_element) * z_element
    n_rt_norm = wp.length(n_rt)
    n_rt_dir = n_rt / (n_rt_norm + 1e-6)
    vr = v_element - wp.dot(v_element, z_element) * z_element
    vr_norm = wp.length(vr)
    r = vr / (vr_norm + 1e-6)
    mask = wp.float32(vr_norm < thresh)
    r_direction_w[env_id, body_id, contact_point_id] = r * (1.0 - mask) + n_rt_dir * mask


@wp.kernel
def compute_t_direction_w(
    z_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    r_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    t_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
):

    env_id, body_id, contact_point_id = wp.tid()
    z_element = z_direction_w[env_id, body_id, contact_point_id]
    r_element = r_direction_w[env_id, body_id, contact_point_id]
    t_direction_w[env_id, body_id, contact_point_id] = wp.cross(z_element, r_element)


@wp.kernel
def compute_intrusion_angle(
    z_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    v_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    r_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    intrusion_angle: wp.array3d(dtype=wp.float32),  # (N, B, C)
):

    env_id, body_id, contact_point_id = wp.tid()
    z_element = z_direction_w[env_id, body_id, contact_point_id]
    v_element = v_direction_w[env_id, body_id, contact_point_id]
    r_element = r_direction_w[env_id, body_id, contact_point_id]

    vdotr = wp.dot(v_element, r_element)
    vdotz = wp.dot(v_element, z_element)
    intrusion_angle[env_id, body_id, contact_point_id] = wp.acos(vdotr) * (
        wp.float32(vdotz < 0.0) - wp.float32(vdotz >= 0.0)
    )


@wp.kernel
def compute_tilt_angle(
    z_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    n_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    r_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    t_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    tilt_angle: wp.array3d(dtype=wp.float32),  # (N, B, C)
):

    env_id, body_id, contact_point_id = wp.tid()
    z_element = z_direction_w[env_id, body_id, contact_point_id]
    n_element = n_direction_w[env_id, body_id, contact_point_id]
    r_element = r_direction_w[env_id, body_id, contact_point_id]
    t_element = t_direction_w[env_id, body_id, contact_point_id]

    ndotr = wp.dot(n_element, r_element)
    ndott = wp.dot(n_element, t_element)
    ndotz = wp.dot(n_element, z_element)
    n_rtz = wp.vec3f(ndotr, ndott, ndotz)
    reflection_matrix = 1.0 - 2.0 * wp.float32(ndotr < 0.0)
    n_rtz = n_rtz * reflection_matrix
    tilt_angle[env_id, body_id, contact_point_id] = -wp.acos(n_rtz[2]) + wp.PI * wp.float32(n_rtz[2] < 0.0)


@wp.kernel
def compute_twist_angle(
    z_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    n_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    r_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    t_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    n_rtz_direction_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    twist_angle: wp.array3d(dtype=wp.float32),  # (N, B, C)
):

    env_id, body_id, contact_point_id = wp.tid()
    z_element = z_direction_w[env_id, body_id, contact_point_id]
    n_element = n_direction_w[env_id, body_id, contact_point_id]
    r_element = r_direction_w[env_id, body_id, contact_point_id]
    t_element = t_direction_w[env_id, body_id, contact_point_id]

    ndotr = wp.dot(n_element, r_element)
    ndott = wp.dot(n_element, t_element)
    ndotz = wp.dot(n_element, z_element)
    n_rtz = wp.vec3f(ndotr, ndott, ndotz)
    reflection_matrix = 1.0 - 2.0 * wp.float32(ndotr < 0.0)
    n_rtz = n_rtz * reflection_matrix
    n_rtz_norm = wp.length(n_rtz)
    n_rtz = n_rtz / (n_rtz_norm + 1e-6)

    thresh = 1e-10
    mask = wp.float32(n_rtz_norm < thresh)
    n_rtz = (1.0 - mask) * n_rtz + mask * r_element

    n_rtz_direction_w[env_id, body_id, contact_point_id] = n_rtz
    twist_angle[env_id, body_id, contact_point_id] = wp.atan2(n_rtz[1], n_rtz[0])


"""
force kernels
"""


@wp.func
def _compute_elementary_force(
    beta: wp.float32,
    gamma: wp.float32,
    psi: wp.float32,
    coef_1: wp.array(dtype=wp.float32),
    coef_2: wp.array(dtype=wp.float32),
    coef_3: wp.array(dtype=wp.float32),
) -> wp.vec3f:

    p1 = wp.sin(gamma)
    p2 = wp.cos(beta)
    p3 = wp.cos(psi) * wp.cos(gamma) * wp.sin(beta) + wp.sin(gamma) * wp.cos(beta)

    # Compute base terms
    base_0 = 1.0
    base_1 = p1
    base_2 = p2
    base_3 = p3
    base_4 = p1 * p1
    base_5 = p2 * p2
    base_6 = p3 * p3
    base_7 = p1 * p2
    base_8 = p2 * p3
    base_9 = p3 * p1
    base_10 = p1 * p1 * p1
    base_11 = p2 * p2 * p2
    base_12 = p3 * p3 * p3
    base_13 = p1 * p2 * p2
    base_14 = p2 * p1 * p1
    base_15 = p2 * p3 * p3
    base_16 = p3 * p2 * p2
    base_17 = p3 * p1 * p1
    base_18 = p1 * p3 * p3
    base_19 = p1 * p2 * p3

    # Compute f1, f2, f3
    f1 = (
        coef_1[0] * base_0
        + coef_1[1] * base_1
        + coef_1[2] * base_2
        + coef_1[3] * base_3
        + coef_1[4] * base_4
        + coef_1[5] * base_5
        + coef_1[6] * base_6
        + coef_1[7] * base_7
        + coef_1[8] * base_8
        + coef_1[9] * base_9
        + coef_1[10] * base_10
        + coef_1[11] * base_11
        + coef_1[12] * base_12
        + coef_1[13] * base_13
        + coef_1[14] * base_14
        + coef_1[15] * base_15
        + coef_1[16] * base_16
        + coef_1[17] * base_17
        + coef_1[18] * base_18
        + coef_1[19] * base_19
    )

    f2 = (
        coef_2[0] * base_0
        + coef_2[1] * base_1
        + coef_2[2] * base_2
        + coef_2[3] * base_3
        + coef_2[4] * base_4
        + coef_2[5] * base_5
        + coef_2[6] * base_6
        + coef_2[7] * base_7
        + coef_2[8] * base_8
        + coef_2[9] * base_9
        + coef_2[10] * base_10
        + coef_2[11] * base_11
        + coef_2[12] * base_12
        + coef_2[13] * base_13
        + coef_2[14] * base_14
        + coef_2[15] * base_15
        + coef_2[16] * base_16
        + coef_2[17] * base_17
        + coef_2[18] * base_18
        + coef_2[19] * base_19
    )

    f3 = (
        coef_3[0] * base_0
        + coef_3[1] * base_1
        + coef_3[2] * base_2
        + coef_3[3] * base_3
        + coef_3[4] * base_4
        + coef_3[5] * base_5
        + coef_3[6] * base_6
        + coef_3[7] * base_7
        + coef_3[8] * base_8
        + coef_3[9] * base_9
        + coef_3[10] * base_10
        + coef_3[11] * base_11
        + coef_3[12] * base_12
        + coef_3[13] * base_13
        + coef_3[14] * base_14
        + coef_3[15] * base_15
        + coef_3[16] * base_16
        + coef_3[17] * base_17
        + coef_3[18] * base_18
        + coef_3[19] * base_19
    )

    alpha_r = f1 * wp.sin(beta) * wp.cos(psi) + f2 * wp.cos(gamma)
    alpha_t = f1 * wp.sin(beta) * wp.sin(psi)
    alpha_z = -f1 * wp.cos(beta) - f2 * wp.sin(gamma) - f3

    return wp.vec3f(alpha_r, alpha_t, alpha_z)


@wp.kernel
def compute_resistive_force(
    foot_pos_w: wp.array2d(dtype=wp.vec3f),  # (N, M, 3)
    foot_velocity_w: wp.array2d(dtype=wp.vec3f),  # (N, M, 3)
    foot_velocity_prev_w: wp.array2d(dtype=wp.vec3f),  # (N, M, 3)
    beta: wp.array2d(dtype=wp.float32),  # (N, M)
    gamma: wp.array2d(dtype=wp.float32),  # (N, M)
    psi: wp.array2d(dtype=wp.float32),  # (N, M)
    r_direction_w: wp.array2d(dtype=wp.vec3f),  # (N, M, 3)
    t_direction_w: wp.array2d(dtype=wp.vec3f),  # (N, M, 3)
    z_direction_w: wp.array2d(dtype=wp.vec3f),  # (N, M, 3)
    n_rtz_direction_w: wp.array2d(dtype=wp.vec3f),  # (N, M, 3)
    # sand parameters
    rho_c: wp.array1d(dtype=wp.float32),  # (N,)
    mu_int: wp.array1d(dtype=wp.float32),  # (N,)
    dynamic_friction_coeff: wp.array1d(dtype=wp.float32),  # (N,)
    # 3d RFT polynomial fit coefficients
    coef_1: wp.array1d(dtype=wp.float32),  # (20,)
    coef_2: wp.array1d(dtype=wp.float32),  # (20,)
    coef_3: wp.array1d(dtype=wp.float32),  # (20,)
    # emf filter cache
    tau_r: wp.array2d(dtype=wp.float32),  # (N, M)
    c_r: wp.float32,
    enable_ema: wp.int32, # 1 = use EMA output, 0 = use raw force_gm
    # intruder parameters
    dA: wp.array1d(dtype=wp.float32),  # (C,) per-contact-point area element
    num_cp: wp.int32,  # number of contact points per body (C)
    # output
    alpha_unfiltered: wp.array2d(dtype=wp.vec3f),  # (N, M, 3)
    alpha_filtered: wp.array2d(dtype=wp.vec3f),  # (N, M, 3)
    resistive_force: wp.array2d(dtype=wp.vec3f),  # (N, M, 3)
):
    env_id, body_id = wp.tid()

    # extract element-wise data
    foot_velocity_element = foot_velocity_w[env_id, body_id]
    foot_velocity_prev_element = foot_velocity_prev_w[env_id, body_id]

    r_element = r_direction_w[env_id, body_id]
    z_element = z_direction_w[env_id, body_id]
    t_element = t_direction_w[env_id, body_id]

    beta_element = beta[env_id, body_id]
    gamma_element = gamma[env_id, body_id]
    psi_element = psi[env_id, body_id]

    depth = -foot_pos_w[env_id, body_id][2]
    is_contact = wp.float32(depth > 0.0)

    # NOTE: orthogonal base is {r, t, z} here.
    alpha_gen = _compute_elementary_force(beta_element, gamma_element, psi_element, coef_1, coef_2, coef_3)

    # media specific params
    g = 9.81
    rho_c_element = rho_c[env_id]
    mu_int_element = mu_int[env_id]
    dynamic_friction_coeff_element = dynamic_friction_coeff[env_id]
    xi = rho_c_element * g * (894.0 * (mu_int_element**3.0) - 386.0 * (mu_int_element**2.0) + 89.0 * mu_int_element)
    alpha = xi * alpha_gen

    n = wp.vec3f(
        wp.sin(beta_element) * wp.cos(psi_element), wp.sin(beta_element) * wp.sin(psi_element), -wp.cos(beta_element)
    )
    alpha_n = wp.dot(alpha, n) * n
    alpha_n_norm = wp.length(alpha_n)
    alpha_tan = alpha - alpha_n
    alpha_tan_norm = wp.length(alpha_tan)

    # friction cone check
    cone_coef = wp.min(wp.vec2f(1.0, (dynamic_friction_coeff_element * alpha_n_norm) / (alpha_tan_norm + 1e-6)))
    alpha = alpha_n + cone_coef * alpha_tan

    # apply EMA filtering
    coef = 0.8
    increment_mask = wp.float32(wp.dot(foot_velocity_element, foot_velocity_prev_element) < 0.0)
    tau_r_element = tau_r[env_id, body_id]
    tau_r_boundary = wp.float32(tau_r_element < 1.0)
    depth_mask = wp.float32(depth > 0.0)
    mask = increment_mask * tau_r_boundary
    tau_r_update = tau_r_element + c_r * mask
    tau_r_update = depth_mask * tau_r_update + (1.0 - depth_mask) * 0.0
    tau_r[env_id, body_id] = tau_r_update

    alpha_unfiltered[env_id, body_id] = alpha
    # filter all axis
    alpha_filtered[env_id, body_id] = (
        (1.0 - coef * tau_r_update) * alpha_unfiltered[env_id, body_id] + coef * tau_r_update * alpha_filtered[env_id, body_id]
    )
    # # filter z only
    # alpha_filtered[env_id, body_id] = alpha
    # alpha_filtered[env_id, body_id][2] = (
    #     (1.0 - coef * tau_r_update) * alpha_unfiltered[env_id, body_id][2] + coef * tau_r_update * alpha_filtered[env_id, body_id][2]
    # )

    if enable_ema == 0:
        alpha_out = alpha_unfiltered[env_id, body_id] * depth_mask
    else:
        alpha_out = alpha_filtered[env_id, body_id] * depth_mask

    # NOTE: orthogonal base is {x, y, z} here.
    cp_id = body_id % num_cp  # contact point index within body
    dA_element = dA[cp_id]
    force_vec = alpha_out * depth * dA_element * is_contact
    sign_fy = 1.0 - 2.0 * wp.float32(n_rtz_direction_w[env_id, body_id][1] < 0.0)
    resistive_force[env_id, body_id] = (
        force_vec[0] * r_element + sign_fy * force_vec[1] * t_element + force_vec[2] * z_element
    )


@wp.kernel
def compute_contact_wrench(
    body_pos: wp.array2d(dtype=wp.vec3f),  # (N, B, 3)
    contact_point_pos_w: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    resistive_force: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    contact_point_force: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    contact_point_torque: wp.array3d(dtype=wp.vec3f),  # (N, B, C, 3)
    contact_force: wp.array2d(dtype=wp.vec3f),  # (N, B, 3)
    contact_torque: wp.array2d(dtype=wp.vec3f),  # (N, B, 3)
):

    env_id, body_id, contact_point_id = wp.tid()
    r = contact_point_pos_w[env_id, body_id, contact_point_id] - body_pos[env_id, body_id]

    contact_point_force[env_id, body_id, contact_point_id] = resistive_force[env_id, body_id, contact_point_id]
    contact_point_torque[env_id, body_id, contact_point_id] = wp.cross(
        r, resistive_force[env_id, body_id, contact_point_id]
    )

    wp.atomic_add(contact_force, env_id, body_id, contact_point_force[env_id, body_id, contact_point_id])
    wp.atomic_add(contact_torque, env_id, body_id, contact_point_torque[env_id, body_id, contact_point_id])


@wp.kernel
def transform_global_wrench_to_body(
    body_quat: wp.array2d(dtype=wp.quatf),  # (N, B, 4)
    contact_force: wp.array2d(dtype=wp.vec3f),  # (N, B, 3)
    contact_torque: wp.array2d(dtype=wp.vec3f),  # (N, B, 3)
    contact_force_body: wp.array2d(dtype=wp.vec3f),  # (N, B, 3)
    contact_torque_body: wp.array2d(dtype=wp.vec3f),  # (N, B, 3)
):

    env_id, body_id = wp.tid()
    body_quat_conj = wp.quat_inverse(body_quat[env_id, body_id])
    contact_force_body[env_id, body_id] = wp.quat_rotate(body_quat_conj, contact_force[env_id, body_id])
    contact_torque_body[env_id, body_id] = wp.quat_rotate(body_quat_conj, contact_torque[env_id, body_id])


"""
reset kernels
"""


@wp.kernel
def reset(
    env_ids: wp.array(dtype=wp.int64),  # (n, )
    alpha_unfiltered: wp.array2d(dtype=wp.vec3f),  # (N, M, 3)
    alpha_filtered: wp.array2d(dtype=wp.vec3f),  # (N, M, 3)
    tau_r: wp.array2d(dtype=wp.float32),  # (N, M)
    contact_point_lin_vel_prev_w: wp.array2d(dtype=wp.vec3f),  # (N, M, 3)
):
    i, j = wp.tid()
    env_id = env_ids[i]

    alpha_filtered[env_id, j] = wp.vec3f(0.0)
    alpha_unfiltered[env_id, j] = wp.vec3f(0.0)
    tau_r[env_id, j] = 0.0
    contact_point_lin_vel_prev_w[env_id, j] = wp.vec3f(0.0)


@wp.kernel
def zero_wrench(
    contact_force: wp.array2d(dtype=wp.vec3f),  # (N, B, 3)
    contact_torque: wp.array2d(dtype=wp.vec3f),  # (N, B, 3)
):
    env_id, body_id = wp.tid()
    contact_force[env_id, body_id] = wp.vec3f(0.0)
    contact_torque[env_id, body_id] = wp.vec3f(0.0)


"""
2D RFT force kernels
"""


@wp.func
def _compute_elementary_force_2d(
    beta: wp.float32,
    gamma: wp.float32,
    A00: wp.float32,
    A10: wp.float32,
    B11: wp.float32,
    B01: wp.float32,
    B_11: wp.float32,
    C11: wp.float32,
    C01: wp.float32,
    C_11: wp.float32,
    D10: wp.float32,
) -> wp.vec2f:
    """
    Fourier series expansion for 2D RFT force coefficients.
    Returns (alpha_x, alpha_z) where:
      alpha_z is the normal (vertical) component
      alpha_x is the horizontal component (not used in current force pipeline)
    See https://www.science.org/doi/10.1126/science.1229163
    """
    alpha_z = wp.float32(0)
    alpha_x = wp.float32(0)

    alpha_z += A00 * wp.cos(2.0 * wp.PI * (0.0 * beta / wp.PI))
    alpha_z += A10 * wp.cos(2.0 * wp.PI * (1.0 * beta / wp.PI))
    alpha_z += B01 * wp.sin(2.0 * wp.PI * (1.0 * gamma / (2.0 * wp.PI)))
    alpha_z += B11 * wp.sin(2.0 * wp.PI * (1.0 * beta / wp.PI + 1.0 * gamma / (2.0 * wp.PI)))
    alpha_z += B_11 * wp.sin(2.0 * wp.PI * (-1.0 * beta / wp.PI + 1.0 * gamma / (2.0 * wp.PI)))

    alpha_x += C01 * wp.sin(2.0 * wp.PI * (1.0 * gamma / (2.0 * wp.PI)))
    alpha_x += C11 * wp.sin(2.0 * wp.PI * (1.0 * beta / wp.PI + 1.0 * gamma / (2.0 * wp.PI)))
    alpha_x += C_11 * wp.sin(2.0 * wp.PI * (-1.0 * beta / wp.PI + 1.0 * gamma / (2.0 * wp.PI)))
    alpha_x += D10 * wp.cos(2.0 * wp.PI * (1.0 * beta / wp.PI))

    return wp.vec2f(alpha_x, alpha_z)


@wp.kernel
def compute_resistive_force_2d(
    foot_pos_w: wp.array2d(dtype=wp.vec3f),           # (N, M)
    foot_velocity_w: wp.array2d(dtype=wp.vec3f),       # (N, M)
    foot_velocity_prev_w: wp.array2d(dtype=wp.vec3f),  # (N, M)
    beta: wp.array2d(dtype=wp.float32),                # (N, M) tilt angle
    gamma: wp.array2d(dtype=wp.float32),               # (N, M) intrusion angle
    z_direction_w: wp.array2d(dtype=wp.vec3f),         # (N, M)
    # per-env material params
    rho: wp.array1d(dtype=wp.float32),                 # (N,) DRFT density
    lam: wp.array1d(dtype=wp.float32),                 # (N,) DRFT coefficient
    dynamic_friction_coef: wp.array1d(dtype=wp.float32),  # (N,)
    kf: wp.array1d(dtype=wp.float32),                  # (N,)
    rho_c: wp.array1d(dtype=wp.float32),               # (N,) critical media density (kg/m^3)
    mu_int: wp.array1d(dtype=wp.float32),              # (N,) internal friction coefficient
    # Fourier coefficients (fixed scalar constants, dimensionless)
    A00: wp.float32,
    A10: wp.float32,
    B11: wp.float32,
    B01: wp.float32,
    B_11: wp.float32,
    C11: wp.float32,
    C01: wp.float32,
    C_11: wp.float32,
    D10: wp.float32,
    # EMA filter state
    force_gm: wp.array2d(dtype=wp.float32),            # (N, M)
    force_ema: wp.array2d(dtype=wp.float32),           # (N, M)
    tau_r: wp.array2d(dtype=wp.float32),               # (N, M)
    c_r: wp.float32,
    enable_ema: wp.int32,                              # 1 = use EMA output, 0 = use raw force_gm
    # per-contact-point area
    dA: wp.array1d(dtype=wp.float32),                  # (C,)
    num_cp: wp.int32,
    # output
    resistive_force: wp.array2d(dtype=wp.vec3f),       # (N, M)
):
    env_id, body_id = wp.tid()

    foot_velocity = foot_velocity_w[env_id, body_id]
    foot_velocity_prev = foot_velocity_prev_w[env_id, body_id]

    depth = -foot_pos_w[env_id, body_id][2]
    is_contact = wp.float32(depth > 0.0)
    depth_mask = wp.float32(depth > 0.0)

    beta_val = beta[env_id, body_id]
    gamma_val = gamma[env_id, body_id]
    z_dir = z_direction_w[env_id, body_id]

    alpha_xz = _compute_elementary_force_2d(beta_val, gamma_val, A00, A10, B11, B01, B_11, C11, C01, C_11, D10)
    alpha_z = alpha_xz[1]

    rho_val = rho[env_id]    # DRFT density
    lam_val = lam[env_id]    # DRFT coefficient
    dynamic_friction = dynamic_friction_coef[env_id]
    kf_val = kf[env_id]

    # Quasistatic stiffness xi: same formula as 3D RFT.
    # alpha_z is dimensionless (Li et al. 2013); xi [N/m^3] provides the force scale.
    g_val = wp.float32(9.81)
    mu_val = mu_int[env_id]
    xi = rho_c[env_id] * g_val * (894.0 * mu_val * mu_val * mu_val - 386.0 * mu_val * mu_val + 89.0 * mu_val)

    cp_id = body_id % num_cp
    dA_val = dA[cp_id]

    force_gm_val = xi * alpha_z * depth * dA_val * is_contact

    # EMA filter on z-force (always computed so the state stays valid for reset)
    coef = 0.8
    increment_mask = wp.float32(foot_velocity[2] * foot_velocity_prev[2] < 0.0)
    tau_r_val = tau_r[env_id, body_id]
    tau_r_boundary = wp.float32(tau_r_val < 1.0)
    mask = increment_mask * tau_r_boundary
    tau_r_val = tau_r_val + c_r * mask
    tau_r_val = depth_mask * tau_r_val
    tau_r[env_id, body_id] = tau_r_val

    force_gm[env_id, body_id] = force_gm_val
    force_ema_prev = force_ema[env_id, body_id]
    force_ema_val = (1.0 - coef * tau_r_val) * force_gm_val + coef * tau_r_val * force_ema_prev
    force_ema_val = depth_mask * force_ema_val
    force_ema[env_id, body_id] = force_ema_val

    # Select filtered or unfiltered quasistatic force (matches torch enable_ema_filter flag)
    fz_qs = force_gm_val
    if enable_ema:
        fz_qs = force_ema_val

    # DRFT inertial term: lam * rho * vn^2 (dynamic RFT)
    vn = foot_velocity[2]
    fz = fz_qs # only quasistatic term
    # TODO: experimental feature
    # fz += is_contact * lam_val * rho_val * vn * vn # DRFT term

    # Coulomb tangential friction in x-y plane.
    # Uses fz directly (not abs), matching the torch implementation exactly.
    vt_x = foot_velocity[0]
    vt_y = foot_velocity[1]
    vt_norm = wp.sqrt(vt_x * vt_x + vt_y * vt_y)
    ft = wp.min(dynamic_friction * fz, kf_val * vt_norm)
    vt_dir = wp.vec3f(vt_x / (vt_norm + 1.0e-6), vt_y / (vt_norm + 1.0e-6), wp.float32(0))

    resistive_force[env_id, body_id] = fz * z_dir - ft * vt_dir


@wp.kernel
def reset_2d(
    env_ids: wp.array(dtype=wp.int64),                      # (n,)
    force_gm: wp.array2d(dtype=wp.float32),                 # (N, M)
    force_ema: wp.array2d(dtype=wp.float32),                # (N, M)
    tau_r: wp.array2d(dtype=wp.float32),                    # (N, M)
    contact_point_lin_vel_prev_w: wp.array2d(dtype=wp.vec3f),  # (N, M)
):
    i, j = wp.tid()
    env_id = env_ids[i]
    force_gm[env_id, j] = 0.0
    force_ema[env_id, j] = 0.0
    tau_r[env_id, j] = 0.0
    contact_point_lin_vel_prev_w[env_id, j] = wp.vec3f(0.0)


"""
Spring-damper force kernel
"""


@wp.kernel
def compute_spring_damper_force(
    foot_pos_w: wp.array2d(dtype=wp.vec3f),             # (N, M)
    foot_velocity_w: wp.array2d(dtype=wp.vec3f),         # (N, M)
    # per-env material params
    k: wp.array1d(dtype=wp.float32),                     # (N,) spring stiffness density (N/m^3)
    b: wp.array1d(dtype=wp.float32),                     # (N,) damping density (N*s/m^3)
    dynamic_friction_coef: wp.array1d(dtype=wp.float32), # (N,)
    kf: wp.array1d(dtype=wp.float32),                    # (N,)
    # per-contact-point area
    dA: wp.array1d(dtype=wp.float32),                    # (C,)
    num_cp: wp.int32,
    # output
    resistive_force: wp.array2d(dtype=wp.vec3f),         # (N, M)
):
    env_id, body_id = wp.tid()

    vel = foot_velocity_w[env_id, body_id]
    depth = -foot_pos_w[env_id, body_id][2]
    is_contact = wp.float32(depth > 0.0)

    cp_id = body_id % num_cp
    dA_val = dA[cp_id]

    # Spring-damper normal force (no tensile: clamped to >= 0)
    vn = vel[2]
    fz = wp.max((k[env_id] * depth - b[env_id] * vn) * dA_val * is_contact, wp.float32(0.0))

    # Coulomb tangential friction — same model as 2D RFT
    vt_x = vel[0]
    vt_y = vel[1]
    vt_norm = wp.sqrt(vt_x * vt_x + vt_y * vt_y)
    ft = wp.min(dynamic_friction_coef[env_id] * fz, kf[env_id] * vt_norm)
    vt_dir = wp.vec3f(vt_x / (vt_norm + 1.0e-6), vt_y / (vt_norm + 1.0e-6), wp.float32(0))

    resistive_force[env_id, body_id] = fz * wp.vec3f(0.0, 0.0, 1.0) - ft * vt_dir
