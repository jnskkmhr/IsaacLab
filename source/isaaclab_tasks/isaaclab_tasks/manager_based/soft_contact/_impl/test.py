import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import warp as wp

from kernels_fix import _compute_elementary_force, _friction_cone_check
from kernels_fix import (
    plane_contact_points, 
    box_contact_points,  
)
from kernels_fix import (
    expand_vec3_to_3d_vec3,
    compute_contact_point_pos_w, 
    compute_contact_point_lin_vel_w, 
    compute_normal_direction_w, 
    compute_v_direction_w, 
    compute_r_direction_w, 
    compute_t_direction_w, 
    compute_n_rt_direction_w,
)

from kernels_fix import compute_intrusion_angle, compute_tilt_angle, compute_twist_angle

class Material3DRFTCfg:
    """
    Material parameters for 3D RFT soft contact model.
    See https://www.pnas.org/doi/10.1073/pnas.2214017120 supplementary material S3.

    c1^k, c2^k, c3^k are the coefficients of 3rd order polynomial that computes the resistive force per volume.
    static_friction_coef: static friction coefficient
    dynamic_friction_coef: dynamic friction coefficient
    rho_c: critical media density (effective media density = packing fraction * grain density)
    mu_int: media internal friction coefficient
    """

    # c1^k
    coef_1: list[float] = [
        0.00212,
        -0.02320,
        -0.20890,
        -0.43083,
        -0.00259,
        0.48872,
        -0.00415,
        0.07204,
        -0.02750,
        -0.08772,
        0.01992,
        -0.45961,
        0.40799,
        -0.10107,
        -0.06576,
        0.05664,
        -0.09269,
        0.01892,
        0.01033,
        0.15120,
    ]
    # c2^k
    coef_2: list[float] = [
        -0.06796,
        -0.10941,
        0.04725,
        -0.06914,
        -0.05835,
        -0.65880,
        -0.11985,
        -0.25739,
        -0.26834,
        0.02692,
        -0.00736,
        0.63758,
        0.08997,
        0.21069,
        0.04748,
        0.20406,
        0.18519,
        0.04934,
        0.13527,
        -0.33207,
    ]
    # c3^k
    coef_3: list[float] = [
        -0.02634,
        -0.03436,
        0.45256,
        0.00835,
        0.02553,
        -1.31290,
        -0.05532,
        0.06790,
        -0.16404,
        0.02287,
        0.02927,
        0.95406,
        -0.00131,
        -0.11028,
        0.01487,
        -0.20770,
        0.10911,
        -0.04097,
        0.07881,
        -0.27519,
    ]

    dynamic_friction_coef: float = 0.15
    mu_int: float = 0.4
    rho_c: float = 3000.0

"""
test elementary force calculation
"""

@wp.kernel 
def compute_elementary_force_kernel(
    beta: wp.array1d(dtype=wp.float32),  # (N,)
    gamma: wp.array1d(dtype=wp.float32),  # (N,)
    psi: wp.array1d(dtype=wp.float32),  # (N,)
    coef_1: wp.array1d(dtype=wp.float32),  # (20,)
    coef_2: wp.array1d(dtype=wp.float32),  # (20,)
    coef_3: wp.array1d(dtype=wp.float32),  # (20 ) 

    alpha_gen: wp.array1d(dtype=wp.vec3),  # (N,)
): 
    i = wp.tid()
    beta_i = beta[i]
    gamma_i = gamma[i]
    psi_i = psi[i]
    alpha_gen_i = _compute_elementary_force(beta_i, gamma_i, psi_i, coef_1, coef_2, coef_3)
    alpha_gen[i] = alpha_gen_i

@wp.kernel
def friction_cone_check_kernel(
    beta: wp.array1d(dtype=wp.float32),  # (N,)
    gamma: wp.array1d(dtype=wp.float32),  # (N,)
    psi: wp.array1d(dtype=wp.float32),  # (N,)
    alpha_gen: wp.array1d(dtype=wp.vec3),  # (N,)
    mu_surf: wp.float32,

    alpha_gen_cone: wp.array1d(dtype=wp.vec3),  # (N,)
):
    i = wp.tid()
    beta_i = beta[i]
    gamma_i = gamma[i]
    psi_i = psi[i]
    alpha_gen_i = alpha_gen[i]

    alpha_gen_cone_i = _friction_cone_check(beta_i, gamma_i, psi_i, mu_surf, alpha_gen_i)
    alpha_gen_cone[i] = alpha_gen_cone_i

@wp.kernel 
def scale_alpha_kernel(
    alpha_gen_cone: wp.array1d(dtype=wp.vec3),  # (N,)
    rho_c: wp.float32, 
    mu_int: wp.float32,

    alpha_gen_cone_scaled: wp.array1d(dtype=wp.vec3),  # (N,)
):
    i = wp.tid()
    alpha_gen_cone_i = alpha_gen_cone[i]

    g = 9.81
    xi = rho_c * g * (894.0 * (mu_int**3.0) - 386.0 * (mu_int**2.0) + 89.0 * mu_int)
    alpha_gen_cone_scaled[i] = alpha_gen_cone_i * xi

@wp.kernel
def transform_local_alpha_to_world_alpha_kernel(
    alpha_rtz: wp.array1d(dtype=wp.vec3),  # (N,)
    r_dir_w: wp.array1d(dtype=wp.vec3),  # (N,)
    t_dir_w: wp.array1d(dtype=wp.vec3),  # (N,)
    z_dir_w: wp.array1d(dtype=wp.vec3),  # (N,)
    alpha_xyz: wp.array1d(dtype=wp.vec3),  # (N,)
):
    i = wp.tid()
    alpha_rtz_i = alpha_rtz[i]
    r_dir_w_i = r_dir_w[i]
    t_dir_w_i = t_dir_w[i]
    z_dir_w_i = z_dir_w[i]

    # transform alpha from local rtz frame to world xyz frame
    alpha_xyz_i = alpha_rtz_i[0] * r_dir_w_i + alpha_rtz_i[1] * t_dir_w_i + alpha_rtz_i[2] * z_dir_w_i
    alpha_xyz[i] = alpha_xyz_i



def test_compute_elementary_force():
    beta_range = (-math.pi/2, math.pi/2)
    gamma_range = (-math.pi/2, math.pi/2)
    psi_range = (0, math.pi/2)
    num_beta = 100 
    num_gamma = 100
    num_psi = 4 

    beta_array = np.linspace(beta_range[0], beta_range[1], num_beta)
    gamma_array = np.linspace(gamma_range[0], gamma_range[1], num_gamma)
    psi_array = np.linspace(psi_range[0], psi_range[1], num_psi)

    beta_grid, gamma_grid, psi_grid = np.meshgrid(beta_array, gamma_array, psi_array, indexing='ij')
    beta = beta_grid.flatten()
    gamma = gamma_grid.flatten()
    psi = psi_grid.flatten()

    beta_wp = wp.from_numpy(beta.astype(np.float32))
    gamma_wp = wp.from_numpy(gamma.astype(np.float32))
    psi_wp = wp.from_numpy(psi.astype(np.float32))

    material_cfg = Material3DRFTCfg()
    coef_1 = wp.array1d(material_cfg.coef_1, dtype=wp.float32)
    coef_2 = wp.array1d(material_cfg.coef_2, dtype=wp.float32)
    coef_3 = wp.array1d(material_cfg.coef_3, dtype=wp.float32)

    alpha_gen = wp.zeros(beta.shape[0], dtype=wp.vec3)
    alpha_gen_cone = wp.zeros(beta.shape[0], dtype=wp.vec3)
    alpha = wp.zeros(beta.shape[0], dtype=wp.vec3)
    wp.launch(
        kernel=compute_elementary_force_kernel,
        dim=beta.shape[0],
        inputs=[beta_wp, gamma_wp, psi_wp, coef_1, coef_2, coef_3],
        outputs=[alpha_gen],
    )
    wp.launch(
        kernel=friction_cone_check_kernel,
        dim=beta.shape[0],
        inputs=[beta_wp, gamma_wp, psi_wp, alpha_gen, material_cfg.dynamic_friction_coef],
        outputs=[alpha_gen_cone],
    )

    wp.launch(
        kernel=scale_alpha_kernel,
        dim=beta.shape[0],
        inputs=[alpha_gen_cone, material_cfg.rho_c, material_cfg.mu_int],
        outputs=[alpha],
        )

    xi = 0.92 * 10**6
    alpha_gen_np = alpha.numpy() / xi

    # separate alpha_gen_cone_np to each beta, gamma, psi
    alpha_gen_np = alpha_gen_np.reshape((num_beta, num_gamma, num_psi, 3))

    # plot
    fig, axs = plt.subplots(num_psi, 3, figsize=(6, 12))
    for i in range(num_psi):
        psi_i = psi_array[i]
        alpha_gen_cone_psi = alpha_gen_np[:, :, i, :]

        # alpha_r
        im0 = axs[i, 0].imshow(-alpha_gen_cone_psi[:, :, 0], extent=(gamma_range[0], gamma_range[1], beta_range[0], beta_range[1]), origin='lower', cmap='jet', vmin=-0.4, vmax=0.4)
        if i == 0:
            axs[i, 0].set_title(f'alpha_r (psi={psi_i:.2f})')
            fig.colorbar(im0, ax=axs[i, 0], orientation='horizontal')
        axs[i, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

        # alpha_theta
        im1 = axs[i, 1].imshow(alpha_gen_cone_psi[:, :, 1], extent=(gamma_range[0], gamma_range[1], beta_range[0], beta_range[1]), origin='lower', cmap='jet', vmin=-0.3, vmax=0.3)
        if i == 0:
            axs[i, 1].set_title(f'alpha_t (psi={psi_i:.2f})')
            fig.colorbar(im1, ax=axs[i, 1], orientation='horizontal')
        axs[i, 1].grid(True, which='both', linestyle='--', linewidth=0.5)

        # alpha_z
        im2 = axs[i, 2].imshow(alpha_gen_cone_psi[:, :, 2], extent=(gamma_range[0], gamma_range[1], beta_range[0], beta_range[1]), origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)
        if i == 0:
            axs[i, 2].set_title(f'alpha_z (psi={psi_i:.2f})')
            fig.colorbar(im2, ax=axs[i, 2], orientation='horizontal')
        axs[i, 2].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


"""
test collider generation
"""
def test_collider_generation():
    num_env = 1
    num_body = 1
    nx, ny = 5, 5

    # """
    # predefine plane collider
    # """
    # contact_edge_x=(-0.1, 0.1)
    # contact_edge_y=(-0.05, 0.05)
    # z_offset = -0.04 # contact point is 4cm below com 

    # contact_points = wp.zeros(nx*ny, dtype=wp.vec3)
    # contact_normals = wp.zeros(nx*ny, dtype=wp.vec3)
    # wp.launch(
    #     kernel=plane_contact_points,
    #     dim=[nx, ny],
    #     inputs=[contact_edge_x, contact_edge_y, z_offset, nx, ny],
    #     outputs=[contact_points, contact_normals],
    # )
    # contact_point_np = contact_points.numpy()
    # contact_normal_np = contact_normals.numpy()

    """
    predefine plane collider
    """
    contact_edge_x=(-0.1, 0.1)
    contact_edge_y=(-0.05, 0.05)
    contact_edge_z=(-0.04, 0.04)

    contact_points = wp.zeros(nx*ny*6, dtype=wp.vec3)
    contact_normals = wp.zeros(nx*ny*6, dtype=wp.vec3)
    wp.launch(
        kernel=box_contact_points,
        dim=[6, nx, ny],
        inputs=[contact_edge_x, contact_edge_y, contact_edge_z, nx, ny],
        outputs=[contact_points, contact_normals],
    )
    contact_point_np = contact_points.numpy()
    contact_normal_np = contact_normals.numpy()

    # plot contact points and normals in 3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(contact_point_np[:, 0], contact_point_np[:, 1], contact_point_np[:, 2], c='b', s=20)
    ax.quiver(
        contact_point_np[:, 0], contact_point_np[:, 1], contact_point_np[:, 2],
        contact_normal_np[:, 0], contact_normal_np[:, 1], contact_normal_np[:, 2],
        length=0.01, normalize=True, color='r',
    )
    ax.set_xlim(contact_edge_x[0]*1.5, contact_edge_x[1]*1.5)
    ax.set_ylim(contact_edge_y[0]*1.5, contact_edge_y[1]*1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Contact Points and Normals')
    plt.tight_layout()
    plt.show()


"""
test contact point pos and lin vel calculation
"""
def test_contact_kinematics():
    num_env = 1
    num_body = 1
    nx, ny = 5, 5

    """
    predefine plane collider
    """
    contact_edge_x=(-0.1, 0.1)
    contact_edge_y=(-0.05, 0.05)
    z_offset = -0.04 # contact point is 4cm below com 

    contact_points = wp.zeros(nx*ny, dtype=wp.vec3)
    contact_normals = wp.zeros(nx*ny, dtype=wp.vec3)
    wp.launch(
        kernel=plane_contact_points,
        dim=[nx, ny],
        inputs=[contact_edge_x, contact_edge_y, z_offset, nx, ny],
        outputs=[contact_points, contact_normals],
    )

    contact_points = contact_points.reshape((num_env, num_body, nx*ny))
    contact_normals = contact_normals.reshape((num_env, num_body, nx*ny))

    """
    calculate global contact point position, linear velocity
    """
    body_pos_w = np.zeros((num_env, num_body, 3))
    body_quat_w = np.zeros((num_env, num_body, 4))
    # pitch -30deg
    body_quat_w[:, :, 0] = 0
    body_quat_w[:, :, 1] = -0.258819
    body_quat_w[:, :, 2] = 0
    body_quat_w[:, :, 3] = 0.9659258
    body_lin_vel_w = np.zeros((num_env, num_body, 3))
    body_lin_vel_w[:, :, 0] = 1.0
    body_lin_vel_w[:, :, 2] = -1.0
    body_ang_vel_w = np.zeros((num_env, num_body, 3))
    body_ang_vel_w[:, :, 1] = 5

    body_pos_w_wp = wp.from_numpy(body_pos_w, dtype=wp.vec3f)
    body_quat_w_wp = wp.from_numpy(body_quat_w, dtype=wp.quatf)
    body_lin_vel_w_wp = wp.from_numpy(body_lin_vel_w, dtype=wp.vec3f)
    body_ang_vel_w_wp = wp.from_numpy(body_ang_vel_w, dtype=wp.vec3f)

    contact_point_pos_w = wp.zeros_like(contact_points)
    contact_point_lin_vel_w = wp.zeros_like(contact_points)

    wp.launch(
        kernel=compute_contact_point_pos_w,
        dim=[num_env, num_body, nx*ny],
        inputs=[body_pos_w_wp, body_quat_w_wp, contact_points],
        outputs=[contact_point_pos_w],
    )

    wp.launch(
        kernel=compute_contact_point_lin_vel_w,
        dim=[num_env, num_body, nx*ny],
        inputs=[body_pos_w_wp, body_lin_vel_w_wp, body_ang_vel_w_wp, contact_point_pos_w],
        outputs=[contact_point_lin_vel_w],
    )

    contact_point_pos_w_np = contact_point_pos_w.numpy()[0, 0, :, :]
    contact_point_lin_vel_w_np = contact_point_lin_vel_w.numpy()[0, 0, :, :]
    print(contact_point_lin_vel_w_np)

    # plot contact point positions and linear velocities in 3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2], c='b', s=20)
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        contact_point_lin_vel_w_np[:, 0], contact_point_lin_vel_w_np[:, 1], contact_point_lin_vel_w_np[:, 2],
        length=0.1, normalize=True, color='r',
    )
    lim = max(contact_edge_x[1], contact_edge_y[1]) * 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Contact Points and velocities')
    plt.tight_layout()
    plt.show()


"""
test contact point coordinate
"""
def test_contact_coordinates():
    num_env = 1
    num_body = 1
    nx, ny = 5, 5

    """
    predefine plane collider
    """
    contact_edge_x=(-0.1, 0.1)
    contact_edge_y=(-0.05, 0.05)
    z_offset = -0.04 # contact point is 4cm below com 

    contact_points = wp.zeros(nx*ny, dtype=wp.vec3)
    contact_normals = wp.zeros(nx*ny, dtype=wp.vec3)
    wp.launch(
        kernel=plane_contact_points,
        dim=[nx, ny],
        inputs=[contact_edge_x, contact_edge_y, z_offset, nx, ny],
        outputs=[contact_points, contact_normals],
    )

    contact_points = contact_points.reshape((num_env, num_body, nx*ny))
    contact_normals = contact_normals.reshape((num_env, num_body, nx*ny))

    """
    calculate global contact point position, linear velocity
    """
    body_pos_w = np.zeros((num_env, num_body, 3))
    body_quat_w = np.zeros((num_env, num_body, 4))

    # identity quat
    # body_quat_w[:, :, 3] = 1.0 # warp uses (x, y, z, w) convention for quaternion

    # yaw 90deg
    body_quat_w[:, :, 0] = 0
    body_quat_w[:, :, 1] = 0
    body_quat_w[:, :, 2] = 0.7071068
    body_quat_w[:, :, 3] = 0.7071068

    # pitch -30deg
    # body_quat_w[:, :, 0] = 0
    # body_quat_w[:, :, 1] = -0.258819
    # body_quat_w[:, :, 2] = 0
    # body_quat_w[:, :, 3] = 0.9659258

    # # roll 15deg, pitch -30deg 
    # body_quat_w[:, :, 0] = 0.1260786
    # body_quat_w[:, :, 1] = -0.2566048
    # body_quat_w[:, :, 2] = -0.0337827
    # body_quat_w[:, :, 3] = 0.9576622

    body_lin_vel_w = np.zeros((num_env, num_body, 3))
    body_lin_vel_w[:, :, 0] = 2.0
    body_lin_vel_w[:, :, 1] = 0.5
    body_lin_vel_w[:, :, 2] = -1.0
    body_ang_vel_w = np.zeros((num_env, num_body, 3))
    body_ang_vel_w[:, :, 1] = 0

    body_pos_w_wp = wp.from_numpy(body_pos_w, dtype=wp.vec3f)
    body_quat_w_wp = wp.from_numpy(body_quat_w, dtype=wp.quatf)
    body_lin_vel_w_wp = wp.from_numpy(body_lin_vel_w, dtype=wp.vec3f)
    body_ang_vel_w_wp = wp.from_numpy(body_ang_vel_w, dtype=wp.vec3f)

    contact_point_pos_w = wp.zeros_like(contact_points)
    contact_point_lin_vel_w = wp.zeros_like(contact_points)

    wp.launch(
        kernel=compute_contact_point_pos_w,
        dim=[num_env, num_body, nx*ny],
        inputs=[body_pos_w_wp, body_quat_w_wp, contact_points],
        outputs=[contact_point_pos_w],
    )

    wp.launch(
        kernel=compute_contact_point_lin_vel_w,
        dim=[num_env, num_body, nx*ny],
        inputs=[body_pos_w_wp, body_lin_vel_w_wp, body_ang_vel_w_wp, contact_point_pos_w],
        outputs=[contact_point_lin_vel_w],
    )

    """
    calculate coordinate vectors
    """

    num_contact_points = nx * ny
    normal_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    v_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    r_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    t_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    z_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    Z_DIR = wp.vec3f(0.0, 0.0, 1.0)
    wp.launch(
        kernel=expand_vec3_to_3d_vec3,
        dim=(num_env, num_body, num_contact_points),
        inputs=[Z_DIR],
        outputs=[z_direction_w],
    )

    wp.launch(
        kernel=compute_normal_direction_w,
        dim=[num_env, num_body, num_contact_points],
        inputs=[body_quat_w_wp, contact_normals],
        outputs=[normal_direction_w],
    )

    wp.launch(
        kernel=compute_v_direction_w,
        dim=[num_env, num_body, num_contact_points],
        inputs=[contact_point_lin_vel_w],
        outputs=[v_direction_w],
    )

    threshold = 1e-3
    wp.launch(
        kernel=compute_r_direction_w,
        dim=[num_env, num_body, num_contact_points],
        inputs=[normal_direction_w, z_direction_w, v_direction_w, r_direction_w, threshold],
    )

    wp.launch(
        kernel=compute_t_direction_w, 
        dim=[num_env, num_body, num_contact_points],
        inputs=[z_direction_w, r_direction_w],
        outputs=[t_direction_w],
    )

    env_id = 0
    body_id = 0
    contact_point_pos_w_np = contact_point_pos_w.numpy()[env_id, body_id, :, :]
    contact_point_lin_vel_w_np = contact_point_lin_vel_w.numpy()[env_id, body_id, :, :]
    n_vec = normal_direction_w.numpy()[env_id, body_id, :, :]
    v_vec = v_direction_w.numpy()[env_id, body_id, :, :]
    r_vec = r_direction_w.numpy()[env_id, body_id, :, :]
    t_vec = t_direction_w.numpy()[env_id, body_id, :, :]
    z_vec = z_direction_w.numpy()[env_id, body_id, :, :]


    # plot contact point positions and linear velocities in 3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2], c='b', s=20)
    # # normal
    # ax.quiver(
    #     contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
    #     n_vec[:, 0], n_vec[:, 1], n_vec[:, 2],
    #     length=0.1, normalize=True, color='r',
    # )
    # velocity
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        v_vec[:, 0], v_vec[:, 1], v_vec[:, 2],
        length=0.1, normalize=True, color='g',
    )
    # r direction
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        r_vec[:, 0], r_vec[:, 1], r_vec[:, 2],
        length=0.1, normalize=True, color='m',
    )
    # t direction
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        t_vec[:, 0], t_vec[:, 1], t_vec[:, 2],
        length=0.1, normalize=True, color='c',
    )
    # z direction
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        z_vec[:, 0], z_vec[:, 1], z_vec[:, 2],
        length=0.1, normalize=True, color='y',
    )

    lim = max(contact_edge_x[1], contact_edge_y[1]) * 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Contact Points and velocities')
    plt.tight_layout()
    plt.show()


"""
test contact point coordinate
"""
def test_characteristic_angle():
    num_env = 1
    num_body = 1
    nx, ny = 5, 5

    """
    predefine plane collider
    """
    contact_edge_x=(-0.1, 0.1)
    contact_edge_y=(-0.05, 0.05)
    z_offset = -0.04 # contact point is 4cm below com 

    contact_points = wp.zeros(nx*ny, dtype=wp.vec3)
    contact_normals = wp.zeros(nx*ny, dtype=wp.vec3)
    wp.launch(
        kernel=plane_contact_points,
        dim=[nx, ny],
        inputs=[contact_edge_x, contact_edge_y, z_offset, nx, ny],
        outputs=[contact_points, contact_normals],
    )

    contact_points = contact_points.reshape((num_env, num_body, nx*ny))
    contact_normals = contact_normals.reshape((num_env, num_body, nx*ny))

    """
    calculate global contact point position, linear velocity
    """
    body_pos_w = np.zeros((num_env, num_body, 3))
    body_quat_w = np.zeros((num_env, num_body, 4))

    # identity quat
    # body_quat_w[:, :, 3] = 1.0 # warp uses (x, y, z, w) convention for quaternion

    # yaw 90deg 
    # 0, 0, 0.7071068, 0.7071068
    # body_quat_w[:, :, 0] = 0
    # body_quat_w[:, :, 1] = 0
    # body_quat_w[:, :, 2] = 0.7071068
    # body_quat_w[:, :, 3] = 0.7071068

    # pitch -30deg
    # body_quat_w[:, :, 0] = 0
    # body_quat_w[:, :, 1] = -0.258819
    # body_quat_w[:, :, 2] = 0
    # body_quat_w[:, :, 3] = 0.9659258

    # roll 15deg, pitch -30deg 
    body_quat_w[:, :, 0] = 0.1260786
    body_quat_w[:, :, 1] = -0.2566048
    body_quat_w[:, :, 2] = -0.0337827
    body_quat_w[:, :, 3] = 0.9576622

    body_lin_vel_w = np.zeros((num_env, num_body, 3))
    body_lin_vel_w[:, :, 0] = 2.0
    body_lin_vel_w[:, :, 1] = 0.5
    body_lin_vel_w[:, :, 2] = -1.0
    body_ang_vel_w = np.zeros((num_env, num_body, 3))
    body_ang_vel_w[:, :, 1] = 0

    body_pos_w_wp = wp.from_numpy(body_pos_w, dtype=wp.vec3f)
    body_quat_w_wp = wp.from_numpy(body_quat_w, dtype=wp.quatf)
    body_lin_vel_w_wp = wp.from_numpy(body_lin_vel_w, dtype=wp.vec3f)
    body_ang_vel_w_wp = wp.from_numpy(body_ang_vel_w, dtype=wp.vec3f)

    contact_point_pos_w = wp.zeros_like(contact_points)
    contact_point_lin_vel_w = wp.zeros_like(contact_points)

    wp.launch(
        kernel=compute_contact_point_pos_w,
        dim=[num_env, num_body, nx*ny],
        inputs=[body_pos_w_wp, body_quat_w_wp, contact_points],
        outputs=[contact_point_pos_w],
    )

    wp.launch(
        kernel=compute_contact_point_lin_vel_w,
        dim=[num_env, num_body, nx*ny],
        inputs=[body_pos_w_wp, body_lin_vel_w_wp, body_ang_vel_w_wp, contact_point_pos_w],
        outputs=[contact_point_lin_vel_w],
    )

    """
    calculate coordinate vectors
    """

    num_contact_points = nx * ny
    normal_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    v_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    r_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    t_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    z_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    Z_DIR = wp.vec3f(0.0, 0.0, 1.0)
    wp.launch(
        kernel=expand_vec3_to_3d_vec3,
        dim=(num_env, num_body, num_contact_points),
        inputs=[Z_DIR],
        outputs=[z_direction_w],
    )

    wp.launch(
        kernel=compute_normal_direction_w,
        dim=[num_env, num_body, num_contact_points],
        inputs=[body_quat_w_wp, contact_normals],
        outputs=[normal_direction_w],
    )

    wp.launch(
        kernel=compute_v_direction_w,
        dim=[num_env, num_body, num_contact_points],
        inputs=[contact_point_lin_vel_w],
        outputs=[v_direction_w],
    )

    threshold = 1e-3
    wp.launch(
        kernel=compute_r_direction_w,
        dim=[num_env, num_body, num_contact_points],
        inputs=[normal_direction_w, z_direction_w, v_direction_w, r_direction_w, threshold],
    )

    wp.launch(
        kernel=compute_t_direction_w, 
        dim=[num_env, num_body, num_contact_points],
        inputs=[z_direction_w, r_direction_w],
        outputs=[t_direction_w],
    )

    intrusion_angle = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.float32) # gamma
    tilt_angle = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.float32) # beta
    twist_angle = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.float32) # psi
    n_rtz_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)

    wp.launch(
        kernel=compute_intrusion_angle,
        dim=[num_env, num_body, num_contact_points],
        inputs=[z_direction_w, v_direction_w, r_direction_w, intrusion_angle],
    )
    wp.launch(
        kernel=compute_tilt_angle,
        dim=[num_env, num_body, num_contact_points],
        inputs=[z_direction_w, normal_direction_w, r_direction_w, t_direction_w, tilt_angle],
    )
    wp.launch(
        kernel=compute_twist_angle,
        dim=[num_env, num_body, num_contact_points],
        inputs=[z_direction_w, normal_direction_w, r_direction_w, t_direction_w, n_rtz_direction_w, twist_angle],
    )


    env_id = 0
    body_id = 0
    contact_point_pos_w_np = contact_point_pos_w.numpy()[env_id, body_id, :, :]
    contact_point_lin_vel_w_np = contact_point_lin_vel_w.numpy()[env_id, body_id, :, :]
    n_vec = normal_direction_w.numpy()[env_id, body_id, :, :]
    v_vec = v_direction_w.numpy()[env_id, body_id, :, :]
    r_vec = r_direction_w.numpy()[env_id, body_id, :, :]
    t_vec = t_direction_w.numpy()[env_id, body_id, :, :]
    z_vec = z_direction_w.numpy()[env_id, body_id, :, :]
    gamma = intrusion_angle.numpy()[env_id, body_id, :]
    beta = tilt_angle.numpy()[env_id, body_id, :]
    psi = twist_angle.numpy()[env_id, body_id, :]

    print("gamma:", gamma * 180.0 / math.pi)
    print("beta:", beta * 180.0 / math.pi)
    print("psi:", psi * 180.0 / math.pi)

    # plot contact point positions and linear velocities in 3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2], c='b', s=20)
    # normal
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        n_vec[:, 0], n_vec[:, 1], n_vec[:, 2],
        length=0.1, normalize=True, color='r',
    )
    # velocity
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        v_vec[:, 0], v_vec[:, 1], v_vec[:, 2],
        length=0.1, normalize=True, color='g',
    )
    # r direction
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        r_vec[:, 0], r_vec[:, 1], r_vec[:, 2],
        length=0.1, normalize=True, color='m',
    )
    # t direction
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        t_vec[:, 0], t_vec[:, 1], t_vec[:, 2],
        length=0.1, normalize=True, color='c',
    )
    # # z direction
    # ax.quiver(
    #     contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
    #     z_vec[:, 0], z_vec[:, 1], z_vec[:, 2],
    #     length=0.1, normalize=True, color='y',
    # )

    lim = max(contact_edge_x[1], contact_edge_y[1]) * 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Contact Points and velocities')
    plt.tight_layout()
    plt.show()


"""
test force calculation
"""
def test_force_calculation():
    num_env = 1
    num_body = 1
    nx, ny = 2, 2

    # """
    # predefine plane collider
    # """
    # contact_edge_x=(-0.1, 0.1)
    # contact_edge_y=(-0.05, 0.05)
    # z_offset = -0.04 # contact point is 4cm below com 
    # num_contact_points = nx * ny

    # contact_points = wp.zeros(nx*ny, dtype=wp.vec3)
    # contact_normals = wp.zeros(nx*ny, dtype=wp.vec3)
    # wp.launch(
    #     kernel=plane_contact_points,
    #     dim=[nx, ny],
    #     inputs=[contact_edge_x, contact_edge_y, z_offset, nx, ny],
    #     outputs=[contact_points, contact_normals],
    # )

    """
    predefine plane collider
    """
    contact_edge_x=(-0.1, 0.1)
    contact_edge_y=(-0.05, 0.05)
    contact_edge_z=(-0.04, 0.04)
    num_contact_points = nx * ny * 6

    contact_points = wp.zeros(nx*ny*6, dtype=wp.vec3)
    contact_normals = wp.zeros(nx*ny*6, dtype=wp.vec3)
    wp.launch(
        kernel=box_contact_points,
        dim=[6, nx, ny],
        inputs=[contact_edge_x, contact_edge_y, contact_edge_z, nx, ny],
        outputs=[contact_points, contact_normals],
    )

    contact_points = contact_points.reshape((num_env, num_body, num_contact_points))
    contact_normals = contact_normals.reshape((num_env, num_body, num_contact_points))

    """
    calculate global contact point position, linear velocity
    """
    body_pos_w = np.zeros((num_env, num_body, 3))
    body_quat_w = np.zeros((num_env, num_body, 4))

    # # identity quat
    # body_quat_w[:, :, 3] = 1.0 # warp uses (x, y, z, w) convention for quaternion

    # # roll -5deg
    # body_quat_w[:, :, 0] = -0.0436194
    # body_quat_w[:, :, 1] = 0
    # body_quat_w[:, :, 2] = 0
    # body_quat_w[:, :, 3] = 0.9990482

    # yaw 90deg 
    # 0, 0, 0.7071068, 0.7071068
    # body_quat_w[:, :, 0] = 0
    # body_quat_w[:, :, 1] = 0
    # body_quat_w[:, :, 2] = 0.7071068
    # body_quat_w[:, :, 3] = 0.7071068

    # pitch -30deg
    body_quat_w[:, :, 0] = 0
    body_quat_w[:, :, 1] = -0.258819
    body_quat_w[:, :, 2] = 0
    body_quat_w[:, :, 3] = 0.9659258

    # # roll 15deg, pitch -30deg 
    # body_quat_w[:, :, 0] = 0.1260786
    # body_quat_w[:, :, 1] = -0.2566048
    # body_quat_w[:, :, 2] = -0.0337827
    # body_quat_w[:, :, 3] = 0.9576622

    body_lin_vel_w = np.zeros((num_env, num_body, 3))
    body_lin_vel_w[:, :, 0] = 0.0
    body_lin_vel_w[:, :, 1] = 1.0
    body_lin_vel_w[:, :, 2] = -1.0
    body_ang_vel_w = np.zeros((num_env, num_body, 3))
    body_ang_vel_w[:, :, 1] = 0

    body_pos_w_wp = wp.from_numpy(body_pos_w, dtype=wp.vec3f)
    body_quat_w_wp = wp.from_numpy(body_quat_w, dtype=wp.quatf)
    body_lin_vel_w_wp = wp.from_numpy(body_lin_vel_w, dtype=wp.vec3f)
    body_ang_vel_w_wp = wp.from_numpy(body_ang_vel_w, dtype=wp.vec3f)

    contact_point_pos_w = wp.zeros_like(contact_points)
    contact_point_lin_vel_w = wp.zeros_like(contact_points)

    wp.launch(
        kernel=compute_contact_point_pos_w,
        dim=[num_env, num_body, num_contact_points],
        inputs=[body_pos_w_wp, body_quat_w_wp, contact_points],
        outputs=[contact_point_pos_w],
    )

    wp.launch(
        kernel=compute_contact_point_lin_vel_w,
        dim=[num_env, num_body, num_contact_points],
        inputs=[body_pos_w_wp, body_lin_vel_w_wp, body_ang_vel_w_wp, contact_point_pos_w],
        outputs=[contact_point_lin_vel_w],
    )

    """
    calculate coordinate vectors
    """

    normal_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    v_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    r_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    t_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    z_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    n_rt_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)
    Z_DIR = wp.vec3f(0.0, 0.0, 1.0)
    wp.launch(
        kernel=expand_vec3_to_3d_vec3,
        dim=(num_env, num_body, num_contact_points),
        inputs=[Z_DIR],
        outputs=[z_direction_w],
    )

    wp.launch(
        kernel=compute_normal_direction_w,
        dim=[num_env, num_body, num_contact_points],
        inputs=[body_quat_w_wp, contact_normals],
        outputs=[normal_direction_w],
    )

    wp.launch(
        kernel=compute_v_direction_w,
        dim=[num_env, num_body, num_contact_points],
        inputs=[contact_point_lin_vel_w],
        outputs=[v_direction_w],
    )

    threshold = 1e-6
    wp.launch(
        kernel=compute_r_direction_w,
        dim=[num_env, num_body, num_contact_points],
        inputs=[normal_direction_w, z_direction_w, v_direction_w, r_direction_w, threshold],
    )

    wp.launch(
        kernel=compute_t_direction_w, 
        dim=[num_env, num_body, num_contact_points],
        inputs=[z_direction_w, r_direction_w],
        outputs=[t_direction_w],
    )

    wp.launch(
        kernel=compute_n_rt_direction_w,
        dim=[num_env, num_body, num_contact_points],
        inputs=[normal_direction_w, z_direction_w], 
        outputs=[n_rt_direction_w],
    )


    intrusion_angle = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.float32) # gamma
    tilt_angle = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.float32) # beta
    twist_angle = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.float32) # psi
    n_rtz_direction_w = wp.zeros((num_env, num_body, num_contact_points), dtype=wp.vec3)

    wp.launch(
        kernel=compute_intrusion_angle,
        dim=[num_env, num_body, num_contact_points],
        inputs=[z_direction_w, v_direction_w, r_direction_w, intrusion_angle],
    )
    wp.launch(
        kernel=compute_tilt_angle,
        dim=[num_env, num_body, num_contact_points],
        inputs=[z_direction_w, normal_direction_w, r_direction_w, t_direction_w, tilt_angle],
    )
    wp.launch(
        kernel=compute_twist_angle,
        dim=[num_env, num_body, num_contact_points],
        inputs=[z_direction_w, normal_direction_w, r_direction_w, t_direction_w, n_rtz_direction_w, twist_angle],
    )


    beta_wp = twist_angle[0, 0]
    gamma_wp = intrusion_angle[0, 0]
    psi_wp = twist_angle[0, 0]
    r_dir_wp = r_direction_w[0, 0]
    t_dir_wp = t_direction_w[0, 0]
    z_dir_wp = z_direction_w[0, 0]

    material_cfg = Material3DRFTCfg()
    coef_1 = wp.array1d(material_cfg.coef_1, dtype=wp.float32)
    coef_2 = wp.array1d(material_cfg.coef_2, dtype=wp.float32)
    coef_3 = wp.array1d(material_cfg.coef_3, dtype=wp.float32)

    alpha_gen = wp.zeros(beta_wp.shape[0], dtype=wp.vec3)
    alpha_gen_cone = wp.zeros(beta_wp.shape[0], dtype=wp.vec3)
    alpha = wp.zeros(beta_wp.shape[0], dtype=wp.vec3)
    alpha_w = wp.zeros(beta_wp.shape[0], dtype=wp.vec3)

    wp.launch(
        kernel=compute_elementary_force_kernel,
        dim=beta_wp.shape[0],
        inputs=[beta_wp, gamma_wp, psi_wp, coef_1, coef_2, coef_3],
        outputs=[alpha_gen],
    )
    wp.launch(
        kernel=friction_cone_check_kernel,
        dim=beta_wp.shape[0],
        inputs=[beta_wp, gamma_wp, psi_wp, alpha_gen, material_cfg.dynamic_friction_coef],
        outputs=[alpha_gen_cone],
    )

    wp.launch(
        kernel=scale_alpha_kernel,
        dim=beta_wp.shape[0],
        inputs=[alpha_gen_cone, material_cfg.rho_c, material_cfg.mu_int],
        outputs=[alpha],
        )

    wp.launch(
        kernel=transform_local_alpha_to_world_alpha_kernel, 
        dim=beta_wp.shape[0],
        inputs=[alpha, r_dir_wp, t_dir_wp, z_dir_wp],
        outputs=[alpha_w],
    )


    env_id = 0
    body_id = 0
    contact_point_pos_w_np = contact_point_pos_w.numpy()[env_id, body_id, :, :]
    contact_point_lin_vel_w_np = contact_point_lin_vel_w.numpy()[env_id, body_id, :, :]
    n_vec = normal_direction_w.numpy()[env_id, body_id, :, :]
    v_vec = v_direction_w.numpy()[env_id, body_id, :, :]
    r_vec = r_direction_w.numpy()[env_id, body_id, :, :]
    t_vec = t_direction_w.numpy()[env_id, body_id, :, :]
    z_vec = z_direction_w.numpy()[env_id, body_id, :, :]
    n_rt_vec = n_rt_direction_w.numpy()[env_id, body_id, :, :]
    gamma = intrusion_angle.numpy()[env_id, body_id, :]
    beta = tilt_angle.numpy()[env_id, body_id, :]
    psi = twist_angle.numpy()[env_id, body_id, :]
    alpha_rtz_np = alpha.numpy()
    alpha_w_np = alpha_w.numpy()

    print("gamma:", gamma * 180.0 / math.pi)
    print("beta:", beta * 180.0 / math.pi)
    print("psi:", psi * 180.0 / math.pi)
    print("alpha_rtz:", alpha_rtz_np)
    print("alpha_w:", alpha_w_np)

    # plot contact point positions and linear velocities in 3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2], c='b', s=20)
    # normal
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        n_vec[:, 0], n_vec[:, 1], n_vec[:, 2],
        length=0.05, normalize=True, color='r',
    )
    # velocity
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        v_vec[:, 0], v_vec[:, 1], v_vec[:, 2],
        length=0.05, normalize=True, color='g',
    )
    # r direction
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        r_vec[:, 0], r_vec[:, 1], r_vec[:, 2],
        length=0.05, normalize=True, color='m',
    )
    # t direction
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        t_vec[:, 0], t_vec[:, 1], t_vec[:, 2],
        length=0.05, normalize=True, color='c',
    )
    # z direction
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        z_vec[:, 0], z_vec[:, 1], z_vec[:, 2],
        length=0.05, normalize=True, color='y',
    )
    # n_rt direction
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        n_rt_vec[:, 0], n_rt_vec[:, 1], n_rt_vec[:, 2],
        length=0.05, normalize=True, color='orange',
    )

    # alpha_w direction
    ax.quiver(
        contact_point_pos_w_np[:, 0], contact_point_pos_w_np[:, 1], contact_point_pos_w_np[:, 2],
        alpha_w_np[:, 0], alpha_w_np[:, 1], alpha_w_np[:, 2],
        length=0.1, normalize=True, color='b',
    )

    # # draw collider surface
    # surf = contact_point_pos_w_np.reshape(nx, ny, 3)
    # ax.plot_surface(surf[:, :, 0], surf[:, :, 1], surf[:, :, 2], alpha=0.3, color='gray')

    lim = max(contact_edge_x[1], contact_edge_y[1]) * 1.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Contact Points and velocities')
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.show()



def test_force_calculation_animation():
    import matplotlib.animation as mpl_animation
    import subprocess

    num_env = 1
    num_body = 1
    nx, ny = 2, 2
    num_frames = 50

    contact_edge_x = (-0.1, 0.1)
    contact_edge_y = (-0.05, 0.05)
    z_offset = -0.04

    contact_points = wp.zeros(nx * ny, dtype=wp.vec3)
    contact_normals = wp.zeros(nx * ny, dtype=wp.vec3)
    wp.launch(
        kernel=plane_contact_points,
        dim=[nx, ny],
        inputs=[contact_edge_x, contact_edge_y, z_offset, nx, ny],
        outputs=[contact_points, contact_normals],
    )
    contact_points = contact_points.reshape((num_env, num_body, nx * ny))
    contact_normals = contact_normals.reshape((num_env, num_body, nx * ny))

    material_cfg = Material3DRFTCfg()
    coef_1 = wp.array1d(material_cfg.coef_1, dtype=wp.float32)
    coef_2 = wp.array1d(material_cfg.coef_2, dtype=wp.float32)
    coef_3 = wp.array1d(material_cfg.coef_3, dtype=wp.float32)

    body_lin_vel_w = np.zeros((num_env, num_body, 3))
    body_lin_vel_w[:, :, 0] = 0.0
    body_lin_vel_w[:, :, 1] = 1.0
    body_lin_vel_w[:, :, 2] = -1.0
    body_ang_vel_w = np.zeros((num_env, num_body, 3))

    roll_angles = np.linspace(0.0, -20.0 * math.pi / 180.0, num_frames)

    # pre-compute all frames
    all_frames = []
    for phi in roll_angles:
        body_pos_w = np.zeros((num_env, num_body, 3))
        body_quat_w = np.zeros((num_env, num_body, 4))
        body_quat_w[:, :, 0] = math.sin(phi / 2.0)   # x  (roll about world-x)
        body_quat_w[:, :, 3] = math.cos(phi / 2.0)   # w

        body_pos_w_wp       = wp.from_numpy(body_pos_w,      dtype=wp.vec3f)
        body_quat_w_wp      = wp.from_numpy(body_quat_w,     dtype=wp.quatf)
        body_lin_vel_w_wp   = wp.from_numpy(body_lin_vel_w,  dtype=wp.vec3f)
        body_ang_vel_w_wp   = wp.from_numpy(body_ang_vel_w,  dtype=wp.vec3f)

        contact_point_pos_w     = wp.zeros_like(contact_points)
        contact_point_lin_vel_w = wp.zeros_like(contact_points)

        wp.launch(compute_contact_point_pos_w,
                  dim=[num_env, num_body, nx * ny],
                  inputs=[body_pos_w_wp, body_quat_w_wp, contact_points],
                  outputs=[contact_point_pos_w])
        wp.launch(compute_contact_point_lin_vel_w,
                  dim=[num_env, num_body, nx * ny],
                  inputs=[body_pos_w_wp, body_lin_vel_w_wp, body_ang_vel_w_wp, contact_point_pos_w],
                  outputs=[contact_point_lin_vel_w])

        num_cp = nx * ny
        normal_direction_w  = wp.zeros((num_env, num_body, num_cp), dtype=wp.vec3)
        v_direction_w       = wp.zeros((num_env, num_body, num_cp), dtype=wp.vec3)
        r_direction_w       = wp.zeros((num_env, num_body, num_cp), dtype=wp.vec3)
        t_direction_w       = wp.zeros((num_env, num_body, num_cp), dtype=wp.vec3)
        z_direction_w       = wp.zeros((num_env, num_body, num_cp), dtype=wp.vec3)
        n_rt_direction_w    = wp.zeros((num_env, num_body, num_cp), dtype=wp.vec3)

        Z_DIR = wp.vec3f(0.0, 0.0, 1.0)
        wp.launch(expand_vec3_to_3d_vec3,    dim=(num_env, num_body, num_cp), inputs=[Z_DIR],                                             outputs=[z_direction_w])
        wp.launch(compute_normal_direction_w, dim=[num_env, num_body, num_cp], inputs=[body_quat_w_wp, contact_normals],                  outputs=[normal_direction_w])
        wp.launch(compute_v_direction_w,      dim=[num_env, num_body, num_cp], inputs=[contact_point_lin_vel_w],                          outputs=[v_direction_w])
        wp.launch(compute_r_direction_w,      dim=[num_env, num_body, num_cp], inputs=[normal_direction_w, z_direction_w, v_direction_w, r_direction_w, 1e-6])
        wp.launch(compute_t_direction_w,      dim=[num_env, num_body, num_cp], inputs=[z_direction_w, r_direction_w],                     outputs=[t_direction_w])
        wp.launch(compute_n_rt_direction_w,   dim=[num_env, num_body, num_cp], inputs=[normal_direction_w, z_direction_w],                outputs=[n_rt_direction_w])

        intrusion_angle   = wp.zeros((num_env, num_body, num_cp), dtype=wp.float32)
        tilt_angle        = wp.zeros((num_env, num_body, num_cp), dtype=wp.float32)
        twist_angle       = wp.zeros((num_env, num_body, num_cp), dtype=wp.float32)
        n_rtz_direction_w = wp.zeros((num_env, num_body, num_cp), dtype=wp.vec3)

        wp.launch(compute_intrusion_angle, dim=[num_env, num_body, num_cp], inputs=[z_direction_w, v_direction_w, r_direction_w, intrusion_angle])
        wp.launch(compute_tilt_angle,      dim=[num_env, num_body, num_cp], inputs=[z_direction_w, normal_direction_w, r_direction_w, t_direction_w, tilt_angle])
        wp.launch(compute_twist_angle,     dim=[num_env, num_body, num_cp], inputs=[z_direction_w, normal_direction_w, r_direction_w, t_direction_w, n_rtz_direction_w, twist_angle])

        beta_wp   = tilt_angle[0, 0]
        gamma_wp  = intrusion_angle[0, 0]
        psi_wp    = twist_angle[0, 0]
        r_dir_wp  = r_direction_w[0, 0]
        t_dir_wp  = t_direction_w[0, 0]
        z_dir_wp  = z_direction_w[0, 0]

        alpha_gen      = wp.zeros(beta_wp.shape[0], dtype=wp.vec3)
        alpha_gen_cone = wp.zeros(beta_wp.shape[0], dtype=wp.vec3)
        alpha          = wp.zeros(beta_wp.shape[0], dtype=wp.vec3)
        alpha_w_arr    = wp.zeros(beta_wp.shape[0], dtype=wp.vec3)

        wp.launch(compute_elementary_force_kernel,           dim=beta_wp.shape[0], inputs=[beta_wp, gamma_wp, psi_wp, coef_1, coef_2, coef_3],                              outputs=[alpha_gen])
        wp.launch(friction_cone_check_kernel,                dim=beta_wp.shape[0], inputs=[beta_wp, gamma_wp, psi_wp, alpha_gen, material_cfg.dynamic_friction_coef],        outputs=[alpha_gen_cone])
        wp.launch(scale_alpha_kernel,                        dim=beta_wp.shape[0], inputs=[alpha_gen_cone, material_cfg.rho_c, material_cfg.mu_int],                         outputs=[alpha])
        wp.launch(transform_local_alpha_to_world_alpha_kernel, dim=beta_wp.shape[0], inputs=[alpha, r_dir_wp, t_dir_wp, z_dir_wp],                                           outputs=[alpha_w_arr])

        all_frames.append({
            'pos':     contact_point_pos_w.numpy()[0, 0],
            'n':       normal_direction_w.numpy()[0, 0],
            'v':       v_direction_w.numpy()[0, 0],
            'r':       r_direction_w.numpy()[0, 0],
            't':       t_direction_w.numpy()[0, 0],
            'z':       z_direction_w.numpy()[0, 0],
            'n_rt':    n_rt_direction_w.numpy()[0, 0],
            'alpha_w': alpha_w_arr.numpy(),
            'beta':    beta_wp.numpy(),
            'gamma':   gamma_wp.numpy(),
            'psi':     psi_wp.numpy(),
        })

    # build animation
    lim = max(contact_edge_x[1], contact_edge_y[1]) * 1.5
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(idx):
        ax.cla()
        d = all_frames[idx]
        roll_deg = roll_angles[idx] * 180.0 / math.pi
        pos = d['pos']

        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='b', s=20)
        ax.quiver(pos[:,0], pos[:,1], pos[:,2], d['n'][:,0],      d['n'][:,1],      d['n'][:,2],      length=0.05, normalize=True, color='r',      label='n')
        ax.quiver(pos[:,0], pos[:,1], pos[:,2], d['v'][:,0],      d['v'][:,1],      d['v'][:,2],      length=0.05, normalize=True, color='g',      label='v')
        ax.quiver(pos[:,0], pos[:,1], pos[:,2], d['r'][:,0],      d['r'][:,1],      d['r'][:,2],      length=0.05, normalize=True, color='m',      label='r')
        ax.quiver(pos[:,0], pos[:,1], pos[:,2], d['t'][:,0],      d['t'][:,1],      d['t'][:,2],      length=0.05, normalize=True, color='c',      label='t')
        ax.quiver(pos[:,0], pos[:,1], pos[:,2], d['z'][:,0],      d['z'][:,1],      d['z'][:,2],      length=0.05, normalize=True, color='y',      label='z')
        ax.quiver(pos[:,0], pos[:,1], pos[:,2], d['n_rt'][:,0],   d['n_rt'][:,1],   d['n_rt'][:,2],   length=0.05, normalize=True, color='orange', label='n_rt')
        ax.quiver(pos[:,0], pos[:,1], pos[:,2], d['alpha_w'][:,0],d['alpha_w'][:,1],d['alpha_w'][:,2],length=0.1,  normalize=True, color='b',      label='α_w')

        surf = pos.reshape(nx, ny, 3)
        ax.plot_surface(surf[:,:,0], surf[:,:,1], surf[:,:,2], alpha=0.3, color='gray')

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        alpha_vec = d['alpha_w'][0]/np.linalg.norm(d['alpha_w'][0])

        ax.set_title(f'Roll = {roll_deg:.1f} deg | α_w = [{alpha_vec[0]:.2f}, {alpha_vec[1]:.2f}, {alpha_vec[2]:.2f}] \n beta={d["beta"][0]*180.0/math.pi:.1f} deg, gamma={d["gamma"][0]*180.0/math.pi:.1f} deg, psi={d["psi"][0]*180.0/math.pi:.1f} deg')
        ax.legend(loc='upper right', fontsize=7)
        ax.view_init(elev=20, azim=45)

    ani = mpl_animation.FuncAnimation(fig, update, frames=num_frames, interval=100)

    save_path = '/tmp/force_roll_animation.gif'
    ani.save(save_path, writer='pillow', fps=10)
    print(f"Saved animation to {save_path}")
    plt.close(fig)
    subprocess.Popen(['xdg-open', save_path])


if __name__ == "__main__":
    # test_compute_elementary_force()
    # test_collider_generation()
    # test_contact_kinematics()
    # test_contact_coordinates()
    # test_characteristic_angle()
    test_force_calculation()
    # test_force_calculation_animation()