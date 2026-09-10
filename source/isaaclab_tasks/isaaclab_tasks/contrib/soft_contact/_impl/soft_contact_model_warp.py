# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
import warp as wp


from .soft_contact_model_data import SoftContactData
from .material import ConeDRFTCfg, Material3DRFTCfg, MaterialCfg, PoppySeedCPCfg, SpringDamperCfg
from .collider import Collider, ColliderCfg, PlaneColliderCfg, BoxColliderCfg, SphereColliderCfg
# from .kernels import (
#     compute_contact_point_lin_vel_w,
#     compute_contact_point_pos_w,
#     compute_contact_wrench,
#     compute_intrusion_angle,
#     compute_normal_direction_w,
#     compute_r_direction_w,
#     compute_contact_force,
#     compute_contact_force_2d,
#     compute_spring_damper_force,
#     compute_t_direction_w,
#     compute_tilt_angle,
#     compute_twist_angle,
#     compute_v_direction_w,
#     expand_vec3_to_3d_vec3,
#     reset,
#     reset_2d,
#     transform_global_wrench_to_body,
#     update_array_with_index,
#     zero_wrench,
# )

from .kernels import (
    compute_cone_drft_force,
    compute_contact_point_lin_vel_w,
    compute_contact_point_pos_w,
    compute_contact_wrench,
    compute_intrusion_angle,
    compute_normal_direction_w,
    compute_r_direction_w,
    compute_contact_force,
    compute_contact_force_2d,
    compute_spring_damper_force,
    compute_t_direction_w,
    compute_tilt_angle,
    compute_twist_angle,
    compute_v_direction_w,
    expand_vec3_to_3d_vec3,
    reset,
    reset_2d,
    reset_cone,
    transform_global_wrench_to_body,
    update_array_with_index,
    update_prev_velocity,
    zero_wrench,
)


class RFT_3D:
    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        device: torch.device | str,
        dt: float,
        material_cfg: Material3DRFTCfg,
        collider_cfg: ColliderCfg,
        history_length: int = 3,
        history_logging_decimation: int = 1,
        contact_threshold: float = 40.0,
        enable_ema_filter: bool = True,
    ) -> None:
        """
        Soft contact model based on 3D RFT proposed in
        https://www.pnas.org/doi/10.1073/pnas.2214017120

        Args:
            num_envs: number of parallel environments
            num_bodies: number of bodies using soft contact model per env
            device: torch device
            dt: simulation time step
            history_length: length of history for force tracking
            material_cfg: material configuration
            collider_cfg: collider geometry configuration
        """

        self.cfg = material_cfg
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.device = device
        self.dt = dt
        self.c_r = 100 / (1 / self.dt)  # 100/f (e.g. f=2000hz -> 0.05)
        self.history_length = history_length
        self.history_logging_decimation = history_logging_decimation
        self._history_step_counter: int = 0
        self.enable_ema_filter = enable_ema_filter
        self.contact_threshold = contact_threshold

        # Build collider geometry
        self.collider = Collider(collider_cfg, num_envs, num_bodies, device)
        self.num_contact_points = self.collider.num_contact_points
        self.surface_area = self.collider.surface_area

        self._data: SoftContactData = SoftContactData()

        self.create_buffers()
        self.initialize_data()

        print("-" * 40)
        print("3D RFT soft contact.")
        print("backend: warp")
        print(f"Number of envs: {self.num_envs}")
        print(f"Number of bodies per env: {self.num_bodies}")
        print(f"Number of contact points per body: {self.num_contact_points}")
        print(f"Contact surface area per body: {self.surface_area:.2f} m^2")
        print(f"mu int: {self.cfg.mu_int:.2f}")
        print(f"rho c: {self.cfg.rho_c:.2f} kg/m^3")
        print(f"mu_surf: {self.cfg.dynamic_friction_coef:.2f}")
        print("-" * 40)
        print("\n")

        self.capture()

    """
    make graph of contact evaluation
    """

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._eval_contacts()
            self.graph = capture.graph
        else:
            self.graph = None

    def create_buffers(self):
        """
        create warp arrays
        """
        # torch buffers
        self.body_pos_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_quat_torch = torch.zeros((self.num_envs, self.num_bodies, 4), device=self.device)
        self.body_lin_vel_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_ang_vel_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)

        # body state copied from torch
        self.body_pos = wp.from_torch(self.body_pos_torch, dtype=wp.vec3f)
        self.body_quat = wp.from_torch(self.body_quat_torch, dtype=wp.quatf)
        self.body_lin_vel = wp.from_torch(self.body_lin_vel_torch, dtype=wp.vec3f)
        self.body_ang_vel = wp.from_torch(self.body_ang_vel_torch, dtype=wp.vec3f)

        # contact points (provided by collider)
        self.contact_point_local = self.collider.contact_point_local
        self.normal_dir_local = self.collider.normal_dir_local
        self.contact_point_pos = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_lin_vel = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_lin_vel_prev = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )

        # characteristics angle
        self.contact_point_tilt_angle = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.float32, device=self.device
        )
        self.contact_point_intrusion_angle = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.float32, device=self.device
        )
        self.contact_point_twist_angle = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.float32, device=self.device
        )

        # local coordinate
        self.n_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.r_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.t_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.z_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.v_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.n_rtz_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )

        # contact forces
        self.contact_point_force = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_torque = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_force = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_torque = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)

        # cache for EMA filter
        self.alpha_unfiltered = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.alpha_filtered = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.resitive_force = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.tau_r = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.float32, device=self.device
        )

        # material parameters
        self.static_friction_coef = wp.full(
            self.num_envs, self.cfg.static_friction_coef, dtype=wp.float32, device=self.device
        )
        self.dynamic_friction_coef = wp.full(
            self.num_envs, self.cfg.dynamic_friction_coef, dtype=wp.float32, device=self.device
        )
        self.rho_c = wp.full(self.num_envs, self.cfg.rho_c, dtype=wp.float32, device=self.device)
        self.mu_int = wp.full(self.num_envs, self.cfg.mu_int, dtype=wp.float32, device=self.device)
        self.coef_1 = wp.array1d(self.cfg.coef_1, dtype=wp.float32, device=self.device)
        self.coef_2 = wp.array1d(self.cfg.coef_2, dtype=wp.float32, device=self.device)
        self.coef_3 = wp.array1d(self.cfg.coef_3, dtype=wp.float32, device=self.device)
        self.kf = wp.full(self.num_envs, self.cfg.kf, dtype=wp.float32, device=self.device)

        # timestamps
        self._timestamp = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._timestamp_last_update = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Bind torch buffers to warp buffers
        self.torch_contact_point_pos = wp.to_torch(self.contact_point_pos)
        self.torch_contact_point_vel = wp.to_torch(self.contact_point_lin_vel)
        self.torch_contact_force = wp.to_torch(self.contact_force)
        self.torch_contact_torque = wp.to_torch(self.contact_torque)
        self.torch_contact_force_b = wp.to_torch(self.contact_force_b)
        self.torch_contact_torque_b = wp.to_torch(self.contact_torque_b)
        self.torch_contact_point_force = wp.to_torch(self.contact_point_force)
        self.torch_contact_point_torque = wp.to_torch(self.contact_point_torque)
        self.torch_friction_coef = wp.to_torch(self.dynamic_friction_coef)
        self.torch_rho_c = wp.to_torch(self.rho_c)
        self.torch_mu_int = wp.to_torch(self.mu_int)

        # extra angles
        self.torch_contact_point_tilt_angle = wp.to_torch(self.contact_point_tilt_angle)
        self.torch_contact_point_intrusion_angle = wp.to_torch(self.contact_point_intrusion_angle)
        self.torch_contact_point_twist_angle = wp.to_torch(self.contact_point_twist_angle)

        # extra local coordinate
        self.torch_n_dir = wp.to_torch(self.n_dir)
        self.torch_r_dir = wp.to_torch(self.r_dir)
        self.torch_t_dir = wp.to_torch(self.t_dir)
        self.torch_v_dir = wp.to_torch(self.v_dir)


    def initialize_data(self) -> None:
        """
        Initialize soft contact data.
        """
        self._data.net_forces_w = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self._data.net_forces_w_history = torch.zeros(
            (self.num_envs, self.history_length, self.num_bodies, 3), device=self.device
        )
        self._data.force_matrix_w = torch.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device
        )
        self._data.force_matrix_w_history = torch.zeros(
            (self.num_envs, self.history_length, self.num_bodies, self.num_contact_points, 3), device=self.device
        )
        self._data.last_air_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.current_air_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.last_contact_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.current_contact_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.is_sensor_active = torch.zeros((self.num_envs, self.num_bodies), dtype=torch.bool, device=self.device)

    """
    properties.
    """

    @property
    def data(self) -> SoftContactData:
        return self._data

    @property
    def contact_wrench(self) -> torch.Tensor:
        contact_wrench = torch.cat(
            (self.torch_contact_force, self.torch_contact_torque), dim=-1
        )  # (num_envs, num_bodies, 6)
        return contact_wrench

    @property
    def contact_wrench_b(self) -> torch.Tensor:
        contact_wrench_b = torch.cat(
            (self.torch_contact_force_b, self.torch_contact_torque_b), dim=-1
        )  # (num_envs, num_bodies, 6)
        return contact_wrench_b

    @property
    def contact_point_wrench(self) -> torch.Tensor:
        contact_point_wrench = torch.cat(
            (self.torch_contact_point_force, self.torch_contact_point_torque), dim=-1
        )  # (num_envs, num_bodies, num_contact_points, 6)
        return contact_point_wrench

    @property
    def terrain_friction(self) -> torch.Tensor:
        return self.torch_friction_coef

    @property
    def terrain_density(self) -> torch.Tensor:
        return self.torch_rho_c

    @property
    def terrain_stiffness(self) -> torch.Tensor:
        return self.torch_mu_int

    """
    operations.
    """

    def update(
        self, body_pos: torch.Tensor, body_quat: torch.Tensor, body_lin_vel: torch.Tensor, body_ang_vel: torch.Tensor
    ):
        """
        Update soft contact model.

        Args:
            body_pos: intruder position. (num_envs, num_bodies, 3)
            body_quat: intruder orientation in quaternion form. (num_envs, num_bodies, 4)
            body_lin_vel: intruder linear velocity wrt global frame. (num_envs, num_bodies, 3)
            body_ang_vel: intruder angular velocity wrt global frame. (num_envs, num_bodies, 3)
        """
        # copy to torch buffer
        self.body_pos_torch[:] = body_pos
        self.body_quat_torch[:] = body_quat[:, :, [1, 2, 3, 0]]  # convert (w, x, y, z) to (x, y, z, w)
        self.body_lin_vel_torch[:] = body_lin_vel
        self.body_ang_vel_torch[:] = body_ang_vel

        # evaluate contact forces
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._eval_contacts()

        # update timestamp and data
        self._timestamp += self.dt
        self._update_data(torch.arange(self.num_envs, device=self.device))
        self._timestamp_last_update[:] = self._timestamp[:]

    def randomize_ground_stiffness(self, env_ids: torch.Tensor, mu_int: torch.Tensor) -> None:
        """
        Update ground stiffness (N/m) for each env.
        Implementation is similar to terrain curriculum used in terrain importer class.
        This can be triggered by curriculum manager.

        Args:
            env_ids: tensor of env ids to update
            mu_int: tensor of mu_int values (len(env_ids), )
        """
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64), wp.from_torch(mu_int, dtype=wp.float32), self.mu_int],
            device=self.device,
        )

    def update_material_density(self, env_ids: torch.Tensor, packing_density: torch.Tensor, bulk_density: torch.Tensor) -> None:
        """
        Update material density for each env.
        This can be triggered by event manager.

        Args:
            env_ids: tensor of env ids to update
            packing_density: tensor of packing densities (len(env_ids), )
            bulk_density: tensor of bulk densities (len(env_ids), )
        """
        rho_c = bulk_density * packing_density
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64), wp.from_torch(rho_c, dtype=wp.float32), self.rho_c],
            device=self.device,
        )

    def update_friction_params(
        self, env_ids: torch.Tensor, static_friction_coef: torch.Tensor, dynamic_friction_coef: torch.Tensor
    ) -> None:
        """
        Update friction coefficients for each env.
        This can be triggered by event manager.

        Args:
            env_ids: tensor of env ids to update
            static_friction_coef: tensor of static friction coefficients (len(env_ids), )
            dynamic_friction_coef: tensor of dynamic friction coefficients (len(env_ids), )
        """
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[
                wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64),
                wp.from_torch(static_friction_coef, dtype=wp.float32),
                self.static_friction_coef,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[
                wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64),
                wp.from_torch(dynamic_friction_coef, dtype=wp.float32),
                self.dynamic_friction_coef,
            ],
            device=self.device,
        )

    """
    data helper functions.
    """

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        """Checks if bodies that have established contact within the last :attr:`dt` seconds.

        This function checks if the bodies have established contact within the last :attr:`dt` seconds
        by comparing the current contact time with the given time period. If the contact time is less
        than the given time period, then the bodies are considered to be in contact.

        Note:
            The function assumes that :attr:`dt` is a factor of the sensor update time-step. In other
            words :math:`dt / dt_sensor = n`, where :math:`n` is a natural number. This is always true
            if the sensor is updated by the physics or the environment stepping time-step and the sensor
            is read by the environment stepping time-step.

        Args:
            dt: The time period since the contact was established.
            abs_tol: The absolute tolerance for the comparison.

        Returns:
            A boolean tensor indicating the bodies that have established contact within the last
            :attr:`dt` seconds. Shape is (N, B), where N is the number of sensors and B is the
            number of bodies in each sensor.

        Raises:
            RuntimeError: If the sensor is not configured to track contact time.
        """
        # check if the bodies are in contact
        currently_in_contact = self._data.current_contact_time > 0.0
        less_than_dt_in_contact = self._data.current_contact_time < (dt + abs_tol)
        return currently_in_contact * less_than_dt_in_contact

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        """Checks if bodies that have broken contact within the last :attr:`dt` seconds.

        This function checks if the bodies have broken contact within the last :attr:`dt` seconds
        by comparing the current air time with the given time period. If the air time is less
        than the given time period, then the bodies are considered to not be in contact.

        Note:
            It assumes that :attr:`dt` is a factor of the sensor update time-step. In other words,
            :math:`dt / dt_sensor = n`, where :math:`n` is a natural number. This is always true if
            the sensor is updated by the physics or the environment stepping time-step and the sensor
            is read by the environment stepping time-step.

        Args:
            dt: The time period since the contract is broken.
            abs_tol: The absolute tolerance for the comparison.

        Returns:
            A boolean tensor indicating the bodies that have broken contact within the last :attr:`dt` seconds.
            Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

        Raises:
            RuntimeError: If the sensor is not configured to track contact time.
        """
        # check if the sensor is configured to track contact time
        currently_detached = self._data.current_air_time > 0.0
        less_than_dt_detached = self._data.current_air_time < (dt + abs_tol)
        return currently_detached * less_than_dt_detached

    def _update_data(self, env_ids: torch.Tensor) -> None:
        """
        Update soft contact data.
        Majority of implementations are from IsaacLab's contact sensor class.

        Args:
            env_ids: tensor of env ids to update
        """
        self._data.net_forces_w[env_ids, :, :] = self.torch_contact_force[env_ids, :, :]  # type: ignore
        self._data.force_matrix_w[env_ids, :, :, :] = self.torch_contact_point_force[env_ids, :, :, :]  # type: ignore
        if self.history_length > 0:
            self._history_step_counter += 1
            if self._history_step_counter >= self.history_logging_decimation:
                self._history_step_counter = 0
                self._data.net_forces_w_history[env_ids] = self._data.net_forces_w_history[env_ids].roll(shifts=1, dims=1)  # type: ignore
                self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]  # type: ignore

                self._data.force_matrix_w_history[env_ids] = self._data.force_matrix_w_history[env_ids].roll(
                    shifts=1, dims=1
                )  # type: ignore
                self._data.force_matrix_w_history[env_ids, 0] = self._data.force_matrix_w[env_ids]  # type: ignore

        # track air time (see contact sensor class)
        elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
        is_contact = torch.norm(self._data.net_forces_w[env_ids, :, :], dim=-1) > self.contact_threshold  # type: ignore
        is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact  # type: ignore
        is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact  # type: ignore
        # -- update the last contact time if body has just become in contact
        self._data.last_air_time[env_ids] = torch.where(  # type: ignore
            is_first_contact,
            self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),  # type: ignore
            self._data.last_air_time[env_ids],  # type: ignore
        )
        # -- increment time for bodies that are not in contact
        self._data.current_air_time[env_ids] = torch.where(  # type: ignore
            ~is_contact,
            self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),
            0.0,  # type: ignore
        )
        # -- update the last contact time if body has just detached
        self._data.last_contact_time[env_ids] = torch.where(  # type: ignore
            is_first_detached,
            self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),  # type: ignore
            self._data.last_contact_time[env_ids],  # type: ignore
        )
        # -- increment time for bodies that are in contact
        self._data.current_contact_time[env_ids] = torch.where(  # type: ignore
            is_contact,
            self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),
            0.0,  # type: ignore
        )

    """
    helper functions.
    """

    def _eval_contacts(self) -> None:
        """
        Update contact points' kinematic states and compute contact forces.
        """
        # step1: calculate contact pos, lin vel, and surface normal
        # compute contact points in global frame
        wp.launch(
            kernel=compute_contact_point_pos_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.body_pos, self.body_quat, self.contact_point_local, self.contact_point_pos],
            device=self.device,
        )

        # compute contact point linear velocity in global frame
        wp.launch(
            kernel=compute_contact_point_lin_vel_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[
                self.body_pos,
                self.body_lin_vel,
                self.body_ang_vel,
                self.contact_point_pos,
                self.contact_point_lin_vel,
            ],
            device=self.device,
        )

        # step2: Find local coordinate frame {r, theta, z}
        # compute unit vectors (z, n, v, r, t)
        # see S7 eq.4 from https://www.pnas.org/doi/10.1073/pnas.2214017120

        # compute contact normal
        # currently, we only consider surface normal at the bottom of foot which faces to -z in local frame.
        wp.launch(
            kernel=compute_normal_direction_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.body_quat, self.normal_dir_local, self.n_dir],
            device=self.device,
        )

        Z_DIR = wp.vec3f(0.0, 0.0, 1.0)
        wp.launch(
            kernel=expand_vec3_to_3d_vec3,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[Z_DIR, self.z_dir],
            device=self.device,
        )
        wp.launch(
            kernel=compute_v_direction_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.contact_point_lin_vel, self.v_dir],
            device=self.device,
        )
        wp.launch(
            kernel=compute_r_direction_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.n_dir, self.z_dir, self.v_dir, self.r_dir, 0.01],
            device=self.device,
        )
        wp.launch(
            kernel=compute_t_direction_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.z_dir, self.r_dir, self.t_dir],
            device=self.device,
        )

        # step3: compute characteristic angles

        # compute contact point velocity angle (gamma)
        # see S7 eq.6 from https://www.pnas.org/doi/10.1073/pnas.2214017120
        wp.launch(
            kernel=compute_intrusion_angle,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.z_dir, self.v_dir, self.r_dir, self.contact_point_intrusion_angle],
            device=self.device,
        )

        # compute contact point tilt angle (beta)
        # see S7 eq.5 from https://www.pnas.org/doi/10.1073/pnas.2214017120
        wp.launch(
            kernel=compute_tilt_angle,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.z_dir, self.n_dir, self.r_dir, self.t_dir, self.contact_point_tilt_angle],
            device=self.device,
        )

        # compute twist angle (psi)
        # see S7 eq.7 from https://www.pnas.org/doi/10.1073/pnas.2214017120
        wp.launch(
            kernel=compute_twist_angle,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.z_dir, self.n_dir, self.r_dir, self.t_dir, self.n_rtz_dir, self.contact_point_twist_angle],
            device=self.device,
        )

        # step4: compute resistive force alpha (N/m^3) and point forces (N)
        # see eq.1 in https://www.pnas.org/doi/10.1073/pnas.2214017120

        wp.launch(
            kernel=compute_contact_force,
            dim=(self.num_envs, self.num_bodies * self.num_contact_points),
            inputs=[
                self.contact_point_pos.reshape((self.num_envs, -1)),
                self.contact_point_lin_vel.reshape((self.num_envs, -1)),
                self.contact_point_lin_vel_prev.reshape((self.num_envs, -1)),

                self.contact_point_tilt_angle.reshape((self.num_envs, -1)),
                self.contact_point_intrusion_angle.reshape((self.num_envs, -1)),
                self.contact_point_twist_angle.reshape((self.num_envs, -1)),

                self.r_dir.reshape((self.num_envs, -1)),
                self.t_dir.reshape((self.num_envs, -1)),
                self.z_dir.reshape((self.num_envs, -1)),
                self.n_dir.reshape((self.num_envs, -1)),

                self.rho_c,
                self.mu_int,
                self.dynamic_friction_coef,
                self.kf,

                self.coef_1,
                self.coef_2,
                self.coef_3,

                self.tau_r,
                self.c_r,
                wp.int32(1 if self.enable_ema_filter else 0),
                self.collider.dA,
                self.num_contact_points,

                self.alpha_unfiltered,
                self.alpha_filtered,
                self.resitive_force,
            ],
        )

        # Step5: sum up contact point forces to get net contact force and torque on the body, and transform to body frame
        # clear contact wrench before atomic add
        wp.launch(
            kernel=zero_wrench,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                self.contact_force,
                self.contact_torque,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=compute_contact_wrench,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[
                self.body_pos,
                self.contact_point_pos,
                self.resitive_force.reshape((self.num_envs, self.num_bodies, self.num_contact_points)),
                self.contact_point_force,
                self.contact_point_torque,
                self.contact_force,
                self.contact_torque,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=transform_global_wrench_to_body,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                self.body_quat,
                self.contact_force,
                self.contact_torque,
                self.contact_force_b,
                self.contact_torque_b,
            ],
            device=self.device,
        )

        wp.copy(self.contact_point_lin_vel_prev, self.contact_point_lin_vel)

    """
    reset.
    """

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # the managers hand out int32 indices, while the kernels below index with int64
        env_ids = env_ids.to(torch.int64)

        wp.launch(
            kernel=reset,
            dim=(len(env_ids), self.num_bodies * self.num_contact_points),
            inputs=[
                env_ids,
                self.alpha_unfiltered,
                self.alpha_filtered,
                self.tau_r,
                self.contact_point_lin_vel_prev.reshape((self.num_envs, -1)),
            ],
            device=self.device,
        )


class RFT_2D:
    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        device: torch.device | str,
        dt: float,
        material_cfg: MaterialCfg = PoppySeedCPCfg(),
        collider_cfg: ColliderCfg = PlaneColliderCfg(),
        history_length: int = 3,
        history_logging_decimation: int = 1,
        contact_threshold: float = 10.0,
        enable_ema_filter: bool = True,
    ) -> None:
        """
        Soft contact model based on 2D RFT proposed in
        https://www.science.org/doi/10.1126/science.1229163
        with dynamic inertial modification (DRFT) and EMA filtering.

        Args:
            num_envs: number of parallel environments
            num_bodies: number of bodies using soft contact model per env
            device: torch device
            dt: simulation time step
            material_cfg: 2D RFT material configuration (Fourier coefficients + friction)
            collider_cfg: collider geometry configuration
            history_length: length of history for force tracking
            contact_threshold: force magnitude threshold for contact detection (N)
            enable_ema_filter: whether to apply EMA filter on z-force
        """
        self.cfg = material_cfg
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.device = device
        self.dt = dt
        self.c_r = 100 / (1 / self.dt)
        self.history_length = history_length
        self.history_logging_decimation = history_logging_decimation
        self._history_step_counter: int = 0
        self.enable_ema_filter = enable_ema_filter
        self.contact_threshold = contact_threshold

        self.collider = Collider(collider_cfg, num_envs, num_bodies, device)
        self.num_contact_points = self.collider.num_contact_points
        self.surface_area = self.collider.surface_area

        self._data: SoftContactData = SoftContactData()

        self.create_buffers()
        self.initialize_data()

        print("-" * 40)
        print("2D RFT soft contact.")
        print("backend: warp")
        print(f"Number of envs: {self.num_envs}")
        print(f"Number of bodies per env: {self.num_bodies}")
        print(f"Number of contact points per body: {self.num_contact_points}")
        print(f"Contact surface area per body: {self.surface_area:.4f} m^2")
        print(f"rho_c: {self.cfg.rho_c:.2f} kg/m^3")
        print(f"mu_int: {self.cfg.mu_int:.2f}")
        print(f"mu_surf: {self.cfg.dynamic_friction_coef:.2f}")
        print("-" * 40)
        print("\n")

        self.capture()

    """
    CUDA graph capture.
    """

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._eval_contacts()
            self.graph = capture.graph
        else:
            self.graph = None

    def create_buffers(self):
        """Create warp arrays for 2D RFT."""
        # torch input buffers
        self.body_pos_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_quat_torch = torch.zeros((self.num_envs, self.num_bodies, 4), device=self.device)
        self.body_lin_vel_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_ang_vel_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)

        # body state warp views
        self.body_pos = wp.from_torch(self.body_pos_torch, dtype=wp.vec3f)
        self.body_quat = wp.from_torch(self.body_quat_torch, dtype=wp.quatf)
        self.body_lin_vel = wp.from_torch(self.body_lin_vel_torch, dtype=wp.vec3f)
        self.body_ang_vel = wp.from_torch(self.body_ang_vel_torch, dtype=wp.vec3f)

        # contact points from collider
        self.contact_point_local = self.collider.contact_point_local
        self.normal_dir_local = self.collider.normal_dir_local
        self.contact_point_pos = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_lin_vel = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_lin_vel_prev = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )

        # characteristic angles (tilt beta, intrusion gamma)
        self.contact_point_tilt_angle = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.float32, device=self.device
        )
        self.contact_point_intrusion_angle = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.float32, device=self.device
        )

        # local coordinate frame vectors
        self.n_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.r_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.t_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.z_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.v_dir = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )

        # contact forces
        self.contact_point_force = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_torque = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_force = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_torque = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)

        # EMA filter state (scalar per contact point)
        self.force_gm = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.float32, device=self.device
        )
        self.force_ema = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.float32, device=self.device
        )
        self.tau_r = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.float32, device=self.device
        )

        # final resistive force vector per contact point (output of compute_resistive_force_2d)
        self.resitive_force = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.vec3f, device=self.device
        )

        # per-env material parameters
        self.rho = wp.full(self.num_envs, self.cfg.rho, dtype=wp.float32, device=self.device)
        self.lam = wp.full(self.num_envs, self.cfg.lam, dtype=wp.float32, device=self.device)
        self.dynamic_friction_coef = wp.full(
            self.num_envs, self.cfg.dynamic_friction_coef, dtype=wp.float32, device=self.device
        )
        self.kf = wp.full(self.num_envs, self.cfg.kf, dtype=wp.float32, device=self.device)
        # quasistatic stiffness parameterised by rho_c and mu_int (same as 3D RFT)
        self.rho_c = wp.full(self.num_envs, self.cfg.rho_c, dtype=wp.float32, device=self.device)
        self.mu_int = wp.full(self.num_envs, self.cfg.mu_int, dtype=wp.float32, device=self.device)

        # timestamps
        self._timestamp = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._timestamp_last_update = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # torch views into warp buffers
        self.torch_contact_point_pos = wp.to_torch(self.contact_point_pos)
        self.torch_contact_point_vel = wp.to_torch(self.contact_point_lin_vel)
        self.torch_contact_force = wp.to_torch(self.contact_force)
        self.torch_contact_torque = wp.to_torch(self.contact_torque)
        self.torch_contact_force_b = wp.to_torch(self.contact_force_b)
        self.torch_contact_torque_b = wp.to_torch(self.contact_torque_b)
        self.torch_contact_point_force = wp.to_torch(self.contact_point_force)
        self.torch_contact_point_torque = wp.to_torch(self.contact_point_torque)
        self.torch_dynamic_friction_coef = wp.to_torch(self.dynamic_friction_coef)
        self.torch_rho_c = wp.to_torch(self.rho_c)
        self.torch_mu_int = wp.to_torch(self.mu_int)

        # extra angles
        self.torch_contact_point_tilt_angle = wp.to_torch(self.contact_point_tilt_angle)
        self.torch_contact_point_intrusion_angle = wp.to_torch(self.contact_point_intrusion_angle)
        self.torch_contact_point_twist_angle = torch.zeros_like(self.torch_contact_point_tilt_angle)

        # extra local coordinate
        self.torch_n_dir = wp.to_torch(self.n_dir)
        self.torch_r_dir = wp.to_torch(self.r_dir)
        self.torch_t_dir = wp.to_torch(self.t_dir)
        self.torch_v_dir = wp.to_torch(self.v_dir)

    def initialize_data(self) -> None:
        """Initialize soft contact data."""
        self._data.net_forces_w = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self._data.net_forces_w_history = torch.zeros(
            (self.num_envs, self.history_length, self.num_bodies, 3), device=self.device
        )
        self._data.force_matrix_w = torch.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device
        )
        self._data.force_matrix_w_history = torch.zeros(
            (self.num_envs, self.history_length, self.num_bodies, self.num_contact_points, 3), device=self.device
        )
        self._data.last_air_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.current_air_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.last_contact_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.current_contact_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.is_sensor_active = torch.zeros(
            (self.num_envs, self.num_bodies), dtype=torch.bool, device=self.device
        )

    """
    Properties.
    """

    @property
    def data(self) -> SoftContactData:
        return self._data

    @property
    def contact_wrench(self) -> torch.Tensor:
        return torch.cat(
            (self.torch_contact_force, self.torch_contact_torque), dim=-1
        )  # (num_envs, num_bodies, 6)

    @property
    def contact_wrench_b(self) -> torch.Tensor:
        return torch.cat(
            (self.torch_contact_force_b, self.torch_contact_torque_b), dim=-1
        )  # (num_envs, num_bodies, 6)

    @property
    def contact_point_wrench(self) -> torch.Tensor:
        return torch.cat(
            (self.torch_contact_point_force, self.torch_contact_point_torque), dim=-1
        )  # (num_envs, num_bodies, num_contact_points, 6)

    @property
    def terrain_friction(self) -> torch.Tensor:
        return self.torch_dynamic_friction_coef

    @property
    def terrain_density(self) -> torch.Tensor:
        return self.torch_rho_c

    @property
    def terrain_stiffness(self) -> torch.Tensor:
        return self.torch_mu_int

    """
    Operations.
    """

    def update(
        self,
        body_pos: torch.Tensor,
        body_quat: torch.Tensor,
        body_lin_vel: torch.Tensor,
        body_ang_vel: torch.Tensor,
    ):
        """
        Update soft contact model.

        Args:
            body_pos: intruder position. (num_envs, num_bodies, 3)
            body_quat: intruder orientation (w, x, y, z). (num_envs, num_bodies, 4)
            body_lin_vel: linear velocity in global frame. (num_envs, num_bodies, 3)
            body_ang_vel: angular velocity in global frame. (num_envs, num_bodies, 3)
        """
        self.body_pos_torch[:] = body_pos
        # (w,x,y,z) -> (x,y,z,w) as warp uses xyzw order
        self.body_quat_torch[:] = body_quat[:, :, [1, 2, 3, 0]]
        self.body_lin_vel_torch[:] = body_lin_vel
        self.body_ang_vel_torch[:] = body_ang_vel

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._eval_contacts()

        self._timestamp += self.dt
        self._update_data(torch.arange(self.num_envs, device=self.device))
        self._timestamp_last_update[:] = self._timestamp[:]

    def randomize_ground_stiffness(self, env_ids: torch.Tensor, mu_int: torch.Tensor) -> None:
        """
        Update per-env internal friction coefficient (controls quasistatic stiffness xi).
        Same API as 3D RFT.

        Args:
            env_ids: (n,) env indices to update
            mu_int: (n,) new internal friction coefficient values
        """
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64), wp.from_torch(mu_int, dtype=wp.float32), self.mu_int],
            device=self.device,
        )

    def update_material_density(
        self, env_ids: torch.Tensor, packing_density: torch.Tensor, bulk_density: torch.Tensor
    ) -> None:
        """
        Update per-env critical media density (rho_c = bulk_density * packing_density).
        Same API as 3D RFT.

        Args:
            env_ids: (n,) env indices to update
            packing_density: (n,) packing fraction
            bulk_density: (n,) bulk density (kg/m^3)
        """
        rho_c = bulk_density * packing_density
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64), wp.from_torch(rho_c, dtype=wp.float32), self.rho_c],
            device=self.device,
        )

    def update_friction_params(
        self,
        env_ids: torch.Tensor,
        static_friction_coef: torch.Tensor,
        dynamic_friction_coef: torch.Tensor,
    ) -> None:
        """
        Update per-env friction coefficients.

        Args:
            env_ids: (n,) env indices to update
            static_friction_coef: (n,) static friction (unused in force model, kept for API consistency)
            dynamic_friction_coef: (n,) dynamic friction
        """
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[
                wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64),
                wp.from_torch(dynamic_friction_coef, dtype=wp.float32),
                self.dynamic_friction_coef,
            ],
            device=self.device,
        )

    """
    Data helper functions.
    """

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        currently_in_contact = self._data.current_contact_time > 0.0
        less_than_dt_in_contact = self._data.current_contact_time < (dt + abs_tol)
        return currently_in_contact * less_than_dt_in_contact

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        currently_detached = self._data.current_air_time > 0.0
        less_than_dt_detached = self._data.current_air_time < (dt + abs_tol)
        return currently_detached * less_than_dt_detached

    def _update_data(self, env_ids: torch.Tensor) -> None:
        self._data.net_forces_w[env_ids, :, :] = self.torch_contact_force[env_ids, :, :]  # type: ignore
        self._data.force_matrix_w[env_ids, :, :, :] = self.torch_contact_point_force[env_ids, :, :, :]  # type: ignore
        if self.history_length > 0:
            self._history_step_counter += 1
            if self._history_step_counter >= self.history_logging_decimation:
                self._history_step_counter = 0
                self._data.net_forces_w_history[env_ids] = self._data.net_forces_w_history[env_ids].roll(  # type: ignore
                    shifts=1, dims=1
                )
                self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]  # type: ignore
                self._data.force_matrix_w_history[env_ids] = self._data.force_matrix_w_history[env_ids].roll(  # type: ignore
                    shifts=1, dims=1
                )
                self._data.force_matrix_w_history[env_ids, 0] = self._data.force_matrix_w[env_ids]  # type: ignore

        elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
        is_contact = torch.norm(self._data.net_forces_w[env_ids, :, :], dim=-1) > self.contact_threshold  # type: ignore
        is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact  # type: ignore
        is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact  # type: ignore
        self._data.last_air_time[env_ids] = torch.where(  # type: ignore
            is_first_contact,
            self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),  # type: ignore
            self._data.last_air_time[env_ids],  # type: ignore
        )
        self._data.current_air_time[env_ids] = torch.where(  # type: ignore
            ~is_contact,
            self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),
            0.0,  # type: ignore
        )
        self._data.last_contact_time[env_ids] = torch.where(  # type: ignore
            is_first_detached,
            self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),  # type: ignore
            self._data.last_contact_time[env_ids],  # type: ignore
        )
        self._data.current_contact_time[env_ids] = torch.where(  # type: ignore
            is_contact,
            self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),
            0.0,  # type: ignore
        )

    """
    Helper functions.
    """

    def _eval_contacts(self) -> None:
        """Compute contact kinematics, characteristic angles, and 2D RFT forces."""

        # step 1: contact point positions and velocities in global frame
        wp.launch(
            kernel=compute_contact_point_pos_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.body_pos, self.body_quat, self.contact_point_local, self.contact_point_pos],
            device=self.device,
        )
        wp.launch(
            kernel=compute_contact_point_lin_vel_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[
                self.body_pos,
                self.body_lin_vel,
                self.body_ang_vel,
                self.contact_point_pos,
            ],
            outputs=[self.contact_point_lin_vel],
            device=self.device,
        )

        # step 2: local coordinate frame {n, r, t, z, v}
        # compute local coordinate vectors in global frame.
        # see S7 eq.4 from https://www.pnas.org/doi/10.1073/pnas.2214017120
        wp.launch(
            kernel=compute_normal_direction_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.body_quat, self.normal_dir_local, self.n_dir],
            device=self.device,
        )
        Z_DIR = wp.vec3f(0.0, 0.0, 1.0)
        wp.launch(
            kernel=expand_vec3_to_3d_vec3,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[Z_DIR, self.z_dir],
            device=self.device,
        )
        wp.launch(
            kernel=compute_v_direction_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.contact_point_lin_vel, self.v_dir],
            device=self.device,
        )
        wp.launch(
            kernel=compute_r_direction_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.n_dir, self.z_dir, self.v_dir, self.r_dir, 0.01],
            device=self.device,
        )
        wp.launch(
            kernel=compute_t_direction_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.z_dir, self.r_dir, self.t_dir],
            device=self.device,
        )

        # step 3: intrusion angle gamma and tilt angle beta
        wp.launch(
            kernel=compute_intrusion_angle,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.z_dir, self.v_dir, self.r_dir, self.contact_point_intrusion_angle],
            device=self.device,
        )
        wp.launch(
            kernel=compute_tilt_angle,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.z_dir, self.n_dir, self.r_dir, self.t_dir, self.contact_point_tilt_angle],
            device=self.device,
        )

        wp.launch(
            kernel=compute_contact_force_2d,
            dim=(self.num_envs, self.num_bodies * self.num_contact_points),
            inputs=[
                self.contact_point_pos.reshape((self.num_envs, -1)),
                self.contact_point_lin_vel.reshape((self.num_envs, -1)),
                self.contact_point_lin_vel_prev.reshape((self.num_envs, -1)),
                self.contact_point_tilt_angle.reshape((self.num_envs, -1)),
                self.contact_point_intrusion_angle.reshape((self.num_envs, -1)),
                self.z_dir.reshape((self.num_envs, -1)),
                self.rho,
                self.lam,
                self.dynamic_friction_coef,
                self.kf,
                self.rho_c,
                self.mu_int,
                wp.float32(self.cfg.A00),
                wp.float32(self.cfg.A10),
                wp.float32(self.cfg.B11),
                wp.float32(self.cfg.B01),
                wp.float32(self.cfg.B_11),
                wp.float32(self.cfg.C11),
                wp.float32(self.cfg.C01),
                wp.float32(self.cfg.C_11),
                wp.float32(self.cfg.D10),
                self.force_gm,
                self.force_ema,
                self.tau_r,
                self.c_r,
                wp.int32(1 if self.enable_ema_filter else 0),
                self.collider.dA,
                self.num_contact_points,
                self.resitive_force,
            ],
            device=self.device,
        )

        # step 5: sum per-contact forces to body wrench and transform to body frame
        wp.launch(
            kernel=zero_wrench,
            dim=(self.num_envs, self.num_bodies),
            inputs=[self.contact_force, self.contact_torque],
            device=self.device,
        )
        wp.launch(
            kernel=compute_contact_wrench,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[
                self.body_pos,
                self.contact_point_pos,
                self.resitive_force.reshape((self.num_envs, self.num_bodies, self.num_contact_points)),
                self.contact_point_force,
                self.contact_point_torque,
                self.contact_force,
                self.contact_torque,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=transform_global_wrench_to_body,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                self.body_quat,
                self.contact_force,
                self.contact_torque,
                self.contact_force_b,
                self.contact_torque_b,
            ],
            device=self.device,
        )

        wp.copy(self.contact_point_lin_vel_prev, self.contact_point_lin_vel)

    """
    Reset.
    """

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # the managers hand out int32 indices, while the kernels below index with int64
        env_ids = env_ids.to(torch.int64)

        wp.launch(
            kernel=reset_2d,
            dim=(len(env_ids), self.num_bodies * self.num_contact_points),
            inputs=[
                wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64),
                self.force_gm,
                self.force_ema,
                self.tau_r,
                self.contact_point_lin_vel_prev.reshape((self.num_envs, -1)),
            ],
            device=self.device,
        )


class SpringDamper:
    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        device: torch.device | str,
        dt: float,
        material_cfg: SpringDamperCfg,
        collider_cfg: ColliderCfg,
        history_length: int = 3,
        history_logging_decimation: int = 1,
        contact_threshold: float = 10.0,
        enable_ema_filter: bool = False,  # unused; kept for API consistency
    ) -> None:
        """
        Spring-damper contact model (warp backend).

        Normal force per contact point: fz = max((k * depth - b * vn) * dA, 0)
        Tangential force: ft = min(mu * fz, kf * vt)  (same Coulomb model as 2D RFT)

        Args:
            num_envs: number of parallel environments
            num_bodies: number of bodies using soft contact model per env
            device: torch device
            dt: simulation time step
            material_cfg: spring-damper material configuration
            collider_cfg: collider geometry configuration
            history_length: length of history for force tracking
            contact_threshold: force magnitude threshold for contact detection (N)
        """
        self.cfg = material_cfg
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.device = device
        self.dt = dt
        self.history_length = history_length
        self.history_logging_decimation = history_logging_decimation
        self._history_step_counter: int = 0
        self.contact_threshold = contact_threshold

        self.collider = Collider(collider_cfg, num_envs, num_bodies, device)
        self.num_contact_points = self.collider.num_contact_points
        self.surface_area = self.collider.surface_area

        self._data: SoftContactData = SoftContactData()

        self.create_buffers()
        self.initialize_data()

        print("-" * 40)
        print("Spring-damper soft contact.")
        print("backend: warp")
        print(f"Number of envs: {self.num_envs}")
        print(f"Number of bodies per env: {self.num_bodies}")
        print(f"Number of contact points per body: {self.num_contact_points}")
        print(f"Contact surface area per body: {self.surface_area:.4f} m^2")
        print(f"k: {self.cfg.k:.2e} N/m^3")
        print(f"b: {self.cfg.b:.2e} N*s/m^3")
        print(f"mu_surf: {self.cfg.dynamic_friction_coef:.2f}")
        print("-" * 40)
        print("\n")

        self.capture()

    """
    CUDA graph capture.
    """

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._eval_contacts()
            self.graph = capture.graph
        else:
            self.graph = None

    def create_buffers(self):
        """Create warp arrays for spring-damper model."""
        # torch input buffers
        self.body_pos_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_quat_torch = torch.zeros((self.num_envs, self.num_bodies, 4), device=self.device)
        self.body_lin_vel_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_ang_vel_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)

        # warp views of body state
        self.body_pos = wp.from_torch(self.body_pos_torch, dtype=wp.vec3f)
        self.body_quat = wp.from_torch(self.body_quat_torch, dtype=wp.quatf)
        self.body_lin_vel = wp.from_torch(self.body_lin_vel_torch, dtype=wp.vec3f)
        self.body_ang_vel = wp.from_torch(self.body_ang_vel_torch, dtype=wp.vec3f)

        # contact geometry from collider
        self.contact_point_local = self.collider.contact_point_local
        self.normal_dir_local = self.collider.normal_dir_local
        self.contact_point_pos = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_lin_vel = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )

        # contact forces
        self.contact_point_force = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_torque = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_force = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_torque = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)

        # resistive force per contact point (output of spring-damper kernel)
        self.resitive_force = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.vec3f, device=self.device
        )

        # per-env material parameters
        self.k = wp.full(self.num_envs, self.cfg.k, dtype=wp.float32, device=self.device)
        self.b = wp.full(self.num_envs, self.cfg.b, dtype=wp.float32, device=self.device)
        self.dynamic_friction_coef = wp.full(
            self.num_envs, self.cfg.dynamic_friction_coef, dtype=wp.float32, device=self.device
        )
        self.kf = wp.full(self.num_envs, self.cfg.kf, dtype=wp.float32, device=self.device)

        # timestamps
        self._timestamp = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._timestamp_last_update = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # torch views into warp buffers
        self.torch_contact_point_pos = wp.to_torch(self.contact_point_pos)
        self.torch_contact_force = wp.to_torch(self.contact_force)
        self.torch_contact_torque = wp.to_torch(self.contact_torque)
        self.torch_contact_force_b = wp.to_torch(self.contact_force_b)
        self.torch_contact_torque_b = wp.to_torch(self.contact_torque_b)
        self.torch_contact_point_force = wp.to_torch(self.contact_point_force)
        self.torch_contact_point_torque = wp.to_torch(self.contact_point_torque)
        self.torch_dynamic_friction_coef = wp.to_torch(self.dynamic_friction_coef)
        self.torch_k = wp.to_torch(self.k)
        self.torch_b = wp.to_torch(self.b)

    def initialize_data(self) -> None:
        """Initialize soft contact data."""
        self._data.net_forces_w = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self._data.net_forces_w_history = torch.zeros(
            (self.num_envs, self.history_length, self.num_bodies, 3), device=self.device
        )
        self._data.force_matrix_w = torch.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device
        )
        self._data.force_matrix_w_history = torch.zeros(
            (self.num_envs, self.history_length, self.num_bodies, self.num_contact_points, 3), device=self.device
        )
        self._data.last_air_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.current_air_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.last_contact_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.current_contact_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.is_sensor_active = torch.zeros(
            (self.num_envs, self.num_bodies), dtype=torch.bool, device=self.device
        )

    """
    Properties.
    """

    @property
    def data(self) -> SoftContactData:
        return self._data

    @property
    def contact_wrench(self) -> torch.Tensor:
        return torch.cat(
            (self.torch_contact_force, self.torch_contact_torque), dim=-1
        )

    @property
    def contact_wrench_b(self) -> torch.Tensor:
        return torch.cat(
            (self.torch_contact_force_b, self.torch_contact_torque_b), dim=-1
        )

    @property
    def contact_point_wrench(self) -> torch.Tensor:
        return torch.cat(
            (self.torch_contact_point_force, self.torch_contact_point_torque), dim=-1
        )

    @property
    def terrain_friction(self) -> torch.Tensor:
        return self.torch_dynamic_friction_coef

    @property
    def terrain_density(self) -> torch.Tensor:
        return self.torch_b  # damping plays the role of density in this model

    @property
    def terrain_stiffness(self) -> torch.Tensor:
        return self.torch_k

    """
    Operations.
    """

    def update(
        self,
        body_pos: torch.Tensor,
        body_quat: torch.Tensor,
        body_lin_vel: torch.Tensor,
        body_ang_vel: torch.Tensor,
    ):
        self.body_pos_torch[:] = body_pos
        self.body_quat_torch[:] = body_quat[:, :, [1, 2, 3, 0]]  # (w,x,y,z) → (x,y,z,w)
        self.body_lin_vel_torch[:] = body_lin_vel
        self.body_ang_vel_torch[:] = body_ang_vel

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._eval_contacts()

        self._timestamp += self.dt
        self._update_data(torch.arange(self.num_envs, device=self.device))
        self._timestamp_last_update[:] = self._timestamp[:]

    def randomize_ground_stiffness(self, env_ids: torch.Tensor, k: torch.Tensor) -> None:
        """Update per-env spring stiffness (N/m^3)."""
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64), wp.from_torch(k, dtype=wp.float32), self.k],
            device=self.device,
        )

    def update_material_density(
        self, env_ids: torch.Tensor, packing_density: torch.Tensor, bulk_density: torch.Tensor
    ) -> None:
        """Update per-env damping coefficient (b = bulk_density * packing_density, N*s/m^3)."""
        b_new = bulk_density * packing_density
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64), wp.from_torch(b_new, dtype=wp.float32), self.b],
            device=self.device,
        )

    def update_friction_params(
        self,
        env_ids: torch.Tensor,
        static_friction_coef: torch.Tensor,
        dynamic_friction_coef: torch.Tensor,
    ) -> None:
        """Update per-env friction coefficients."""
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[
                wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64),
                wp.from_torch(dynamic_friction_coef, dtype=wp.float32),
                self.dynamic_friction_coef,
            ],
            device=self.device,
        )

    """
    Data helper functions.
    """

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        currently_in_contact = self._data.current_contact_time > 0.0
        less_than_dt_in_contact = self._data.current_contact_time < (dt + abs_tol)
        return currently_in_contact * less_than_dt_in_contact

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        currently_detached = self._data.current_air_time > 0.0
        less_than_dt_detached = self._data.current_air_time < (dt + abs_tol)
        return currently_detached * less_than_dt_detached

    def _update_data(self, env_ids: torch.Tensor) -> None:
        self._data.net_forces_w[env_ids, :, :] = self.torch_contact_force[env_ids, :, :]  # type: ignore
        self._data.force_matrix_w[env_ids, :, :, :] = self.torch_contact_point_force[env_ids, :, :, :]  # type: ignore
        if self.history_length > 0:
            self._history_step_counter += 1
            if self._history_step_counter >= self.history_logging_decimation:
                self._history_step_counter = 0
                self._data.net_forces_w_history[env_ids] = self._data.net_forces_w_history[env_ids].roll(  # type: ignore
                    shifts=1, dims=1
                )
                self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]  # type: ignore
                self._data.force_matrix_w_history[env_ids] = self._data.force_matrix_w_history[env_ids].roll(  # type: ignore
                    shifts=1, dims=1
                )
                self._data.force_matrix_w_history[env_ids, 0] = self._data.force_matrix_w[env_ids]  # type: ignore

        elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
        is_contact = torch.norm(self._data.net_forces_w[env_ids, :, :], dim=-1) > self.contact_threshold  # type: ignore
        is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact  # type: ignore
        is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact  # type: ignore
        self._data.last_air_time[env_ids] = torch.where(  # type: ignore
            is_first_contact,
            self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),  # type: ignore
            self._data.last_air_time[env_ids],  # type: ignore
        )
        self._data.current_air_time[env_ids] = torch.where(  # type: ignore
            ~is_contact,
            self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),
            0.0,  # type: ignore
        )
        self._data.last_contact_time[env_ids] = torch.where(  # type: ignore
            is_first_detached,
            self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),  # type: ignore
            self._data.last_contact_time[env_ids],  # type: ignore
        )
        self._data.current_contact_time[env_ids] = torch.where(  # type: ignore
            is_contact,
            self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),
            0.0,  # type: ignore
        )

    """
    Helper functions.
    """

    def _eval_contacts(self) -> None:
        """Compute contact kinematics and spring-damper forces."""

        # step 1: contact point positions and velocities in global frame
        wp.launch(
            kernel=compute_contact_point_pos_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.body_pos, self.body_quat, self.contact_point_local, self.contact_point_pos],
            device=self.device,
        )
        wp.launch(
            kernel=compute_contact_point_lin_vel_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[
                self.body_pos,
                self.body_lin_vel,
                self.body_ang_vel,
                self.contact_point_pos,
                self.contact_point_lin_vel,
            ],
            device=self.device,
        )

        # step 2: spring-damper force per contact point
        wp.launch(
            kernel=compute_spring_damper_force,
            dim=(self.num_envs, self.num_bodies * self.num_contact_points),
            inputs=[
                self.contact_point_pos.reshape((self.num_envs, -1)),
                self.contact_point_lin_vel.reshape((self.num_envs, -1)),
                self.k,
                self.b,
                self.dynamic_friction_coef,
                self.kf,
                self.collider.dA,
                self.num_contact_points,
                self.resitive_force,
            ],
            device=self.device,
        )

        # step 3: sum per-contact forces to body wrench and transform to body frame
        wp.launch(
            kernel=zero_wrench,
            dim=(self.num_envs, self.num_bodies),
            inputs=[self.contact_force, self.contact_torque],
            device=self.device,
        )
        wp.launch(
            kernel=compute_contact_wrench,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[
                self.body_pos,
                self.contact_point_pos,
                self.resitive_force.reshape((self.num_envs, self.num_bodies, self.num_contact_points)),
                self.contact_point_force,
                self.contact_point_torque,
                self.contact_force,
                self.contact_torque,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=transform_global_wrench_to_body,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                self.body_quat,
                self.contact_force,
                self.contact_torque,
                self.contact_force_b,
                self.contact_torque_b,
            ],
            device=self.device,
        )

    """
    Reset.
    """

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass  # no stateful buffers (no EMA, no prev velocity)

class ConeDRFT:
    """
    ConeDRFT (granular jammed-cone) soft contact model (warp backend).

    Implements Choi et al., "Learning quadrupedal locomotion on deformable terrain",
    Sci. Robotics 2023, supplementary sections S11-S16. The foot is modelled as a single
    point intruder: one contact point at the bottom-center of the foot. The vertical
    granular reaction is computed in closed form from the cone integrals as a function of
    penetration depth z, rate z_dot and acceleration z_ddot (see compute_cone_drft_force).

    The hydraulic radius r_h is derived once from the collider footprint:
      box / plane:  r_h = lx * ly / (lx + ly)
      sphere:       r_h = radius
    and is held constant during intrusion (exact for a box with vertical walls).
    """

    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        device: torch.device | str,
        dt: float,
        material_cfg: ConeDRFTCfg,
        collider_cfg: ColliderCfg,
        history_length: int = 3,
        history_logging_decimation: int = 1,
        contact_threshold: float = 10.0,
        enable_ema_filter: bool | None = None,
    ) -> None:
        self.cfg = material_cfg
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.device = device
        self.dt = dt
        self.history_length = history_length
        self.history_logging_decimation = history_logging_decimation
        self._history_step_counter: int = 0
        self.contact_threshold = contact_threshold

        # Build collider and get single bottom-center contact point
        self.collider = Collider(collider_cfg, num_envs, num_bodies, device)
        self._bottom_center_local, self.r_h = self.collider.get_cone_single_point()
        self.num_contact_points = 1

        # EMA / numerical flags (cfg drives them; explicit arg overrides for API parity)
        self.enable_ema = bool(self.cfg.enable_ema_filter if enable_ema_filter is None else enable_ema_filter)
        self.enable_added_mass = bool(self.cfg.enable_added_mass)
        # transition-coefficient step c_r = 100 / f = 100 * dt  (Eq. S16)
        self.c_r = 100.0 * self.dt

        self._data: SoftContactData = SoftContactData()

        self.create_buffers()
        self.initialize_data()

        print("-" * 40)
        print("ConeDRFT (granular jammed-cone) soft contact.")
        print("backend: warp")
        print(f"Number of envs: {self.num_envs}")
        print(f"Number of bodies per env: {self.num_bodies}")
        print("Single point contact (bottom-center of foot).")
        print(f"Hydraulic radius r_h: {self.r_h * 100.0:.3f} cm")
        print(f"Bottom-center (local): {tuple(round(v, 4) for v in self._bottom_center_local)}")
        print(f"sigma_flat: {self.cfg.sigma_flat:.2e} N/m^3")
        print(f"sigma_cone: {self.cfg.sigma_cone:.2e} N/m^3")
        print(f"theta: {self.cfg.theta:.3f} rad, nu: {self.cfg.nu:.3f}")
        print(f"mu_surf: {self.cfg.dynamic_friction_coef:.2f}")
        print(f"EMA filter: {self.enable_ema}, added mass: {self.enable_added_mass}")
        print("-" * 40)
        print("\n")

        self.capture()

    """
    CUDA graph capture.
    """

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._eval_contacts()
            self.graph = capture.graph
        else:
            self.graph = None

    def create_buffers(self):
        """Create warp arrays for the ConeDRFT model."""
        # torch input buffers
        self.body_pos_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_quat_torch = torch.zeros((self.num_envs, self.num_bodies, 4), device=self.device)
        self.body_lin_vel_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_ang_vel_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)

        # warp views of body state
        self.body_pos = wp.from_torch(self.body_pos_torch, dtype=wp.vec3f)
        self.body_quat = wp.from_torch(self.body_quat_torch, dtype=wp.quatf)
        self.body_lin_vel = wp.from_torch(self.body_lin_vel_torch, dtype=wp.vec3f)
        self.body_ang_vel = wp.from_torch(self.body_ang_vel_torch, dtype=wp.vec3f)

        # single contact point (bottom-center) in body frame, shape (N, B, 1) vec3f
        local = torch.tensor(self._bottom_center_local, device=self.device, dtype=torch.float32)
        self.contact_point_local_torch = (
            local.view(1, 1, 1, 3).expand(self.num_envs, self.num_bodies, 1, 3).contiguous()
        )
        self.contact_point_local = wp.from_torch(self.contact_point_local_torch, dtype=wp.vec3f)

        self.contact_point_pos = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_lin_vel = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        # previous-step contact-point velocity (accel + EMA velocity-sign test), shape (N, B)
        self.contact_point_lin_vel_prev = wp.zeros(
            (self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device
        )

        # contact forces
        self.contact_point_force = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_torque = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_force = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_torque = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)

        # resistive force per contact point (output of cone kernel), shape (N, B*1)
        self.resitive_force = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.vec3f, device=self.device
        )

        # per-env terrain stiffness (randomizable)
        self.sigma_flat = wp.full(self.num_envs, self.cfg.sigma_flat, dtype=wp.float32, device=self.device)
        self.sigma_cone = wp.full(self.num_envs, self.cfg.sigma_cone, dtype=wp.float32, device=self.device)
        self.dynamic_friction_coef = wp.full(
            self.num_envs, self.cfg.dynamic_friction_coef, dtype=wp.float32, device=self.device
        )
        self.kf = wp.full(self.num_envs, self.cfg.kf, dtype=wp.float32, device=self.device)

        # per-env material properties (randomizable, can be updated via update_material_density)
        self.rho = wp.full(self.num_envs, self.cfg.rho, dtype=wp.float32, device=self.device)
        self.phi = wp.full(self.num_envs, self.cfg.phi, dtype=wp.float32, device=self.device)

        # stateful buffers (plastic deformation + anti-drift EMA), shape (N, B)
        self.z_max = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.float32, device=self.device)
        self.force_gm = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.float32, device=self.device)
        self.force_ema = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.float32, device=self.device)
        self.tau_r = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.float32, device=self.device)

        # timestamps
        self._timestamp = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._timestamp_last_update = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # torch views into warp buffers
        self.torch_contact_point_pos = wp.to_torch(self.contact_point_pos)
        self.torch_contact_force = wp.to_torch(self.contact_force)
        self.torch_contact_torque = wp.to_torch(self.contact_torque)
        self.torch_contact_force_b = wp.to_torch(self.contact_force_b)
        self.torch_contact_torque_b = wp.to_torch(self.contact_torque_b)
        self.torch_contact_point_force = wp.to_torch(self.contact_point_force)
        self.torch_contact_point_torque = wp.to_torch(self.contact_point_torque)
        self.torch_dynamic_friction_coef = wp.to_torch(self.dynamic_friction_coef)
        self.torch_sigma_flat = wp.to_torch(self.sigma_flat)
        self.torch_sigma_cone = wp.to_torch(self.sigma_cone)
        self.torch_rho = wp.to_torch(self.rho)
        self.torch_phi = wp.to_torch(self.phi)

    def initialize_data(self) -> None:
        """Initialize soft contact data."""
        self._data.net_forces_w = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self._data.net_forces_w_history = torch.zeros(
            (self.num_envs, self.history_length, self.num_bodies, 3), device=self.device
        )
        self._data.force_matrix_w = torch.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device
        )
        self._data.force_matrix_w_history = torch.zeros(
            (self.num_envs, self.history_length, self.num_bodies, self.num_contact_points, 3), device=self.device
        )
        self._data.last_air_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.current_air_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.last_contact_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.current_contact_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.is_sensor_active = torch.zeros(
            (self.num_envs, self.num_bodies), dtype=torch.bool, device=self.device
        )

    """
    Properties.
    """

    @property
    def data(self) -> SoftContactData:
        return self._data

    @property
    def contact_wrench(self) -> torch.Tensor:
        return torch.cat((self.torch_contact_force, self.torch_contact_torque), dim=-1)

    @property
    def contact_wrench_b(self) -> torch.Tensor:
        return torch.cat((self.torch_contact_force_b, self.torch_contact_torque_b), dim=-1)

    @property
    def contact_point_wrench(self) -> torch.Tensor:
        return torch.cat((self.torch_contact_point_force, self.torch_contact_point_torque), dim=-1)

    @property
    def terrain_friction(self) -> torch.Tensor:
        return self.torch_dynamic_friction_coef

    @property
    def terrain_density(self) -> torch.Tensor:
        # rho is a scalar material constant in this model; expose sigma_cone as the secondary
        # depth-dependent stiffness term for logging/randomization parity.
        # return self.torch_sigma_cone
        return self.torch_rho * self.torch_phi

    @property
    def terrain_stiffness(self) -> torch.Tensor:
        return self.torch_sigma_flat * self.torch_sigma_cone

    @property
    def terrain_sigma_flat(self) -> torch.Tensor:
        return self.torch_sigma_flat

    @property
    def terrain_sigma_cone(self) -> torch.Tensor:
        return self.torch_sigma_cone

    """
    Operations.
    """

    def update(
        self,
        body_pos: torch.Tensor,
        body_quat: torch.Tensor,
        body_lin_vel: torch.Tensor,
        body_ang_vel: torch.Tensor,
    ):
        self.body_pos_torch[:] = body_pos
        self.body_quat_torch[:] = body_quat[:, :, [1, 2, 3, 0]]  # (w,x,y,z) -> (x,y,z,w)
        self.body_lin_vel_torch[:] = body_lin_vel
        self.body_ang_vel_torch[:] = body_ang_vel

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._eval_contacts()

        self._timestamp += self.dt
        self._update_data(torch.arange(self.num_envs, device=self.device))
        self._timestamp_last_update[:] = self._timestamp[:]

    def randomize_ground_stiffness(
        self, env_ids: torch.Tensor, sigma_flat: torch.Tensor, sigma_cone: torch.Tensor | None = None
    ) -> None:
        """Update per-env depth-dependent stiffness stresses (N/m^3)."""
        env_ids_wp = wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64)
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[env_ids_wp, wp.from_torch(sigma_flat, dtype=wp.float32), self.sigma_flat],
            device=self.device,
        )
        if sigma_cone is not None:
            wp.launch(
                kernel=update_array_with_index,
                dim=len(env_ids),
                inputs=[env_ids_wp, wp.from_torch(sigma_cone, dtype=wp.float32), self.sigma_cone],
                device=self.device,
            )

    def update_material_density(
        self, env_ids: torch.Tensor, packing_density: torch.Tensor, bulk_density: torch.Tensor
    ) -> None:
        """Update per-env material density parameters (phi = packing_density, rho = bulk_density)."""
        # phi (packing fraction) = packing_density
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[
                wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64),
                wp.from_torch(packing_density, dtype=wp.float32),
                self.phi,
            ],
            device=self.device,
        )
        # rho (grain density) = bulk_density
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[
                wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64),
                wp.from_torch(bulk_density, dtype=wp.float32),
                self.rho,
            ],
            device=self.device,
        )

    def update_friction_params(
        self,
        env_ids: torch.Tensor,
        static_friction_coef: torch.Tensor,
        dynamic_friction_coef: torch.Tensor,
    ) -> None:
        """Update per-env friction coefficients."""
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[
                wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64),
                wp.from_torch(dynamic_friction_coef, dtype=wp.float32),
                self.dynamic_friction_coef,
            ],
            device=self.device,
        )

    """
    Data helper functions.
    """

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        currently_in_contact = self._data.current_contact_time > 0.0
        less_than_dt_in_contact = self._data.current_contact_time < (dt + abs_tol)
        return currently_in_contact * less_than_dt_in_contact

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        currently_detached = self._data.current_air_time > 0.0
        less_than_dt_detached = self._data.current_air_time < (dt + abs_tol)
        return currently_detached * less_than_dt_detached

    def _update_data(self, env_ids: torch.Tensor) -> None:
        self._data.net_forces_w[env_ids, :, :] = self.torch_contact_force[env_ids, :, :]  # type: ignore
        self._data.force_matrix_w[env_ids, :, :, :] = self.torch_contact_point_force[env_ids, :, :, :]  # type: ignore
        if self.history_length > 0:
            self._history_step_counter += 1
            if self._history_step_counter >= self.history_logging_decimation:
                self._history_step_counter = 0
                self._data.net_forces_w_history[env_ids] = self._data.net_forces_w_history[env_ids].roll(  # type: ignore
                    shifts=1, dims=1
                )
                self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]  # type: ignore
                self._data.force_matrix_w_history[env_ids] = self._data.force_matrix_w_history[env_ids].roll(  # type: ignore
                    shifts=1, dims=1
                )
                self._data.force_matrix_w_history[env_ids, 0] = self._data.force_matrix_w[env_ids]  # type: ignore

        elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
        is_contact = torch.norm(self._data.net_forces_w[env_ids, :, :], dim=-1) > self.contact_threshold  # type: ignore
        is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact  # type: ignore
        is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact  # type: ignore
        self._data.last_air_time[env_ids] = torch.where(  # type: ignore
            is_first_contact,
            self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),  # type: ignore
            self._data.last_air_time[env_ids],  # type: ignore
        )
        self._data.current_air_time[env_ids] = torch.where(  # type: ignore
            ~is_contact,
            self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),
            0.0,  # type: ignore
        )
        self._data.last_contact_time[env_ids] = torch.where(  # type: ignore
            is_first_detached,
            self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),  # type: ignore
            self._data.last_contact_time[env_ids],  # type: ignore
        )
        self._data.current_contact_time[env_ids] = torch.where(  # type: ignore
            is_contact,
            self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),
            0.0,  # type: ignore
        )

    """
    Helper functions.
    """

    def _eval_contacts(self) -> None:
        """Compute contact kinematics and ConeDRFT forces."""

        # step 1: contact point position and velocity (single bottom-center point) in world frame
        wp.launch(
            kernel=compute_contact_point_pos_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.body_pos, self.body_quat, self.contact_point_local, self.contact_point_pos],
            device=self.device,
        )
        wp.launch(
            kernel=compute_contact_point_lin_vel_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[
                self.body_pos,
                self.body_lin_vel,
                self.body_ang_vel,
                self.contact_point_pos,
                self.contact_point_lin_vel,
            ],
            device=self.device,
        )

        # step 2: ConeDRFT force at the single contact point
        wp.launch(
            kernel=compute_cone_drft_force,
            dim=(self.num_envs, self.num_bodies * self.num_contact_points),
            inputs=[
                self.contact_point_pos.reshape((self.num_envs, -1)),
                self.contact_point_lin_vel.reshape((self.num_envs, -1)),
                self.contact_point_lin_vel_prev,
                self.sigma_flat,
                self.sigma_cone,
                self.dynamic_friction_coef,
                self.kf,
                wp.float32(self.r_h),
                wp.float32(self.cfg.nu),
                wp.float32(self.cfg.theta),
                wp.float32(self.cfg.c_g),
                wp.float32(self.cfg.c_d),
                self.phi,
                self.rho,
                wp.int32(1 if self.enable_added_mass else 0),
                wp.float32(self.dt),
                self.z_max,
                wp.float32(self.cfg.eps_f),
                self.force_gm,
                self.force_ema,
                self.tau_r,
                wp.float32(self.c_r),
                wp.int32(1 if self.enable_ema else 0),
                self.resitive_force,
            ],
            device=self.device,
        )

        # step 3: store current contact-point velocity for next step's accel / EMA test
        wp.launch(
            kernel=update_prev_velocity,
            dim=(self.num_envs, self.num_bodies),
            inputs=[self.contact_point_lin_vel.reshape((self.num_envs, -1)), self.contact_point_lin_vel_prev],
            device=self.device,
        )

        # step 4: sum per-contact forces to body wrench and transform to body frame
        wp.launch(
            kernel=zero_wrench,
            dim=(self.num_envs, self.num_bodies),
            inputs=[self.contact_force, self.contact_torque],
            device=self.device,
        )
        wp.launch(
            kernel=compute_contact_wrench,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[
                self.body_pos,
                self.contact_point_pos,
                self.resitive_force.reshape((self.num_envs, self.num_bodies, self.num_contact_points)),
                self.contact_point_force,
                self.contact_point_torque,
                self.contact_force,
                self.contact_torque,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=transform_global_wrench_to_body,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                self.body_quat,
                self.contact_force,
                self.contact_torque,
                self.contact_force_b,
                self.contact_torque_b,
            ],
            device=self.device,
        )

    """
    Reset.
    """

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # the managers hand out int32 indices, while the kernels below index with int64
        env_ids = env_ids.to(torch.int64)
        wp.launch(
            kernel=reset_cone,
            dim=(len(env_ids), self.num_bodies),
            inputs=[
                wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64),
                self.z_max,
                self.force_gm,
                self.force_ema,
                self.tau_r,
                self.contact_point_lin_vel_prev,
            ],
            device=self.device,
        )

class ConeDRFTMultiPoint:
    """
    ConeDRFT variant with multiple contact points distributed across the footprint.

    Unlike ConeDRFT which uses a single bottom-center contact point, this variant splits
    the geometry into a grid of contact points and applies ConeDRFT independently to each.
    The hydraulic radius for each sub-contact point is scaled to preserve the per-point
    force contribution. Total force is the sum of forces from all contact points.

    This approach better captures distributed pressure and heterogeneous intrusion,
    especially for larger footprints or when the terrain exhibits significant property variations.
    """

    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        device: torch.device | str,
        dt: float,
        material_cfg: ConeDRFTCfg,
        collider_cfg: ColliderCfg,
        history_length: int = 3,
        history_logging_decimation: int = 1,
        contact_threshold: float = 10.0,
        enable_ema_filter: bool | None = None,
    ) -> None:
        self.cfg = material_cfg
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.device = device
        self.dt = dt
        self.history_length = history_length
        self.history_logging_decimation = history_logging_decimation
        self._history_step_counter: int = 0
        self.contact_threshold = contact_threshold

        # Build collider and reuse its contact points for ConeDRFT grid
        self.collider = Collider(collider_cfg, num_envs, num_bodies, device)
        self.contact_points_local, self.r_h_per_point = (
            self.collider.get_cone_grid_points()
        )
        self.num_contact_points = len(self.contact_points_local)

        self.enable_ema = bool(self.cfg.enable_ema_filter if enable_ema_filter is None else enable_ema_filter)
        self.enable_added_mass = bool(self.cfg.enable_added_mass)
        self.c_r = 100.0 * self.dt

        self._data: SoftContactData = SoftContactData()

        self.create_buffers()
        self.initialize_data()

        print("-" * 40)
        print("ConeDRFT-MultiPoint (granular jammed-cone) soft contact.")
        print("backend: warp")
        print(f"Number of envs: {self.num_envs}")
        print(f"Number of bodies per env: {self.num_bodies}")
        print(f"Number of contact points per body: {self.num_contact_points}")
        print(f"Hydraulic radius per point: {self.r_h_per_point:.4f} m")
        print(f"sigma_flat: {self.cfg.sigma_flat:.2e} N/m^3")
        print(f"sigma_cone: {self.cfg.sigma_cone:.2e} N/m^3")
        print(f"theta: {self.cfg.theta:.3f} rad, nu: {self.cfg.nu:.3f}")
        print(f"mu_surf: {self.cfg.dynamic_friction_coef:.2f}")
        print(f"EMA filter: {self.enable_ema}, added mass: {self.enable_added_mass}")
        print("-" * 40)
        print("\n")

        self.capture()

    """
    CUDA graph capture.
    """

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._eval_contacts()
            self.graph = capture.graph
        else:
            self.graph = None

    def create_buffers(self):
        """Create warp arrays for the ConeDRFT-MultiPoint model."""
        # torch input buffers
        self.body_pos_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_quat_torch = torch.zeros((self.num_envs, self.num_bodies, 4), device=self.device)
        self.body_lin_vel_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_ang_vel_torch = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)

        # warp views of body state
        self.body_pos = wp.from_torch(self.body_pos_torch, dtype=wp.vec3f)
        self.body_quat = wp.from_torch(self.body_quat_torch, dtype=wp.quatf)
        self.body_lin_vel = wp.from_torch(self.body_lin_vel_torch, dtype=wp.vec3f)
        self.body_ang_vel = wp.from_torch(self.body_ang_vel_torch, dtype=wp.vec3f)

        # multiple contact points in body frame, shape (N, B, num_contact_points) vec3f
        # TODO: refactor as this is very messy
        local_tensor = torch.tensor(self.contact_points_local, device=self.device, dtype=torch.float32)
        self.contact_point_local_torch = (
            local_tensor.view(1, 1, -1, 3).expand(self.num_envs, self.num_bodies, -1, 3).contiguous()
        )
        self.contact_point_local = wp.from_torch(self.contact_point_local_torch, dtype=wp.vec3f)

        self.contact_point_pos = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_lin_vel = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        # previous-step contact-point velocity per contact point, shape (N, B, num_contact_points)
        self.contact_point_lin_vel_prev = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )

        # contact forces
        self.contact_point_force = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_point_torque = wp.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.vec3f, device=self.device
        )
        self.contact_force = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_torque = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_force_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)
        self.contact_torque_b = wp.zeros((self.num_envs, self.num_bodies), dtype=wp.vec3f, device=self.device)

        # resistive force per contact point (output of cone kernel), shape (N, B*num_contact_points)
        self.resitive_force = wp.zeros(
            (self.num_envs, self.num_bodies * self.num_contact_points), dtype=wp.vec3f, device=self.device
        )

        # per-env terrain stiffness (randomizable)
        self.sigma_flat = wp.full(self.num_envs, self.cfg.sigma_flat, dtype=wp.float32, device=self.device)
        self.sigma_cone = wp.full(self.num_envs, self.cfg.sigma_cone, dtype=wp.float32, device=self.device)
        self.dynamic_friction_coef = wp.full(
            self.num_envs, self.cfg.dynamic_friction_coef, dtype=wp.float32, device=self.device
        )
        self.kf = wp.full(self.num_envs, self.cfg.kf, dtype=wp.float32, device=self.device)

        # per-env material properties (randomizable, can be updated via update_material_density)
        self.rho = wp.full(self.num_envs, self.cfg.rho, dtype=wp.float32, device=self.device)
        self.phi = wp.full(self.num_envs, self.cfg.phi, dtype=wp.float32, device=self.device)

        # stateful buffers per contact point, shape (N, B, num_contact_points)
        self.z_max = wp.zeros((self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.float32, device=self.device)
        self.force_gm = wp.zeros((self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.float32, device=self.device)
        self.force_ema = wp.zeros((self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.float32, device=self.device)
        self.tau_r = wp.zeros((self.num_envs, self.num_bodies, self.num_contact_points), dtype=wp.float32, device=self.device)

        # timestamps
        self._timestamp = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._timestamp_last_update = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # torch views into warp buffers
        self.torch_contact_point_pos = wp.to_torch(self.contact_point_pos)
        self.torch_contact_force = wp.to_torch(self.contact_force)
        self.torch_contact_torque = wp.to_torch(self.contact_torque)
        self.torch_contact_force_b = wp.to_torch(self.contact_force_b)
        self.torch_contact_torque_b = wp.to_torch(self.contact_torque_b)
        self.torch_contact_point_force = wp.to_torch(self.contact_point_force)
        self.torch_contact_point_torque = wp.to_torch(self.contact_point_torque)
        self.torch_dynamic_friction_coef = wp.to_torch(self.dynamic_friction_coef)
        self.torch_sigma_flat = wp.to_torch(self.sigma_flat)
        self.torch_sigma_cone = wp.to_torch(self.sigma_cone)
        self.torch_rho = wp.to_torch(self.rho)
        self.torch_phi = wp.to_torch(self.phi)

    def initialize_data(self) -> None:
        """Initialize soft contact data."""
        self._data.net_forces_w = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self._data.net_forces_w_history = torch.zeros(
            (self.num_envs, self.history_length, self.num_bodies, 3), device=self.device
        )
        self._data.force_matrix_w = torch.zeros(
            (self.num_envs, self.num_bodies, self.num_contact_points, 3), device=self.device
        )
        self._data.force_matrix_w_history = torch.zeros(
            (self.num_envs, self.history_length, self.num_bodies, self.num_contact_points, 3), device=self.device
        )
        self._data.last_air_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.current_air_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.last_contact_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.current_contact_time = torch.zeros((self.num_envs, self.num_bodies), device=self.device)
        self._data.is_sensor_active = torch.zeros(
            (self.num_envs, self.num_bodies), dtype=torch.bool, device=self.device
        )

    """
    Properties.
    """

    @property
    def data(self) -> SoftContactData:
        return self._data

    @property
    def contact_wrench(self) -> torch.Tensor:
        return torch.cat((self.torch_contact_force, self.torch_contact_torque), dim=-1)

    @property
    def contact_wrench_b(self) -> torch.Tensor:
        return torch.cat((self.torch_contact_force_b, self.torch_contact_torque_b), dim=-1)

    @property
    def contact_point_wrench(self) -> torch.Tensor:
        return torch.cat((self.torch_contact_point_force, self.torch_contact_point_torque), dim=-1)

    @property
    def terrain_friction(self) -> torch.Tensor:
        return self.torch_dynamic_friction_coef

    @property
    def terrain_density(self) -> torch.Tensor:
        # rho is a scalar material constant in this model; expose sigma_cone as the secondary
        # depth-dependent stiffness term for logging/randomization parity.
        return self.torch_rho * self.torch_phi

    @property
    def terrain_stiffness(self) -> torch.Tensor:
        return self.torch_sigma_flat * self.torch_sigma_cone

    @property
    def terrain_sigma_flat(self) -> torch.Tensor:
        return self.torch_sigma_flat

    @property
    def terrain_sigma_cone(self) -> torch.Tensor:
        return self.torch_sigma_cone

    """
    Operations.
    """

    def update(
        self,
        body_pos: torch.Tensor,
        body_quat: torch.Tensor,
        body_lin_vel: torch.Tensor,
        body_ang_vel: torch.Tensor,
    ):
        self.body_pos_torch[:] = body_pos
        self.body_quat_torch[:] = body_quat[:, :, [1, 2, 3, 0]]  # (w,x,y,z) -> (x,y,z,w)
        self.body_lin_vel_torch[:] = body_lin_vel
        self.body_ang_vel_torch[:] = body_ang_vel

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._eval_contacts()

        self._timestamp += self.dt
        self._update_data(torch.arange(self.num_envs, device=self.device))
        self._timestamp_last_update[:] = self._timestamp[:]

    def randomize_ground_stiffness(
        self, env_ids: torch.Tensor, sigma_flat: torch.Tensor, sigma_cone: torch.Tensor | None = None
    ) -> None:
        """Update per-env depth-dependent stiffness stresses (N/m^3)."""
        env_ids_wp = wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64)
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[env_ids_wp, wp.from_torch(sigma_flat, dtype=wp.float32), self.sigma_flat],
            device=self.device,
        )
        if sigma_cone is not None:
            wp.launch(
                kernel=update_array_with_index,
                dim=len(env_ids),
                inputs=[env_ids_wp, wp.from_torch(sigma_cone, dtype=wp.float32), self.sigma_cone],
                device=self.device,
            )

    def update_material_density(
        self, env_ids: torch.Tensor, packing_density: torch.Tensor, bulk_density: torch.Tensor
    ) -> None:
        """Update per-env material density parameters (phi = packing_density, rho = bulk_density)."""
        # phi (packing fraction) = packing_density
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[
                wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64),
                wp.from_torch(packing_density, dtype=wp.float32),
                self.phi,
            ],
            device=self.device,
        )
        # rho (grain density) = bulk_density
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[
                wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64),
                wp.from_torch(bulk_density, dtype=wp.float32),
                self.rho,
            ],
            device=self.device,
        )

    def update_friction_params(
        self,
        env_ids: torch.Tensor,
        static_friction_coef: torch.Tensor,
        dynamic_friction_coef: torch.Tensor,
    ) -> None:
        """Update per-env friction coefficients."""
        wp.launch(
            kernel=update_array_with_index,
            dim=len(env_ids),
            inputs=[
                wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64),
                wp.from_torch(dynamic_friction_coef, dtype=wp.float32),
                self.dynamic_friction_coef,
            ],
            device=self.device,
        )

    """
    Data helper functions.
    """

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        currently_in_contact = self._data.current_contact_time > 0.0
        less_than_dt_in_contact = self._data.current_contact_time < (dt + abs_tol)
        return currently_in_contact * less_than_dt_in_contact

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
        currently_detached = self._data.current_air_time > 0.0
        less_than_dt_detached = self._data.current_air_time < (dt + abs_tol)
        return currently_detached * less_than_dt_detached

    def _update_data(self, env_ids: torch.Tensor) -> None:
        self._data.net_forces_w[env_ids, :, :] = self.torch_contact_force[env_ids, :, :]  # type: ignore
        self._data.force_matrix_w[env_ids, :, :, :] = self.torch_contact_point_force[env_ids, :, :, :]  # type: ignore
        if self.history_length > 0:
            self._history_step_counter += 1
            if self._history_step_counter >= self.history_logging_decimation:
                self._history_step_counter = 0
                self._data.net_forces_w_history[env_ids] = self._data.net_forces_w_history[env_ids].roll(  # type: ignore
                    shifts=1, dims=1
                )
                self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]  # type: ignore
                self._data.force_matrix_w_history[env_ids] = self._data.force_matrix_w_history[env_ids].roll(  # type: ignore
                    shifts=1, dims=1
                )
                self._data.force_matrix_w_history[env_ids, 0] = self._data.force_matrix_w[env_ids]  # type: ignore

        elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
        is_contact = torch.norm(self._data.net_forces_w[env_ids, :, :], dim=-1) > self.contact_threshold  # type: ignore
        is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact  # type: ignore
        is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact  # type: ignore
        self._data.last_air_time[env_ids] = torch.where(  # type: ignore
            is_first_contact,
            self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),  # type: ignore
            self._data.last_air_time[env_ids],  # type: ignore
        )
        self._data.current_air_time[env_ids] = torch.where(  # type: ignore
            ~is_contact,
            self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),
            0.0,  # type: ignore
        )
        self._data.last_contact_time[env_ids] = torch.where(  # type: ignore
            is_first_detached,
            self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),  # type: ignore
            self._data.last_contact_time[env_ids],  # type: ignore
        )
        self._data.current_contact_time[env_ids] = torch.where(  # type: ignore
            is_contact,
            self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),
            0.0,  # type: ignore
        )

    """
    Helper functions.
    """

    def _eval_contacts(self) -> None:
        """Compute contact kinematics and ConeDRFT forces for multiple contact points."""

        # step 1: contact point positions and velocities (multiple points) in world frame
        wp.launch(
            kernel=compute_contact_point_pos_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[self.body_pos, self.body_quat, self.contact_point_local, self.contact_point_pos],
            device=self.device,
        )
        wp.launch(
            kernel=compute_contact_point_lin_vel_w,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[
                self.body_pos,
                self.body_lin_vel,
                self.body_ang_vel,
                self.contact_point_pos,
                self.contact_point_lin_vel,
            ],
            device=self.device,
        )

        # step 2: ConeDRFT force at each contact point
        wp.launch(
            kernel=compute_cone_drft_force,
            dim=(self.num_envs, self.num_bodies * self.num_contact_points),
            inputs=[
                self.contact_point_pos.reshape((self.num_envs, -1)),
                self.contact_point_lin_vel.reshape((self.num_envs, -1)),
                self.contact_point_lin_vel_prev.reshape((self.num_envs, -1)),
                self.sigma_flat,
                self.sigma_cone,
                self.dynamic_friction_coef,
                self.kf,
                wp.float32(self.r_h_per_point),
                wp.float32(self.cfg.nu),
                wp.float32(self.cfg.theta),
                wp.float32(self.cfg.c_g),
                wp.float32(self.cfg.c_d),
                self.phi,
                self.rho,
                wp.int32(1 if self.enable_added_mass else 0),
                wp.float32(self.dt),
                self.z_max.reshape((self.num_envs, -1)),
                wp.float32(self.cfg.eps_f),
                self.force_gm.reshape((self.num_envs, -1)),
                self.force_ema.reshape((self.num_envs, -1)),
                self.tau_r.reshape((self.num_envs, -1)),
                wp.float32(self.c_r),
                wp.int32(1 if self.enable_ema else 0),
                self.resitive_force,
            ],
            device=self.device,
        )

        # step 3: store current contact-point velocity for next step's accel / EMA test
        wp.launch(
            kernel=update_prev_velocity,
            dim=(self.num_envs, self.num_bodies * self.num_contact_points),
            inputs=[self.contact_point_lin_vel.reshape((self.num_envs, -1)), self.contact_point_lin_vel_prev.reshape((self.num_envs, -1))],
            device=self.device,
        )

        # step 4: sum per-contact forces to body wrench and transform to body frame
        wp.launch(
            kernel=zero_wrench,
            dim=(self.num_envs, self.num_bodies),
            inputs=[self.contact_force, self.contact_torque],
            device=self.device,
        )
        wp.launch(
            kernel=compute_contact_wrench,
            dim=(self.num_envs, self.num_bodies, self.num_contact_points),
            inputs=[
                self.body_pos,
                self.contact_point_pos,
                self.resitive_force.reshape((self.num_envs, self.num_bodies, self.num_contact_points)),
                self.contact_point_force,
                self.contact_point_torque,
                self.contact_force,
                self.contact_torque,
            ],
            device=self.device,
        )
        wp.launch(
            kernel=transform_global_wrench_to_body,
            dim=(self.num_envs, self.num_bodies),
            inputs=[
                self.body_quat,
                self.contact_force,
                self.contact_torque,
                self.contact_force_b,
                self.contact_torque_b,
            ],
            device=self.device,
        )

    """
    Reset.
    """

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # the managers hand out int32 indices, while the kernels below index with int64
        env_ids = env_ids.to(torch.int64)
        wp.launch(
            kernel=reset_cone,
            dim=(len(env_ids), self.num_bodies * self.num_contact_points),
            inputs=[
                wp.from_torch(env_ids.to(torch.int64), dtype=wp.int64),
                self.z_max.reshape((self.num_envs, -1)),
                self.force_gm.reshape((self.num_envs, -1)),
                self.force_ema.reshape((self.num_envs, -1)),
                self.tau_r.reshape((self.num_envs, -1)),
                self.contact_point_lin_vel_prev.reshape((self.num_envs, -1)),
            ],
            device=self.device,
        )


if __name__ == "__main__":
    num_envs = 4096
    num_bodies = 2
    device = "cuda"
    dt = 1 / 200
    material_cfg = Material3DRFTCfg(
        coef_1=[
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
        ],
        coef_2=[
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
        ],
        coef_3=[
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
        ],
    )
    collider_cfg = PlaneColliderCfg(
        contact_edge_x=(-0.1, 0.1),
        contact_edge_y=(-0.05, 0.05),
        contact_edge_z=(-0.02, 0.0),
        resolution=(5, 5),
    )

    rft_3d = RFT_3D(
        num_envs=num_envs,
        num_bodies=num_bodies,
        device=device,
        dt=dt,
        material_cfg=material_cfg,
        collider_cfg=collider_cfg,
    )

    body_pos = torch.zeros((num_envs, num_bodies, 3), device=device)
    body_quat = torch.zeros((num_envs, num_bodies, 4), device=device)
    # identity quaternion, whose real part is the last element in the (x, y, z, w) convention
    body_quat[..., 3] = 1.0
    body_lin_vel = torch.zeros((num_envs, num_bodies, 3), device=device)
    body_ang_vel = torch.zeros((num_envs, num_bodies, 3), device=device)
    rft_3d.update(body_pos, body_quat, body_lin_vel, body_ang_vel)
    rft_3d.reset()
