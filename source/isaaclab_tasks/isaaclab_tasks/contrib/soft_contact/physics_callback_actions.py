# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.markers import VisualizationMarkers

from ._impl.material import (
    DefaultConeDRFTCfg,
    DefaultSpringDamperCfg,
    GenericMaterialCfg,
    Material3DRFTCfg,
    # PoppySeedCPCfg,
)
from ._impl.soft_contact_model_torch import (
    RFT_2D,
    RFT_3D,
)
from ._impl.soft_contact_model_warp import RFT_2D as RFT_2D_WARP
from ._impl.soft_contact_model_warp import RFT_3D as RFT_3D_WARP
from ._impl.soft_contact_model_warp import ConeDRFT as ConeDRFT_WARP
from ._impl.soft_contact_model_warp import ConeDRFTMultiPoint as ConeDRFTMultiPoint_WARP
from ._impl.soft_contact_model_warp import SpringDamper as SpringDamper_WARP

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg

SUPPORTED_BACKENDS = ["3D-warp", "2D-warp", "spring-damper", "cone-drft", "cone-drft-multipoint"]


class PhysicsCallbackAction(ActionTerm):
    cfg: actions_cfg.PhysicsCallbackActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.PhysicsCallbackActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # process body ids
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_names)
        self._body_ids = body_ids

        # action buffer
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # wrench buffer
        self.contact_wrench = torch.zeros(self.num_envs, len(body_ids), 6, device=self.device)
        self.contact_wrench_b = torch.zeros(self.num_envs, len(body_ids), 6, device=self.device)

        # get physics backend
        self.backend = self.cfg.backend
        if self.cfg.backend == "2D":
            material_cfg = GenericMaterialCfg()
            num_bodies = len(body_ids)
            self.contact_solver = RFT_2D(
                material_cfg=material_cfg,
                num_envs=self.num_envs,
                num_bodies=num_bodies,
                device=self.device,
                dt=env.physics_dt,
                contact_threshold=self.cfg.contact_threshold,
                enable_ema_filter=self.cfg.enable_ema_filter,
                collider_cfg=self.cfg.intruder_geometry_cfg,
                history_length=self.cfg.contact_data_history_length,
                history_logging_decimation=self.cfg.history_logging_decimation,
            )
        elif self.cfg.backend == "3D":
            material_cfg = Material3DRFTCfg()
            num_bodies = len(body_ids)
            self.contact_solver = RFT_3D(
                material_cfg=material_cfg,
                num_envs=self.num_envs,
                num_bodies=num_bodies,
                device=self.device,
                dt=env.physics_dt,
                contact_threshold=self.cfg.contact_threshold,
                enable_ema_filter=self.cfg.enable_ema_filter,
                collider_cfg=self.cfg.intruder_geometry_cfg,
                history_length=self.cfg.contact_data_history_length,
                history_logging_decimation=self.cfg.history_logging_decimation,
            )
        elif self.cfg.backend == "3D-warp":
            material_cfg = Material3DRFTCfg()
            num_bodies = len(body_ids)
            self.contact_solver = RFT_3D_WARP(
                material_cfg=material_cfg,
                num_envs=self.num_envs,
                num_bodies=num_bodies,
                device=self.device,
                dt=env.physics_dt,
                contact_threshold=self.cfg.contact_threshold,
                enable_ema_filter=self.cfg.enable_ema_filter,
                collider_cfg=self.cfg.intruder_geometry_cfg,
                history_length=self.cfg.contact_data_history_length,
                history_logging_decimation=self.cfg.history_logging_decimation,
            )
        elif self.cfg.backend == "2D-warp":
            material_cfg = GenericMaterialCfg()
            num_bodies = len(body_ids)
            self.contact_solver = RFT_2D_WARP(
                material_cfg=material_cfg,
                num_envs=self.num_envs,
                num_bodies=num_bodies,
                device=self.device,
                dt=env.physics_dt,
                contact_threshold=self.cfg.contact_threshold,
                enable_ema_filter=self.cfg.enable_ema_filter,
                collider_cfg=self.cfg.intruder_geometry_cfg,
                history_length=self.cfg.contact_data_history_length,
                history_logging_decimation=self.cfg.history_logging_decimation,
            )
        elif self.cfg.backend == "spring-damper":
            material_cfg = DefaultSpringDamperCfg()
            num_bodies = len(body_ids)
            self.contact_solver = SpringDamper_WARP(
                material_cfg=material_cfg,
                num_envs=self.num_envs,
                num_bodies=num_bodies,
                device=self.device,
                dt=env.physics_dt,
                contact_threshold=self.cfg.contact_threshold,
                collider_cfg=self.cfg.intruder_geometry_cfg,
                history_length=self.cfg.contact_data_history_length,
                history_logging_decimation=self.cfg.history_logging_decimation,
            )
        elif self.cfg.backend == "cone-drft":
            material_cfg = DefaultConeDRFTCfg()
            num_bodies = len(body_ids)
            self.contact_solver = ConeDRFT_WARP(
                material_cfg=material_cfg,
                num_envs=self.num_envs,
                num_bodies=num_bodies,
                device=self.device,
                dt=env.physics_dt,
                contact_threshold=self.cfg.contact_threshold,
                enable_ema_filter=self.cfg.enable_ema_filter,
                collider_cfg=self.cfg.intruder_geometry_cfg,
                history_length=self.cfg.contact_data_history_length,
                history_logging_decimation=self.cfg.history_logging_decimation,
            )
        elif self.cfg.backend == "cone-drft-multipoint":
            material_cfg = DefaultConeDRFTCfg()
            num_bodies = len(body_ids)
            self.contact_solver = ConeDRFTMultiPoint_WARP(
                material_cfg=material_cfg,
                num_envs=self.num_envs,
                num_bodies=num_bodies,
                device=self.device,
                dt=env.physics_dt,
                contact_threshold=self.cfg.contact_threshold,
                enable_ema_filter=self.cfg.enable_ema_filter,
                collider_cfg=self.cfg.intruder_geometry_cfg,
                history_length=self.cfg.contact_data_history_length,
                history_logging_decimation=self.cfg.history_logging_decimation,
            )
        else:
            raise ValueError(f"Unsupported RFT backend: {self.cfg.backend}")

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "contact_visualizer"):
                self.contact_visualizer = VisualizationMarkers(self.cfg.contact_visualizer_cfg)
            # create markers if necessary for the first time
            if not hasattr(self, "contact_force_visualizer"):
                self.contact_force_visualizer = VisualizationMarkers(self.cfg.contact_force_visualizer_cfg)
            # set their visibility to true
            self.contact_visualizer.set_visibility(True)
            self.contact_force_visualizer.set_visibility(True)
        else:
            if hasattr(self, "contact_visualizer"):
                self.contact_visualizer.set_visibility(False)
            if hasattr(self, "contact_force_visualizer"):
                self.contact_force_visualizer.set_visibility(False)

    """
    properties.
    """

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def body_pos(self) -> torch.Tensor:
        body_pos = self._asset.data.body_link_pose_w.torch[:, self._body_ids, :3]
        return body_pos

    @property
    def body_quat(self) -> torch.Tensor:
        # return self._asset.data.body_com_pose_w.torch[:, self._body_ids, 3:7]
        return self._asset.data.body_link_pose_w.torch[:, self._body_ids, 3:7]

    @property
    def body_lin_vel(self) -> torch.Tensor:
        # return self._asset.data.body_com_vel_w.torch[:, self._body_ids, :3]
        # return self._asset.data.root_lin_vel_w.torch.unsqueeze(1)
        return self._asset.data.body_link_lin_vel_w.torch[:, self._body_ids, :]

    @property
    def body_ang_vel(self) -> torch.Tensor:
        # return self._asset.data.body_com_vel_w.torch[:, self._body_ids, 3:6]
        # return self._asset.data.root_ang_vel_w.torch.unsqueeze(1)
        return self._asset.data.body_link_ang_vel_w.torch[:, self._body_ids, :]

    """
    operations.
    """

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        if self.cfg.disable:
            return
        body_pos = self.body_pos.clone()
        body_quat = self.body_quat.clone()[:, :, [3, 0, 1, 2]]  # convert to [w, x, y, z]
        body_lin_vel = self.body_lin_vel.clone()
        body_ang_vel = self.body_ang_vel.clone()
        self.contact_solver.update(body_pos, body_quat, body_lin_vel, body_ang_vel)

        self.contact_wrench = self.contact_solver.contact_wrench.clone()  # global wrench (num_envs, num_bodies, 6)
        self.contact_wrench_b = self.contact_solver.contact_wrench_b.clone()  # body wrench (num_envs, num_bodies, 6)

        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self.contact_wrench_b[:, :, :3],
            torques=self.contact_wrench_b[:, :, 3:6],
            body_ids=self._body_ids,
        )

        # track if sensor if active or not
        self.contact_solver.data.is_sensor_active = (
            torch.max(torch.norm(self.contact_solver.data.net_forces_w_history, dim=-1), dim=1)[0]
            > self.cfg.contact_threshold
        )

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._asset.is_initialized:
            return

        # handle contact point visualization
        if self.cfg.backend in SUPPORTED_BACKENDS:
            contact_pos = self.contact_solver.torch_contact_point_pos.reshape(-1, 3)
        elif self.cfg.backend in ("3D", "2D"):
            contact_pos = self.contact_solver.contact_point_pos.reshape(-1, 3)
        scale = (
            torch.tensor(
                self.contact_force_visualizer.cfg.markers["arrow"].scale,
                device=self.device,  # type: ignore
            )
            .unsqueeze(0)
            .repeat(contact_pos.shape[0], 1)
        )
        self.contact_visualizer.visualize(contact_pos, None, scale)

        # handle contact force visualization
        # get marker location
        body_pos_w = self.body_pos.clone()
        body_pos_w[:, :, 2] += 0.04
        # body_quat_w = self.body_quat.clone()

        # # get scale
        # # TODO: handle tangential components too
        # grf_scale = self.contact_wrench_b[:, :, :3] / self.cfg.contact_vis_max_force
        # scale = (
        #     torch.tensor(
        #         self.contact_force_visualizer.cfg.markers["arrow"].scale,
        #         device=self.device,  # type: ignore
        #     )
        #     .unsqueeze(0)
        #     .repeat(self.num_envs * len(self._body_ids), 1)
        # )
        # # scale[:, 2] = grf_scale[:, :, 2].reshape(-1)
        # scale = grf_scale.reshape(-1, 3)

        # # display markers
        # self.contact_force_visualizer.visualize(
        #     body_pos_w.reshape(-1, 3),
        #     body_quat_w.reshape(-1, 4),
        #     scale.reshape(-1, 3),
        # )

        force = self.contact_wrench[:, :, :3].reshape(-1, 3)  # global frame
        # Arrow orientation from force direction
        arrow_quat = self.force_to_arrow_quat(force)  # (N, 4)
        # Arrow scale: fixed width, force-proportional length
        base_scale = (
            torch.tensor(self.contact_force_visualizer.cfg.markers["arrow"].scale, device=self.device)
            .unsqueeze(0)
            .repeat(force.shape[0], 1)
        )
        force_magnitude = torch.norm(force, dim=-1) / self.cfg.contact_vis_scale
        force_magnitude[force_magnitude < (self.cfg.contact_vis_force_threshold / self.cfg.contact_vis_scale)] = 0.0
        scale = base_scale.clone()
        scale[:, 2] = force_magnitude  # Z = length
        self.contact_force_visualizer.visualize(
            body_pos_w.reshape(-1, 3),
            arrow_quat,  # <-- orientation from force direction
            scale,  # <-- length proportional to magnitude
        )

    def reset(self, env_ids: torch.Tensor):
        self.contact_wrench_b[env_ids] = 0.0
        self.contact_solver.reset(env_ids)

    @staticmethod
    def force_to_arrow_quat(force: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Convert a 3D force vector (global frame) to a quaternion that rotates
        the default arrow axis (+Z) to point along the force direction.

        Args:
            force: (N, 3) force vectors in global frame
            eps: small value to avoid divide-by-zero

        Returns:
            quat: (N, 4) quaternion [w, x, y, z]
        """
        # Normalize force direction
        force_norm = torch.norm(force, dim=-1, keepdim=True).clamp(min=eps)
        force_dir = force / force_norm  # (N, 3)
        # Default arrow axis in Isaac Lab is +Z = [0, 0, 1]
        arrow_axis = torch.zeros_like(force_dir)
        arrow_axis[:, 2] = 1.0  # +Z
        # Compute rotation axis = cross(arrow_axis, force_dir)
        rot_axis = torch.linalg.cross(arrow_axis, force_dir)  # (N, 3)
        rot_axis_norm = torch.norm(rot_axis, dim=-1, keepdim=True).clamp(min=eps)
        rot_axis_unit = rot_axis / rot_axis_norm
        # Compute rotation angle = arccos(dot(arrow_axis, force_dir))
        dot = (arrow_axis * force_dir).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)  # (N, 1)
        angle = torch.acos(dot)  # (N, 1)
        # Convert axis-angle to quaternion: q = [cos(a/2), sin(a/2)*axis]
        half_angle = angle / 2.0
        w = torch.cos(half_angle)  # (N, 1)
        xyz = torch.sin(half_angle) * rot_axis_unit  # (N, 3)
        quat = torch.cat([w, xyz], dim=-1)  # (N, 4) [w, x, y, z]
        # Handle degenerate case: force already along +Z (angle ≈ 0)
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=force.device).expand(force.shape[0], -1)
        # Handle anti-parallel case: force along -Z (angle ≈ π), rotate 180° around +X
        flip = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=force.device).expand(force.shape[0], -1)
        near_zero = (angle < eps).squeeze(-1)
        near_pi = (angle > (torch.pi - eps)).squeeze(-1)
        quat = torch.where(near_zero.unsqueeze(-1), identity, quat)
        quat = torch.where(near_pi.unsqueeze(-1), flip, quat)
        return quat
