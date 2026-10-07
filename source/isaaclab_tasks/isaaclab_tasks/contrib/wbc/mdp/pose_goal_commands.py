# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Fixed navigation goals with pelvis-relative torso and torso-relative hand poses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
    wrap_to_pi,
    yaw_quat,
)

from .dataset import WholeBodyDataset
from .pose_goal_dataset import endpoint_kinematics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .pose_goal_commands_cfg import PoseGoalCommandCfg


class PoseGoalCommand(CommandTerm):
    """Expose a fixed world destination and fixed goals in measured parent link frames."""

    def __init__(self, cfg: PoseGoalCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        cfg.validate_config()
        self.robot = env.scene["robot"]
        self.joint_ids = self.robot.find_joints(cfg.joint_names, preserve_order=True)[0]
        names = [cfg.pelvis_name, cfg.torso_name, *cfg.hand_names, *cfg.foot_names]
        self.body_ids = self.robot.find_bodies(names, preserve_order=True)[0]
        self.pelvis_id, self.torso_id = self.body_ids[:2]
        self.hand_ids, self.foot_ids = self.body_ids[2:4], self.body_ids[4:]
        self.sensor = env.scene["contact_forces"]
        self.sensor_foot_ids = self.sensor.find_bodies(cfg.foot_names, preserve_order=True)[0]
        dataset_names = [cfg.pelvis_name, *cfg.foot_names, *cfg.hand_names]
        self.dataset = WholeBodyDataset(cfg.dataset_path, cfg.joint_names, dataset_names, "cpu")
        if not self.dataset.is_static:
            raise ValueError("WBC pose goals require static endpoints, not motion trajectories")
        root_pose = torch.cat((self.dataset.body_pos_w[:, 0], self.dataset.body_quat_w[:, 0]), dim=-1)
        poses, corners = endpoint_kinematics(
            self.robot.cfg.spawn.usd_path, cfg.joint_names, self.dataset.joint_pos, root_pose, names
        )
        pelvis_quat, torso_quat = poses[:, 0, 3:], poses[:, 1, 3:]
        self.endpoint_torso_quat_p = quat_mul(quat_conjugate(pelvis_quat), torso_quat).to(self.device)
        torso_inverse = quat_conjugate(torso_quat)[:, None].expand(-1, 2, -1)
        self.endpoint_hand_pos_t = quat_apply_inverse(
            torso_quat[:, None].expand(-1, 2, -1), poses[:, 2:4, :3] - poses[:, 1:2, :3]
        ).to(self.device)
        self.endpoint_hand_quat_t = quat_mul(torso_inverse, poses[:, 2:4, 3:]).to(self.device)
        self.endpoint_height = poses[:, 0, 2].to(self.device)
        self.initial_frames = (self.endpoint_height >= cfg.initial_min_height).nonzero().flatten()
        if not len(self.initial_frames):
            raise ValueError("WBC dataset has no initial poses above initial_min_height")
        self.initial_joint_pos = self.dataset.joint_pos.to(self.device)
        self.initial_root_quat = pelvis_quat.to(self.device)
        self.sole_corners = corners.to(self.device)
        self.initial_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_frame = torch.zeros_like(self.initial_frame)
        self.goal_pelvis_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pelvis_yaw_w = torch.zeros(self.num_envs, device=self.device)
        self.goal_torso_quat_p = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_torso_quat_p[:, 3] = 1
        self.goal_hand_pos_t = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.goal_hand_quat_t = torch.zeros(self.num_envs, 2, 4, device=self.device)
        self.goal_hand_quat_t[..., 3] = 1
        self.hold_time = torch.zeros(self.num_envs, device=self.device)
        self.goal_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name in (
            "pelvis_xy_error_m",
            "pelvis_height_error_m",
            "pelvis_yaw_error_rad",
            "torso_error_rad",
            "hand_position_error_m",
            "hand_orientation_error_rad",
            "sole_clearance_m",
            "goal_reached",
            "goals_completed",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Goal encoding (N, 29): heading-frame pelvis error, yaw sin/cos, torso 6D, hand XYZ/6D."""
        position = quat_apply_inverse(yaw_quat(self.pelvis_quat_w), self.pelvis_position_error_w)
        yaw_error = self.pelvis_yaw_error
        return torch.cat(
            (
                position,
                torch.sin(yaw_error)[:, None],
                torch.cos(yaw_error)[:, None],
                matrix_from_quat(self.goal_torso_quat_p)[..., :2].flatten(1),
                self.goal_hand_pos_t.flatten(1),
                matrix_from_quat(self.goal_hand_quat_t)[..., :2].flatten(1),
            ),
            dim=-1,
        )

    @property
    def pelvis_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w.torch[:, self.pelvis_id]

    @property
    def pelvis_position_error_w(self) -> torch.Tensor:
        return self.goal_pelvis_pos_w - self.robot.data.body_pos_w.torch[:, self.pelvis_id]

    @property
    def pelvis_yaw_error(self) -> torch.Tensor:
        return wrap_to_pi(self.goal_pelvis_yaw_w - euler_xyz_from_quat(self.pelvis_quat_w)[2])

    @property
    def torso_quat_p(self) -> torch.Tensor:
        return quat_mul(quat_conjugate(self.pelvis_quat_w), self.robot.data.body_quat_w.torch[:, self.torso_id])

    @property
    def hand_pos_t(self) -> torch.Tensor:
        torso_quat = self.robot.data.body_quat_w.torch[:, self.torso_id]
        displacement = (
            self.robot.data.body_pos_w.torch[:, self.hand_ids]
            - self.robot.data.body_pos_w.torch[:, self.torso_id, None]
        )
        return quat_apply_inverse(torso_quat[:, None].expand(-1, 2, -1), displacement)

    @property
    def hand_quat_t(self) -> torch.Tensor:
        torso_inverse = quat_conjugate(self.robot.data.body_quat_w.torch[:, self.torso_id])
        return quat_mul(torso_inverse[:, None].expand(-1, 2, -1), self.robot.data.body_quat_w.torch[:, self.hand_ids])

    @property
    def foot_forces_w(self) -> torch.Tensor:
        """Latest normal forces, left/right order, without history-max lift-off delay."""
        return self.sensor.data.net_normal_forces_w.torch[:, self.sensor_foot_ids]

    @property
    def foot_contacts(self) -> torch.Tensor:
        return self.foot_forces_w.norm(dim=-1) > self.cfg.contact_threshold

    @property
    def sole_clearance(self) -> torch.Tensor:
        """Minimum box-corner height above the flat plane, shape (N, 2)."""
        quaternion = self.robot.data.body_quat_w.torch[:, self.foot_ids, None].expand(-1, -1, 8, -1)
        corners = self.sole_corners[None].expand(self.num_envs, -1, -1, -1)
        heights = quat_apply(quaternion, corners)[..., 2] + self.robot.data.body_pos_w.torch[:, self.foot_ids, 2, None]
        return heights.min(dim=-1).values - self._env.scene.env_origins[:, 2, None]

    @property
    def navigation_gate(self) -> torch.Tensor:
        """Smooth movement gate for translation or heading error; zero inside both tolerances."""
        distance = self.pelvis_position_error_w[:, :2].norm(dim=-1)
        distance_gate = ((distance - self.cfg.position_tolerance) / self.cfg.position_tolerance).clamp(0, 1)
        yaw_gate = (
            (self.pelvis_yaw_error.abs() - self.cfg.orientation_tolerance) / self.cfg.orientation_tolerance
        ).clamp(0, 1)
        return torch.maximum(distance_gate, yaw_gate)

    @property
    def settled(self) -> torch.Tensor:
        """All task errors and world/relative velocities within configured tolerances."""
        cfg, data = self.cfg, self.robot.data
        position_error = self.pelvis_position_error_w
        torso_error = quat_error_magnitude(self.torso_quat_p, self.goal_torso_quat_p)
        hand_position_error = (self.hand_pos_t - self.goal_hand_pos_t).norm(dim=-1)
        hand_orientation_error = quat_error_magnitude(self.hand_quat_t, self.goal_hand_quat_t)
        torso_position = data.body_pos_w.torch[:, self.torso_id]
        torso_velocity = data.body_lin_vel_w.torch[:, self.torso_id]
        torso_angular_velocity = data.body_ang_vel_w.torch[:, self.torso_id]
        displacement = data.body_pos_w.torch[:, self.hand_ids] - torso_position[:, None]
        hand_relative_velocity = (
            data.body_lin_vel_w.torch[:, self.hand_ids]
            - torso_velocity[:, None]
            - torch.cross(torso_angular_velocity[:, None].expand(-1, 2, -1), displacement, dim=-1)
        )
        hand_relative_angular = data.body_ang_vel_w.torch[:, self.hand_ids] - torso_angular_velocity[:, None]
        torso_relative_angular = torso_angular_velocity - data.body_ang_vel_w.torch[:, self.pelvis_id]
        return (
            self.foot_contacts.any(dim=-1)
            & (position_error[:, :2].norm(dim=-1) < cfg.position_tolerance)
            & (position_error[:, 2].abs() < cfg.height_tolerance)
            & (self.pelvis_yaw_error.abs() < cfg.orientation_tolerance)
            & (torso_error < cfg.orientation_tolerance)
            & (hand_position_error.max(dim=-1).values < cfg.hand_position_tolerance)
            & (hand_orientation_error.max(dim=-1).values < cfg.orientation_tolerance)
            & (data.body_lin_vel_w.torch[:, self.pelvis_id].norm(dim=-1) < cfg.linear_velocity_tolerance)
            & (data.body_ang_vel_w.torch[:, self.pelvis_id].norm(dim=-1) < cfg.angular_velocity_tolerance)
            & (hand_relative_velocity.norm(dim=-1).max(dim=-1).values < cfg.relative_velocity_tolerance)
            & (hand_relative_angular.norm(dim=-1).max(dim=-1).values < cfg.angular_velocity_tolerance)
            & (torso_relative_angular.norm(dim=-1) < cfg.angular_velocity_tolerance)
        )

    def set_goals(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice,
        pelvis_pos_w: torch.Tensor,
        pelvis_yaw_w: torch.Tensor,
        torso_quat_p: torch.Tensor,
        hand_pos_t: torch.Tensor,
        hand_quat_t: torch.Tensor,
    ) -> None:
        """Accept fixed external goals and disable periodic resampling for selected environments.

        Inputs have leading size len(env_ids), positions in metres and normalized XYZW
        rotations in the documented frames. World positions include environment origins.
        Episode reset restores sampled goals. No goal moves until this method is called again.
        """
        ids = (
            torch.arange(self.num_envs, device=self.device)[env_ids]
            if isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )
        if ids.ndim != 1 or torch.any((ids < 0) | (ids >= self.num_envs)) or ids.unique().numel() != ids.numel():
            raise ValueError("WBC goal environment IDs must be unique valid indices")
        values = (pelvis_pos_w, pelvis_yaw_w, torso_quat_p, hand_pos_t, hand_quat_t)
        shapes = ((len(ids), 3), (len(ids),), (len(ids), 4), (len(ids), 2, 3), (len(ids), 2, 4))
        converted = [value.to(device=self.device, dtype=torch.float32) for value in values]
        for value, shape in zip(converted, shapes):
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"WBC goal must have shape {shape} and finite values")
        for quaternion in (converted[2], converted[4]):
            if not torch.allclose(quaternion.norm(dim=-1), torch.ones_like(quaternion[..., 0]), atol=1e-4):
                raise ValueError("WBC goal quaternions must be normalized XYZW")
        for destination, value in zip(
            (
                self.goal_pelvis_pos_w,
                self.goal_pelvis_yaw_w,
                self.goal_torso_quat_p,
                self.goal_hand_pos_t,
                self.goal_hand_quat_t,
            ),
            converted,
        ):
            destination[ids] = value
        self.hold_time[ids] = 0
        self.goal_completed[ids] = False
        self.time_left[ids] = torch.inf

    def reset_robot(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        """Reset a stable initial endpoint independently of the subsequently sampled goal."""
        ids = (
            torch.arange(self.num_envs, device=self.device)[env_ids]
            if isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )
        frames = self.initial_frames[torch.randint(len(self.initial_frames), (len(ids),), device=self.device)]
        self.initial_frame[ids] = frames
        yaw = torch.empty(len(ids), device=self.device).uniform_(-torch.pi, torch.pi)
        rotation = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        root_state = self.robot.data.default_root_state.torch[ids].clone()
        root_state[:, :3] = self._env.scene.env_origins[ids]
        root_state[:, 2] += self.endpoint_height[frames]
        root_state[:, 3:7] = quat_mul(rotation, self.initial_root_quat[frames])
        root_state[:, 7:] = 0
        joint_position = self.robot.data.default_joint_pos.torch[ids].clone()
        joint_position[:, self.joint_ids] = self.initial_joint_pos[frames]
        self.robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=ids)
        self.robot.write_joint_state_to_sim_index(
            position=joint_position, velocity=torch.zeros_like(joint_position), env_ids=ids
        )
        self.hold_time[ids] = 0
        self.goal_completed[ids] = False

    def _resample_command(self, env_ids: Sequence[int] | slice) -> None:
        ids = (
            torch.arange(self.num_envs, device=self.device)[env_ids]
            if isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )
        frames = torch.randint(len(self.endpoint_height), (len(ids),), device=self.device)
        self.goal_frame[ids] = frames
        progress = (
            min(1.0, self._env.common_step_counter / max(1, self.cfg.curriculum_steps))
            if self.cfg.curriculum_steps
            else 1.0
        )
        distance_limit = self.cfg.initial_distance + progress * (self.cfg.final_distance - self.cfg.initial_distance)
        yaw_limit = self.cfg.initial_yaw_change + progress * (self.cfg.final_yaw_change - self.cfg.initial_yaw_change)
        angle = torch.rand(len(ids), device=self.device) * (2 * torch.pi)
        distance = torch.sqrt(torch.rand(len(ids), device=self.device)) * distance_limit
        yaw_change = torch.empty(len(ids), device=self.device).uniform_(-yaw_limit, yaw_limit)
        standing = torch.rand(len(ids), device=self.device) < self.cfg.standing_probability
        distance[standing], yaw_change[standing] = 0, 0
        self.goal_pelvis_pos_w[ids] = self.robot.data.root_pos_w.torch[ids]
        self.goal_pelvis_pos_w[ids, :2] += distance[:, None] * torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        self.goal_pelvis_pos_w[ids, 2] = self._env.scene.env_origins[ids, 2] + self.endpoint_height[frames]
        self.goal_pelvis_yaw_w[ids] = wrap_to_pi(
            euler_xyz_from_quat(self.robot.data.root_quat_w.torch[ids])[2] + yaw_change
        )
        self.goal_torso_quat_p[ids] = self.endpoint_torso_quat_p[frames]
        self.goal_hand_pos_t[ids] = self.endpoint_hand_pos_t[frames]
        self.goal_hand_quat_t[ids] = self.endpoint_hand_quat_t[frames]
        self.hold_time[ids] = 0
        self.goal_completed[ids] = False

    def _update_command(self) -> None:
        self.hold_time[:] = torch.where(self.settled, self.hold_time + self._env.step_dt, 0)
        completed = self.hold_time >= self.cfg.hold_duration
        self.metrics["goals_completed"] += (completed & ~self.goal_completed).float()
        self.goal_completed |= completed

    def _update_metrics(self) -> None:
        error = self.pelvis_position_error_w
        self.metrics["pelvis_xy_error_m"][:] = error[:, :2].norm(dim=-1)
        self.metrics["pelvis_height_error_m"][:] = error[:, 2].abs()
        self.metrics["pelvis_yaw_error_rad"][:] = self.pelvis_yaw_error.abs()
        self.metrics["torso_error_rad"][:] = quat_error_magnitude(self.torso_quat_p, self.goal_torso_quat_p)
        self.metrics["hand_position_error_m"][:] = (self.hand_pos_t - self.goal_hand_pos_t).norm(dim=-1).mean(dim=-1)
        self.metrics["hand_orientation_error_rad"][:] = quat_error_magnitude(
            self.hand_quat_t, self.goal_hand_quat_t
        ).mean(dim=-1)
        self.metrics["sole_clearance_m"][:] = self.sole_clearance.mean(dim=-1)
        self.metrics["goal_reached"][:] = (self.hold_time >= self.cfg.hold_duration).float()

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis and not hasattr(self, "goal_visualizer"):
            cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/WBCPoseGoals",
                markers={
                    "x": sim_utils.CuboidCfg(
                        size=(0.15, 0.008, 0.008),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "y": sim_utils.CuboidCfg(
                        size=(0.008, 0.15, 0.008),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                    "z": sim_utils.CuboidCfg(
                        size=(0.008, 0.008, 0.15),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    ),
                },
            )
            self.goal_visualizer = VisualizationMarkers(cfg)
        if hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event: object) -> None:
        count = min(self.num_envs, self.cfg.max_visualized_envs)
        data = self.robot.data
        torso_pos = data.body_pos_w.torch[:count, self.torso_id]
        torso_quat = data.body_quat_w.torch[:count, self.torso_id]
        hand_pos = torso_pos[:, None] + quat_apply(torso_quat[:, None].expand(-1, 2, -1), self.goal_hand_pos_t[:count])
        hand_quat = quat_mul(torso_quat[:, None].expand(-1, 2, -1), self.goal_hand_quat_t[:count])
        yaw = self.goal_pelvis_yaw_w[:count]
        pelvis_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        desired_torso = quat_mul(self.pelvis_quat_w[:count], self.goal_torso_quat_p[:count])
        positions = torch.cat((self.goal_pelvis_pos_w[:count, None], torso_pos[:, None], hand_pos), dim=1).reshape(
            -1, 3
        )
        orientations = torch.cat((pelvis_quat[:, None], desired_torso[:, None], hand_quat), dim=1).reshape(-1, 4)
        offsets = torch.eye(3, device=self.device) * 0.075
        offsets = quat_apply(orientations[:, None].expand(-1, 3, -1), offsets[None])
        self.goal_visualizer.visualize(
            translations=(positions[:, None] + offsets).reshape(-1, 3),
            orientations=orientations[:, None].expand(-1, 3, -1).reshape(-1, 4),
            marker_indices=torch.arange(3, device=self.device).repeat(len(positions)),
            environment_ids=torch.arange(count, device=self.device).repeat_interleave(12),
        )
