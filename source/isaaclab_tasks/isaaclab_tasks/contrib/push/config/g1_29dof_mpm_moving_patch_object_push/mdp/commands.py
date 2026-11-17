# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-heading-relative direction and smooth, proximity-dependent teacher targets."""

import numpy as np
import torch
import warp as wp

from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_mul,
    quat_unique,
    yaw_quat,
)


class PushDirectionCommand(CommandTerm):
    """A unit XY vector or zero; teacher speed and posture remain derived quantities."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.object = env.scene[cfg.object_name]
        self.torso_id = self.robot.find_bodies(cfg.reference_body_name)[0][0]
        self.direction = torch.zeros(self.num_envs, 2, device=self.device)
        self.velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading = torch.zeros(self.num_envs, 4, device=self.device)
        self.heading[:, 3] = 1
        self.height = torch.full((self.num_envs, 1), cfg.neutral_height, device=self.device)
        self.pitch = torch.zeros_like(self.height)
        self.hand_pose = torch.tensor(cfg.neutral_hand_poses, device=self.device).expand(self.num_envs, -1, -1).clone()
        self.stationary_position = self.robot.data.root_pos_w.torch.clone()
        self.engagement = torch.zeros(self.num_envs, device=self.device)
        self.metrics["object_forward_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hand_target_error"] = torch.zeros(self.num_envs, device=self.device)
        self.hand_ids = self.robot.find_bodies(cfg.hand_body_names, preserve_order=True)[0]
        self.contact_ids = self.robot.find_bodies(cfg.contact_body_names)[0]
        self._neutral = torch.tensor(cfg.neutral_hand_poses, device=self.device)
        self._contact_points = torch.tensor(cfg.contact_points, device=self.device)
        self.metrics["hand_contact_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["standing_drift"] = torch.zeros(self.num_envs, device=self.device)
        with np.load(cfg.workspace_path) as data:
            self.workspace = [
                torch.as_tensor(data[key], dtype=torch.float32, device=self.device)
                for key in ("left_pose", "right_pose")
            ]
        self.metrics["workspace_projection_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return self.direction

    @property
    def standing(self):
        return self.direction.norm(dim=-1) < 1e-6

    @property
    def hand_contact(self):
        """Whether a wrist/finger receives measured object contact, excluding knees and feet."""
        return (
            self._env.object_contact_force[:, self.contact_ids].norm(dim=-1).amax(-1) > self.cfg.contact_force_threshold
        )

    @property
    def teacher_command(self):
        return torch.cat((self.velocity, self.heading, self.height), dim=-1)

    def _resample_command(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        angle = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.angle_range)
        self.direction[env_ids] = torch.stack((angle.cos(), angle.sin()), dim=-1)
        stop = torch.rand(len(env_ids), device=self.device) < self.cfg.standing_probability
        self.direction[env_ids[stop]] = 0
        self.heading[env_ids] = yaw_quat(self.robot.data.root_quat_w.torch[env_ids])
        self.stationary_position[env_ids] = self.robot.data.root_pos_w.torch[env_ids]

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.engagement[env_ids] = 0
        return super().reset(env_ids)

    def _update_command(self):
        stop = self.standing
        self.velocity[:, 0] = torch.where(stop, 0.0, self.cfg.forward_speed)
        self.velocity[:, 1] = 0
        angle = torch.atan2(self.direction[:, 1], self.direction[:, 0])
        self.velocity[:, 2] = torch.where(
            stop, 0.0, (angle * self.cfg.steering_gain).clamp(-self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)
        )
        zero = torch.zeros_like(angle)
        delta = quat_from_euler_xyz(zero, zero, self.velocity[:, 2] * self._env.step_dt)
        self.heading[:] = quat_unique(quat_mul(delta, self.heading))
        current = yaw_quat(self.robot.data.root_quat_w.torch)
        wrong_heading = (current * self.heading).sum(-1).abs() < 0.7071
        self.heading[wrong_heading] = current[wrong_heading]
        distance = (self.object.data.root_pos_w.torch[:, :2] - self.robot.data.root_pos_w.torch[:, :2]).norm(dim=-1)
        target = (
            (self.cfg.neutral_distance - distance) / (self.cfg.neutral_distance - self.cfg.contact_distance)
        ).clamp(0, 1)
        target[stop] = 0
        rate = self._env.step_dt / self.cfg.posture_transition_s
        self.engagement += (target - self.engagement).clamp(-rate, rate)
        a = self.engagement
        points = wp.from_torch(self.object.data.root_pos_w.torch.clone().contiguous(), dtype=wp.vec3)
        self._env.scene.terrain.background_mesh.sample_surface_heights(points)
        surface = wp.to_torch(points)[:, 2]
        height = (
            self.object.data.root_pos_w.torch[:, 2]
            - surface
            + self.cfg.height_reference_offset
            + self.cfg.object_center_height_offset
        ).clamp(*self.cfg.contact_height_range)
        self.height[:, 0] = self.cfg.neutral_height + a * (height - self.cfg.neutral_height)
        self.pitch[:, 0] = a * self.cfg.contact_pitch
        # Rear face in object coordinates; wrist offset accounts for the closed fingers.
        local = self._contact_points.expand(self.num_envs, -1, -1)
        oq = yaw_quat(self.object.data.root_quat_w.torch)[:, None].expand(-1, 2, -1)
        target_w = self.object.data.root_pos_w.torch[:, None] + quat_apply(oq, local)
        torso_p = self.robot.data.body_pos_w.torch[:, self.torso_id, None]
        torso_q = self.robot.data.body_quat_w.torch[:, self.torso_id, None].expand(-1, 2, -1)
        target_b = quat_apply_inverse(torso_q, target_w - torso_p)
        neutral = self._neutral
        self.hand_pose[:, :, :3] = torch.lerp(neutral[None, :, :3], target_b, a[:, None, None])
        self.hand_pose[:, :, 3:] = neutral[None, :, 3:]
        projection_error = torch.zeros(self.num_envs, device=self.device)
        for hand, workspace in enumerate(self.workspace):
            requested = self.hand_pose[:, hand, :3]
            distance = torch.cdist(requested, workspace[:, :3]).square()
            orientation = 1.0 - (neutral[hand, 3:] @ workspace[:, 3:].T).abs()
            selected = (distance + self.cfg.workspace_orientation_weight * orientation[None]).argmin(-1)
            feasible = workspace[selected]
            projection_error += (feasible[:, :3] - requested).norm(dim=-1) / 2
            # Keep the validated neutral pose exactly; project only the contact request.
            self.hand_pose[:, hand] = torch.where((a > 0)[:, None], feasible, self.hand_pose[:, hand])
        self.metrics["workspace_projection_error"][:] = projection_error

    def _update_metrics(self):
        self.metrics["hand_contact_fraction"][:] = self.hand_contact.float()
        self.metrics["standing_drift"][:] = (
            self.robot.data.root_pos_w.torch[:, :2] - self.stationary_position[:, :2]
        ).norm(dim=-1) * self.standing
        self.metrics["object_forward_speed"][:] = quat_apply_inverse(
            yaw_quat(self.robot.data.root_quat_w.torch), self.object.data.root_lin_vel_w.torch
        )[:, 0]
        torso_p = self.robot.data.body_pos_w.torch[:, self.torso_id, None]
        torso_q = self.robot.data.body_quat_w.torch[:, self.torso_id, None].expand(-1, 2, -1)
        hand = quat_apply_inverse(torso_q, self.robot.data.body_pos_w.torch[:, self.hand_ids] - torso_p)
        self.metrics["hand_target_error"][:] = (hand - self.hand_pose[:, :, :3]).norm(dim=-1).mean(-1)

    def _set_debug_vis_impl(self, debug_vis):
        if debug_vis and not hasattr(self, "direction_visualizer"):
            self.direction_visualizer = VisualizationMarkers(self.cfg.direction_visualizer_cfg)
        if hasattr(self, "direction_visualizer"):
            self.direction_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        position = self.object.data.root_pos_w.torch.clone()
        position[:, 2] += 0.6
        angle = torch.atan2(self.direction[:, 1], self.direction[:, 0])
        zero = torch.zeros_like(angle)
        orientation = quat_mul(yaw_quat(self.robot.data.root_quat_w.torch), quat_from_euler_xyz(zero, zero, angle))
        scale = torch.full_like(position, 0.12)
        scale[:, 0] = 0.5
        scale[self.standing] = 0
        self.direction_visualizer.visualize(position, orientation, scale)
