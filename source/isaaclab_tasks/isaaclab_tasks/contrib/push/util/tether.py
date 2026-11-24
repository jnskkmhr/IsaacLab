# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A rope between a robot body and a free object, applied as external forces."""

from __future__ import annotations

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.scene import InteractiveScene


class Tether:
    """A rope that pulls only when taut: a one-sided spring and damper between two points.

    One point is fixed on a body of the robot articulation, the other on a rigid object. :meth:`apply`
    computes the tension from the current state and sets it on both bodies, equal and opposite, as a
    permanent wrench that stays until the next call.
    """

    def __init__(self, cfg, scene: InteractiveScene, robot_name: str = "robot"):
        """
        Args:
            cfg: Rope settings with the fields of the tow task's ``TetherCfg``.
            scene: Scene holding the robot and the towed object.
            robot_name: Scene name of the robot.
        """
        self.cfg = cfg
        self._robot = scene[robot_name]
        self._object = scene[cfg.object_name]
        self._body_id = self._robot.find_bodies(cfg.robot_body)[0][0]
        device, count = scene.device, scene.num_envs
        self._robot_offset = torch.tensor(cfg.robot_offset, device=device).repeat(count, 1)
        self._object_offset = torch.tensor(cfg.object_offset, device=device).repeat(count, 1)
        self.enabled = torch.ones(count, dtype=torch.bool, device=device)
        """Environments in which the rope exists, shape ``(num_envs,)``."""
        self.force_w = torch.zeros(count, 3, device=device)
        """Rope force on the robot in world frame [N], shape ``(num_envs, 3)``."""
        self._ends = (torch.zeros(count, 3, device=device), torch.zeros(count, 3, device=device))

    @property
    def tension(self) -> torch.Tensor:
        """Rope tension [N], shape ``(num_envs,)``."""
        return torch.linalg.norm(self.force_w, dim=-1)

    def apply(self) -> None:
        """Compute the rope tension from the current state and apply it to the robot body and the object."""
        cfg = self.cfg
        robot, body = self._robot.data, self._body_id
        body_com = robot.body_com_pos_w.torch[:, body]
        robot_point = robot.body_pos_w.torch[:, body] + math_utils.quat_apply(
            robot.body_quat_w.torch[:, body], self._robot_offset
        )
        robot_point_vel = robot.body_lin_vel_w.torch[:, body] + torch.cross(
            robot.body_ang_vel_w.torch[:, body], robot_point - body_com, dim=-1
        )

        obj = self._object.data
        object_com = obj.root_com_pos_w.torch
        object_point = obj.root_pos_w.torch + math_utils.quat_apply(obj.root_quat_w.torch, self._object_offset)
        object_point_vel = obj.root_lin_vel_w.torch + torch.cross(
            obj.root_ang_vel_w.torch, object_point - object_com, dim=-1
        )

        # Unit vector from the robot's end of the rope to the object's end.
        delta = object_point - robot_point
        distance = torch.linalg.norm(delta, dim=-1)
        direction = delta / distance.clamp_min(1.0e-6).unsqueeze(-1)
        stretch = distance - cfg.length
        stretch_rate = torch.sum((object_point_vel - robot_point_vel) * direction, dim=-1)
        tension = (cfg.stiffness * stretch + cfg.damping * stretch_rate).clamp(0.0, cfg.max_force)
        tension = torch.where((stretch > 0.0) & self.enabled, tension, torch.zeros_like(tension))
        torch.nan_to_num_(tension, nan=0.0, posinf=0.0, neginf=0.0)

        self.force_w = tension.unsqueeze(-1) * direction
        self._ends = (robot_point, object_point)
        self._robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self.force_w.unsqueeze(1),
            torques=torch.cross(robot_point - body_com, self.force_w, dim=-1).unsqueeze(1),
            body_ids=[body],
            is_global=True,
        )
        self._object.permanent_wrench_composer.set_forces_and_torques_index(
            forces=-self.force_w.unsqueeze(1),
            torques=torch.cross(object_point - object_com, -self.force_w, dim=-1).unsqueeze(1),
            body_ids=[0],
            is_global=True,
        )

    def reapply(self) -> None:
        """Add the held rope force again, without recomputing it.

        Use it after another term replaced the permanent wrenches of the robot or the object with its own, for
        example an analytical contact force that is set once per physics step.
        """
        robot_point, object_point = self._ends
        body_com = self._robot.data.body_com_pos_w.torch[:, self._body_id]
        object_com = self._object.data.root_com_pos_w.torch
        self._robot.permanent_wrench_composer.add_forces_and_torques_index(
            forces=self.force_w.unsqueeze(1),
            torques=torch.cross(robot_point - body_com, self.force_w, dim=-1).unsqueeze(1),
            body_ids=[self._body_id],
            is_global=True,
        )
        self._object.permanent_wrench_composer.add_forces_and_torques_index(
            forces=-self.force_w.unsqueeze(1),
            torques=torch.cross(object_point - object_com, -self.force_w, dim=-1).unsqueeze(1),
            body_ids=[0],
            is_global=True,
        )

    def draw(self, visualizer) -> None:
        """Draw the rope of every environment in a Newton visualizer: red when taut, grey when slack.

        Call it from the visualizer's frame hook.
        """
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is None:
            return
        start, end = self._ends
        # A row of small spheres along each rope: the RTX viewer draws points but not lines.
        count = 48
        fraction = torch.linspace(0.0, 1.0, count, device=start.device).view(1, count, 1)
        points = (start.unsqueeze(1) + fraction * (end - start).unsqueeze(1)).reshape(-1, 3)
        taut = (self.tension > 1.0).view(-1, 1, 1).expand(-1, count, 1).reshape(-1, 1)
        red = torch.tensor([0.9, 0.1, 0.1], device=start.device)
        grey = torch.tensor([0.5, 0.5, 0.5], device=start.device)
        colors = torch.where(taut, red, grey)
        device = viewer.device
        viewer.log_points(
            "tether",
            wp.array(points.detach().cpu().numpy(), dtype=wp.vec3, device=device),
            radii=0.012,
            colors=wp.array(colors.cpu().numpy(), dtype=wp.vec3, device=device),
        )
