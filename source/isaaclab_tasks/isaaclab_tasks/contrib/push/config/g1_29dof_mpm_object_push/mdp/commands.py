# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object velocity commands with the heading state expected by the granular locomotion teacher."""

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse, quat_from_matrix, yaw_quat

from isaaclab_tasks.contrib.push.config.g1_29dof_object_push.mdp.commands import (
    BoxVelocityCommand,
    NearestBoxFaceCommand,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.commands.velocity_command import (
    UniformVelocityYawCommand,
)


class ObjectVelocityHeadingCommand(BoxVelocityCommand):
    """Retain object tracking metrics and three student commands while integrating teacher heading."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.heading_target = torch.zeros((self.num_envs, 4), device=self.device)
        self.heading_target[:, 3] = 1.0

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.heading_target[env_ids] = yaw_quat(self.robot.data.root_quat_w.torch[env_ids])

    def _update_command(self):
        # Reuse the teacher's integration and large-heading-error reset semantics.
        UniformVelocityYawCommand._update_command(self)


def velocity_heading_command(env, command_name: str) -> torch.Tensor:
    """Read velocity and integrated world yaw quaternion in the teacher's seven-column order."""
    command = env.command_manager.get_term(command_name)
    return torch.cat((command.command, command.heading_target), dim=-1)


class ObjectSurfaceHandPoseCommand(NearestBoxFaceCommand):
    """Choose wrist targets on cuboid, sphere or cylinder surfaces before workspace projection.

    Shape membership and dimensions come from the scene's clone plan, so target geometry
    cannot drift from the actual spawner choice. Sphere targets use world-up even while rolling.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.is_sphere = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_cylinder = torch.zeros_like(self.is_sphere)
        self.radius = torch.ones(self.num_envs, device=self.device)
        self.half_height = torch.ones_like(self.radius)
        plan = env.scene.clone_plan
        assigned = torch.zeros_like(self.is_sphere)
        for prototype in cloner.path.get_asset_prototypes(plan, self.object.cfg.prim_path):
            worlds, _ = cloner.query.get_asset_prototype_world_index(plan.topology, int(prototype))
            ids = torch.as_tensor(worlds, device=self.device, dtype=torch.long)
            spawn = plan.asset_cfgs[prototype].spawn
            if isinstance(spawn, sim_utils.MultiAssetSpawnerCfg):
                spawn = spawn.assets_cfg[0]
            if isinstance(spawn, (sim_utils.SphereCfg, sim_utils.MeshSphereCfg)):
                if not abs(cfg.sphere_contact_height) < spawn.radius:
                    raise ValueError("Sphere contact height must lie strictly inside its radius.")
                self.is_sphere[ids] = True
                self.radius[ids] = spawn.radius
            elif isinstance(spawn, (sim_utils.CylinderCfg, sim_utils.MeshCylinderCfg)):
                if spawn.axis != "Z":
                    raise ValueError("Object surface targets require cylinders with local Z axes.")
                self.is_cylinder[ids] = True
                self.radius[ids] = spawn.radius
                self.half_height[ids] = spawn.height / 2
            elif isinstance(spawn, (sim_utils.CuboidCfg, sim_utils.MeshCuboidCfg)):
                if tuple(spawn.size) != tuple(cfg.box_size):
                    raise ValueError("Hand-target bounding dimensions must match the cuboid dimensions.")
            else:
                raise ValueError(f"Unsupported pushing shape: {type(spawn).__name__}")
            assigned[ids] = True
        if not assigned.all():
            raise ValueError("Every environment must have one supported pushing object.")

    def object_target_pose_w(self):
        cube_position, cube_rotation = super().object_target_pose_w()
        frame = self.object.data.root_quat_w.torch.clone()
        frame[self.is_sphere] = frame.new_tensor([0.0, 0.0, 0.0, 1.0])
        robot_local = quat_apply_inverse(frame, self.robot.data.root_pos_w.torch - self.object.data.root_pos_w.torch)
        near = robot_local.clone()
        near[:, 2] = 0.0
        length = near.norm(dim=-1, keepdim=True)
        near = torch.where(length > 1.0e-6, near / length.clamp_min(1.0e-6), near.new_tensor([-1.0, 0.0, 0.0]))
        left = torch.stack((near[:, 1], -near[:, 0], torch.zeros_like(near[:, 0])), dim=-1)
        height = torch.where(self.is_sphere, self.cfg.sphere_contact_height, self.half_height - self.cfg.top_clearance)
        circle_radius = torch.where(
            self.is_sphere, (self.radius.square() - height.square()).clamp_min(0).sqrt(), self.radius
        )
        lateral = torch.minimum(torch.full_like(circle_radius, self.cfg.lateral_offset), 0.8 * circle_radius)
        forward = (circle_radius.square() - lateral.square()).clamp_min(0).sqrt()
        signs = near.new_tensor([1.0, -1.0])
        position = (
            forward[:, None, None] * near[:, None] + lateral[:, None, None] * signs[None, :, None] * left[:, None]
        )
        position[..., 2] = height[:, None]
        normal = position.clone()
        normal[..., 2] = torch.where(self.is_sphere[:, None], normal[..., 2], 0.0)
        normal = torch.nn.functional.normalize(normal, dim=-1)
        x_axis = -normal
        up = torch.zeros_like(normal)
        up[..., 2] = 1.0
        y_axis = torch.nn.functional.normalize(torch.cross(up, x_axis, dim=-1), dim=-1)
        z_axis = torch.cross(x_axis, y_axis, dim=-1)
        rotation = quat_from_matrix(torch.stack((x_axis, y_axis, z_axis), dim=-1))
        position += self.cfg.wrist_surface_offset * normal
        position, rotation = combine_frame_transforms(
            self.object.data.root_pos_w.torch[:, None].expand(-1, 2, -1),
            frame[:, None].expand(-1, 2, -1),
            position,
            rotation,
        )
        curved = (self.is_sphere | self.is_cylinder)[:, None, None]
        return torch.where(curved, position, cube_position), torch.where(curved, rotation, cube_rotation)
