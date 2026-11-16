# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direction progress, hand contact, and stationary behavior without object attraction."""

import torch
import warp as wp

from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, yaw_quat


def direction_progress(env, command_name, speed_scale):
    c = env.command_manager.get_term(command_name)
    velocity = quat_apply_inverse(yaw_quat(c.robot.data.root_quat_w.torch), c.object.data.root_lin_vel_w.torch)[:, :2]
    # Reward signed progress, not an exact continuous velocity target.
    return ((velocity * c.direction).sum(-1) / speed_scale).clamp(-1, 1) * (~c.standing) * c.hand_contact


def object_turning(env, command_name, std):
    c = env.command_manager.get_term(command_name)
    error = c.object.data.root_ang_vel_w.torch[:, 2] - c.velocity[:, 2]
    return torch.exp(-error.square() / std**2) * (~c.standing) * c.hand_contact


def stand_still(env, command_name, position_std, velocity_std):
    c = env.command_manager.get_term(command_name)
    drift = (c.robot.data.root_pos_w.torch[:, :2] - c.stationary_position[:, :2]).square().sum(-1)
    speed = c.robot.data.root_lin_vel_w.torch.square().sum(-1) + c.robot.data.root_ang_vel_w.torch.square().sum(-1)
    return torch.exp(-drift / position_std**2 - speed / velocity_std**2) * c.standing


def hand_contact(env, command_name, asset_cfg, force_scale):
    c = env.command_manager.get_term(command_name)
    forces = torch.stack(
        [env.object_contact_force[:, group.body_ids].norm(dim=-1).amax(-1) for group in asset_cfg], dim=-1
    )
    return (forces > force_scale).float().mean(-1) * (~c.standing)


def hand_tracking(env, command_name, std):
    c = env.command_manager.get_term(command_name)
    p = c.robot.data.body_pos_w.torch[:, c.hand_ids] - c.robot.data.body_pos_w.torch[:, c.torso_id, None]
    q = c.robot.data.body_quat_w.torch[:, c.torso_id, None].expand(-1, 2, -1)
    error = (quat_apply_inverse(q, p) - c.hand_pose[:, :, :3]).square().sum(-1).mean(-1)
    return torch.exp(-error / std**2)


def no_touch_when_standing(env, command_name):
    c = env.command_manager.get_term(command_name)
    return env.object_contact_force.norm(dim=-1).sum(-1).clamp(max=100) / 100 * c.standing


def posture_tracking(env, command_name, height_std, pitch_std):
    c = env.command_manager.get_term(command_name)
    # The terrain chart stores the initial sand surface; root height is measured above it.
    points = wp.from_torch(c.robot.data.root_pos_w.torch.clone().contiguous(), dtype=wp.vec3)
    env.scene.terrain.background_mesh.sample_surface_heights(points)
    heights = wp.to_torch(points)[:, 2]
    height = c.robot.data.root_pos_w.torch[:, 2] - heights
    roll, pitch, _ = euler_xyz_from_quat(c.robot.data.body_quat_w.torch[:, c.torso_id])
    roll = torch.atan2(roll.sin(), roll.cos())
    pitch = torch.atan2(pitch.sin(), pitch.cos())
    return torch.exp(
        -(height - c.height[:, 0]).square() / height_std**2
        - (pitch - c.pitch[:, 0]).square() / pitch_std**2
        - roll.square() / pitch_std**2
    )


def feet_slide(env, asset_cfg):
    velocity = env.scene[asset_cfg.name].data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return (velocity.norm(dim=-1) * env.foot_contact).sum(-1)


def stance_foot_angle_penalty(env, asset_cfg, angle_tolerance):
    quaternion = env.scene[asset_cfg.name].data.body_quat_w.torch[:, asset_cfg.body_ids]
    forward = torch.zeros_like(quaternion[..., :3])
    forward[..., 0] = 1
    forward = quat_apply(quaternion, forward)
    pitch = torch.atan2(forward[..., 2], forward[..., :2].norm(dim=-1))
    return ((pitch.abs() - angle_tolerance).clamp_min(0).square() * env.foot_contact).sum(-1)


def feet_air_time(env, command_name, threshold):
    contact = env.foot_contact_time > 0
    duration = torch.where(contact, env.foot_contact_time, env.foot_air_time)
    single = contact.sum(-1) == 1
    return torch.where(single, duration.min(-1).values, 0).clamp(max=threshold) * (
        ~env.command_manager.get_term(command_name).standing
    )


def landing_force(env, command_name):
    return (env.foot_contact_force.norm(dim=-1) * env.foot_first_contact).sum(-1) * (
        ~env.command_manager.get_term(command_name).standing
    )


def leg_posture_when_standing(env, command_name, asset_cfg):
    r = env.scene[asset_cfg.name]
    error = (
        (r.data.joint_pos.torch[:, asset_cfg.joint_ids] - r.data.default_joint_pos.torch[:, asset_cfg.joint_ids])
        .abs()
        .sum(-1)
    )
    return error * env.command_manager.get_term(command_name).standing


def minimum_foot_separation(env, asset_cfg, min_separation):
    r = env.scene[asset_cfg.name]
    vector = r.data.body_pos_w.torch[:, asset_cfg.body_ids[0]] - r.data.body_pos_w.torch[:, asset_cfg.body_ids[1]]
    lateral = quat_apply_inverse(yaw_quat(r.data.root_quat_w.torch), vector)[:, 1]
    return (min_separation - lateral).clamp_min(0).square()
