# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reconstruct teacher inputs without changing the student's joint-position offsets."""

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.string import resolve_matching_names_values


class joint_pos_reference(ManagerTermBase):
    """Joint positions relative to explicitly configured teacher defaults, in requested order."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        self.reference = torch.zeros(asset.num_joints, device=env.device)
        ids, _, values = resolve_matching_names_values(cfg.params["reference_positions"], asset.joint_names)
        self.reference[ids] = torch.tensor(values, device=env.device)

    def __call__(self, env, reference_positions: dict[str, float], asset_cfg: SceneEntityCfg):
        return (
            env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids] - self.reference[asset_cfg.joint_ids]
        )


def action_subset(env, indices: list[int]):
    """Read the previous student actions corresponding to one teacher's controlled joints."""
    return env.action_manager.action[:, indices]


def teacher_command(env, command_name):
    return env.command_manager.get_term(command_name).teacher_command


def hand_targets(env, command_name):
    return env.command_manager.get_term(command_name).hand_pose.flatten(1)


def object_pose_b(env, object_name="object", robot_name="robot"):
    from isaaclab.utils.math import subtract_frame_transforms

    robot, obj = env.scene[robot_name], env.scene[object_name]
    p, q = subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, obj.data.root_pos_w.torch, obj.data.root_quat_w.torch
    )
    return torch.cat((p, q), dim=-1)


def object_velocity_b(env, object_name="object", robot_name="robot"):
    from isaaclab.utils.math import quat_apply_inverse

    r, o = env.scene[robot_name], env.scene[object_name]
    q = r.data.root_quat_w.torch
    return torch.cat(
        (quat_apply_inverse(q, o.data.root_lin_vel_w.torch), quat_apply_inverse(q, o.data.root_ang_vel_w.torch)), dim=-1
    )


def body_pose_local(env, target_frame, parent_frame):
    from isaaclab.utils.math import subtract_frame_transforms

    r = env.scene[target_frame.name]
    parent = parent_frame.body_ids[0]
    p, q = subtract_frame_transforms(
        r.data.body_pos_w.torch[:, parent, None],
        r.data.body_quat_w.torch[:, parent, None].expand(-1, len(target_frame.body_ids), -1),
        r.data.body_pos_w.torch[:, target_frame.body_ids],
        r.data.body_quat_w.torch[:, target_frame.body_ids],
    )
    return torch.cat((p, q), dim=-1).flatten(1)
