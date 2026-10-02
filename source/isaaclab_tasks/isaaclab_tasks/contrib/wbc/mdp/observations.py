# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher and student contracts, with task targets in the measured pelvis heading frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_conjugate, quat_mul, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def proprioception(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Concatenate angular velocity, gravity, named joint state, and previous actions (93 values)."""
    reference = env.command_manager.get_term("whole_body")
    state = reference.robot.data
    return torch.cat(
        (
            state.root_ang_vel_b.torch,
            state.projected_gravity_b.torch,
            state.joint_pos.torch[:, reference.joint_ids] - state.default_joint_pos.torch[:, reference.joint_ids],
            state.joint_vel.torch[:, reference.joint_ids] * 0.05,
            env.action_manager.action,
        ),
        dim=-1,
    )


def task_space_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Five target poses as XYZ displacement plus the first two rotation-matrix columns.

    Targets are expressed in the current yaw-aligned pelvis frame. Generating these errors from
    world-frame goals requires a pelvis pose estimate at deployment. No reference joint state
    enters this observation. Rotation columns are flattened in row-major order.
    """
    reference = env.command_manager.get_term("whole_body")
    robot = reference.robot
    heading = yaw_quat(robot.data.root_quat_w.torch)[:, None, :].expand(-1, 5, -1)
    position = quat_apply_inverse(heading, reference.target_body_pos_w - robot.data.root_pos_w.torch[:, None, :])
    orientation = quat_mul(quat_conjugate(heading), reference.target_body_quat_w)
    rotation_columns = matrix_from_quat(orientation)[..., :2].reshape(env.num_envs, 5, 6)
    return torch.cat((position, rotation_columns), dim=-1).flatten(1)


def student_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Proprioception and five task-space targets: 138 values."""
    return torch.cat((proprioception(env), task_space_command(env)), dim=-1)


def teacher_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The student observation plus privileged reference joint positions: 167 values."""
    return torch.cat((student_observation(env), env.command_manager.get_term("whole_body").command), dim=-1)


def critic_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Teacher observation plus base velocity, reference joint velocity, and support labels."""
    reference = env.command_manager.get_term("whole_body")
    return torch.cat(
        (
            teacher_observation(env),
            reference.robot.data.root_lin_vel_b.torch,
            reference.dataset.joint_vel[reference.frame] * 0.05,
            reference.dataset.foot_contact[reference.frame],
        ),
        dim=-1,
    )
