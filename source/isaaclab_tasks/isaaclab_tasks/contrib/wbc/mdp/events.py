# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_from_reference(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice) -> None:
    """Sample once and initialize joints and floating base from that exact reference frame."""
    reference = env.command_manager.get_term("whole_body")
    reference.sample(env_ids)
    frames = reference.frame[env_ids]
    dataset = reference.dataset
    robot = reference.robot
    joint_position = robot.data.default_joint_pos.torch[env_ids].clone()
    joint_velocity = torch.zeros_like(joint_position)
    joint_position[:, reference.joint_ids] = dataset.joint_pos[frames]
    joint_velocity[:, reference.joint_ids] = dataset.joint_vel[frames]
    root_state = torch.cat(
        (
            dataset.body_pos_w[frames, 0] + env.scene.env_origins[env_ids],
            dataset.body_quat_w[frames, 0],
            dataset.body_lin_vel_w[frames, 0],
            dataset.body_ang_vel_w[frames, 0],
        ),
        dim=-1,
    )
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
