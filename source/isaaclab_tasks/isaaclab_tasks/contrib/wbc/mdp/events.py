# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
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


def push_body(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice,
    force_range: dict[str, tuple[float, float]],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Apply a one-step world-frame wrench to selected bodies, regardless of foot contact.

    Each force component is sampled independently in newtons; omitted axes receive zero
    force. Torques are sampled in newton-metres. The instantaneous composer clears this
    wrench after the next physics write, so it does not persist until the next interval.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    num_envs = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
    body_ids = asset_cfg.body_ids
    num_bodies = len(range(asset.num_bodies)[body_ids]) if isinstance(body_ids, slice) else len(body_ids)
    ranges = torch.tensor([force_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")], device=asset.device)
    size = (num_envs, num_bodies, 3)
    forces = sample_uniform(ranges[:, 0], ranges[:, 1], size, asset.device)
    torques = sample_uniform(*torque_range, size, asset.device)
    asset.instantaneous_wrench_composer.add_forces_and_torques_index(
        forces=forces, torques=torques, body_ids=body_ids, env_ids=env_ids, is_global=True
    )


def update_ghost_pose(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | None) -> None:
    """Display commanded joint angles and world-frame pelvis pose, without changing the robot.

    Refresh all displayed targets, including after partial episode resets. The ghost
    is disabled during training by default and has no physics representation.
    """
    ghost = env.scene["ghost"]
    if not ghost.cfg.enabled:
        return
    reference = env.command_manager.get_term("whole_body")
    root_pose = torch.cat((reference.target_body_pos_w[:, 0], reference.target_body_quat_w[:, 0]), dim=-1)
    ghost.write_pose(root_pose, reference.command, reference.cfg.joint_names)
