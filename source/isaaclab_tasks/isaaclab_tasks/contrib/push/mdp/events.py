# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object-pushing events."""

import torch

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, yaw_quat


def reset_box(
    env,
    env_ids,
    distance_range: tuple[float, float] = (0.85, 0.95),
    lateral_range: tuple[float, float] = (0.0, 0.0),
    yaw_range: tuple[float, float] = (0.0, 0.0),
    object_name: str = "box",
    spawn_height: float | None = None,
):
    """Sample an upright box pose relative to the newly reset robot's heading.

    Forward and lateral offsets are in meters; yaw is in radians relative to the
    robot's heading. The box bottom starts 1 cm above the environment origin's
    ground height, with zero linear and angular velocity.
    """
    box, robot = env.scene[object_name], env.scene["robot"]
    ids = slice(None) if env_ids is None else env_ids
    position = robot.data.root_pos_w.torch[ids]
    orientation = yaw_quat(robot.data.root_quat_w.torch[ids])
    offset = torch.zeros_like(position)
    offset[:, 0].uniform_(*distance_range)
    offset[:, 1].uniform_(*lateral_range)
    yaw = torch.empty(len(position), device=position.device, dtype=position.dtype).uniform_(*yaw_range)
    zeros = torch.zeros_like(yaw)
    box_orientation = quat_mul(orientation, quat_from_euler_xyz(zeros, zeros, yaw))
    pose = torch.cat((position + quat_apply(orientation, offset), box_orientation), dim=-1)
    if spawn_height is None:
        spawn_height = getattr(env.cfg.scene, object_name).spawn.size[2] / 2 + 0.01
    pose[:, 2] = env.scene.env_origins[ids, 2] + spawn_height
    box.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    box.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(pose), 6), device=env.device), env_ids=ids)
