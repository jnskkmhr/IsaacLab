# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination for simulation blow-ups that leave the robot state finite but implausible."""

import torch

from isaaclab.managers import SceneEntityCfg


def robot_velocity_implausible(
    env,
    max_joint_vel: float,
    max_root_lin_vel: float,
    max_root_ang_vel: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when any robot velocity is non-finite or far beyond what the hardware can reach.

    A coupled-solver blow-up can leave velocities finite but many orders of magnitude too large.
    The squared-velocity and acceleration penalties then dominate the return of a whole batch.

    Args:
        env: The environment.
        max_joint_vel: Joint speed limit [rad/s].
        max_root_lin_vel: Base linear speed limit [m/s].
        max_root_ang_vel: Base angular speed limit [rad/s].
        asset_cfg: The robot.
    """
    data = env.scene[asset_cfg.name].data
    joint = data.joint_vel.torch.abs().amax(dim=-1)
    linear = torch.linalg.vector_norm(data.root_lin_vel_w.torch, dim=-1)
    angular = torch.linalg.vector_norm(data.root_ang_vel_w.torch, dim=-1)
    # Negated comparisons so that NaN counts as implausible.
    return ~((joint <= max_joint_vel) & (linear <= max_root_lin_vel) & (angular <= max_root_ang_vel))
