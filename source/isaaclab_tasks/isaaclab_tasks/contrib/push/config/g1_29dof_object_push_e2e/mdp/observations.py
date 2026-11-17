# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-frame command and privileged box-contact observations."""

import torch

from isaaclab.managers import SceneEntityCfg

from .rewards import box_contact_force


def velocity_command(env, command_name: str) -> torch.Tensor:
    """Commanded forward, lateral, and yaw velocity in the robot frame.

    Omits the world-frame heading target stored by the command term, which a robot
    without global yaw estimation cannot use.
    """
    return env.command_manager.get_command(command_name)[:, :3]


def box_contact_state(
    env, left_sensor_cfg: SceneEntityCfg, right_sensor_cfg: SceneEntityCfg, other_sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Log-compressed box contact force on the left wrist, right wrist, and all other bodies."""
    forces = torch.stack(
        [box_contact_force(env, cfg) for cfg in (left_sensor_cfg, right_sensor_cfg, other_sensor_cfg)], dim=-1
    )
    return torch.log1p(forces)
