# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reference_finished(env: ManagerBasedRLEnv) -> torch.Tensor:
    reference = env.command_manager.get_term("whole_body")
    if reference.dataset.is_static:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return reference.frame >= reference.end_frame - 1


def fallen(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    height = robot.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return (height < 0.3) | (robot.data.projected_gravity_b.torch[:, 2] > -0.4)


def tracking_lost(env: ManagerBasedRLEnv) -> torch.Tensor:
    reference = env.command_manager.get_term("whole_body")
    error = reference.robot.data.root_pos_w.torch - reference.target_body_pos_w[:, 0]
    return (error.norm(dim=-1) > 0.5) & (reference.time_since_switch >= reference.cfg.tracking_grace_period)
