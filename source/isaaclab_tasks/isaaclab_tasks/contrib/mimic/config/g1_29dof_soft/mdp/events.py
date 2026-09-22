# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Disturbances for soft-contact motion tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.events import push_by_setting_velocity
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.mdp.observations import foot_contact_hybrid

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def push_by_setting_velocity_in_stance(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    sensor_cfg: SceneEntityCfg,
    command_name: str = "motion",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Apply interval pushes only during fully weighted stance with at least one grounded foot.

    Contact is selected from the active rigid or soft solver. Events outside stance or without
    foot contact are skipped, not deferred to touchdown. The stance blend is excluded so pushes
    do not interfere with the transition to motion tracking.

    Args:
        env: Environment to perturb.
        env_ids: Environments whose push interval has elapsed.
        velocity_range: Per-axis velocity increment ranges [m/s] for x/y/z and [rad/s] for
            roll/pitch/yaw. Omitted axes receive no increment.
        sensor_cfg: Rigid contact sensor with foot bodies in soft-solver foot order.
        command_name: Motion command defining the stance intervals.
        asset_cfg: Robot receiving the velocity increment.
    """
    command = env.command_manager.get_term(command_name)
    env_ids = env_ids[command.standing_weight[env_ids] >= 1.0 - 1.0e-6]
    if env_ids.numel() == 0:
        return
    contacts = foot_contact_hybrid(env, rigid_contact_sensor_cfg=sensor_cfg)
    env_ids = env_ids[contacts[env_ids].bool().any(dim=-1)]
    if env_ids.numel() == 0:
        return
    push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
