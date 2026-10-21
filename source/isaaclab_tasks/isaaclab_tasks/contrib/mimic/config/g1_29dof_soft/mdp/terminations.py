# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tracking-error terminations with soft-task stance gating."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.mimic.mdp import terminations as mimic_terminations

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_anchor_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, only_when_tracking: bool = True
) -> torch.Tensor:
    """Check anchor position error, optionally excluding stance intervals."""
    bad = mimic_terminations.bad_anchor_pos(env, command_name, threshold)
    return _gate_on_tracking(env, command_name, bad, only_when_tracking)


def bad_anchor_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, only_when_tracking: bool = True
) -> torch.Tensor:
    """Check anchor height error, optionally excluding stance intervals."""
    bad = mimic_terminations.bad_anchor_pos_z_only(env, command_name, threshold)
    return _gate_on_tracking(env, command_name, bad, only_when_tracking)


def bad_anchor_ori(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
    only_when_tracking: bool = True,
) -> torch.Tensor:
    """Check anchor orientation error, optionally excluding stance intervals."""
    bad = mimic_terminations.bad_anchor_ori(env, asset_cfg, command_name, threshold)
    return _gate_on_tracking(env, command_name, bad, only_when_tracking)


def bad_motion_body_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    body_names: list[str] | None = None,
    only_when_tracking: bool = True,
) -> torch.Tensor:
    """Check body position error, optionally excluding stance intervals."""
    bad = mimic_terminations.bad_motion_body_pos(env, command_name, threshold, body_names)
    return _gate_on_tracking(env, command_name, bad, only_when_tracking)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    body_names: list[str] | None = None,
    only_when_tracking: bool = True,
) -> torch.Tensor:
    """Check body height error, optionally excluding stance intervals."""
    bad = mimic_terminations.bad_motion_body_pos_z_only(env, command_name, threshold, body_names)
    return _gate_on_tracking(env, command_name, bad, only_when_tracking)


def _gate_on_tracking(
    env: ManagerBasedRLEnv, command_name: str, terminated: torch.Tensor, only_when_tracking: bool
) -> torch.Tensor:
    """Suppress tracking failures during stance; fall termination remains active."""
    if not only_when_tracking:
        return terminated
    return terminated & ~env.command_manager.get_term(command_name).is_stance
