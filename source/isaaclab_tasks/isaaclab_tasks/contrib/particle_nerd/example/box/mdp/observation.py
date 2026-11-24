# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation and reset terms for the `box` two-way coupling task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.managers import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import ActionTermCfg


class BoxWrenchAction(ActionTerm):
    """Apply world-frame 3D force and torque at the rigid box center of mass."""

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._scale = float(getattr(cfg, "scale", 250.0))
        self._actions = torch.zeros((env.num_envs, 6), device=env.device)
        self._processed = torch.zeros((env.num_envs, 6), device=env.device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor) -> None:
        self._actions.copy_(actions)
        clamped = torch.clamp(actions, -1.0, 1.0)
        self._processed[:, :3] = clamped[:, :3] * self._scale
        self._processed[:, 3:] = clamped[:, 3:] * (0.04 * self._scale)

    def apply_actions(self) -> None:
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._processed[:, None, :3].contiguous(),
            torques=self._processed[:, None, 3:].contiguous(),
            is_global=True,
        )


def box_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return box position relative to env origin and orientation quaternion [num_envs, 7]."""
    pose = env.scene["box"].data.root_pose_w.torch.clone()
    pose[:, :3] -= env.scene.env_origins
    return pose


def box_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return box world linear and angular velocity [num_envs, 6]."""
    return env.scene["box"].data.root_vel_w.torch


def box_target_error(env: ManagerBasedRLEnv, target_pos: tuple[float, float, float] = (0.0, 0.0, 0.25)) -> torch.Tensor:
    """Return 3D offset from box position (in env frame) to target support position."""
    pos = env.scene["box"].data.root_pose_w.torch[:, :3] - env.scene.env_origins
    target = pos.new_tensor(target_pos).unsqueeze(0)
    return target - pos


def reset_sand_particles(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Restore particle positions and reset MPM / Particle NeRD solver histories on reset."""
    ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    if ids.numel() == 0:
        return
    sand = env.scene["sand"]
    sand.write_nodal_state_to_sim_index(sand.data.default_nodal_state_w.torch[ids].clone(), env_ids=ids)
    mask = torch.zeros(env.num_envs + 1, dtype=torch.bool, device=env.device)
    mask[ids] = True
    mask_wp = None if len(ids) == env.num_envs else wp.from_torch(mask, dtype=wp.bool)
    solver = NewtonManager._solver
    if solver is not None:
        for state in (NewtonManager.get_state_1(), NewtonManager.get_state_0()):
            if state is not None:
                solver.reset(
                    state,
                    world_mask=mask_wp,
                    flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
                )
