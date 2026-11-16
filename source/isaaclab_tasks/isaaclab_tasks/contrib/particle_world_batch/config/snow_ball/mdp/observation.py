# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation, action, and reset terms for the `snow_ball` task (`example_mpm_snow_ball.py`)."""

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.managers import ActionTerm


class SnowBallWrenchAction(ActionTerm):
    """Apply world-frame 3D force and torque to the rolling snow-ball collider."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._scale = float(getattr(cfg, "scale", 40.0))
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
        self._processed = torch.clamp(actions, -1.0, 1.0) * self._scale

    def apply_actions(self) -> None:
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._processed[:, None, :3].contiguous(),
            torques=self._processed[:, None, 3:].contiguous(),
            is_global=True,
        )


def snow_ball_pose(env) -> torch.Tensor:
    """Return snow-ball position relative to env origin and orientation quaternion [num_envs, 7]."""
    pose = env.scene["snow_ball"].data.root_pose_w.torch.clone()
    pose[:, :3] -= env.scene.env_origins
    return pose


def snow_ball_lin_vel(env) -> torch.Tensor:
    """Return snow-ball world linear and angular velocity [num_envs, 6]."""
    return env.scene["snow_ball"].data.root_vel_w.torch


def snow_ball_slope_progress(env, target_x: float = 0.65) -> torch.Tensor:
    """Return remaining down-slope distance along +x and lateral y offset [num_envs, 2]."""
    pos = env.scene["snow_ball"].data.root_pose_w.torch[:, :3] - env.scene.env_origins
    dx = target_x - pos[:, 0]
    dy = -pos[:, 1]
    return torch.stack((dx, dy), dim=-1)


def reset_snow_particles(env, env_ids: torch.Tensor) -> None:
    """Restore snow pack particle positions and reset MPM / WorldParticle solver histories on reset."""
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
        for state in (NewtonManager.get_state_0(), NewtonManager.get_state_1()):
            if state is not None:
                solver.reset(
                    state,
                    world_mask=mask_wp,
                    flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
                )
