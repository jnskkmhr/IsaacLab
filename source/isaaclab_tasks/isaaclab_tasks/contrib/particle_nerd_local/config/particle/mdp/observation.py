# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation and reset terms for the `particle` task."""

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager


def block_center_of_mass(env) -> torch.Tensor:
    """Return the mean particle position of the sand block relative to the env origin [num_envs, 3]."""
    positions = wp.to_torch(NewtonManager.get_state_0().particle_q).view(env.num_envs, -1, 3)
    return positions.mean(dim=1) - env.scene.env_origins


def reset_particle_block(env, env_ids: torch.Tensor, velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
    """Restore the sand block to its release pose with a uniform initial velocity and reset solver history.

    Args:
        env: The environment.
        env_ids: Environments to reset.
        velocity: World-frame particle velocity at release (m/s). The Particle NeRD solver encodes it in its
            position history; implicit MPM uses it directly.
    """
    ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    if ids.numel() == 0:
        return
    sand = env.scene["sand"]
    default_state = sand.data.default_nodal_state_w.torch
    default_state[:, :, 3:6] = default_state.new_tensor(velocity)
    sand.write_nodal_state_to_sim_index(default_state[ids].clone(), env_ids=ids)
    mask = torch.zeros(env.num_envs + 1, dtype=torch.bool, device=env.device)
    mask[ids] = True
    mask_wp = None if len(ids) == env.num_envs else wp.from_torch(mask, dtype=wp.bool)
    solver = NewtonManager._solver
    if solver is not None:
        for state in (NewtonManager.get_state_1(), NewtonManager.get_state_0()):
            if state is not None:
                solver.reset(state, world_mask=mask_wp, flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE)
