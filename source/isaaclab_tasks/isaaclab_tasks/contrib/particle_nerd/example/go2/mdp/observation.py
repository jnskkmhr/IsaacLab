# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation and reset terms for the `go2` quadruped granular locomotion task."""

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager


def go2_base_pos(env) -> torch.Tensor:
    """Return Unitree Go2 trunk position relative to environment origin [num_envs, 3]."""
    pos = env.scene["robot"].data.root_pos_w.torch.clone()
    return pos - env.scene.env_origins


def reset_sand_particles(env, env_ids: torch.Tensor) -> None:
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
