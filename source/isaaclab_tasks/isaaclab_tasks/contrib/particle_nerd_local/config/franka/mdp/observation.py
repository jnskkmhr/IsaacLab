# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation and reset terms for the `franka` sand-pushing task."""

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager


def franka_ee_pos(env) -> torch.Tensor:
    """Return `panda_hand` position relative to environment origin [num_envs, 3]."""
    robot = env.scene["robot"]
    hand_idx = robot.body_names.index("panda_hand")
    pos = robot.data.body_pos_w.torch[:, hand_idx, :3].clone()
    return pos - env.scene.env_origins


def reset_sand_particles(env, env_ids: torch.Tensor) -> None:
    """Restore particle positions and reset MPM / ParticleNeRD-Local solver histories on reset."""
    ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    if ids.numel() == 0:
        return
    sand = env.scene["sand"]
    solver = NewtonManager._solver
    if solver is not None and not getattr(sand, "_settled_default_synced", False):
        for entry in getattr(solver, "_entries", {}).values():
            if hasattr(entry.solver, "settled_particle_q"):
                q_set = wp.to_torch(entry.solver.settled_particle_q).view(env.num_envs, -1, 3)
                sand.data.default_nodal_state_w.torch[:, :, :3].copy_(q_set)
                sand._settled_default_synced = True
                break
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
