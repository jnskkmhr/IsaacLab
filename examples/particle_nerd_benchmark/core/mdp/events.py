# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Restore particle positions and native constitutive/coupling history on reset."""

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonMPMManager


def reset_sand(env, env_ids):
    ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    sand = env.scene["sand"]
    sand.write_nodal_state_to_sim_index(sand.data.default_nodal_state_w.torch[ids].clone(), env_ids=ids)
    mask = torch.zeros(env.num_envs + 1, dtype=torch.bool, device=env.device)
    mask[ids] = True
    NewtonMPMManager.reset_solver_state(
        world_mask=None if len(ids) == env.num_envs else wp.from_torch(mask, dtype=wp.bool),
        flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
    )
