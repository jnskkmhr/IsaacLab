# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset particle state and MPM history together between independent episodes."""

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonMPMManager


def reset_material(env, env_ids):
    """Restore material particles, then clear MPM and proxy-feedback history."""
    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    sand = env.scene["sand"]
    sand.write_particle_state_to_sim_index(sand.data.default_particle_state_w.torch[env_ids].clone(), env_ids=env_ids)
    mask = torch.zeros(env.num_envs + 1, dtype=torch.bool, device=env.device)
    mask[env_ids] = True
    NewtonMPMManager.reset_solver_state(
        world_mask=None if len(env_ids) == env.num_envs else wp.from_torch(mask, dtype=wp.bool),
        flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
    )
