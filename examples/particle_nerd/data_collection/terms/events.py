# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset particle state and MPM history together between independent episodes."""

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonMPMManager


def reset_material(env, env_ids, jitter_fraction=0.0, particle_spacing=(0.016, 0.016, 0.016)):
    """Restore material particles, then clear MPM and proxy-feedback history."""
    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    sand = env.scene["sand"]
    state = sand.data.default_particle_state_w.torch[env_ids].clone()
    if jitter_fraction:
        # Independent per-episode offsets; never inject noise during integration.
        spacing = torch.tensor(particle_spacing, dtype=state.dtype, device=env.device)
        state[..., :3] += (2 * torch.rand_like(state[..., :3]) - 1) * spacing * jitter_fraction
    sand.write_particle_state_to_sim_index(state, env_ids=env_ids)
    mask = torch.zeros(env.num_envs + 1, dtype=torch.bool, device=env.device)
    mask[env_ids] = True
    NewtonMPMManager.reset_solver_state(
        world_mask=None if len(env_ids) == env.num_envs else wp.from_torch(mask, dtype=wp.bool),
        flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
    )
