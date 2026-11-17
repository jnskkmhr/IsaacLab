# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Moving-patch pushing environment with a bounded step reward."""

import torch

from isaaclab_tasks.contrib.push.config.g1_29dof_mpm_moving_patch_object_push.object_push_env import (
    G1MovingPatchObjectPushEnv,
)


class G1MovingPatchObjectPushE2EEnv(G1MovingPatchObjectPushEnv):
    """Bound the step reward so one blown-up simulation step cannot dominate a training batch.

    The step in which the coupled solver blows up is still rewarded before the episode is ended by
    ``implausible_velocity``, and its squared-velocity penalties can reach 1e9.
    """

    def step(self, action):
        obs, reward, terminated, truncated, extras = super().step(action)
        limit = self.cfg.step_reward_limit
        reward = torch.nan_to_num(reward, nan=0.0, posinf=limit, neginf=-limit).clamp(-limit, limit)
        return obs, reward, terminated, truncated, extras
