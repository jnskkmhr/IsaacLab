# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""World-frame forces and torques applied through the action manager."""

import torch

from isaaclab.managers import ActionTerm


class BodyWrenchAction(ActionTerm):
    """Apply [Fx, Fy, Fz, Tx, Ty, Tz] at the dynamic body's center of mass.

    Force units are N, torque units are N m. No pose or velocity is prescribed.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._actions = torch.zeros((env.num_envs, 6), device=env.device)

    @property
    def action_dim(self):
        return 6

    @property
    def raw_actions(self):
        return self._actions

    @property
    def processed_actions(self):
        return self._actions

    def process_actions(self, actions):
        self._actions.copy_(actions)

    def apply_actions(self):
        self._asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._actions[:, None, :3].contiguous(),
            torques=self._actions[:, None, 3:].contiguous(),
            is_global=True,
        )

    def reset(self, env_ids=None):
        self._actions[slice(None) if env_ids is None else env_ids] = 0
        self._asset.permanent_wrench_composer.reset(env_ids)
