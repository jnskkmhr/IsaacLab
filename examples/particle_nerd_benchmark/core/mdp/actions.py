# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prescribed object motion and external wrench actions."""

import torch

from isaaclab.managers import ActionTerm


class ObjectStateAction(ActionTerm):
    """Set a kinematic collider pose (xyzw) and world-frame linear/angular velocity."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._actions = torch.zeros((env.num_envs, 13), device=env.device)
        self._actions[:, 6] = 1

    @property
    def action_dim(self):
        return 13

    @property
    def raw_actions(self):
        return self._actions

    @property
    def processed_actions(self):
        return self._actions

    def process_actions(self, actions):
        self._actions.copy_(actions)

    def apply_actions(self):
        pose = self._actions[:, :7].clone()
        pose[:, :3] += self._env.scene.env_origins
        self._asset.write_root_pose_to_sim_index(root_pose=pose)
        self._asset.write_root_velocity_to_sim_index(root_velocity=self._actions[:, 7:].contiguous())


class BodyWrenchAction(ActionTerm):
    """Apply world-frame force/torque at the dynamic object's centre of mass."""

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
