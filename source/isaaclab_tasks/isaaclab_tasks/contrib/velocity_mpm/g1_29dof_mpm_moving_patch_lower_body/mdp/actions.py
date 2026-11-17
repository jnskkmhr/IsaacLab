# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AGILE random arm trajectories with episode-local trajectory state."""

import torch

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp.actions import (
    RandomActionCfg,
    RandomPositionAction,
)


class RandomArmAction(RandomPositionAction):
    """Reset arm trajectories together with the selected robot instances."""

    def reset(self, env_ids=None):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        super().reset(env_ids)
        self._velocity_profile.reset(env_ids)
        initial = self._offset[env_ids]
        self._processed_actions[env_ids] = initial
        self._target_processed_actions[env_ids] = initial
        self._velocity_profile.set_target(initial, initial, env_ids)
        self._time_since_last_sample[env_ids] = 0.0
        self._target_write_pending = True


@configclass
class RandomArmActionCfg(RandomActionCfg):
    class_type: type = RandomArmAction
