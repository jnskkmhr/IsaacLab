# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def modify_reward_std(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, std: float, num_steps: int):
    """Curriculum that modifies a exponential reward std a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        std: The std of the exponential reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params["std"] = std
        env.reward_manager.set_term_cfg(term_name, term_cfg)

def modify_command_range(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, ranges: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.command_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.ranges = ranges
        env.command_manager.set_term_cfg(term_name, term_cfg)