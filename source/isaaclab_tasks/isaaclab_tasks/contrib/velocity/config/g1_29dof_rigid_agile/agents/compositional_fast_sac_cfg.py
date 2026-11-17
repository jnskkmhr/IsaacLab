# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FastSAC on the Agile task, retaining its action bounds and replay reward curricula."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.agents.compositional_fast_sac_cfg import (
    G1RoughFastSACRunnerCfg as RigidFastSACRunnerCfg,
)
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp import symmetry

from .compositional_flash_sac_cfg import G1RoughFlashSACRunnerCfg


@configclass
class G1RoughFastSACRunnerCfg(RigidFastSACRunnerCfg):
    """Use Holosoma-style FastSAC MLPs with the Agile task's observation contract."""

    replay_cfg = G1RoughFlashSACRunnerCfg().replay_cfg
    max_iterations = 100_000
    wandb_project = "g1_29dof_rigid_agile_rough_fast_sac"
    experiment_name = "g1_29dof_rigid_agile_rough_fast_sac"

    def __post_init__(self):
        self.networks.actor.input = ["policy"]
        self.networks.critic.input = ["policy", "critic", "height_scan"]
        self.algorithm.symmetry_cfg.data_augmentation_func = symmetry.compute_symmetric_states
        # Keep the task's 0.5-radian environment multiplier and 1 -> 4 action-bound curriculum.
        # FastSAC multiplies its tanh output by action_scale * action_support.
        self.algorithm.curriculum_cfg = G1RoughFlashSACRunnerCfg().algorithm.curriculum_cfg
