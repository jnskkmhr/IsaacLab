# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Agile FlashSAC settings with independent actor and critic residual networks."""

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.agents.compositional_flash_sac_cfg import (
    G1RoughFlashSACRunnerCfg,
)


@configclass
class G1FlatFlashSACRunnerCfg(G1RoughFlashSACRunnerCfg):
    """Match native learner settings and concatenate policy history with privileged state."""

    experiment_name = "g1_29dof_rigid_agile_native_flash_sac"
    wandb_project = "g1_29dof_rigid_agile_native_flash_sac"

    def __post_init__(self):
        self.networks.critic_encoder.input = ["policy", "critic"]
