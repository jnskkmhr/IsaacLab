# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner for the G1 29-DoF locomotion task among trees."""

from isaaclab.utils.configclass import configclass

from ...g1_29dof_rigid.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class G1TreePPORunnerCfg(G1FlatPPORunnerCfg):
    """Flat-ground PPO settings, logged under their own run name."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # self.wandb_project = "g1_29dof_tree"
        # self.experiment_name = "g1_29dof_tree"
        self.wandb_project = "g1_29dof_soft_vanilla_ppo"
        self.experiment_name = "g1_29dof_soft_vanilla_ppo"


@configclass
class G1BarPPORunnerCfg(G1FlatPPORunnerCfg):
    """Flat-ground PPO settings for the bar task, logged under their own run name."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # self.wandb_project = "g1_29dof_tree_bar"
        # self.experiment_name = "g1_29dof_tree_bar"

        self.wandb_project = "g1_29dof_soft_vanilla_ppo"
        self.experiment_name = "g1_29dof_soft_vanilla_ppo"
