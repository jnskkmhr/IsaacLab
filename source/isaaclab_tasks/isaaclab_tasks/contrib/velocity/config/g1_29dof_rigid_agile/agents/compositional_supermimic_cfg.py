# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SuperMimic with Agile's actor/critic inputs, rewards and 15 policy actions."""

from compositional_rsl_rl.algorithm import CompositionalSuperMimicAlgorithmCfg

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.agents.compositional_supermimic_cfg import (
    G1SuperMimicRunnerCfg,
    SuperMimicNetworksCfg,
)


@configclass
class G1RoughSuperMimicRunnerCfg(G1SuperMimicRunnerCfg):
    networks = SuperMimicNetworksCfg()
    networks.goal_encoder = None
    algorithm = CompositionalSuperMimicAlgorithmCfg(
        task_cfg={"class_name": "isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp.supermimic:AgileTask"},
        replay_buffer_size=4_096_000,
        # dt * (linear tracking 5 + angular tracking 7.5 + height 2.5 + orientation 5).
        max_step_reward=0.4,
    )
    obs_groups = {name: [name] for name in ("state", "goal", "context", "labels", "prev_action", "prev_prev_action")}
    seed = 1
    logger = "wandb"
    experiment_name = "g1_29dof_rigid_agile_supermimic"
    wandb_project = "g1_29dof_rigid_agile_supermimic"
