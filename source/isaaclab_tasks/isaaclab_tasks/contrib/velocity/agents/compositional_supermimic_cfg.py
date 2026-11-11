# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PR #255 learning schedule with composable deterministic network terms."""

from compositional_rsl_rl.algorithm import CompositionalSuperMimicAlgorithmCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, MLPNetworkCfg, NetworkCfg
from compositional_rsl_rl.runner import CompositionalSuperMimicRunnerCfg

from isaaclab.utils import configclass


@configclass
class SuperMimicNetworksCfg(NetworkCfg):
    actor = ActorHeadCfg(input=["policy"], hidden_dims=[2048, 1024, 512, 256], distribution_cfg=None)
    goal_encoder = MLPNetworkCfg(input=["goal"], hidden_dims=[2048, 1024, 512, 256], output_dim=32)
    critic = CriticHeadCfg(input=["critic"], hidden_dims=[256, 128], obs_normalization=True)
    world_model = MLPNetworkCfg(input=["dynamics"], hidden_dims=[2048, 2048, 2048], activation="swish", layer_norm=True)
    event_model = MLPNetworkCfg(input=["events"], hidden_dims=[512, 512], layer_norm=True)


@configclass
class G1SuperMimicRunnerCfg(CompositionalSuperMimicRunnerCfg):
    networks = SuperMimicNetworksCfg()
    algorithm = CompositionalSuperMimicAlgorithmCfg(
        task_cfg={"class_name": "isaaclab_tasks.core.velocity.supermimic:VelocityTask"},
        replay_buffer_size=4_096_000,
        max_step_reward=0.046,
    )
    obs_groups = {name: [name] for name in ("state", "goal", "context", "labels")}
    experiment_name = "g1_29dof_rigid_supermimic"
    wandb_project = "g1_29dof_rigid_supermimic"
