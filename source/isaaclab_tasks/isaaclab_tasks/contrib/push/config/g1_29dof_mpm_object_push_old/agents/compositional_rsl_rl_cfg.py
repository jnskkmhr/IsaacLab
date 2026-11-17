# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compositional PPO with whole-body proprioception and hand/object observations."""

from compositional_rsl_rl.algorithm import CompositionalPPOAlgorithmCfg
from compositional_rsl_rl.loss import LossesCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, NetworkCfg
from compositional_rsl_rl.runner import CompositionalRunnerCfg

from isaaclab.utils.configclass import configclass


@configclass
class G1ObjectPushPPORunnerCfg(CompositionalRunnerCfg):
    """Train independent actor and critic MLPs with task-local pushing settings."""

    env_interface = None  # Flat terrain has no terrain-level curriculum to checkpoint.
    num_steps_per_env = 24
    max_iterations = 20_000
    save_interval = 500
    networks = NetworkCfg(
        actor=ActorHeadCfg(
            input=["policy", "command", "manipulation"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=CriticHeadCfg(
            input=["critic", "command", "privileged", "manipulation", "object_privileged"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
        ),
    )
    loss = LossesCfg()
    algorithm = CompositionalPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,  # Box and hand mirror transforms are not defined.
    )
    logger = "wandb"
    wandb_project = "g1_29dof_mpm_object_push"
    experiment_name = "g1_29dof_mpm_object_push"
