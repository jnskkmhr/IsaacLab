# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MLP actor and critic trained with compositional PPO and resumable curriculum progress."""

from compositional_rsl_rl.algorithm import CompositionalPPOAlgorithmCfg
from compositional_rsl_rl.env_interface import EnvInterfaceCfg
from compositional_rsl_rl.loss import LossesCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, NetworkCfg
from compositional_rsl_rl.runner import CompositionalRunnerCfg

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.mdp import compute_mirrored_states

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg


@configclass
class G1MovingPatchPPORunnerCfg(CompositionalRunnerCfg):
    """Keep the moving-patch policy inputs and PPO settings with two independent MLP heads."""

    env_interface = EnvInterfaceCfg(restore_terrain=True)
    num_steps_per_env = 24
    max_iterations = 20_000
    save_interval = 500
    networks = NetworkCfg(
        actor=ActorHeadCfg(
            input=["policy", "command"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=CriticHeadCfg(
            input=["critic", "command", "privileged"],
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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=compute_mirrored_states,
        ),
    )
    logger = "wandb"
    wandb_project = "g1_29dof_mpm_terrain"
    experiment_name = "g1_29dof_mpm_terrain"


@configclass
class G1MixedTerrainPPORunnerCfg(G1MovingPatchPPORunnerCfg):
    """Keep the existing actor/critic contract and separate mixed-terrain training logs."""

    experiment_name = "g1_29dof_mpm_mixed_terrain"
    wandb_project = "g1_29dof_mpm_mixed_terrain"
