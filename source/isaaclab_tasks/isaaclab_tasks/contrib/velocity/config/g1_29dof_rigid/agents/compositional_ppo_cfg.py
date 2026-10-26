# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compositional PPO configurations for G1 locomotion with rigid contacts."""

from compositional_rsl_rl.algorithm import CompositionalPPOAlgorithmCfg
from compositional_rsl_rl.loss import LossesCfg
from compositional_rsl_rl.network import (
    ActorHeadCfg,
    CriticHeadCfg,
    NetworkCfg,
    RecurrentActorHeadCfg,
    RecurrentCriticHeadCfg,
)
from compositional_rsl_rl.runner import CompositionalRunnerCfg

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp import symmetry


@configclass
class G1RoughPPORunnerCfg(CompositionalRunnerCfg):
    """Independent actor and critic MLPs for rigid rough-terrain locomotion."""

    num_steps_per_env = 24
    max_iterations = 30_000
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
            data_augmentation_func=symmetry.compute_symmetric_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
        ),
    )
    logger = "wandb"
    neptune_project = "isaaclab"
    wandb_project = "g1_29dof_rigid_rough_ppo"
    experiment_name = "g1_29dof_rigid_rough_ppo"


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()  # pyright: ignore[reportAttributeAccessIssue]

        self.max_iterations = 30_000
        self.wandb_project = "g1_29dof_rigid_flat_ppo"
        self.experiment_name = "g1_29dof_rigid_flat_ppo"


@configclass
class G1FlatPPORunnerRecurrentCfg(G1FlatPPORunnerCfg):
    """Independent LSTM actor and critic with the original flat-task recurrent settings."""

    networks = NetworkCfg(
        actor=RecurrentActorHeadCfg(
            input=["policy", "command"],
            hidden_dims=[128, 128, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=1.0),
            rnn_type="lstm",
            rnn_hidden_dim=256,
            rnn_num_layers=1,
        ),
        critic=RecurrentCriticHeadCfg(
            input=["critic", "command", "privileged"],
            hidden_dims=[128, 128, 128],
            activation="elu",
            obs_normalization=False,
            rnn_type="lstm",
            rnn_hidden_dim=256,
            rnn_num_layers=1,
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # RSL-RL does not support symmetry augmentation for recurrent policies.
        self.algorithm.symmetry_cfg = None
