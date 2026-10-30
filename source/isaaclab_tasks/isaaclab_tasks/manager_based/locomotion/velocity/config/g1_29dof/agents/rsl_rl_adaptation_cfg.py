# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPEncoderModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlTCNAttentionModelCfg,
    RslRlTCNModelCfg,
)


@configclass
class G1AdaptationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # max_iterations = 20_000
    max_iterations = 30_000
    save_interval = 500
    obs_groups = {"actor": ["policy"], "critic": ["critic", "privileged"], "privileged": ["privileged"]}
    actor = RslRlMLPEncoderModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        encoder_obs_set="privileged",
        encoder_output_dim=64,
        encoder_hidden_dims=[256, 128],
        encoder_activation="elu",
        encoder_obs_normalization=False,
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
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
    )
    logger = "wandb"
    wandb_project = "g1_29dof_adaptation"
    experiment_name = "g1_29dof_adaptation"


@configclass
class G1AdaptationDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    # num_steps_per_env = 120
    # max_iterations = 300
    # save_interval = 50
    num_steps_per_env = 24
    max_iterations = 10_000
    save_interval = 100
    logger = "wandb"
    wandb_project = "g1_29dof_adaptation_distillation"
    experiment_name = "g1_29dof_adaptation_distillation"
    obs_groups = {
        "student": ["policy"],
        "teacher": ["policy"],
        "teacher_privileged": ["teacher_privileged"],
        "student_encoder": ["student_encoder"],
    }
    teacher = RslRlMLPEncoderModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        encoder_obs_set="teacher_privileged",
        encoder_output_dim=64,
        encoder_hidden_dims=[256, 128],
        encoder_activation="elu",
    )
    student = RslRlTCNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        encoder_obs_set="student_encoder",
        encoder_output_dim=64,
        encoder_hidden_dims=[256, 128],
        encoder_activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )
