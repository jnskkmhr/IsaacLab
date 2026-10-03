# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from isaaclab_tasks.contrib.velocity.config.vel_mdp import compute_mirrored_states


def teacher_model() -> RslRlMLPModelCfg:
    """Keep the PPO actor and distillation teacher checkpoint architectures identical."""
    return RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )


@configclass
class G1WholeBodyTeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 250
    obs_groups = {"actor": ["teacher"], "critic": ["critic"]}
    actor = teacher_model()
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=compute_mirrored_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
        ),
    )
    experiment_name = "g1_29dof_wbc_teacher"
    logger = "wandb"
    wandb_project = "g1_29dof_wbc_teacher"


@configclass
class G1WholeBodyStudentRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 250
    obs_groups = {"student": ["student"], "teacher": ["teacher"]}
    teacher = teacher_model()
    student = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5, learning_rate=1.0e-3, gradient_length=24, max_grad_norm=1.0
    )
    experiment_name = "g1_29dof_wbc_student"
    logger = "wandb"
    wandb_project = "g1_29dof_wbc_student"
