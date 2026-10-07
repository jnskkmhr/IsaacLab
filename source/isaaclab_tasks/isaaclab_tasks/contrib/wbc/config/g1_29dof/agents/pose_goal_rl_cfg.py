# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Asymmetric PPO for the G1 WBC 2.0 terminal-goal actor."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from ....mdp.pose_goal_symmetry import compute_mirrored_states


@configclass
class G1PoseGoalRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Train direct goal reaching; privileged forces are restricted to the critic."""

    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 250
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False)
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

    experiment_name = "g1_29dof_wbc_pose_goal"
    logger = "wandb"
    wandb_project = "g1_29dof_wbc_pose_goal"
