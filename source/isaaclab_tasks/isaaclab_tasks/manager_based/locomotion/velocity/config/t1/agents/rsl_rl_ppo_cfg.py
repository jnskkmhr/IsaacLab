# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symmetry import t1


@configclass
class T1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "t1_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        # class_name="ActorCriticRecurrent",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class T1FlatPPORunnerCfg(T1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 30000
        self.experiment_name = "t1_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]

        self.logger = "wandb"
        self.wandb_project = "t1_flat_zh"
        self.experiment_name = "t1_flat_zh"
        self.run_name = "t1_flat_zh"


@configclass
class T1FlatPPORunnerWithSymmetryCfg(T1FlatPPORunnerCfg):
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
            use_data_augmentation=True, data_augmentation_func=t1.compute_symmetric_states
        ),
    )