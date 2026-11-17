# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standard PPO for whole-body G1 pushing on moving-patch MPM sand; same networks as the rigid-ground task."""

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.mdp import compute_mirrored_states

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

MIN_ACTION_STD = 0.5
"""Lower bound of the action noise standard deviation during sand fine-tuning."""


@configclass
class G1ObjectPushPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Train independent MLP actor and critic heads with standard PPO."""

    num_steps_per_env = 24
    max_iterations = 20_000
    save_interval = 500
    obs_groups = {
        "actor": ["policy", "command"],
        "critic": ["critic", "critic_command", "privileged", "object_privileged"],
    }
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        # The floor keeps exploration at the rigid-ground policy's level (mean std 0.56 at convergence);
        # without it the std fell to about 0.3 on sand while the policy settled on standing.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_range=(MIN_ACTION_STD, 1.0e6)),
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
        # Every observation and action term declares a reflection rule.
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=compute_mirrored_states,
        ),
    )
    logger = "wandb"

    wandb_project = "g1_29dof_mpm_moving_patch_object_push_e2e"
    experiment_name = "g1_29dof_mpm_moving_patch_object_push_e2e"
