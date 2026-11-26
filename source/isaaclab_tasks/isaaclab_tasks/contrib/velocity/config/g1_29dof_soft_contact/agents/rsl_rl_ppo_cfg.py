# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPM MLP PPO settings, with augmentation and the requested 0.1 mirror loss."""

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.mdp import compute_mirrored_states

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Train independent MLP actor and critic heads with standard PPO."""

    num_steps_per_env = 24
    max_iterations = 20_000
    save_interval = 500
    obs_groups = {
        "actor": ["policy", "command"],
        "critic": ["critic", "command", "privileged"],
    }
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
            data_augmentation_func=compute_mirrored_states,
        ),
    )
    check_for_nan = True
    run_name = "rft_flat_mpm_align_box"
    logger = "wandb"

    wandb_project = "g1_29dof_contact_benchmark"
    experiment_name = "g1_29dof_contact_benchmark"


@configclass
class G1RoughPPORunnerCfg(G1FlatPPORunnerCfg):
    """Same PPO and symmetry settings on rough terrain."""

    run_name = "rft_rough_mpm_align_box"
