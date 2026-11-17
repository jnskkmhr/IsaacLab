# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Holosoma FastSAC configurations using the jnskkmhr/holosoma library branches.

Use ``env.actions=joint_range_action`` to map normalized actions to per-joint target ranges.
"""

from compositional_rsl_rl.algorithm import CompositionalFastSACAlgorithmCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, NetworkCfg
from compositional_rsl_rl.runner import CompositionalFastSACRunnerCfg

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp import symmetry


@configclass
class G1RoughFastSACRunnerCfg(CompositionalFastSACRunnerCfg):
    """Independent LayerNorm MLP actor and categorical Q ensemble for rigid locomotion."""

    seed = 1
    num_steps_per_env = 1
    updates_per_step = 8
    n_step = 1
    batch_size = 8192
    batch_size_is_global = True
    # Retain the rigid task's replay budget for its observation histories.
    replay_steps_per_env = None
    replay_capacity = 1_000_000
    replay_max_bytes = 16 * 1024**3
    learning_starts = 10
    warmup_transitions = 1
    max_iterations = 50_000
    save_interval = 500

    networks = NetworkCfg(
        actor=ActorHeadCfg(
            input=["policy", "command"],
            hidden_dims=[512, 256, 128],
            activation="swish",
            layer_norm=True,
            obs_normalization=False,
            distribution_cfg=ActorHeadCfg.SquashedGaussianDistributionCfg(
                log_std_min=-5.0,
                log_std_max=0.0,
                zero_init_output=True,
            ),
        ),
        critic=CriticHeadCfg(
            input=["critic", "command", "privileged"],
            hidden_dims=[768, 384, 192],
            activation="swish",
            layer_norm=True,
            num_ensembles=2,
            obs_normalization=False,
            distribution_cfg=CriticHeadCfg.CategoricalValueDistributionCfg(
                num_bins=101,
                value_min=-20.0,
                value_max=20.0,
            ),
        ),
    )
    algorithm = CompositionalFastSACAlgorithmCfg(
        # Append env.num_actions internally before the critic MLP.
        critic_action_input="critic",
        gamma=0.97,
        target_tau=0.125,
        actor_update_period=4,
        learning_rate=3e-4,
        num_bins=101,
        value_max=20.0,
        temperature_initial=0.001,
        target_entropy_ratio=0.0,
        use_autotune=True,
        weight_decay=0.001,
        max_grad_norm=0.0,
        # The learner normalizes observations, excluding critic action inputs.
        obs_normalization=True,
        # Keep policy/replay actions normalized; env.actions=joint_range_action owns physical scaling.
        action_scale=1.0,
        use_amp=False,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=symmetry.compute_symmetric_states,
            use_mirror_loss=False,
            mirror_loss_coeff=0.0,
        ),
    )

    logger = "wandb"
    wandb_project = "g1_29dof_rigid_rough_fast_sac"
    experiment_name = "g1_29dof_rigid_rough_fast_sac"


@configclass
class G1FlatFastSACRunnerCfg(G1RoughFastSACRunnerCfg):
    """Use the same observation contract with separate flat-terrain training logs."""

    wandb_project = "g1_29dof_rigid_flat_fast_sac"
    experiment_name = "g1_29dof_rigid_flat_fast_sac"
