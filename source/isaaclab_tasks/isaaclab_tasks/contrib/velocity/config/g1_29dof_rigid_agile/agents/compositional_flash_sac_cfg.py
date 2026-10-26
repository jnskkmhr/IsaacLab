# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compositional FlashSAC examples for G1 locomotion with rigid contacts.

Uses the FlashSAC-enabled RSL-RL fork and compositional extension through OffPolicyRunner.
"""

from compositional_rsl_rl.algorithm import CompositionalFlashSACAlgorithmCfg, SchedulerCfg
from compositional_rsl_rl.components.network import ResidualNetworkCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, NetworkCfg
from compositional_rsl_rl.runner import CompositionalOffPolicyRunnerCfg

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from ..mdp import symmetry


@configclass
class G1FlashSACNetworksCfg(NetworkCfg):
    """Independent residual feature terms feeding actor and Q output heads."""

    actor_encoder: ResidualNetworkCfg | None = None
    critic_encoder: ResidualNetworkCfg | None = None


@configclass
class G1RoughFlashSACRunnerCfg(CompositionalOffPolicyRunnerCfg):
    """Reference residual actor and categorical Q ensemble for the rough-terrain observation groups."""

    seed = 1
    max_time_s = 3500.0
    save_interval_updates = 5000
    replay_cfg = {
        "class_name": (
            "isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp.curriculums:ReplayRewardCurriculum"
        ),
    }
    num_steps_per_env = 1
    updates_per_step = 4
    n_step = 3
    batch_size = 2048
    # Match Agile capacity, subject to its 3 GiB allocation budget.
    replay_capacity = 1_000_000
    replay_max_bytes = 3 * 1024**3
    warmup_transitions = 100_000
    max_iterations = 1_000_000_000
    save_interval = 1250  # 5000 learner updates after warmup.

    networks = G1FlashSACNetworksCfg(
        actor_encoder=ResidualNetworkCfg(
            input=["policy"],
            output_dim=128,
            num_blocks=2,
        ),
        critic_encoder=ResidualNetworkCfg(
            input=["policy", "critic"],
            output_dim=256,
            num_blocks=2,
            num_ensembles=2,
        ),
        actor=ActorHeadCfg(
            class_name="compositional_rsl_rl.components.network:LinearNetwork",
            input=["actor_encoder"],
            hidden_dims=[],
            obs_normalization=False,
            distribution_cfg=ActorHeadCfg.SquashedGaussianDistributionCfg(
                log_std_min=-10.0,
                log_std_max=2.0,
            ),
        ),
        critic=CriticHeadCfg(
            class_name="compositional_rsl_rl.components.network:LinearNetwork",
            input=["critic_encoder"],
            hidden_dims=[],
            obs_normalization=False,
            distribution_cfg=CriticHeadCfg.CategoricalValueDistributionCfg(
                num_bins=101,
                value_min=-5.0,
                value_max=5.0,
            ),
        ),
    )
    # Learner hyperparameters from agile/algorithms/flash_sac/learner.py:FlashSACConfig.
    algorithm = CompositionalFlashSACAlgorithmCfg(
        # Append env.num_actions internally before encoding the critic inputs.
        critic_action_input="critic_encoder",
        gamma=0.99,
        target_tau=0.01,
        actor_update_period=2,
        learning_rate=3e-4,
        final_learning_rate=1.5e-4,
        decay_updates=100_000,
        num_bins=101,
        value_max=5.0,
        temperature_initial=0.01,
        temperature_target_sigma=0.15,
        normalize_reward=True,
        noise_zeta_exponent=2.0,
        noise_max_repeat=16,
        action_support=1.0,
        curriculum_cfg={
            "action_support": SchedulerCfg(
                type="linear",
                params={
                    "start_step": 5000,
                    "end_step": 10000,
                    "start_value": 1.0,
                    "end_value": 4.0,
                },
            ),
        },
        use_amp=False,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=symmetry.compute_symmetric_states,
            # Agile pairs replay samples with their mirrors without a policy mirror penalty.
            use_mirror_loss=False,
            mirror_loss_coeff=0.0,
        ),
    )

    logger = "wandb"
    wandb_project = "g1_29dof_rigid_agile_rough_flash_sac"
    experiment_name = "g1_29dof_rigid_agile_rough_flash_sac"


@configclass
class G1FlatFlashSACRunnerCfg(G1RoughFlashSACRunnerCfg):
    """Use the same observation contract with separate flat-terrain training logs."""

    wandb_project = "g1_29dof_rigid_agile_flat_flash_sac"
    experiment_name = "g1_29dof_rigid_agile_flat_flash_sac"
