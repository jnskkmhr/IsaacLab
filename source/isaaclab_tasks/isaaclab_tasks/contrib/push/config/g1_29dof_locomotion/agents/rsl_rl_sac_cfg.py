# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native RSL-RL SAC for G1 locomotion with independently trained Q critics."""

# Disabled while using the pre-SAC SuperMimic RSL-RL revision.
# Restore the SAC-compatible dependencies before enabling this config and its registration.

# from isaaclab.utils import configclass

# from isaaclab_contrib.mdp import compute_mirrored_states

# from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlMLPModelCfg, RslRlSymmetryCfg


# @configclass
# class SquashedGaussianDistributionCfg(RslRlMLPModelCfg.DistributionCfg):
#     """Bound normalized actions to +/-4, giving joint offsets up to +/-1 radian."""

#     class_name = "SquashedGaussianDistribution"
#     init_std: float = 0.25
#     std_range: tuple[float, float] = (0.01, 1.0)
#     action_range: tuple[float, float] = (-4.0, 4.0)


# @configclass
# class SACAlgorithmCfg:
#     """SAC settings; replay capacity, warm-up transitions, and batch size are per GPU."""

#     class_name: str = "SAC"
#     num_critics: int = 2
#     replay_buffer_capacity: int = 1_000_000
#     max_replay_memory_fraction: float = 0.25
#     num_warmup_transitions: int = 100_000
#     batch_size: int = 1024
#     num_updates: int = 24
#     num_return_steps: int = 1
#     gamma: float = 0.99
#     tau: float = 0.005
#     actor_update_interval: int = 1
#     actor_learning_rate: float = 3.0e-4
#     critic_learning_rate: float = 3.0e-4
#     alpha_learning_rate: float = 3.0e-4
#     init_alpha: float = 0.01
#     target_entropy_std: float = 0.15
#     critic_loss_type: str = "mse"
#     reward_normalization: bool = False
#     use_mixed_precision: bool = False
#     use_cuda_graph: bool = False
#     symmetry_cfg = RslRlSymmetryCfg(
#         use_data_augmentation=True,
#         data_augmentation_func=compute_mirrored_states,
#         use_mirror_loss=True,
#         mirror_loss_coeff=0.1,
#     )


# @configclass
# class G1LocomotionSACRunnerCfg(RslRlBaseRunnerCfg):
#     """Use the standard SAC runner while retaining the PPO task's curriculum time scale."""

#     class_name: str = "OffPolicyRunner"
#     num_steps_per_env = 24
#     max_iterations = 20_000
#     save_interval = 500
#     clip_actions = None
#     obs_groups = {"actor": ["policy"], "critic": ["critic"]}
#     actor = RslRlMLPModelCfg(
#         hidden_dims=[512, 256, 128],
#         activation="elu",
#         obs_normalization=False,
#         distribution_cfg=SquashedGaussianDistributionCfg(),
#     )
#     critic = RslRlMLPModelCfg(
#         class_name="QMLPModel",
#         hidden_dims=[512, 256, 128],
#         activation="elu",
#         obs_normalization=False,
#     )
#     algorithm = SACAlgorithmCfg()
#     logger = "wandb"
#     wandb_project = "g1_29dof_locomotion"
#     experiment_name = "g1_29dof_locomotion_sac"
