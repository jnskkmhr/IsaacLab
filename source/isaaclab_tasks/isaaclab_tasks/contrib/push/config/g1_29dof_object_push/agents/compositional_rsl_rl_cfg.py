# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with two frozen teachers supervising disjoint student action subsets."""

from compositional_rsl_rl.algorithm import CompositionalPPOAlgorithmCfg, SchedulerCfg
from compositional_rsl_rl.loss import DistillationLossCfg, LossesCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, NetworkCfg, TeacherPolicyCfg
from compositional_rsl_rl.runner import CompositionalRunnerCfg

from isaaclab.paths import ISAACLAB_ROOT
from isaaclab.utils import configclass

from isaaclab_contrib.mdp import compute_mirrored_states

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from ..env_cfg.action_cfg import ALL_JOINTS, LOWER_JOINTS, UPPER_JOINTS


@configclass
class TeacherNetworksCfg(NetworkCfg):
    actor = ActorHeadCfg(
        input=["policy"],
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=0.1),
    )
    critic = CriticHeadCfg(
        input=["critic"],
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    locomotion_teacher = TeacherPolicyCfg(
        input=["locomotion_teacher_obs"],
        output_dim=len(LOWER_JOINTS),
        hidden_dims=[512, 256, 128],
        pretrained_cfg={
            "path": (
                f"{ISAACLAB_ROOT}/logs/rsl_rl/g1_29dof_locomotion/wandb/"
                "2026-11-06_03-36-53_ppo_z4rc1ja0_reward_obs_reproduction_4gpu_2jiewizp/model_16000.pt"
            ),
            "state_dict_key": "actor_state_dict",
            "prefix": "",
        },
    )
    manipulation_teacher = TeacherPolicyCfg(
        input=["manipulation_teacher_obs"],
        output_dim=len(UPPER_JOINTS),
        hidden_dims=[512, 256, 128],
        pretrained_cfg={
            "path": (
                f"{ISAACLAB_ROOT}/logs/rsl_rl/g1_29dof_manipulation/wandb/"
                "2026-11-06_23-52-50_closed_hands_feasible_targets_4gpu_6pbvl2df/model_9500.pt"
            ),
            "state_dict_key": "actor_state_dict",
            "prefix": "",
        },
    )


@configclass
class TeacherLossesCfg(LossesCfg):
    locomotion = DistillationLossCfg(
        target="locomotion_teacher",
        prediction_indices=list(range(len(LOWER_JOINTS))),
        weight=1.0,
    )
    manipulation = DistillationLossCfg(
        target="manipulation_teacher",
        prediction_indices=list(range(len(LOWER_JOINTS), len(ALL_JOINTS))),
        weight=1.0,
    )


@configclass
class G1ObjectPushPPORunnerCfg(CompositionalRunnerCfg):
    """The student acts; PPO and teacher action-mean losses update the same student."""

    env_interface = None
    num_steps_per_env = 24
    max_iterations = 20_000
    save_interval = 500
    networks = TeacherNetworksCfg()
    loss = TeacherLossesCfg()
    # Clone teacher action means for 2,000 iterations, then ramp the complete PPO objective.
    # A fixed learning rate prevents PPO KL adaptation from suppressing the cloning updates.
    curriculum_cfg = {
        "ppo_loss_weight": SchedulerCfg(
            type="linear",
            params={"start_step": 2000 * 24, "end_step": 10000 * 24, "start_value": 0.0, "end_value": 1.0},
        ),
        "loss.locomotion.weight": SchedulerCfg(
            type="linear",
            params={"start_step": 2000 * 24, "end_step": 10000 * 24, "start_value": 1.0, "end_value": 0.1},
        ),
        "loss.manipulation.weight": SchedulerCfg(
            type="linear",
            params={"start_step": 2000 * 24, "end_step": 10000 * 24, "start_value": 1.0, "end_value": 0.1},
        ),
    }
    algorithm = CompositionalPPOAlgorithmCfg(
        ppo_loss_weight=0.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=compute_mirrored_states,
            use_mirror_loss=False,
            mirror_loss_coeff=0.1,
        ),
    )
    logger = "wandb"
    wandb_project = "g1_29dof_object_push"
    experiment_name = "g1_29dof_object_push"
