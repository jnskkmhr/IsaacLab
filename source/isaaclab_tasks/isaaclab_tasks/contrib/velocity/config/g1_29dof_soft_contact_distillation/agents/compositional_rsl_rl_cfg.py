# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VAE teacher and temporal student using named networks and auxiliary PPO losses."""

from compositional_rsl_rl.algorithm import CompositionalPPOAlgorithmCfg, SchedulerCfg
from compositional_rsl_rl.components.network import GaussianLatentNetworkCfg, TCNNetworkCfg, VAEEncoderNetworkCfg
from compositional_rsl_rl.env_interface import EnvInterfaceCfg
from compositional_rsl_rl.loss import DistillationLossCfg, LossesCfg, RegularizationLossCfg, SupervisedLossCfg
from compositional_rsl_rl.network import ActorHeadCfg, CriticHeadCfg, MLPNetworkCfg, NetworkCfg, TeacherPolicyCfg
from compositional_rsl_rl.runner import CompositionalRunnerCfg

from isaaclab.utils import configclass


@configclass
class TeacherNetworksCfg(NetworkCfg):
    encoder = VAEEncoderNetworkCfg(
        input=["teacher_history", "dynamics_privileged", "terrain_privileged"], output_dim=128, hidden_dims=[256, 128]
    )
    latent = GaussianLatentNetworkCfg(input=["encoder"], output_dim=64)
    latent_sample = GaussianLatentNetworkCfg(input=["encoder"], output_dim=64, sample=True)
    decoder = MLPNetworkCfg(input=["latent_sample"], output_dim=1, hidden_dims=[128, 256])
    actor = ActorHeadCfg(
        input=["policy", "latent"],
        hidden_dims=[512, 256, 128],
        distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = CriticHeadCfg(input=["critic", "dynamics_privileged", "terrain_privileged"], hidden_dims=[512, 256, 128])


@configclass
class TeacherLossesCfg(LossesCfg):
    reconstruction = SupervisedLossCfg(
        prediction="decoder", target="terrain_privileged", trains=["encoder", "decoder"], weight=1.0
    )
    kl = RegularizationLossCfg(
        prediction="encoder",
        trains=["encoder"],
        weight=0.01,
        loss_fn="compositional_rsl_rl.components.loss:gaussian_kl",
    )


@configclass
class G1TeacherRunnerCfg(CompositionalRunnerCfg):
    networks = TeacherNetworksCfg()
    loss = TeacherLossesCfg()
    env_interface = EnvInterfaceCfg(restore_terrain=True)
    algorithm = CompositionalPPOAlgorithmCfg(entropy_coef=0.005)
    num_steps_per_env = 24
    max_iterations = 28000
    save_interval = 500
    experiment_name = "g1_29dof_soft_vae_teacher"
    wandb_project = "g1_29dof_soft_vae_teacher"
    logger = "wandb"


@configclass
class StudentNetworksCfg(NetworkCfg):
    student_encoder = TCNNetworkCfg(input=["student_history"], output_dim=64, hidden_dims=[256, 128])
    actor = ActorHeadCfg(
        input=["policy", "student_encoder"],
        hidden_dims=[512, 256, 128],
        distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=0.1),
        pretrained_cfg={"path": "", "state_dict_key": "network_state_dict", "prefix": "terms.actor."},
    )
    critic = CriticHeadCfg(
        input=["critic", "dynamics_privileged", "terrain_privileged"],
        hidden_dims=[512, 256, 128],
        pretrained_cfg={"path": "", "state_dict_key": "network_state_dict", "prefix": "terms.critic."},
    )
    teacher_encoder = VAEEncoderNetworkCfg(
        input=["teacher_history", "dynamics_privileged", "terrain_privileged"],
        output_dim=128,
        hidden_dims=[256, 128],
        frozen=True,
        pretrained_cfg={"path": "", "state_dict_key": "network_state_dict", "prefix": "terms.encoder."},
    )
    teacher_latent = GaussianLatentNetworkCfg(input=["teacher_encoder"], output_dim=64, frozen=True)
    teacher_policy = TeacherPolicyCfg(
        input=["policy", "teacher_latent"],
        output_dim=29,
        hidden_dims=[512, 256, 128],
        distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=1.0),
        pretrained_cfg={"path": "", "state_dict_key": "network_state_dict", "prefix": "terms.actor."},
    )

    # Two frozen copies of the same teacher decoder evaluate the two latent inputs.
    # Frozen weights still pass gradients back to the student encoder.
    student_decoder = MLPNetworkCfg(
        input=["student_encoder"],
        output_dim=1,
        hidden_dims=[128, 256],
        frozen=True,
        pretrained_cfg={"path": "", "state_dict_key": "network_state_dict", "prefix": "terms.decoder."},
    )
    teacher_decoder = MLPNetworkCfg(
        input=["teacher_latent"],
        output_dim=1,
        hidden_dims=[128, 256],
        frozen=True,
        pretrained_cfg={"path": "", "state_dict_key": "network_state_dict", "prefix": "terms.decoder."},
    )


@configclass
class StudentLossesCfg(LossesCfg):
    action_imitation = DistillationLossCfg(target="teacher_policy", trains=["actor", "student_encoder"], weight=1.0)
    feature_matching = DistillationLossCfg(
        prediction="student_encoder", target="teacher_latent", trains=["student_encoder"], weight=1.0
    )

    decoder_matching = DistillationLossCfg(
        prediction="student_decoder", target="teacher_decoder", trains=["student_encoder"], weight=1.0
    )


@configclass
class G1StudentRunnerCfg(CompositionalRunnerCfg):
    networks = StudentNetworksCfg()
    loss = StudentLossesCfg()
    env_interface = EnvInterfaceCfg(restore_terrain=True)
    algorithm = CompositionalPPOAlgorithmCfg(entropy_coef=0.005)
    num_steps_per_env = 24
    max_iterations = 25000
    save_interval = 500
    experiment_name = "g1_29dof_soft_vae_student_tcn"
    wandb_project = "g1_29dof_soft_vae_student_tcn"
    logger = "wandb"
    # Original schedule: imitation=max(0.1, 1-iteration/12500), PPO=1-imitation.
    curriculum_cfg = {
        "ppo_loss_weight": SchedulerCfg(
            type="linear", params={"start_step": 0, "end_step": 270000, "start_value": 0.0, "end_value": 0.9}
        ),
        "loss.action_imitation.weight": SchedulerCfg(
            type="linear", params={"start_step": 0, "end_step": 270000, "start_value": 1.0, "end_value": 0.1}
        ),
    }
