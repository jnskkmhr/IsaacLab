# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AME height-map PPO using the simulator-independent compositional RSL-RL library."""

from compositional_rsl_rl.algorithm import CompositionalPPOAlgorithmCfg
from compositional_rsl_rl.loss import LossesCfg
from compositional_rsl_rl.mdp.network import AttentionMapNetworkCfg
from compositional_rsl_rl.network import ActorHeadCfg, CNNNetworkCfg, CriticHeadCfg, NetworkCfg
from compositional_rsl_rl.runner import CompositionalRunnerCfg

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.mdp import compute_mirrored_states

"""
network backbone config
"""

@configclass
class CNNNetworksCfg(NetworkCfg):
    map_encoder: CNNNetworkCfg | None = None


@configclass
class AMENetworkCfg(NetworkCfg):
    """One actor-visible map encoder shared by the policy and value heads."""

    map_encoder: AttentionMapNetworkCfg | None = None


"""
main config
"""

@configclass
class G1CNNPPORunnerCfg(CompositionalRunnerCfg):
    """Train the map encoder and both heads jointly through PPO."""

    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500
    networks = CNNNetworksCfg(
        actor=ActorHeadCfg(
            input=["policy", "command", "map_encoder"],
            hidden_dims=[256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=CriticHeadCfg(
            input=["critic", "privileged", "command", "map_encoder"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
        ),
        map_encoder=CNNNetworkCfg(
            input=["elevation_map"],
            output_dim=64,
            hidden_dims=[128],
            activation="elu",
            obs_normalization=False,
            cnn_cfg={
                "output_channels": [16, 32, 32],
                "kernel_size": 3,
                "stride": [2, 2, 1],
                "padding": "zeros",
                "activation": "elu",
                "norm": "none",
                "global_pool": "none",
                "flatten": True,
            },
        ),
    )
    loss = LossesCfg()
    algorithm = CompositionalPPOAlgorithmCfg(
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
        symmetry_cfg={
            "use_data_augmentation": True,
            "use_mirror_loss": False,
            "data_augmentation_func": compute_mirrored_states,
            "mirror_loss_coeff": 0.1,
        },
    )
    logger = "wandb"
    experiment_name = "g1_29dof_perception_cnn"
    wandb_project = "g1_29dof_perception_cnn"

@configclass
class G1AMEPPORunnerCfg(CompositionalRunnerCfg):
    """Train the map encoder and both heads jointly through PPO."""

    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500
    networks = AMENetworkCfg(
        actor=ActorHeadCfg(
            input=["policy", "command", "map_encoder"],
            hidden_dims=[256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=ActorHeadCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=CriticHeadCfg(
            input=["critic", "privileged", "command", "map_encoder"],
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
        ),
        map_encoder=AttentionMapNetworkCfg(
            input=["elevation_map", "policy", "command"],
            map_input="elevation_map",
            map_shape=None,  # Inferred from the structured (33, 21, 3) observation.
            output_dim=64,
            feature_dim=64,
            num_heads=16,
            cnn_channels=16,
            cnn_downsample=True,
            cnn_normalization="batch",
            attach_global=False,
            hidden_dims=[],
            obs_normalization=True,
            frozen=False,
            pretrained_cfg=None,
        ),
    )
    loss = LossesCfg()
    algorithm = CompositionalPPOAlgorithmCfg(
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
        symmetry_cfg={
            "use_data_augmentation": True,
            "use_mirror_loss": False,
            "data_augmentation_func": compute_mirrored_states,
            "mirror_loss_coeff": 0.1,
        },
    )
    logger = "wandb"
    experiment_name = "g1_29dof_perception_ame"
    wandb_project = "g1_29dof_perception_ame"


@configclass
class G1AMEPretrainedPPORunnerCfg(G1AMEPPORunnerCfg):
    """Load and freeze the existing AME term without changing its inputs or architecture."""

    run_name = "encoder_pretrained"
    networks = G1AMEPPORunnerCfg().networks
    networks.map_encoder.frozen = True
    networks.map_encoder.pretrained_cfg = {
        "path": "???",
        "state_dict_key": "network_state_dict",
        "prefix": "terms.map_encoder.",
        "strict": True,
    }


@configclass
class G1AMEFinetunePPORunnerCfg(G1AMEPretrainedPPORunnerCfg):
    """Load the same AME term and train it with the existing PPO settings."""

    run_name = "encoder_finetune"
    networks = G1AMEPretrainedPPORunnerCfg().networks
    networks.map_encoder.frozen = False


@configclass
class G1CNNPretrainedPPORunnerCfg(G1CNNPPORunnerCfg):
    """Load and freeze the existing CNN term without changing its inputs or architecture."""

    run_name = "encoder_pretrained"
    networks = G1CNNPPORunnerCfg().networks
    networks.map_encoder.frozen = True
    networks.map_encoder.pretrained_cfg = G1AMEPretrainedPPORunnerCfg().networks.map_encoder.pretrained_cfg


@configclass
class G1CNNFinetunePPORunnerCfg(G1CNNPretrainedPPORunnerCfg):
    """Load the same CNN term and train it with the existing PPO settings."""

    run_name = "encoder_finetune"
    networks = G1CNNPretrainedPPORunnerCfg().networks
    networks.map_encoder.frozen = False
