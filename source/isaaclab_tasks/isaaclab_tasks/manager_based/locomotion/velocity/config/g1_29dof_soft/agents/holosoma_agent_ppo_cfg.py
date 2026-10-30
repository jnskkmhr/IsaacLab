# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.holosoma_agent import LayerConfig, ModuleConfig, PPOConfig, PPOModuleDictConfig, SymmetryConfig


@configclass
class G1HolosomaPPOAgentCfg(PPOConfig):
    module_dict = PPOModuleDictConfig(
        actor=ModuleConfig(
            module_type="MLP",
            obs_keys=["policy"],
            layer_config=LayerConfig(
                hidden_dims=[512, 256, 128],
                activation="ELU",
                dropout_prob=0.0,
                use_layer_norm=False,
            ),
            min_noise_std=0.1,
            min_mean_noise_std=0.1,
        ),
        critic=ModuleConfig(
            module_type="MLP",
            obs_keys=["critic"],
            layer_config=LayerConfig(
                hidden_dims=[512, 256, 128],
                activation="ELU",
                dropout_prob=0.0,
                use_layer_norm=False,
            ),
            min_noise_std=0.1,
            min_mean_noise_std=0.1,
        ),
    )
    init_noise_std = 1.0

    num_learning_iterations = 15_000
    logging_interval = 1
    save_interval = 500
    load_optimizer = True
    init_at_random_ep_len = True

    num_steps_per_env = 24
    num_learning_epochs = 5
    num_mini_batches = 4
    clip_param = 0.2
    gamma = 0.99
    lam = 0.95
    value_loss_coef = 1.0
    entropy_coef = 0.005
    max_grad_norm = 1.0

    actor_learning_rate = 1e-5
    actor_optimizer_weight_decay = 0.001
    critic_learning_rate = 1e-5
    critic_optimizer_weight_decay = 0.001
    schedule = "adaptive"
    desired_kl = 0.01
    max_actor_learning_rate = None
    min_actor_learning_rate = None
    max_critic_learning_rate = None
    min_critic_learning_rate = None

    use_symmetry = True
    symmetry_config = SymmetryConfig(
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        symmetry_joint_names={
            "left_hip_pitch_joint": "right_hip_pitch_joint",
            "left_hip_roll_joint": "right_hip_roll_joint",
            "left_hip_yaw_joint": "right_hip_yaw_joint",
            "left_knee_joint": "right_knee_joint",
            "left_ankle_pitch_joint": "right_ankle_pitch_joint",
            "left_ankle_roll_joint": "right_ankle_roll_joint",
            "right_hip_pitch_joint": "left_hip_pitch_joint",
            "right_hip_roll_joint": "left_hip_roll_joint",
            "right_hip_yaw_joint": "left_hip_yaw_joint",
            "right_knee_joint": "left_knee_joint",
            "right_ankle_pitch_joint": "left_ankle_pitch_joint",
            "right_ankle_roll_joint": "left_ankle_roll_joint",
            # Upper body joints
            "left_shoulder_pitch_joint": "right_shoulder_pitch_joint",
            "left_shoulder_roll_joint": "right_shoulder_roll_joint",
            "left_shoulder_yaw_joint": "right_shoulder_yaw_joint",
            "left_elbow_joint": "right_elbow_joint",
            "left_wrist_roll_joint": "right_wrist_roll_joint",
            "left_wrist_pitch_joint": "right_wrist_pitch_joint",
            "left_wrist_yaw_joint": "right_wrist_yaw_joint",
            "right_shoulder_pitch_joint": "left_shoulder_pitch_joint",
            "right_shoulder_roll_joint": "left_shoulder_roll_joint",
            "right_shoulder_yaw_joint": "left_shoulder_yaw_joint",
            "right_elbow_joint": "left_elbow_joint",
            "right_wrist_roll_joint": "left_wrist_roll_joint",
            "right_wrist_pitch_joint": "left_wrist_pitch_joint",
            "right_wrist_yaw_joint": "left_wrist_yaw_joint",
            # Central joints (map to themselves)
            "waist_yaw_joint": "waist_yaw_joint",
            "waist_roll_joint": "waist_roll_joint",
            "waist_pitch_joint": "waist_pitch_joint",
        },
        sign_flip_joints=[
            # Hip roll and yaw joints
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            # Ankle roll joints
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
            # Waist roll and yaw joints
            "waist_roll_joint",
            "waist_yaw_joint",
            # Shoulder roll and yaw joints
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            # Wrist roll and yaw joints
            "left_wrist_roll_joint",
            "left_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_yaw_joint",
        ],
    )

    logger = "wandb"
    experiment_name = "g1_29dof_rigid_ppo"
    wandb_project = "g1_29dof_rigid_ppo"
