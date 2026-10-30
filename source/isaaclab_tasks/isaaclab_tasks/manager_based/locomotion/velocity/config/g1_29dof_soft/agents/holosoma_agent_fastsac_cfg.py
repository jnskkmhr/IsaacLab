# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.holosoma_agent import FastSACConfig, SymmetryConfig


@configclass
class G1HolosomaFastSACAgentCfg(FastSACConfig):
    num_learning_iterations = 150_000
    learning_starts = 10
    save_interval = 1000

    module_type = "MLP"
    actor_hidden_dim = [512, 256, 128]
    critic_hidden_dim = [768, 384, 192]
    use_layer_norm = True
    num_q_networks = 2
    num_atoms = 101
    v_min = -20.0
    v_max = 20.0

    encoder_obs_key = None
    encoder_hidden_dim = (1, 13, 9)

    actor_obs_keys = ["policy"]
    critic_obs_keys = ["critic", "privileged"]

    buffer_size = 1024
    num_steps = 1
    batch_size = 8192

    gamma = 0.98
    tau = 0.125
    policy_frequency = 4
    num_updates = 5

    alpha_init = 0.001
    use_autotune = True
    target_entropy_ratio = 0.0

    # use_tanh = True
    use_tanh = False
    log_std_max = 0.0
    log_std_min = -5.0

    critic_learning_rate = 3e-4
    actor_learning_rate = 3e-4
    alpha_learning_rate = 3e-4
    weight_decay = 0.001
    max_grad_norm = 0.0

    compile = True
    amp = True
    amp_dtype = "bf16"

    # augmentation
    obs_normalization = True
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

    logging_interval = 100
    logger = "wandb"
    experiment_name = "g1_29dof_soft_fastsac"
    wandb_project = "g1_29dof_soft_fastsac"
