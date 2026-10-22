# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp
from isaaclab_tasks.contrib.mimic import MOTION_TRACKING_DATA_DIR

MOTION_DATA_DIR = Path(MOTION_TRACKING_DATA_DIR) / "motions" / "data"


JOINT_NAMES = [
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
]
BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]


@configclass
class G1CommandsCfg:
    """Command specifications for the MDP."""

    motion = mimic_mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        assist_eta=0.8,
        assist_beta_max=0.8,
        # Select tracked joints/bodies in policy order; NPZ metadata supplies source ordering.
        joint_names=JOINT_NAMES,
        body_names=BODY_NAMES,
        anchor_body_name="torso_link",
        # Load terrain from each NPZ when present; flat clips keep the scene plane.
        terrain_mode="auto",
        motion_files=[
            str(MOTION_DATA_DIR / "parc" / "jump_001" / "motion.npz"),
            # str(MOTION_DATA_DIR / "parc" / "jump_002" / "motion.npz"),
            # str(MOTION_DATA_DIR / "parc" / "jump_003" / "motion.npz"),
            # str(MOTION_DATA_DIR / "parc" / "jump_004" / "motion.npz"),
            # str(MOTION_DATA_DIR / "parc" / "jump_005" / "motion.npz"),
            # str(MOTION_DATA_DIR / "deep_wholebody_parkour" / "jump_sit" / "motion.npz"),
            # str(MOTION_DATA_DIR / "deep_wholebody_parkour" / "kneel_climb" / "motion.npz"),
            # str(MOTION_DATA_DIR / "omniretarget" / "climb_06" / "motion.npz"),
            # str(MOTION_DATA_DIR / "omniretarget" / "climb_08" / "motion.npz"),
            # str(MOTION_DATA_DIR / "omniretarget" / "climb_14" / "motion.npz"), # climb up
            # str(MOTION_DATA_DIR / "omniretarget" / "climb_16" / "motion.npz"),
            # str(MOTION_DATA_DIR / "omniretarget" / "climb_20" / "motion.npz"),
            # str(MOTION_DATA_DIR / "omniretarget" / "climb_26" / "motion.npz"),
            # str(MOTION_DATA_DIR / "omniretarget" / "climb_28" / "motion.npz"),
        ],
    )
