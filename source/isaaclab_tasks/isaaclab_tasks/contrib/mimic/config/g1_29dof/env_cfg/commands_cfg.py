# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp
from isaaclab_tasks.contrib.mimic import MOTION_TRACKING_DATA_DIR


@configclass
class G1CommandsCfg:
    """Command specifications for the MDP."""

    motion = mimic_mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        # debug_vis=True,
        debug_vis=False,
        # No stance-phase gating: the CMU clip below wasn't measured for a static-stance interval
        # the way the old stance5s file was. Retune this (and stance_blend_time) if the clip turns
        # out to include one.
        stance_phase_ranges=[],
        stance_blend_time=0.2,
        # ZEST S6 assistive wrench: a bin gets help while its failure rate is above 1 - assist_eta,
        # with the gain capped at assist_beta_max so assistance stays partial.
        assist_eta=1.0,
        assist_beta_max=0.9,
        # `motion_joint_names` intentionally left unset (None) for the CMU clip below: unlike the
        # legacy stance5s file, this NPZ carries no per-joint name metadata to validate a reorder
        # list against, so this assumes the file's joint columns already match the robot's own
        # joint order. Verify with the replay script before trusting a policy trained on this.
        anchor_body_name="torso_link",
        # body names present in motion reference
        body_names=[
            "pelvis",
            "left_hip_pitch_link",
            "right_hip_pitch_link",
            "waist_yaw_link",
            "left_hip_roll_link",
            "right_hip_roll_link",
            "waist_roll_link",
            "left_hip_yaw_link",
            "right_hip_yaw_link",
            "torso_link",
            "left_knee_link",
            "right_knee_link",
            "left_shoulder_pitch_link",
            "right_shoulder_pitch_link",
            "left_ankle_pitch_link",
            "right_ankle_pitch_link",
            "left_shoulder_roll_link",
            "right_shoulder_roll_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_shoulder_yaw_link",
            "right_shoulder_yaw_link",
            "left_elbow_link",
            "right_elbow_link",
            "left_wrist_roll_link",
            "right_wrist_roll_link",
            "left_wrist_pitch_link",
            "right_wrist_pitch_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ],
        # motion_file=f"{MOTION_TRACKING_DATA_DIR}/motions/npz/g1_TO_jump_forward.npz"
        # motion_file=f"{MOTION_TRACKING_DATA_DIR}/motions/npz/leap_g1_retargeted.npz"
        # motion_file=f"{MOTION_TRACKING_DATA_DIR}/motions/npz/leap_wide_g1_retargeted_mirror_right.npz"
        # motion_file=f"{MOTION_TRACKING_DATA_DIR}/motions/npz/leap_wide_g1_retargeted_mirror_right_stance5s.npz",
        # motion_file=f"{MOTION_TRACKING_DATA_DIR}/motions/npz/leap_wide_g1_retargeted.npz"
        motion_file=f"{MOTION_TRACKING_DATA_DIR}/motions/npz/cmu_bvh_convert.npz",
    )
