# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.utils import configclass

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
        # Where the clip is a static stance rather than a trajectory: tracking rewards/terminations
        # are gated off here and the standing rewards take over. Measured on the motion file below
        # (the reference is at rest from ~64% of the clip through the appended hold); retune these
        # if you point `motion_file` at a different clip.
        stance_phase_ranges=[(0.0, 0.344), (0.589, 1.0)],
        stance_blend_time=0.2,
        # ZEST S6 assistive wrench: a bin gets help while its failure rate is above 1 - assist_eta,
        # with the gain capped at assist_beta_max so assistance stays partial.
        assist_eta=1.0,
        assist_beta_max=0.9,
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
        motion_file=f"{MOTION_TRACKING_DATA_DIR}/motions/npz/leap_wide_g1_retargeted_mirror_right_stance5s.npz",
        # motion_file=f"{MOTION_TRACKING_DATA_DIR}/motions/npz/leap_wide_g1_retargeted.npz"
    )
