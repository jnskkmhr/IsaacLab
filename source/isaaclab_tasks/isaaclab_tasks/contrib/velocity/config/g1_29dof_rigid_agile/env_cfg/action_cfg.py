# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.mdp import MirrorJointPositionActionCfg

import isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.mdp as g1_mdp

from ..mdp import symmetry

CONTROLLED_JOINT_NAMES = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
] + ["waist_roll_joint", "waist_pitch_joint"]


@configclass
class G1ActionsCfg:
    """Agile's ordered fourteen joint-position targets."""

    joint_pos = MirrorJointPositionActionCfg(
        mirror=symmetry.mirror_g1_joints,
        mirror_params={"joint_names": CONTROLLED_JOINT_NAMES},
        asset_name="robot",
        joint_names=CONTROLLED_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
        clip={".*": (-10.0, 10.0)},
    )

    random_pos = g1_mdp.RandomActionCfg(
        asset_name="robot",
        joint_names_exclude=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "waist_.*_joint"],
        sample_range=(0.1, 2.5),
        preserve_order=True,
        velocity_profile_cfg=g1_mdp.TrapezoidalVelocityProfileCfg(
            acceleration_range=(1.0, 20.0),
            max_velocity_range=(10.0, 20.0),
            min_cruise_ratio=0.1,
            synchronize_joints=True,
            time_scaling_method="max_time",
            use_smooth_start=False,
            position_tolerance=0.001,
            velocity_tolerance=0.01,
            enable_position_limits=True,
            enable_velocity_limits=True,
        ),
    )
