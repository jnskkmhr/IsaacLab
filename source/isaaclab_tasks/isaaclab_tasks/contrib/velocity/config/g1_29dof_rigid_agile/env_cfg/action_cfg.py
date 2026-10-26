# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.mdp import MirrorJointPositionActionCfg

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
