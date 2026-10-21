# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion command configuration for soft-contact G1 tasks."""

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.mimic.config.g1_29dof.env_cfg.commands_cfg import BODY_NAMES, JOINT_NAMES, LAFAN_MOTION_DIR

from .. import mdp as soft_mdp


@configclass
class G1CommandsCfg:
    """Reference tracking with optional stance intervals."""

    motion = soft_mdp.SoftMotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        assist_eta=0.8,
        assist_beta_max=0.8,
        joint_names=JOINT_NAMES,
        body_names=BODY_NAMES,
        anchor_body_name="torso_link",
        motion_files=[
            str(LAFAN_MOTION_DIR / "walking" / "motion_60fps.npz"),
            str(LAFAN_MOTION_DIR / "running" / "motion_60fps.npz"),
            str(LAFAN_MOTION_DIR / "sprinting" / "motion_60fps.npz"),
        ],
    )
