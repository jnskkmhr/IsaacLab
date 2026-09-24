# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.contrib.mimic.mdp as mimic_mdp

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

@configclass
class G1EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mimic_mdp.randomize_rigid_body_material, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    # NOTE: robust policy randomization
    add_joint_default_pos = EventTerm(
        func=mimic_mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
            "joint_action_name": "joint_pos",
        },
    )

    base_com = EventTerm(
        func=mimic_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # reset
    # NOTE: these two carry the state writes that used to live inside `MotionCommand._resample_command`.
    # They ask the command term for this episode's start frame and place the robot on it. Drop them and
    # the robot starts wherever the previous episode left it, with no relation to the reference.
    reset_joints = EventTerm(
        func=mimic_mdp.reset_joint_state_from_reference,
        mode="reset",
        params={"command_name": "motion", "position_range": (-0.1, 0.1)},
    )

    reset_base = EventTerm(
        func=mimic_mdp.reset_root_state_from_reference,
        mode="reset",
        params={
            "command_name": "motion",
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )

    # interval
    # NOTE: model-based wrench at the base that carries the robot toward the reference. Its gain is
    # the ZEST S6 automatic curriculum: set per env at reset from the measured failure rate of the
    # clip bin the episode starts in, so it retires itself as the policy masters each part of the
    # motion. Tuned via `assist_eta` / `assist_beta_max` on the motion command.
    assistive_wrench = EventTerm(
        func=mimic_mdp.AssistiveWrench, # type: ignore
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=True,
        params={
            "command_term_name": "motion",
            "scale": 1.0,
            "adaptive_scale": True,
            "asset_cfg": SceneEntityCfg("robot"),
            "base_body_name": "pelvis",
        },
    )

    # # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 3.0),
    #     params={"velocity_range": VELOCITY_RANGE},
    # )
