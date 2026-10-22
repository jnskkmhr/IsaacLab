# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as velocity_mdp

from .. import mdp as goal_mdp


@configclass
class G1RewardsCfg:
    """Reward terms for the MDP."""

    # -- task terms for tracking the motion
    motion_global_anchor_pos = RewTerm(
        func=goal_mdp.imitation_reward,
        weight=2.0,
        params={"reward_name": "motion_global_anchor_position_error_exp", "command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=goal_mdp.imitation_reward,
        weight=2.0,
        params={"reward_name": "motion_global_anchor_orientation_error_exp", "command_name": "motion", "std": 0.4},
    )
    motion_global_anchor_lin_vel = RewTerm(
        func=goal_mdp.imitation_reward,
        weight=1.0,
        params={"reward_name": "motion_global_anchor_linear_velocity_error_exp", "command_name": "motion", "std": 1.0},
    )
    motion_global_anchor_ang_vel = RewTerm(
        func=goal_mdp.imitation_reward,
        weight=1.0,
        params={
            "reward_name": "motion_global_anchor_angular_velocity_error_exp",
            "command_name": "motion",
            "std": 3.14,
        },
    )

    motion_body_pos = RewTerm(
        func=goal_mdp.imitation_reward,
        weight=1.0,
        params={"reward_name": "motion_relative_body_position_error_exp", "command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=goal_mdp.imitation_reward,
        weight=1.0,
        params={"reward_name": "motion_relative_body_orientation_error_exp", "command_name": "motion", "std": 0.4},
    )
    # Track both ankle-link velocities using the validated walking/jogging reward.
    motion_feet_lin_vel = RewTerm(
        func=goal_mdp.imitation_reward,
        weight=1.0,
        params={
            "reward_name": "motion_global_body_linear_velocity_error_exp",
            "command_name": "motion",
            "std": 1.0,
            "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
        },
    )

    goal_progress = RewTerm(func=goal_mdp.goal_progress, weight=2.0)
    """Reward decreasing root-to-destination distance in task episodes."""
    goal_position = RewTerm(func=goal_mdp.goal_position, weight=4.0, params={"std": 2.0})
    goal_position_fine = RewTerm(func=goal_mdp.goal_position, weight=4.0, params={"std": 0.5})
    goal_heading = RewTerm(func=goal_mdp.goal_heading, weight=2.0, params={"std": 0.5})
    """Align destination heading after reaching its position."""
    goal_settle = RewTerm(func=goal_mdp.goal_settle, weight=-0.5)
    """Penalize base motion near the destination; leave the jump unconstrained."""

    alive_reward = RewTerm(func=mdp.is_alive, weight=1.0)

    # -- penalties
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )

    feet_slide = RewTerm(
        func=velocity_mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
