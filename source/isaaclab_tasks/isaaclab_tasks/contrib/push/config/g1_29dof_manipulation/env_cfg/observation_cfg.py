# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured joint state includes the uncontrolled arms during locomotion."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_contrib.mdp import MirrorObservationTermCfg as ObsTerm

from isaaclab_tasks.contrib.push.mdp.observations import hand_pose_commands
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp.symmetry import mirror_g1_joints

from ..mdp.symmetry import mirror_hand_poses
from .action_cfg import UPPER_JOINTS, WAIST_JOINTS


@configclass
class PolicyCfg(ObservationGroupCfg):
    """Actor observations, with explicit joint order and mirror rules."""

    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_JOINTS + WAIST_JOINTS, preserve_order=True)},
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": UPPER_JOINTS + WAIST_JOINTS},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_JOINTS + WAIST_JOINTS, preserve_order=True)},
        mirror=mirror_g1_joints,
        mirror_params={"joint_names": UPPER_JOINTS + WAIST_JOINTS},
        scale=0.05,
        noise=Unoise(n_min=-0.5, n_max=0.5),
    )
    actions = ObsTerm(func=mdp.last_action, mirror=mirror_g1_joints, mirror_params={"joint_names": UPPER_JOINTS})
    hand_targets = ObsTerm(
        func=hand_pose_commands, params={"command_names": ("left_ee_pose", "right_ee_pose")}, mirror=mirror_hand_poses
    )
    hand_pose = ObsTerm(
        func=mdp.body_pose_local,
        params={
            "target_frame": SceneEntityCfg(
                "robot", body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True
            ),
            "parent_frame": SceneEntityCfg("robot", body_names="torso_link"),
        },
        mirror=mirror_hand_poses,
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CriticCfg(PolicyCfg):
    """Noise-free measurements for value estimation."""

    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    policy = PolicyCfg()
    critic = CriticCfg()
