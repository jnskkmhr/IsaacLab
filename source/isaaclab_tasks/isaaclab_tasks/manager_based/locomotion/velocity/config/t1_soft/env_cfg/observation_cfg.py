# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.t1_soft.mdp as t1_mdp

@configclass
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    clock = ObsTerm(
        func=t1_mdp.clock, # type: ignore
    )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel, # type: ignore
        scale=1,
        noise=Gnoise(mean=0.0, std=0.15),
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity, # type: ignore
        noise=Gnoise(mean=0.0, std=0.075),
    )
    velocity_commands = ObsTerm(
        func=mdp.generated_commands, # type: ignore
        scale=1,
        params={"command_name": "base_velocity"},
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos, # type: ignore
        scale=1,
        noise=Gnoise(mean=0.0, std=0.175),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
            joint_names=[
                "AAHead_yaw",
                "Head_pitch",
                "Left_Shoulder_Pitch",
                "Left_Shoulder_Roll",
                "Left_Elbow_Pitch",
                "Left_Elbow_Yaw",
                "Left_Wrist_Pitch",
                "Left_Wrist_Yaw",
                "Left_Hand_Roll",

                "Right_Shoulder_Pitch",
                "Right_Shoulder_Roll",
                "Right_Elbow_Pitch",
                "Right_Elbow_Yaw",
                "Right_Wrist_Pitch",
                "Right_Wrist_Yaw",
                "Right_Hand_Roll",

                "Waist",
                "Left_Hip_Pitch",
                "Left_Hip_Roll",
                "Left_Hip_Yaw",
                "Left_Knee_Pitch",
                "Left_Ankle_Pitch",
                "Left_Ankle_Roll",
                "Right_Hip_Pitch",
                "Right_Hip_Roll",
                "Right_Hip_Yaw",
                "Right_Knee_Pitch",
                "Right_Ankle_Pitch",
                "Right_Ankle_Roll",
            ],
                preserve_order=True,
            )
        },
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel, # type: ignore
        scale=1,
        noise=Gnoise(mean=0.0, std=0.175),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
            joint_names=[
                "AAHead_yaw",
                "Head_pitch",
                "Left_Shoulder_Pitch",
                "Left_Shoulder_Roll",
                "Left_Elbow_Pitch",
                "Left_Elbow_Yaw",
                "Left_Wrist_Pitch",
                "Left_Wrist_Yaw",
                "Left_Hand_Roll",

                "Right_Shoulder_Pitch",
                "Right_Shoulder_Roll",
                "Right_Elbow_Pitch",
                "Right_Elbow_Yaw",
                "Right_Wrist_Pitch",
                "Right_Wrist_Yaw",
                "Right_Hand_Roll",

                "Waist",
                "Left_Hip_Pitch",
                "Left_Hip_Roll",
                "Left_Hip_Yaw",
                "Left_Knee_Pitch",
                "Left_Ankle_Pitch",
                "Left_Ankle_Roll",
                "Right_Hip_Pitch",
                "Right_Hip_Roll",
                "Right_Hip_Yaw",
                "Right_Knee_Pitch",
                "Right_Ankle_Pitch",
                "Right_Ankle_Roll",
            ],
                preserve_order=True,
            )
        },
    )
    actions = ObsTerm(func=mdp.last_action) # type: ignore

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 5

@configclass
class CriticCfg(ObsGroup):
    """Observations for critic group."""

    # observation terms (order preserved)
    clock = ObsTerm(func=t1_mdp.clock) # type: ignore
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel) # type: ignore
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel) # type: ignore
    projected_gravity = ObsTerm(func=mdp.projected_gravity) # type: ignore
    velocity_commands = ObsTerm(
        func=mdp.generated_commands, # type: ignore
        params={"command_name": "base_velocity"},
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos, # type: ignore
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
            joint_names=[
                "AAHead_yaw",
                "Head_pitch",
                "Left_Shoulder_Pitch",
                "Left_Shoulder_Roll",
                "Left_Elbow_Pitch",
                "Left_Elbow_Yaw",
                "Left_Wrist_Pitch",
                "Left_Wrist_Yaw",
                "Left_Hand_Roll",

                "Right_Shoulder_Pitch",
                "Right_Shoulder_Roll",
                "Right_Elbow_Pitch",
                "Right_Elbow_Yaw",
                "Right_Wrist_Pitch",
                "Right_Wrist_Yaw",
                "Right_Hand_Roll",

                "Waist",
                "Left_Hip_Pitch",
                "Left_Hip_Roll",
                "Left_Hip_Yaw",
                "Left_Knee_Pitch",
                "Left_Ankle_Pitch",
                "Left_Ankle_Roll",
                "Right_Hip_Pitch",
                "Right_Hip_Roll",
                "Right_Hip_Yaw",
                "Right_Knee_Pitch",
                "Right_Ankle_Pitch",
                "Right_Ankle_Roll",
            ],
                preserve_order=True,
            )
        },
        )

    joint_vel = ObsTerm(
        func=mdp.joint_vel, # type: ignore
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
            joint_names=[
                "AAHead_yaw",
                "Head_pitch",
                "Left_Shoulder_Pitch",
                "Left_Shoulder_Roll",
                "Left_Elbow_Pitch",
                "Left_Elbow_Yaw",
                "Left_Wrist_Pitch",
                "Left_Wrist_Yaw",
                "Left_Hand_Roll",

                "Right_Shoulder_Pitch",
                "Right_Shoulder_Roll",
                "Right_Elbow_Pitch",
                "Right_Elbow_Yaw",
                "Right_Wrist_Pitch",
                "Right_Wrist_Yaw",
                "Right_Hand_Roll",
                
                "Waist",
                "Left_Hip_Pitch",
                "Left_Hip_Roll",
                "Left_Hip_Yaw",
                "Left_Knee_Pitch",
                "Left_Ankle_Pitch",
                "Left_Ankle_Roll",
                "Right_Hip_Pitch",
                "Right_Hip_Roll",
                "Right_Hip_Yaw",
                "Right_Knee_Pitch",
                "Right_Ankle_Pitch",
                "Right_Ankle_Roll",
            ],
                preserve_order=True,
            )
        },
    )
    actions = ObsTerm(func=mdp.last_action) # type: ignore

    # privileged observations
    root_state_w = ObsTerm(func=t1_mdp.root_state_w) # type: ignore
    root_lin_vel = ObsTerm(func=mdp.root_lin_vel_w) # type: ignore
    root_ang_vel = ObsTerm(func=mdp.root_ang_vel_w) # type: ignore

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 5


@configclass
class ContactCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    
    # hard_contact_forces_lf = ObsTerm(
    #     func=t1_mdp.foot_hard_contact_forces, # type: ignore
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces_LF", 
    #                                          )},
    # )
    # hard_contact_forces_rf = ObsTerm(
    #     func=t1_mdp.foot_hard_contact_forces, # type: ignore
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces_RF", 
    #                                          )},
    # )
    
    soft_contact_forces = ObsTerm(
        func=t1_mdp.soft_contact_forces, # type: ignore
        params={"action_term_name": "physics_callback"},
    )
    foot_pos = ObsTerm(
            func=t1_mdp.foot_pos_w, # type: ignore
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*foot_link")},
        )
    

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True

@configclass
class LoggingObsCfg(ObsGroup):
    """Observations for policy group."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel) # type: ignore
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel) # type: ignore
    velocity_commands = ObsTerm(
        func=mdp.generated_commands, # type: ignore
        params={"command_name": "base_velocity"},
    )
    

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True

@configclass
class T1ObservationsCfg:
    """Observation specifications for the MDP."""
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    contact: ContactCfg = ContactCfg()
    logging: LoggingObsCfg = LoggingObsCfg()