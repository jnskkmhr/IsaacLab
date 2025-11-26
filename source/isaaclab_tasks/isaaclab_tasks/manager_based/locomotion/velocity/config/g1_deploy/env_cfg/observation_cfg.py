# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.g1_deploy.mdp as g1_mdp

"""
leggedlab
"""

@configclass
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) 
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) 
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)) 
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)) 
    actions = ObsTerm(func=mdp.last_action)
    height_scan = ObsTerm(
        func=mdp.height_scan, 
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 5 # unitree_rl_lab uses 5
        # self.history_length = 10 # legged_lab uses 10

@configclass
class CriticCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel) 
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) 
    joint_pos = ObsTerm(func=mdp.joint_pos_rel) 
    joint_vel = ObsTerm(func=mdp.joint_vel_rel) 
    actions = ObsTerm(func=mdp.last_action)
    height_scan = ObsTerm(
        func=mdp.height_scan, 
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )

    # privileged observations
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    feet_contact = ObsTerm(
        func=vel_mdp.foot_contact, 
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")}, 
        )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 5 # unitree_rl_lab uses 5
        # self.history_length = 10 # legged_lab uses 10

"""
mjlab
"""

# @configclass
# class PolicyCfg(ObsGroup):
#     """Observations for policy group."""

#     # observation terms (order preserved)
#     base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
#     base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) 
#     projected_gravity = ObsTerm(
#         func=mdp.projected_gravity,
#         noise=Unoise(n_min=-0.05, n_max=0.05),
#     )
#     joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)) 
#     joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)) 
#     actions = ObsTerm(func=mdp.last_action)
#     velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) 
#     height_scan = ObsTerm(
#         func=mdp.height_scan, 
#         params={"sensor_cfg": SceneEntityCfg("height_scanner")},
#         noise=Unoise(n_min=-0.1, n_max=0.1),
#         clip=(-1.0, 1.0),
#     )

#     def __post_init__(self):
#         self.enable_corruption = True
#         self.concatenate_terms = True

# @configclass
# class CriticCfg(ObsGroup):
#     """Observations for policy group."""

#     # observation terms (order preserved)
#     base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
#     base_ang_vel = ObsTerm(func=mdp.base_ang_vel) 
#     projected_gravity = ObsTerm(
#         func=mdp.projected_gravity,
#     )
#     velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) 
#     joint_pos = ObsTerm(func=mdp.joint_pos_rel) 
#     joint_vel = ObsTerm(func=mdp.joint_vel_rel) 
#     actions = ObsTerm(func=mdp.last_action)
#     height_scan = ObsTerm(
#         func=mdp.height_scan, 
#         params={"sensor_cfg": SceneEntityCfg("height_scanner")},
#         clip=(-1.0, 1.0),
#     )

#     # -- privileged observations
#     foot_height = ObsTerm(
#         func=vel_mdp.foot_height, 
#         params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
#     )

#     foot_airtime = ObsTerm(
#         func=vel_mdp.foot_air_time, 
#         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")}, 
#         )

#     foot_contact = ObsTerm(
#         func=vel_mdp.foot_contact, 
#         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")}, 
#         )
    
#     foot_contact_froces = ObsTerm(
#         func=vel_mdp.foot_contact_forces, 
#         params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")}, 
#         )

#     def __post_init__(self):
#         self.enable_corruption = False
#         self.concatenate_terms = True


"""
obs for logging
"""
        
@configclass
class ContactCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    
    # hard_contact_forces_lf = ObsTerm(
    #     func=g1_mdp.foot_hard_contact_forces, 
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces_LF", 
    #                                          )},
    # )
    # hard_contact_forces_rf = ObsTerm(
    #     func=g1_mdp.foot_hard_contact_forces, 
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces_RF", 
    #                                          )},
    # )
    
    # soft_contact_forces = ObsTerm(
    #     func=g1_mdp.soft_contact_forces, 
    #     params={"action_term_name": "physics_callback"},
    # )

    foot_pos = ObsTerm(
        func=g1_mdp.foot_pos_w, 
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )
    

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
    
@configclass
class LoggingObsCfg(ObsGroup):
    """Observations for policy group."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel) 
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel) 
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
    )
    

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True

@configclass
class G1ObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    contact: ContactCfg = ContactCfg()
    logging: LoggingObsCfg = LoggingObsCfg()