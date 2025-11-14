# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.g1_soft.mdp as g1_mdp


@configclass
class G1ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)) # type: ignore
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # type: ignore
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, # type: ignore
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) # type: ignore
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)) # type: ignore
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)) # type: ignore
        actions = ObsTerm(func=mdp.last_action) # type: ignore
        height_scan = ObsTerm(
            func=mdp.height_scan, # type: ignore
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
    @configclass
    class ContactCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        
        # hard_contact_forces_lf = ObsTerm(
        #     func=g1_mdp.foot_hard_contact_forces, # type: ignore
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces_LF", 
        #                                          )},
        # )
        # hard_contact_forces_rf = ObsTerm(
        #     func=g1_mdp.foot_hard_contact_forces, # type: ignore
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces_RF", 
        #                                          )},
        # )
        
        soft_contact_forces = ObsTerm(
            func=g1_mdp.soft_contact_forces, # type: ignore
            params={"action_term_name": "physics_callback"},
        )

        foot_pos = ObsTerm(
            func=g1_mdp.foot_pos_w, # type: ignore
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
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

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    contact: ContactCfg = ContactCfg()
    logging: LoggingObsCfg = LoggingObsCfg()