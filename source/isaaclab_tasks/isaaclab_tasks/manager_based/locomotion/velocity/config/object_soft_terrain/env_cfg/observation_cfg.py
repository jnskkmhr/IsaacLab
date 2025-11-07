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
import isaaclab_tasks.manager_based.locomotion.velocity.config.object_soft_terrain.mdp as object_mdp


@configclass
class ObjectObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        root_pos = ObsTerm(
            func=mdp.root_pos_w, # type: ignore
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, # type: ignore
            noise=Unoise(n_min=-0.1, n_max=0.1), 
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, # type: ignore
            noise=Unoise(n_min=-0.2, n_max=0.2),
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, # type: ignore
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, # type: ignore
            params={"command_name": "base_velocity"}, 
        ) 

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            
    @configclass
    class ContactCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        
        # hard_contact_forces_lf = ObsTerm(
        #     func=hector_mdp.foot_hard_contact_forces, # type: ignore
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces_LF", 
        #                                          )},
        # )
        # hard_contact_forces_rf = ObsTerm(
        #     func=hector_mdp.foot_hard_contact_forces, # type: ignore
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces_RF", 
        #                                          )},
        # )
        
        soft_contact_forces = ObsTerm(
            func=object_mdp.soft_contact_forces, # type: ignore
            params={"action_term_name": "physics_callback"},
        )
        

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    contact: ContactCfg = ContactCfg()