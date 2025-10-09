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


@configclass
class ObjectObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
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
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()