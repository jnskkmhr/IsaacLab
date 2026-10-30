# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.soft_contact.mdp as contact_mdp


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        root_pos = ObsTerm(func=mdp.root_pos_w)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

    @configclass
    class LoggingCfg(ObsGroup):
        """Observations for logging group."""

        # observation terms (order preserved)
        root_pos = ObsTerm(func=mdp.root_pos_w)
        root_quat = ObsTerm(func=mdp.root_quat_w)
        root_lin_vel = ObsTerm(func=mdp.root_lin_vel_w)
        root_ang_vel = ObsTerm(func=mdp.root_ang_vel_w)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # contact_forces = ObsTerm(func=contact_mdp.foot_contact_forces_raw,
        #                          params={
        #                              "action_term_name": "physics_callback",
        #                              "threshold": 40.0,
        #                          })
        contact_forces = ObsTerm(
            func=contact_mdp.foot_contact_forces_raw_hybrid,
            params={
                "rigid_contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
                "soft_contact_sensor_name": "physics_callback",
                "rigid_force_filter_threshold": 5.0,
                "soft_force_filter_threshold": 40.0,
            },
        )

    @configclass
    class ContactAnglesCfg(ObsGroup):
        """Observations for logging group."""

        # observation terms (order preserved)
        contact_angle = ObsTerm(
            func=contact_mdp.contact_angle,
            params={
                "action_term_name": "physics_callback",
            },
        )

    @configclass
    class ContactVectorsCfg(ObsGroup):
        """Observations for logging group."""

        # observation terms (order preserved)
        contact_angle = ObsTerm(
            func=contact_mdp.contact_coordinate_dir,
            params={
                "action_term_name": "physics_callback",
            },
        )

    @configclass
    class ContactPointPosCfg(ObsGroup):
        """Observations for logging group."""

        # observation terms (order preserved)
        contact_angle = ObsTerm(
            func=contact_mdp.contact_point_pos,
            params={
                "action_term_name": "physics_callback",
            },
        )

    @configclass
    class ContactPointVelCfg(ObsGroup):
        """Observations for logging group."""

        # observation terms (order preserved)
        contact_angle = ObsTerm(
            func=contact_mdp.contact_point_vel,
            params={
                "action_term_name": "physics_callback",
            },
        )

    @configclass
    class RootVelCfg(ObsGroup):
        """Observations for logging group."""

        # observation terms (order preserved)
        root_lin_vel = ObsTerm(
            func=contact_mdp.root_lin_vel,
            params={
                "action_term_name": "physics_callback",
            },
        )
        # root_lin_vel = ObsTerm(func=mdp.root_lin_vel_w)

    @configclass
    class ContactPointForceCfg(ObsGroup):
        """Observations for logging group."""

        # observation terms (order preserved)
        contact_angle = ObsTerm(
            func=contact_mdp.contact_point_force,
            params={
                "action_term_name": "physics_callback",
            },
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg(enable_corruption=True, concatenate_terms=True)
    critic: CriticCfg = CriticCfg(enable_corruption=False, concatenate_terms=True)
    logging: LoggingCfg = LoggingCfg(enable_corruption=True, concatenate_terms=True)
    # contact_angles: ContactAnglesCfg = ContactAnglesCfg(enable_corruption=False, concatenate_terms=True)
    # contact_vectors: ContactVectorsCfg = ContactVectorsCfg(enable_corruption=False, concatenate_terms=True)
    # contact_point_pos: ContactPointPosCfg = ContactPointPosCfg(enable_corruption=False, concatenate_terms=True)
    # contact_point_vel: ContactPointVelCfg = ContactPointVelCfg(enable_corruption=False, concatenate_terms=True)
    # contact_point_force: ContactPointForceCfg = ContactPointForceCfg(enable_corruption=False, concatenate_terms=True)
    # root_lin_vel: RootVelCfg = RootVelCfg(enable_corruption=False, concatenate_terms=True)
