# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards of the G1 29-DoF locomotion task among trees.

The rigid task's terms are kept with their weights; the five that read the contact sensor are
re-expressed through the foot kinematics of :mod:`..mdp`, since a coupled solver supports no
contact sensor. Four of them are exact counterparts. The landing penalty is not: it now measures
the touchdown speed [m/s] rather than the impact force [N], so it carries its own weight.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from ...g1_29dof_rigid.env_cfg.reward_cfg import G1RewardsCfg
from .. import mdp as tree_mdp

FOOT_CFG = SceneEntityCfg("robot", body_names=[".*ankle_roll.*"], preserve_order=True)
"""Feet the contact-driven terms act on."""

NON_FOOT_CFG = SceneEntityCfg("robot", body_names="(?!.*ankle.*).*")
"""Links that should stay off the ground."""


@configclass
class G1TreeRewardsCfg(G1RewardsCfg):
    """Reward terms for the MDP."""

    undesired_contacts = RewTerm(
        func=tree_mdp.undesired_ground_proximity,
        weight=-1.0,
        params={"asset_cfg": NON_FOOT_CFG, "height_threshold": 0.1},
    )
    feet_pitch_contact = RewTerm(
        func=tree_mdp.feet_pitch_contact,
        weight=-4.0,
        params={"asset_cfg": FOOT_CFG},
    )
    feet_air_time = RewTerm(
        func=tree_mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": FOOT_CFG,
            "threshold": 0.5,
            "velocity_threshold": 0.05,
        },
    )
    feet_slide = RewTerm(
        func=tree_mdp.feet_slide,
        weight=-0.25,
        params={"asset_cfg": FOOT_CFG},
    )
    # The force-based weight of -5e-3 acted on footfalls of a few hundred newtons; a touchdown of
    # the same severity is about a metre per second, so the weight is rescaled by that ratio.
    contact_impulse = RewTerm(
        func=tree_mdp.soft_landing,
        weight=-1.0,
        params={"command_name": "base_velocity", "asset_cfg": FOOT_CFG, "command_threshold": 0.05},
    )
