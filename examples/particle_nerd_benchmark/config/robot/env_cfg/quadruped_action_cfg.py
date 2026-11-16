# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Four independent damped least-squares leg IK action terms."""

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass


@configclass
class QuadrupedActionsCfg:
    LF = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["FL_.*"],
        body_name="FL_foot",
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
    )
    RF = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["FR_.*"],
        body_name="FR_foot",
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
    )
    LH = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["RL_.*"],
        body_name="RL_foot",
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
    )
    RH = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["RR_.*"],
        body_name="RR_foot",
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
    )
