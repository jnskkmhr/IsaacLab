# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

# from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from . import physics_callback_actions
from . import mpc_actions

"""
Physics callbacks
"""

@configclass
class PhysicsCallbackActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = physics_callback_actions.PhysicsCallbackAction
    body_names: list[str] = MISSING # type: ignore
    """List of joint names or regex expressions that the action will be mapped to."""
    max_terrain_level: int = MISSING # type: ignore
    """Maximum terrain stiffness level. This number is multiplied to terrain stiffness calculated by RFT."""
    backend: Literal["2D", "3D"] = "2D"
    """The RFT backend to use. Options are '2D' or '3D'."""
    
    
"""
MPC controller
"""
@configclass
class BlindLocomotionMPCActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = mpc_actions.BlindLocomotionMPCAction

    joint_names: list[str] = MISSING # type: ignore
    """List of joint names or regex expressions that the action will be mapped to."""
    action_range: tuple[float, float] | tuple[tuple[float, ...], tuple[float, ...]] = (-1.0, 1.0)
    """action range to deal with assymetric action space. """
    negative_action_clip_idx: list[int] = None  # type: ignore
    """List of indices of the action that negative action value should be clipped."""
    command_name: str = "base_velocity"
    """Name of the command to be used for the action term."""
    nominal_height: float = 0.55
    """Reference height of the robot."""
    nominal_swing_height : float = 0.08
    """Nominal swing height of the robot."""
    nominal_stepping_frequency: float = 1.0
    """Nominal stepping frequency of the robot."""
    horizon_length: int = 10
    """Horizon length of the robot."""
    friction_cone_coef: float = 1.0
    """Friction cone coefficient of the robot."""

    control_iteration_between_mpc: int = 10
    """Number of control iterations between MPC updates."""

    # # -- horizon is entire walking step
    # nominal_mpc_dt: float = 0.04
    # """Nominal MPC dt of the robot."""
    # double_support_duration: int = 1 # 0.05s double support
    # """Double support duration of the robot."""
    # single_support_duration: int = 5 # 0.2s single support
    # """Single support duration of the robot."""

    # -- horizon is half of walking step (one foot swing)
    nominal_mpc_dt: float = 0.025
    """Nominal MPC dt of the robot."""
    double_support_duration: int = 2 # 0.05s double support
    """Double support duration of the robot."""
    single_support_duration: int = 8 # 0.2s single support
    """Single support duration of the robot."""

    nominal_cp1_coef: float = 1/3
    """Nominal cp1 coefficient of the robot."""
    nominal_cp2_coef: float = 2/3
    """Nominal cp2 coefficient of the robot."""
    foot_placement_planner: Literal["LIP", "Raibert"] = "Raibert"
    """Foot placement planner to be used. Can be either "LIP" or "Raibert"."""
    swing_foot_reference_frame: Literal["world", "base"] = "base"
    """Swing foot reference frame to be used. Can be either "world" or "base"."""
    gait_num: int = 2 # 1: stand, 2: walk
    """Type of gait to be used. 1: stand, 2: walk."""

    debug_vis: bool = False

@configclass
class BlindLocomotionMPCActionCfgDyn(BlindLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.BlindLocomotionMPCActionDyn

@configclass
class BlindLocomotionMPCActionCfgSwing(BlindLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.BlindLocomotionMPCActionSwing

@configclass
class BlindLocomotionMPCActionCfgGait(BlindLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.BlindLocomotionMPCActionGait

@configclass
class BlindLocomotionMPCActionCfgDynGait(BlindLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.BlindLocomotionMPCActionDynGait

@configclass
class BlindLocomotionMPCActionCfgDynSwing(BlindLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.BlindLocomotionMPCActionDynSwing

@configclass
class BlindLocomotionMPCActionCfgSwingGait(BlindLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.BlindLocomotionMPCActionSwingGait

@configclass
class BlindLocomotionMPCActionCfgSimpleDynSwingGait(BlindLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.BlindLocomotionMPCActionSimpleDynSwingGait

@configclass
class BlindLocomotionMPCActionCfgResAll(BlindLocomotionMPCActionCfg):
    class_type: type[ActionTerm] = mpc_actions.BlindLocomotionMPCActionResAll