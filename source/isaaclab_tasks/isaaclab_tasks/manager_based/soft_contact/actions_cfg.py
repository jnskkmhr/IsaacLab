# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

# from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from .soft_contact_model import IntruderGeometryCfg
from . import physics_callback_actions



@configclass
class PhysicsCallbackActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = physics_callback_actions.PhysicsCallbackAction
    body_names: list[str] = MISSING # type: ignore
    """List of joint names or regex expressions that the action will be mapped to."""
    backend: Literal["2D", "3D"] = "3D"
    """The RFT backend to use. Options are '2D' or '3D'."""
    disable: bool = False
    """Whether to disable this action term."""
    enable_ema_filter: bool = True
    """Whether to enable an exponential moving average filter on the input actions."""
    contact_threshold: float = 10.0
    """Threshold for contact detection (N)."""
    intruder_geometry_cfg: IntruderGeometryCfg = IntruderGeometryCfg()