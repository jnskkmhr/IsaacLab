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



@configclass
class PhysicsCallbackActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = physics_callback_actions.PhysicsCallbackAction
    body_names: list[str] = MISSING # type: ignore
    """List of joint names or regex expressions that the action will be mapped to."""