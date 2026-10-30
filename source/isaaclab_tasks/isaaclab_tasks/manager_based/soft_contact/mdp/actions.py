# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict
from dataclasses import MISSING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

class VelocityStage(TypedDict):
    step: int = 0
    lin_vel_x: tuple[float, float] | None
    lin_vel_y: tuple[float, float] | None
    lin_vel_z: tuple[float, float] | None
    ang_vel_z: tuple[float, float] | None


class VelocitySetAction(ActionTerm):

    cfg: VelocitySetActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    
    
    def __init__(self, cfg: VelocitySetActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        
        # action buffer
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

    
    """
    properties.
    """

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
 
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    operations.
    """
    
    def process_actions(self, actions: torch.Tensor):
        pass
    
    def apply_actions(self):

        # get default root state
        root_states = self._asset.data.default_root_state.clone()
        
        velocities = torch.zeros((self.num_envs, 6), device=self._asset.device) + root_states[:, 7:13]
        for stage in self.cfg.velocity_stages:
            # step_counter = self._env.common_step_counter % self._env.episode_length_steps
            step_counter = self._env.episode_length_buf[0] # type: ignore
            if step_counter > stage["step"]: # type: ignore
                if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
                    velocities[:, 0] = torch.empty(self.num_envs, device=self._asset.device).uniform_(stage["lin_vel_x"][0], stage["lin_vel_x"][1])
                if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
                    velocities[:, 1] = torch.empty(self.num_envs, device=self._asset.device).uniform_(stage["lin_vel_y"][0], stage["lin_vel_y"][1])
                if "lin_vel_z" in stage and stage["lin_vel_z"] is not None:
                    velocities[:, 2] = torch.empty(self.num_envs, device=self._asset.device).uniform_(stage["lin_vel_z"][0], stage["lin_vel_z"][1])
                if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
                    velocities[:, 5] = torch.empty(self.num_envs, device=self._asset.device).uniform_(stage["ang_vel_z"][0], stage["ang_vel_z"][1])

        # set into the physics simulation
        self._asset.write_root_velocity_to_sim(velocities) # type: ignore

@configclass
class VelocitySetActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = VelocitySetAction
    velocity_stages: list[VelocityStage] = MISSING  # type: ignore