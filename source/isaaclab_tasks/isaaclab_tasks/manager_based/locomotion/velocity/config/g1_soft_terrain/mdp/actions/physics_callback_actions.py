# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1_soft_terrain.mdp import PoppySeedCPCfg, PoppySeedLPCfg, RFT_EMF

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg


class PhysicsCallbackAction(ActionTerm):

    cfg: actions_cfg.PhysicsCallbackActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    
    
    def __init__(self, cfg: actions_cfg.PhysicsCallbackActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        
        # process body ids
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_names)
        self._body_ids = body_ids
        
        # action buffer
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        
        # wrench buffer
        self.contact_wrench = torch.zeros(self.num_envs, len(body_ids), 6, device=self.device)
        self.contact_wrench_b = torch.zeros(self.num_envs, len(body_ids), 6, device=self.device)
        
        # get physics backend
        material_cfg = PoppySeedCPCfg()
        num_bodies = len(body_ids)
        num_contact_points = int(40*40) # 40 x 40 grid
        self.rft = RFT_EMF(cfg=material_cfg, num_envs=self.num_envs, num_bodies=num_bodies, num_contact_points=num_contact_points, device=self.device)
        
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
    
    @property
    def body_pos(self)->torch.Tensor:
        return self._asset.data.body_com_pose_w[:, self._body_ids, :3] - self._env.scene.env_origins.unsqueeze(1)
    
    @property
    def body_quat(self)->torch.Tensor:
        return self._asset.data.body_com_pose_w[:, self._body_ids, 3:7]
    
    @property
    def body_lin_vel(self)->torch.Tensor:
        return self._asset.data.body_com_vel_w[:, self._body_ids, :3]
    
    @property
    def body_ang_vel(self)->torch.Tensor:
        return self._asset.data.body_com_vel_w[:, self._body_ids, 3:6]
    
    """
    operations.
    """
    
    def process_actions(self, actions: torch.Tensor):
        pass
    
    def apply_actions(self):
        body_pos = self.body_pos.clone()
        body_quat = self.body_quat.clone()
        body_lin_vel = self.body_lin_vel.clone()
        body_ang_vel = self.body_ang_vel.clone()
        self.rft.update(body_pos, body_quat, body_lin_vel, body_ang_vel)
        
        self.contact_wrench = self.rft.contact_wrench # (num_envs, num_bodies, 6)
        self.contact_wrench_b = self.rft.contact_wrench_b # (num_envs, num_bodies, 6)
        self._asset.set_external_force_and_torque(
            forces = self.contact_wrench_b[:, :, :3],
            torques = self.contact_wrench_b[:, :, 3:6],
            body_ids = self._body_ids,
            is_global=False,
        )
    
    def reset(self, env_ids: torch.Tensor):
        self.contact_wrench_b[env_ids] = 0.0
        self.rft.reset(env_ids)
    