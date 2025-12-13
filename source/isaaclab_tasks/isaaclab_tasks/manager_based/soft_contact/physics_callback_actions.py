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

from isaaclab_tasks.manager_based.soft_contact.soft_contact_model import (
    PoppySeedCPCfg, PoppySeedLPCfg, RFT_2D, Material3DRFTCfg, RFT_3D
)


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

import time
from contextlib import contextmanager
@contextmanager
def timeblock(name="block"):
    t0 = time.time()
    yield
    dt = 1000 * (time.time() - t0)
    print(f"[TIME] {name}: {dt:.3f} ms")

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
        if self.cfg.backend == "2D":
            material_cfg = PoppySeedCPCfg()
            num_bodies = len(body_ids)
            self.contact_solver = RFT_2D(
                material_cfg=material_cfg, 
                num_envs=self.num_envs, 
                num_bodies=num_bodies, 
                device=self.device, 
                dt=env.physics_dt,
                contact_threshold=self.cfg.contact_threshold,
                enable_ema_filter=self.cfg.enable_ema_filter,
                intruder_cfg=self.cfg.intruder_geometry_cfg,
                )
        elif self.cfg.backend == "3D":
            material_cfg = Material3DRFTCfg()
            num_bodies = len(body_ids)
            self.contact_solver = RFT_3D(
                material_cfg=material_cfg, 
                num_envs=self.num_envs, 
                num_bodies=num_bodies, 
                device=self.device, 
                dt=env.physics_dt,
                contact_threshold=self.cfg.contact_threshold,
                enable_ema_filter=self.cfg.enable_ema_filter,
                intruder_cfg=self.cfg.intruder_geometry_cfg,
                )
        else:
            raise ValueError(f"Unsupported RFT backend: {self.cfg.backend}")
        
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
        """
        get body position in local env frame except z pos is measured wrt world frame.
        """
        # body_pos = self._asset.data.body_com_pose_w[:, self._body_ids, :3]
        body_pos = self._asset.data.body_link_pose_w[:, self._body_ids, :3]
        body_pos[:, :, :2] -= self._env.scene.env_origins.unsqueeze(1)[:, :, :2]
        return body_pos
    
    @property
    def body_quat(self)->torch.Tensor:
        # return self._asset.data.body_com_pose_w[:, self._body_ids, 3:7]
        return self._asset.data.body_link_pose_w[:, self._body_ids, 3:7]

    @property
    def body_lin_vel(self)->torch.Tensor:
        return self._asset.data.body_com_vel_w[:, self._body_ids, :3]
        # return self._asset.data.body_link_vel_w[:, self._body_ids, :3]

    @property
    def body_ang_vel(self)->torch.Tensor:
        return self._asset.data.body_com_vel_w[:, self._body_ids, 3:6]
        # return self._asset.data.body_link_vel_w[:, self._body_ids, 3:6]

    """
    operations.
    """
    
    def process_actions(self, actions: torch.Tensor):
        pass
    
    def apply_actions(self):

        if self.cfg.disable:
            return
        body_pos = self.body_pos.clone()
        body_quat = self.body_quat.clone()
        body_lin_vel = self.body_lin_vel.clone()
        body_ang_vel = self.body_ang_vel.clone()
        self.contact_solver.update(body_pos, body_quat, body_lin_vel, body_ang_vel)
        
        # self.contact_wrench = self.contact_solver.contact_wrench # (num_envs, num_bodies, 6)
        self.contact_wrench_b = self.contact_solver.contact_wrench_b # (num_envs, num_bodies, 6)
        self._asset.set_external_force_and_torque(
            forces = self.contact_wrench_b[:, :, :3],
            torques = self.contact_wrench_b[:, :, 3:6],
            body_ids = self._body_ids,
            # is_global=True,
        )
        
        # with timeblock("apply action"):
        #     if self.cfg.disable:
        #         return
        #     with timeblock("body pos"):
        #         body_pos = self.body_pos.clone()
        #     with timeblock("body quat"):
        #         body_quat = self.body_quat.clone()
        #     with timeblock("body lin vel"):
        #         body_lin_vel = self.body_lin_vel.clone()
        #     with timeblock("body ang vel"):
        #         body_ang_vel = self.body_ang_vel.clone()
        #     with timeblock("contact solver update"):
        #         self.contact_solver.update(body_pos, body_quat, body_lin_vel, body_ang_vel)
            
        #     # self.contact_wrench = self.contact_solver.contact_wrench # (num_envs, num_bodies, 6)
        #     self.contact_wrench_b = self.contact_solver.contact_wrench_b # (num_envs, num_bodies, 6)
        #     self._asset.set_external_force_and_torque(
        #         forces = self.contact_wrench_b[:, :, :3],
        #         torques = self.contact_wrench_b[:, :, 3:6],
        #         body_ids = self._body_ids,
        #         # is_global=True,
        #     )
    
    def reset(self, env_ids: torch.Tensor):
        self.contact_wrench_b[env_ids] = 0.0
        self.contact_solver.reset(env_ids)
    