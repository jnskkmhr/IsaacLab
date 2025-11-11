"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.sensors import ContactSensor

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def clock(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Clock time using sin and cos from the phase of the simulation."""
    phase = env.get_phase()
    return torch.cat(
        [
            torch.sin(2 * torch.pi * phase).unsqueeze(1),
            torch.cos(2 * torch.pi * phase).unsqueeze(1),
        ],
        dim=1,
    ).to(env.device)


def root_state_w(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root state in world frame."""
    robot = env.scene[asset_cfg.name]
    robot: Articulation
    return torch.cat([robot.data.root_pos_w, robot.data.root_quat_w], dim=-1).to(
        env.device
    )

"""
root states
"""

def base_pos_z(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    body_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height = (root_pos_z - body_pos_z).max(dim=1).values 
    return height.unsqueeze(-1).to(env.device) # (num_envs, 1)

"""
body states
"""

def body_mass_states(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base mass states."""
    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses().clone()
    coms = asset.root_physx_view.get_coms().clone()

    mass_states = torch.cat(
        [masses[:, asset_cfg.body_ids].unsqueeze(-1), coms[:, asset_cfg.body_ids, :]], dim=-1
    ) # (num_envs, num_bodies, 4)
    return mass_states.view(env.num_envs, -1).to(env.device) # (num_envs, num_bodies*4)