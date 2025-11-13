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


"""
Gait
"""
def should_stand(env: ManagerBasedRLEnv, zero_threshold: float = 0.05) -> torch.Tensor:
    command = env.command_manager.get_command("base_velocity")
    return torch.norm(command, dim=1) < zero_threshold


def should_walk(env: ManagerBasedRLEnv, zero_threshold: float = 0.05) -> torch.Tensor:
    command = env.command_manager.get_command("base_velocity")
    return torch.norm(command, dim=1) >= zero_threshold

def conditioned_get_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    # TODO: decide phase generator or env.get_phase 
    # phase = env.get_phase()
    phase = env.command_manager.get_command("phase")
    standing = should_stand(env)
    return torch.where(standing, torch.zeros_like(phase), phase)

# TODO: decide phase generator or env.get_phase 
# def clock(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """Clock time using sin and cos from the phase of the simulation."""
#     phase = env.get_phase()
#     condition = should_walk(env).unsqueeze(1)
#     return torch.cat(
#         [
#             torch.sin(2 * torch.pi * phase).unsqueeze(1) * condition,
#             torch.cos(2 * torch.pi * phase).unsqueeze(1) * condition,
#         ],
#         dim=1,
#     ).to(env.device)

def clock(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Clock time using sin and cos from the phase of the simulation."""
    phase = env.command_manager.get_command("phase")
    condition = should_walk(env).unsqueeze(1)  # shape: (num_envs, 1)
    return torch.cat(
        [
            torch.cos(2 * torch.pi * phase).unsqueeze(1) * condition,
            torch.sin(2 * torch.pi * phase).unsqueeze(1) * condition,
        ],
        dim=1,
    )

"""
root state
"""
def root_state_w(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root state in world frame."""
    robot = env.scene[asset_cfg.name]
    robot: Articulation
    return torch.cat([robot.data.root_pos_w, robot.data.root_quat_w], dim=-1).to(
        env.device
    )

def base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Return the base height (z position) of the robot's root.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)

def base_height_general(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("sensor"), 
) -> torch.Tensor:
    """
    Return the base height (z position) of the robot's root above the ground
    We use distance from torso to stance foot to handle uneven terrain.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    contacts = (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=2) > 1.0).float()
    root_pos_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    body_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    height = (root_pos_z - contacts*body_pos_z).max(dim=1).values
    return height.unsqueeze(-1)


def base_mass_scaled(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the mass scale factor from the scale_base_mass event for the base body.

    This function returns the scale factor applied to the base body mass during
    the scale_base_mass event. The scale factor represents how much the mass was
    scaled from its default value.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The mass scale factor for the base body [num_envs, 1].
    """

    # Access the scale_base_mass event term from the event manager
    # We need to find the event term that uses randomize_rigid_body_mass_class
    scale_base_mass_term = None

    # Try to find the event term by looking through the startup mode terms
    # We'll use a more direct approach by checking if the event manager has the term
    try:
        # Get the event term configuration for scale_base_mass
        term_cfg = env.event_manager.get_term_cfg("scale_base_mass")
        if hasattr(term_cfg, 'func') and hasattr(term_cfg.func, '__name__'):
            if term_cfg.func.__name__ == 'randomize_rigid_body_mass_class':
                scale_base_mass_term = term_cfg.func
    except ValueError:
        # If scale_base_mass term is not found, try to find any mass randomization term
        for mode in ["startup", "reset", "interval"]:
            if mode in env.event_manager.active_terms:
                for term_name in env.event_manager.active_terms[mode]:
                    try:
                        term_cfg = env.event_manager.get_term_cfg(term_name)
                        if (hasattr(term_cfg, 'func') and hasattr(term_cfg.func, '__name__')
                                and term_cfg.func.__name__ == 'randomize_rigid_body_mass_class'):
                            scale_base_mass_term = term_cfg.func
                            break
                    except ValueError:
                        continue
                if scale_base_mass_term is not None:
                    break

    if scale_base_mass_term is None:
        # If no scale_base_mass event is found, return zeros
        return torch.zeros(env.num_envs, 1, device=env.device)

    # Get the mass noise (which for scale operation is scale_factor - 1.0)
    mass_noise = scale_base_mass_term.get_mass_noise_for_body(asset_cfg.body_ids[0])

    # Convert back to scale factor (add 1.0 to get the actual scale factor)
    scale_factor = mass_noise + 1.0

    return scale_factor.unsqueeze(-1)


def base_com_xyz_offset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the center of mass offset from the add_base_com_xyz event.

    This function returns the CoM offset that was applied by the add_base_com_xyz 
    event during initialization. It retrieves the stored noise values directly 
    from the event term instead of calculating the difference between current 
    and default CoM.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The CoM offset for the base body [num_envs, 3] (x, y, z).
    """
    # Access the add_base_com_xyz event term from the event manager
    # We need to find the event term that uses randomize_rigid_body_com_class
    add_base_com_xyz_term = None

    try:
        # Get the event term configuration for add_base_com_xyz
        term_cfg = env.event_manager.get_term_cfg("add_base_com_xyz")
        if hasattr(term_cfg, 'func'):
            # Check if it's a class instance (already instantiated)
            if hasattr(term_cfg.func, 'get_com_noise_for_body'):
                add_base_com_xyz_term = term_cfg.func
            # Check if it's a class that needs to be instantiated
            elif hasattr(term_cfg.func, '__name__') and term_cfg.func.__name__ == 'randomize_rigid_body_com_class':
                # Instantiate the class if it hasn't been instantiated yet
                add_base_com_xyz_term = term_cfg.func(cfg=term_cfg, env=env)
                # Update the term_cfg to store the instance
                term_cfg.func = add_base_com_xyz_term
    except ValueError:
        # If add_base_com_xyz term is not found, try to find any CoM randomization term
        for mode in ["startup", "reset", "interval"]:
            if mode in env.event_manager.active_terms:
                for term_name in env.event_manager.active_terms[mode]:
                    try:
                        term_cfg = env.event_manager.get_term_cfg(term_name)
                        if (hasattr(term_cfg, 'func')
                            and ((hasattr(term_cfg.func, 'get_com_noise_for_body')) or
                                 (hasattr(term_cfg.func, '__name__') and term_cfg.func.__name__ == 'randomize_rigid_body_com_class'))):
                            if hasattr(term_cfg.func, 'get_com_noise_for_body'):
                                add_base_com_xyz_term = term_cfg.func
                            else:
                                # Instantiate the class
                                add_base_com_xyz_term = term_cfg.func(cfg=term_cfg, env=env)
                                term_cfg.func = add_base_com_xyz_term
                            break
                    except ValueError:
                        continue
                if add_base_com_xyz_term is not None:
                    break

    if add_base_com_xyz_term is None:
        # If no add_base_com_xyz event is found, return zeros
        return torch.zeros(env.num_envs, 3, device=env.device)

    # Get the CoM noise values stored by the event term
    # The noise is stored as (num_envs, num_bodies, 3), we want the first body (base)
    com_noise = add_base_com_xyz_term.get_com_noise_for_body(asset_cfg.body_ids[0])

    return com_noise