# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None) # type: ignore
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data

"""
joint randomization
"""

def reset_robot_upper_joints_from_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # envCfg:ManagerBasedEnvCfg,
    # joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Reset the designated upper body joints to 90% of their joint limits.    
    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        joint_names: List of joint names to reset.
        asset_cfg: SceneEntityCfg for the robot asset.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Get joint indices for the specified joint names
    # joint_indices = []
    # for name in joint_names:
    #     idx = asset.joint_names.index(name) if name in asset.joint_names else None
    #     if idx is not None:
    #         joint_indices.append(idx)
    # if not joint_indices:
    #     return  # nothing to do    # joint_indices = torch.tensor(joint_indices, device=asset.device, dtype=torch.long)
    joint_indices = asset_cfg.joint_ids    # Get joint limits for the selected joints
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_indices, :]
    # Compute 90% of the way from lower to upper limit
    lower = joint_pos_limits[..., 0]
    upper = joint_pos_limits[..., 1]
    # Randomly sample from lower + 0.1*(upper-lower) to lower + 0.9*(upper-lower) for each joint
    min_pos = lower + 0.1 * (upper - lower)
    max_pos = lower + 0.9 * (upper - lower)
    target_pos = math_utils.sample_uniform(min_pos, max_pos, lower.shape, device=lower.device)    
    # Set velocities to zero for these joints
    target_vel = torch.zeros_like(target_pos, device=asset.device)

    # Get current joint positions/velocities for all joints
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()    
    # Overwrite only the selected joints
    joint_pos[:, joint_indices] = target_pos
    joint_vel[:, joint_indices] = target_vel

    # Set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)

"""
joint drive reset (for non-actuated dofs)
"""
def reset_joints_target_by_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # get default joint state
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    # scale these values randomly
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set joint target pos
    # asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    # asset.set_joint_velocity_target(joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_joints_target_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # get default joint state
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set joint target pos
    # asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.set_joint_position_target(joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    # asset.set_joint_velocity_target(joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)

"""
mass randomization.
"""

class randomize_rigid_body_com_class(ManagerTermBase):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    This class allows randomizing the center of mass of the bodies of the asset. The class samples random values from the
    given ranges and adds them to the current CoM values.

    The class stores the randomized noise values for later access, allowing you to retrieve the exact CoM offset
    that was applied to each body in each environment.

    .. note::
        This class uses CPU tensors to assign the CoM. It is recommended to use this class
        only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not an Articulation.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, Articulation):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_com' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'. Only Articulation assets are supported."
            )

        # resolve body indices
        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            self.body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

        # storage for randomized CoM noise values
        # shape: (num_envs, num_bodies, 3) - stores the CoM offset applied to each body
        self.com_noise = torch.zeros(
            (env.scene.num_envs, len(self.body_ids), 3),
            dtype=torch.float32,
            device="cpu"
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        com_range: dict[str, tuple[float, float]],
    ):
        """Apply CoM randomization and store the noise values.

        Args:
            env: The environment instance.
            env_ids: The environment indices to randomize.
            asset_cfg: The asset configuration.
            com_range: Dictionary with 'x', 'y', 'z' keys and (min, max) tuples for CoM offset ranges.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # sample random CoM values
        range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device="cpu")
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)
        # get the current com of the bodies (num_assets, num_bodies)
        coms = self.asset.root_physx_view.get_coms().clone()

        # store the noise values (the random samples that will be added)
        self.com_noise = rand_samples.clone().detach()

        # Randomize the com in range
        coms[:, self.body_ids, :3] += rand_samples

        # Set the new coms
        self.asset.root_physx_view.set_coms(coms, env_ids)

    def get_com_noise(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get the stored CoM noise values.

        Args:
            env_ids: The environment indices to retrieve noise for. If None, returns all environments.

        Returns:
            The CoM noise values. Shape: (num_envs, num_bodies, 3) if env_ids is None,
            otherwise (len(env_ids), num_bodies, 3).
        """
        if env_ids is None:
            return self.com_noise
        else:
            return self.com_noise[env_ids]

    def get_com_noise_for_body(self, body_id: int, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get the stored CoM noise values for a specific body.

        Args:
            body_id: The body index to retrieve noise for.
            env_ids: The environment indices to retrieve noise for. If None, returns all environments.

        Returns:
            The CoM noise values for the specified body. Shape: (num_envs, 3) if env_ids is None,
            otherwise (len(env_ids), 3).
        """
        if env_ids is None:
            return self.com_noise[:, body_id]
        else:
            return self.com_noise[env_ids, body_id]

    def get_com_noise_for_axis(self, axis: str, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get the stored CoM noise values for a specific axis.

        Args:
            axis: The axis to retrieve noise for ('x', 'y', or 'z').
            env_ids: The environment indices to retrieve noise for. If None, returns all environments.

        Returns:
            The CoM noise values for the specified axis. Shape: (num_envs, num_bodies) if env_ids is None,
            otherwise (len(env_ids), num_bodies).
        """
        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis not in axis_map:
            raise ValueError(f"Invalid axis '{axis}'. Must be one of 'x', 'y', 'z'.")

        axis_idx = axis_map[axis]
        if env_ids is None:
            return self.com_noise[:, :, axis_idx]
        else:
            return self.com_noise[env_ids, :, axis_idx]
        
class randomize_rigid_body_mass_class(ManagerTermBase):
    """Randomize the mass of the bodies by adding, scaling, or setting random values.

    This class allows randomizing the mass of the bodies of the asset. The class samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    If the ``recompute_inertia`` flag is set to ``True``, the class recomputes the inertia tensor of the bodies
    after setting the mass. This is useful when the mass is changed significantly, as the inertia tensor depends
    on the mass. It assumes the body is a uniform density object. If the body is not a uniform density object,
    the inertia tensor may not be accurate.

    The class stores the randomized noise values for later access, allowing you to retrieve the exact noise
    that was applied to each body in each environment.

    .. tip::
        This class uses CPU tensors to assign the body masses. It is recommended to use this class
        only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_mass' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # resolve body indices
        if self.asset_cfg.body_ids == slice(None):
            self.body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
        else:
            self.body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

        # storage for randomized noise values
        # shape: (num_envs, num_bodies) - stores the noise applied to each body
        self.mass_noise = torch.zeros(
            (env.scene.num_envs, len(self.body_ids)),
            dtype=torch.float32,
            device="cpu"
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        recompute_inertia: bool = True,
    ):
        """Apply mass randomization and store the noise values.

        Args:
            env: The environment instance.
            env_ids: The environment indices to randomize.
            asset_cfg: The asset configuration.
            mass_distribution_params: The distribution parameters for mass randomization.
            operation: The operation to perform on the mass values.
            distribution: The distribution to sample from.
            recompute_inertia: Whether to recompute inertia tensors after mass change.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # get the current masses of the bodies (num_assets, num_bodies)
        masses = self.asset.root_physx_view.get_masses()

        # apply randomization on default values
        # this is to make sure when calling the function multiple times, the randomization is applied on the
        # default values and not the previously randomized values
        masses[env_ids[:, None], self.body_ids] = self.asset.data.default_mass[env_ids[:, None], self.body_ids].clone()

        # sample from the given range and store the noise
        # note: we modify the masses in-place for all environments
        #   however, the setter takes care that only the masses of the specified environments are modified
        masses_before = masses[env_ids[:, None], self.body_ids].clone()
        masses = _randomize_prop_by_op(
            masses, mass_distribution_params, env_ids, self.body_ids, operation=operation, distribution=distribution
        )
        masses_after = masses[env_ids[:, None], self.body_ids]

        # compute and store the noise that was applied
        if operation == "add":
            # for add operation, noise is the difference
            noise = masses_after - masses_before
        elif operation == "scale":
            # for scale operation, noise is the scale factor - 1
            noise = masses_after / masses_before
        else:  # operation == "abs"
            # for abs operation, noise is the difference from default
            noise = masses_after - masses_before

        # store the noise values
        self.mass_noise = noise.clone().detach()

        # set the mass into the physics simulation
        self.asset.root_physx_view.set_masses(masses, env_ids)

        # recompute inertia tensors if needed
        if recompute_inertia:
            # compute the ratios of the new masses to the initial masses
            ratios = masses[env_ids[:, None], self.body_ids] / self.asset.data.default_mass[env_ids[:, None], self.body_ids]
            # scale the inertia tensors by the the ratios
            # since mass randomization is done on default values, we can use the default inertia tensors
            inertias = self.asset.root_physx_view.get_inertias()
            if isinstance(self.asset, Articulation):
                # inertia has shape: (num_envs, num_bodies, 9) for articulation
                inertias[env_ids[:, None], self.body_ids] = (
                    self.asset.data.default_inertia[env_ids[:, None], self.body_ids] * ratios[..., None]
                )
            else:
                # inertia has shape: (num_envs, 9) for rigid object
                inertias[env_ids] = self.asset.data.default_inertia[env_ids] * ratios
            # set the inertia tensors into the physics simulation
            self.asset.root_physx_view.set_inertias(inertias, env_ids)

    def get_mass_noise(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get the stored mass noise values.

        Args:
            env_ids: The environment indices to retrieve noise for. If None, returns all environments.

        Returns:
            The mass noise values. Shape: (num_envs, num_bodies) if env_ids is None,
            otherwise (len(env_ids), num_bodies).
        """
        if env_ids is None:
            return self.mass_noise
        else:
            return self.mass_noise[env_ids]

    def get_mass_noise_for_body(self, body_id: int, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get the stored mass noise values for a specific body.

        Args:
            body_id: The body index to retrieve noise for.
            env_ids: The environment indices to retrieve noise for. If None, returns all environments.

        Returns:
            The mass noise values for the specified body. Shape: (num_envs,) if env_ids is None,
            otherwise (len(env_ids),).
        """
        if env_ids is None:
            return self.mass_noise[:, body_id]
        else:
            return self.mass_noise[env_ids, body_id]