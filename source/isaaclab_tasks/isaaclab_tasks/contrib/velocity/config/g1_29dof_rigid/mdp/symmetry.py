# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for G1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    two symmetrical transformations: original and left-right. The symmetry transformations are
    beneficial for reinforcement learning tasks by providing additional diverse data without
    requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        history_length = 10  # hardcoding
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(
            env.unwrapped, obs["policy"], history_length
        )
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        history_length = 10  # hardcoding
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions, history_length)
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right(
    env: ManagerBasedRLEnv, obs: torch.Tensor, history_length: int = 1
) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    Observation layout (per time step, repeated history_length times):
        clock             : 2   — [sin(phase), cos(phase)]
        base_ang_vel      : 3   — [roll_rate, pitch_rate, yaw_rate]
        projected_gravity : 3   — [gx, gy, gz]
        velocity_commands : 7   — [vx, vy, wz, orientation in quaternion]
        joint_pos         : 29
        joint_vel         : 29
        last_actions      : 29

    For history_length > 1, each group is stored as h consecutive blocks of dim each:
        [group_t0 | group_t1 | ... | group_t(h-1)]

    Args:
        env: The environment instance.
        obs: Observation tensor of shape (N, total_dim).
        history_length: Number of history steps stacked in obs.

    Returns:
        Transformed observation tensor with left-right symmetry applied.
    """
    obs = obs.clone()
    device = obs.device

    # --- per-step size of each observation group (in order) ---
    obs_dims = {
        "clock": 2,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "velocity_commands": 3 + 4,  # (vx, vy, wz, sin_h, cos_h, sin_p, cos_p)
        "joint_pos": 29,
        "joint_vel": 29,
        "last_actions": 29,
    }

    # --- sign-flip pattern per single time step (None → use joint swap instead) ---
    # Left-right symmetry:
    #   clock:             phase shifts by 0.5 → sin and cos both negate
    #   base_ang_vel:      roll_rate flips, pitch_rate keeps, yaw_rate flips
    #   projected_gravity: gx keeps, gy flips, gz keeps
    #   velocity_commands: vy and wz flip;
    #                      heading: sin_h flips (sin(-h)=-sin(h)), cos_h keeps (cos(-h)=cos(h));
    #                      gait phase clock: same as "clock" → both negate
    sign_flips: dict[str, list[float] | None] = {
        "clock": [-1.0, -1.0],
        "base_ang_vel": [-1.0, 1.0, -1.0],
        "projected_gravity": [1.0, -1.0, 1.0],
        "velocity_commands": [1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        "joint_pos": None,  # handled by joint swap
        "joint_vel": None,  # handled by joint swap
        "last_actions": None,  # handled by joint swap
    }

    cursor = 0
    for group, dim in obs_dims.items():
        total_dim = dim * history_length
        start, end = cursor, cursor + total_dim

        flip = sign_flips[group]
        if flip is not None:
            # tile single-step pattern over all history steps: [p0,p1,...] * h
            pattern = torch.tensor(flip * history_length, device=device)  # (total_dim,)
            obs[:, start:end] = obs[:, start:end] * pattern
        else:
            # reshape to (N, history_length, dim) → swap joints per step → reshape back
            data = obs[:, start:end].view(-1, history_length, dim)
            obs[:, start:end] = _switch_g1_joints_left_right(data).view(-1, total_dim)

        cursor = end

    # height scan (optional): hard-coded for grid size (1.6, 1.0) → 11×17
    if "height_scan" in env.observation_manager.active_terms["policy"]:
        h_start = cursor
        h_end = h_start + 11 * 17
        obs[:, h_start:h_end] = obs[:, h_start:h_end].view(-1, 11, 17).flip(dims=[1]).view(-1, 11 * 17)

    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor, history_length: int = 1) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    Args:
        actions: The actions tensor to be transformed, shape (N, 29).
        history_length: Unused for actions (kept for API consistency).

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_g1_joints_left_right(actions[:])
    return actions


"""
Helper functions for symmetry.

In Isaac Sim, the joint ordering is as follows:
[
    'LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA',
    'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE',
    'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE'
]

Correspondingly, the joint ordering for the ANYmal robot is:

* LF = left front --> [0, 4, 8]
* LH = left hind --> [1, 5, 9]
* RF = right front --> [2, 6, 10]
* RH = right hind --> [3, 7, 11]
"""


def _switch_g1_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)
    # left <-- right[
    joint_data_switched[..., [0, 1, 2, 3, 4, 5]] = joint_data_switched[..., [6, 7, 8, 9, 10, 11]]  # foot
    joint_data_switched[..., [15, 16, 17, 18, 19, 20, 21]] = joint_data_switched[
        ..., [22, 23, 24, 25, 26, 27, 28]
    ]  # arm
    # right <-- left
    joint_data_switched[..., [6, 7, 8, 9, 10, 11]] = joint_data_switched[..., [0, 1, 2, 3, 4, 5]]  # foot
    joint_data_switched[..., [22, 23, 24, 25, 26, 27, 28]] = joint_data_switched[
        ..., [15, 16, 17, 18, 19, 20, 21]
    ]  # arm

    # # Flip the sign of the HAA joints
    # joint_data_switched[..., [0, 1, 2, 3]] *= -1.0

    return joint_data_switched
