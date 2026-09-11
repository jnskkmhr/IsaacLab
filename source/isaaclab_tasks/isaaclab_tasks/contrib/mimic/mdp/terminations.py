# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import TerminationTermCfg
from isaaclab.sensors import ContactSensor

from .commands import MotionCommand
from .utils import get_body_indices


def _gate_on_tracking(command: MotionCommand, terminated: torch.Tensor, only_when_tracking: bool) -> torch.Tensor:
    """Suppress a tracking-error termination while the reference is in a stance interval.

    Stance has no reference trajectory to follow in principle -- the robot only has to stay balanced
    -- so a tracking-error bound there just kills otherwise-healthy episodes. Falling during stance is
    caught by the base-height termination instead.
    """
    if not only_when_tracking:
        return terminated
    return terminated & ~command.is_stance


def bad_anchor_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, only_when_tracking: bool = True
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    bad = torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
    return _gate_on_tracking(command, bad, only_when_tracking)


def bad_anchor_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, only_when_tracking: bool = True
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    bad = torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold
    return _gate_on_tracking(command, bad, only_when_tracking)


def bad_anchor_ori(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
    only_when_tracking: bool = True,
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    motion_projected_gravity_b = quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    bad = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold
    return _gate_on_tracking(command, bad, only_when_tracking)


def bad_motion_body_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    body_names: list[str] | None = None,
    only_when_tracking: bool = True,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore

    body_indexes = get_body_indices(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return _gate_on_tracking(command, torch.any(error > threshold, dim=-1), only_when_tracking)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    body_names: list[str] | None = None,
    only_when_tracking: bool = True,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore

    body_indexes = get_body_indices(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return _gate_on_tracking(command, torch.any(error > threshold, dim=-1), only_when_tracking)


def end_of_reference(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate once the reference motion has played through to its last frame.

    The motion command holds at the final frame rather than resetting itself, so this term is what
    actually ends the episode. Register it with ``time_out=True``: running out of reference is the
    successful end of the clip, not a failure, so it must not be bootstrapped as one nor counted as
    a failure by the command's adaptive sampling.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)  # type: ignore
    return command.has_reached_end


def root_height_below_minimum_adaptive(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    min_foot_height = (asset.data.body_pos_w[:, asset_cfg.body_ids, 2]).min(dim=1).values

    return asset.data.root_pos_w[:, 2] - min_foot_height < minimum_height


def base_ang_vel_exceed(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Check if the base angular velocity exceeds the threshold."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]

    # check if any of the errors exceed the threshold
    # TODO: check if the base angular velocity exceeds the threshold
    # Hint: use the root_ang_vel_b property of the robot articulation
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


class ExtremeJointPositionAction(ManagerTermBase):
    """Terminate when the policy's commanded position target would saturate a joint's
    PD torque well past its effort limit if the joint were sitting at that limit.

    For a position-controlled joint with PD gain ``kp``, joint range ``[q_min, q_max]``,
    and effort limit ``tau_max``, the spring torque applied if the joint is at ``q_max``
    and the target is ``q_des`` is ``kp * (q_des - q_max)``. The action is considered
    "too extreme into the upper limit" when this would exceed ``torque_fraction * tau_max``:

        q_des > q_max + torque_fraction * tau_max / kp

    and symmetrically for the lower limit:

        q_des < q_min - torque_fraction * tau_max / kp
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        action_name: str = cfg.params["action_name"]  # type: ignore
        torque_fraction: float = cfg.params.get("torque_fraction", 10.0)  # type: ignore

        self.action_term = env.action_manager.get_term(action_name)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.joint_ids = cfg.params["asset_cfg"].joint_ids
        self.torque_fraction = torque_fraction

        # Joint ids in articulation space, one per action dimension.
        action_joint_ids = self._resolve_joint_ids(self.joint_ids, self.asset.num_joints)
        action_dim = len(action_joint_ids)

        # Snapshot kp and tau_max from the owning actuators. PD gains and effort limits
        # are configured at startup and not currently randomized; if that changes, this
        # cache will need to be refreshed.
        stiffness = torch.zeros(self.num_envs, action_dim, device=self.device)
        effort_limit = torch.zeros_like(stiffness)
        action_covered = [False] * action_dim
        for actuator in self.asset.actuators.values():
            actuator_joint_ids = self._resolve_joint_ids(actuator.joint_indices, self.asset.num_joints)
            actuator_by_joint = {joint_id: actuator_id for actuator_id, joint_id in enumerate(actuator_joint_ids)}
            # Create a list of the actuator joint ids for the current actuator that correspond to an field in the action
            action_ids: list[int] = []
            actuator_ids: list[int] = []
            for action_id, joint_id in enumerate(action_joint_ids):
                if joint_id in actuator_by_joint:
                    action_ids.append(action_id)
                    actuator_ids.append(actuator_by_joint[joint_id])
                    action_covered[action_id] = True
            if action_ids:
                action_ids_tensor = torch.tensor(action_ids, device=self.device, dtype=torch.long)
                actuator_ids_tensor = torch.tensor(actuator_ids, device=self.device, dtype=torch.long)
                stiffness[:, action_ids_tensor] = actuator.stiffness[:, actuator_ids_tensor]
                effort_limit[:, action_ids_tensor] = actuator.effort_limit[:, actuator_ids_tensor]

        if not all(action_covered):
            missing = [
                self.asset.joint_names[action_joint_ids[action_id]]
                for action_id, covered in enumerate(action_covered)
                if not covered
            ]
            raise RuntimeError(
                f"ExtremeJointPositionAction: action joints {missing} have no owning actuator on"
                f" asset '{self.asset.cfg.prim_path}'."
            )

        active = stiffness > 0
        margin = self.torque_fraction * effort_limit / torch.where(active, stiffness, torch.ones_like(stiffness))
        self._margin = torch.where(active, margin, torch.full_like(margin, float("inf")))

    @staticmethod
    def _resolve_joint_ids(joint_ids, total: int) -> list[int]:
        if isinstance(joint_ids, slice):
            return list(range(total))[joint_ids]
        if isinstance(joint_ids, torch.Tensor):
            return joint_ids.tolist()
        return list(joint_ids)

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, action_name: str, torque_fraction: float = 10.0
    ) -> torch.Tensor:
        target = self.action_term.processed_actions
        pos_limits = self.asset.data.joint_pos_limits[:, self.joint_ids]
        q_min = pos_limits[..., 0]
        q_max = pos_limits[..., 1]

        too_high = target > q_max + self._margin
        too_low = target < q_min - self._margin
        return (too_high | too_low).any(dim=1)
