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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import TerminationTermCfg


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
