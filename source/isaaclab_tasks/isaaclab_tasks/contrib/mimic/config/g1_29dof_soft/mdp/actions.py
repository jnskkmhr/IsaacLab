# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics-step application of event-triggered ankle pitch disturbances."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_soft.mdp.observations import foot_contact_hybrid

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class AnklePitchPerturbation(ActionTerm):
    """Apply smooth external local-y torque pulses without consuming policy actions.

    Pulses stop on loss of foot contact, exit from full stance, or environment reset.
    This term must follow the soft-contact action so that contact state is current.
    """

    cfg: AnklePitchPerturbationCfg

    def __init__(self, cfg: AnklePitchPerturbationCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._body_ids, _ = self._asset.find_bodies(cfg.body_names, preserve_order=True)
        self.cfg.sensor_cfg.resolve(env.scene)
        self._actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._amplitudes = torch.zeros(self.num_envs, len(self._body_ids), device=self.device)
        self._elapsed = torch.zeros(self.num_envs, device=self.device)
        self._duration = torch.zeros_like(self._elapsed)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Accept the empty policy action slice; interval events schedule pulses."""
        pass

    def trigger(
        self, env_ids: torch.Tensor, torque_range: tuple[float, float], duration_range_s: tuple[float, float]
    ) -> None:
        """Schedule independent foot torque amplitudes [N m] and pulse durations [s].

        Active pulses are not restarted. Durations must span at least one physics step.
        """
        if not all(math.isfinite(value) for value in (*torque_range, *duration_range_s)):
            raise ValueError("Torque and duration ranges must be finite.")
        if torque_range[0] > torque_range[1]:
            raise ValueError("Torque range must be ordered.")
        if not self._env.physics_dt <= duration_range_s[0] <= duration_range_s[1]:
            raise ValueError("Duration range must be ordered and at least one physics step.")
        contacts = self._grounded_stance()
        env_ids = env_ids[(self._elapsed[env_ids] >= self._duration[env_ids]) & contacts[env_ids].any(dim=-1)]
        if env_ids.numel() == 0:
            return
        self._amplitudes[env_ids] = (
            math_utils.sample_uniform(*torque_range, (len(env_ids), len(self._body_ids)), self.device)
            * contacts[env_ids]
        )
        self._duration[env_ids] = math_utils.sample_uniform(*duration_range_s, (len(env_ids),), self.device)
        self._elapsed[env_ids] = 0.0

    def apply_actions(self) -> None:
        """Add the current pulse torque [N m] to the instantaneous wrench buffer."""
        env_ids = torch.nonzero(self._elapsed < self._duration, as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        # Cancel individual feet on contact loss; do not resume their pulse at touchdown.
        self._amplitudes[env_ids] *= self._grounded_stance()[env_ids]
        phase = ((self._elapsed[env_ids] + 0.5 * self._env.physics_dt) / self._duration[env_ids]).clamp(max=1.0)
        envelope = torch.sin(torch.pi * phase).square()
        torques = torch.zeros(len(env_ids), len(self._body_ids), 3, device=self.device)
        torques[:, :, 1] = self._amplitudes[env_ids] * envelope[:, None]
        self._asset.instantaneous_wrench_composer.add_forces_and_torques_index(
            torques=torques, body_ids=self._body_ids, env_ids=env_ids, is_global=False
        )
        self._elapsed[env_ids] += self._env.physics_dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear scheduled pulses for reset environments."""
        ids = slice(None) if env_ids is None else env_ids
        self._amplitudes[ids] = 0.0
        self._elapsed[ids] = 0.0
        self._duration[ids] = 0.0

    def _grounded_stance(self) -> torch.Tensor:
        contacts = foot_contact_hybrid(self._env, rigid_contact_sensor_cfg=self.cfg.sensor_cfg).bool()
        if contacts.shape[1] != len(self._body_ids):
            raise ValueError("Perturbed bodies and hybrid contact feet must have matching order and count.")
        command = self._env.command_manager.get_term(self.cfg.command_name)
        return contacts & (command.standing_weight >= 1.0 - 1.0e-6)[:, None]


@configclass
class AnklePitchPerturbationCfg(ActionTermCfg):
    """Configure external ankle-pitch torques with matching left/right contact order."""

    class_type: type[ActionTerm] = AnklePitchPerturbation
    body_names: list[str] = ["left_ankle_pitch_link", "right_ankle_pitch_link"]
    """Bodies whose local y axes align with the ankle-pitch axes."""
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )
    """Contact feet in the same order as the bodies and soft-contact solver."""
    command_name: str = "motion"
    """Motion command defining full stance."""
