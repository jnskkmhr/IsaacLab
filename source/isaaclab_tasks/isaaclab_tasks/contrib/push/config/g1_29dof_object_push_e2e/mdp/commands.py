# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity command that also reports how well the box follows it."""

import torch

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid.mdp import (
    UniformVelocityYawCommand,
    UniformVelocityYawCommandCfg,
)

from .rewards import box_contact_forces, progress_ratio


class PushVelocityCommand(UniformVelocityYawCommand):
    """Robot-frame velocity command with episode-mean pushing metrics.

    The command itself is unchanged. These metrics are logged at reset:

    * ``box_speed_ratio``: box speed along the command over the commanded speed, moving commands only.
    * ``hand_contact_fraction``: fraction of wrists touching the box, moving commands only.
    * ``non_hand_contact_force``: box-filtered force on every non-wrist robot body [N].
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._box = env.scene[cfg.object_name]
        self._sensor = env.scene[cfg.contact_sensor_name]
        self._hand_ids = [self._sensor.find_bodies(name)[0][0] for name in cfg.hand_body_names]
        hand_ids = set(self._hand_ids)
        self._other_ids = [i for i in range(self._sensor.num_bodies) if i not in hand_ids]
        self._moving_steps = torch.zeros(self.num_envs, device=self.device)
        self._push_steps = torch.zeros(self.num_envs, device=self.device)
        self._push_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("box_speed_ratio", "hand_contact_fraction", "non_hand_contact_force")
        }
        for name in self._push_sums:
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids=None):
        # Finalize episode means before the base class logs and clears the metrics.
        ids = slice(None) if env_ids is None else env_ids
        moving_steps = self._moving_steps[ids].clamp(min=1.0)
        self.metrics["box_speed_ratio"][ids] = self._push_sums["box_speed_ratio"][ids] / moving_steps
        self.metrics["hand_contact_fraction"][ids] = self._push_sums["hand_contact_fraction"][ids] / moving_steps
        self.metrics["non_hand_contact_force"][ids] = self._push_sums["non_hand_contact_force"][ids] / self._push_steps[
            ids
        ].clamp(min=1.0)
        extras = super().reset(env_ids)
        self._moving_steps[ids] = 0.0
        self._push_steps[ids] = 0.0
        for value in self._push_sums.values():
            value[ids] = 0.0
        return extras

    def episode_box_speed_ratio(self, env_ids) -> tuple[torch.Tensor, torch.Tensor]:
        """Box speed ratio averaged over the running episode, for the given environments.

        Valid until this term is reset, so callers at episode end must read it before the command reset.

        Returns:
            The mean ratio over moving-command steps, and a mask of environments that had any such step.
        """
        steps = self._moving_steps[env_ids]
        return self._push_sums["box_speed_ratio"][env_ids] / steps.clamp(min=1.0), steps > 0

    def _update_metrics(self):
        super()._update_metrics()
        forces = box_contact_forces(self._env, self._sensor)
        contact = (forces[:, self._hand_ids] > self.cfg.force_threshold).float().mean(dim=-1)
        command = self.vel_command_b[:, :2]
        moving = (torch.linalg.vector_norm(command, dim=-1) > self.cfg.command_threshold).float()
        ratio = progress_ratio(
            command, self.robot.data.root_quat_w.torch, self._box.data.root_lin_vel_w.torch, self.cfg.command_threshold
        )
        self._push_steps += 1.0
        self._moving_steps += moving
        self._push_sums["box_speed_ratio"] += ratio
        self._push_sums["hand_contact_fraction"] += contact * moving
        self._push_sums["non_hand_contact_force"] += forces[:, self._other_ids].sum(dim=-1)


@configclass
class PushVelocityCommandCfg(UniformVelocityYawCommandCfg):
    """Velocity command plus the inputs needed for its pushing metrics."""

    class_type: type = PushVelocityCommand
    object_name: str = "box"
    """Rigid object whose speed is compared with the command."""
    contact_sensor_name: str = "box_contacts"
    """Contact sensor covering all robot bodies, filtered against the object.

    Coupled-sand environments name the robot articulation here instead; see
    :func:`~.rewards.box_contact_forces`.
    """
    hand_body_names: tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    """Bodies that are expected to push the object."""
    force_threshold: float = 1.0
    """Object-filtered force above which a hand body counts as touching [N]."""
    command_threshold: float = 0.05
    """Planar command speed below which the command counts as standing [m/s]."""
