# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based G1 29-DoF locomotion environment on a coupled MPM granular bed."""

from __future__ import annotations

import re
from collections.abc import Sequence

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager, NewtonMPMManager

from isaaclab.envs import ManagerBasedRLEnv

from .env_cfg.physics_cfg import MPM_ENTRY, RIGID_ENTRY, configure_sparse_mpm_capacities
from .g1_mpm_env_cfg import G1MPMEnvCfg


class G1MPMEnv(ManagerBasedRLEnv):
    """G1 walking on a granular bed simulated by Newton's implicit MPM solver.

    The MDP terms of ``g1_29dof_soft`` read foot contact from a contact sensor, which a coupled
    solver cannot provide because contact forces live in per-entry buffers. Here the feet are
    virtual proxies of the rigid entry inside the MPM entry, so the granular reaction on each foot
    is exactly the proxy feedback wrench that the coupler hands back to the rigid solver. This
    class harvests that wrench once per environment step and derives the contact flag and the gait
    timers from it, so that every MDP term in :mod:`.mdp` is a cheap read of a cached tensor.
    """

    cfg: G1MPMEnvCfg

    def __init__(self, cfg: G1MPMEnvCfg, render_mode: str | None = None, **kwargs):
        # command-line overrides such as --num_envs land on the config before the scene is created
        configure_sparse_mpm_capacities(cfg)
        self._contact_state_ready = False
        super().__init__(cfg, render_mode, **kwargs)

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        # Rewards are computed from the pre-reset state, so the step on which a world diverges
        # carries a NaN reward even though `solver_diverged` already reset it. One NaN sample is
        # enough to destroy a policy update, so it is scrubbed here.
        torch.nan_to_num_(reward, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, reward, terminated, truncated, extras

    def load_managers(self) -> None:
        # MDP terms resolve their parameters against this state, so it must exist beforehand
        self._setup_contact_state()
        super().load_managers()

    @property
    def foot_count(self) -> int:
        """Number of feet tracked against the granular bed."""
        return self._foot_body_ids.shape[1]

    """
    Contact state accessors used by the MDP terms.
    """

    def foot_contact_force(self) -> torch.Tensor:
        """Granular reaction force on each foot in world frame [N], shape ``(num_envs, foot_count, 3)``."""
        self.update_contact_state()
        return self._foot_contact_force

    def foot_contact(self) -> torch.Tensor:
        """Contact flag per foot as ``float``, shape ``(num_envs, foot_count)``."""
        self.update_contact_state()
        return self._foot_contact.float()

    def foot_first_contact(self) -> torch.Tensor:
        """Whether a foot touched down this step as ``float``, shape ``(num_envs, foot_count)``."""
        self.update_contact_state()
        return self._foot_first_contact.float()

    def foot_air_time(self) -> torch.Tensor:
        """Time since each foot last left the bed [s], shape ``(num_envs, foot_count)``."""
        self.update_contact_state()
        return self._foot_air_time

    def foot_contact_time(self) -> torch.Tensor:
        """Time since each foot last touched the bed [s], shape ``(num_envs, foot_count)``."""
        self.update_contact_state()
        return self._foot_contact_time

    def update_contact_state(self) -> None:
        """Harvest the proxy feedback wrench and advance the gait timers, at most once per step."""
        if self._contact_state_step == self.common_step_counter:
            return
        previous_contact = self._foot_contact.clone()
        self._refresh_contact_forces()
        self._contact_state_step = self.common_step_counter
        self._foot_first_contact = self._foot_contact & ~previous_contact

        # A mid-step reset invalidates the force cache a second time, but the gait timers must
        # still advance exactly once per step, so they carry their own guard.
        if self._gait_timer_step == self.common_step_counter:
            return
        # mirrors ContactSensor: the timer of the mode a foot is in keeps running
        self._foot_air_time = torch.where(
            self._foot_contact, torch.zeros_like(self._foot_air_time), self._foot_air_time + self.step_dt
        )
        self._foot_contact_time = torch.where(
            self._foot_contact, self._foot_contact_time + self.step_dt, torch.zeros_like(self._foot_contact_time)
        )
        self._gait_timer_step = self.common_step_counter

    def reset_sand_bed(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Restore the bed and clear the solver history of the selected environments.

        Args:
            env_ids: Indices of the environments to reset.
        """
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        particle_state = self._sand.data.default_particle_state_w.torch[env_ids].clone()
        if self.cfg.reset_particle_jitter > 0.0:
            jitter = (
                2.0 * torch.rand((env_ids.numel(), self._sand.particles_per_object, 3), device=self.device) - 1.0
            ) * self.cfg.reset_particle_jitter
            particle_state[..., :3] += jitter
        particle_state[..., 3:] = 0.0
        self._sand.write_particle_state_to_sim_index(particle_state, env_ids=env_ids)

        # Clear the constitutive, contact and proxy-feedback history of exactly the reset worlds.
        # Newton reset masks carry one trailing slot for global (world -1) entities.
        world_mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        world_mask[env_ids] = True
        NewtonMPMManager.reset_solver_state(
            world_mask=wp.from_torch(world_mask, dtype=wp.bool),
            flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
        )

        self._foot_air_time[env_ids] = 0.0
        self._foot_contact_time[env_ids] = 0.0
        self._foot_first_contact[env_ids] = False
        # the cleared feedback is not visible in the solver buffers until the next solver step
        self._contact_state_step = -1

    """
    Internal helpers.
    """

    def _setup_contact_state(self) -> None:
        """Resolve the proxy feedback buffer and allocate the per-foot caches."""
        NewtonMPMManager.get_model().particle_max_velocity = self.cfg.particle_max_velocity
        self._robot = self.scene["robot"]
        self._sand = self.scene["sand"]

        _, foot_names = self._robot.find_bodies(self.cfg.foot_body_expr, preserve_order=True)
        self._foot_body_ids = self._resolve_newton_body_ids(foot_names)
        self._coupling_forces = self._resolve_proxy_feedback_buffer()

        foot_shape = (self.num_envs, len(foot_names))
        self._foot_contact_force = torch.zeros((*foot_shape, 3), device=self.device)
        self._foot_contact = torch.zeros(foot_shape, dtype=torch.bool, device=self.device)
        self._foot_first_contact = torch.zeros(foot_shape, dtype=torch.bool, device=self.device)
        self._foot_air_time = torch.zeros(foot_shape, device=self.device)
        self._foot_contact_time = torch.zeros(foot_shape, device=self.device)

        self._contact_state_step = -1
        self._gait_timer_step = -1
        self._contact_state_ready = True
        self._refresh_contact_forces()

    def _resolve_newton_body_ids(self, foot_names: Sequence[str]) -> torch.Tensor:
        """Map the foot link names to Newton body indices, per environment.

        Newton labels bodies by their full prim path, so the environment index and the link name
        can be recovered from the label without relying on the model's body ordering.

        Args:
            foot_names: Foot link names in the order used by the MDP terms.

        Returns:
            The Newton body indices, shape ``(num_envs, len(foot_names))``.
        """
        pattern = re.compile(r"^/World/envs/env_(\d+)/Robot/.*/([^/]+)$")
        body_ids = torch.full((self.num_envs, len(foot_names)), -1, dtype=torch.long, device=self.device)
        name_to_column = {name: column for column, name in enumerate(foot_names)}
        for body_id, label in enumerate(NewtonManager.get_model().body_label):
            match = pattern.match(label)
            if match is None:
                continue
            column = name_to_column.get(match.group(2))
            env_id = int(match.group(1))
            if column is not None and env_id < self.num_envs:
                body_ids[env_id, column] = body_id
        if bool((body_ids < 0).any()):
            raise RuntimeError(f"Could not resolve Newton body indices for the foot bodies {list(foot_names)}.")
        return body_ids

    def _resolve_proxy_feedback_buffer(self) -> wp.array:
        """Return the coupler buffer holding the granular reaction wrench of the proxy bodies.

        Returns:
            The per-body spatial forces, shape ``(body_count,)``, indexed by Newton body index.
        """
        solver = NewtonManager._solver
        for mapping in getattr(solver, "_proxy_mappings", ()):
            if mapping.src_name == RIGID_ENTRY and mapping.dst_name == MPM_ENTRY:
                return mapping.coupling_forces
        raise RuntimeError(
            f"The coupled solver has no body proxy mapping from {RIGID_ENTRY!r} to {MPM_ENTRY!r}, so the granular"
            " reaction on the feet cannot be observed."
        )

    def _refresh_contact_forces(self) -> None:
        """Read the proxy feedback wrench and rebuild the contact flag from it."""
        # the coupler stores the feedback as (force, torque); only the linear part is a contact force
        forces = wp.to_torch(self._coupling_forces, requires_grad=False)[:, :3]
        self._foot_contact_force = forces[self._foot_body_ids].to(self.device)
        force_magnitude = torch.norm(self._foot_contact_force, dim=-1)
        self._foot_contact = force_magnitude > self.cfg.foot_contact_force_threshold

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)
        if not self._contact_state_ready:
            return
        self._foot_air_time[env_ids] = 0.0
        self._foot_contact_time[env_ids] = 0.0
        self._foot_first_contact[env_ids] = False
        self._contact_state_step = -1
