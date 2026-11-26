# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 environment with two-way MPM coupling; the MPM solver settings choose how the sand is simulated."""

from __future__ import annotations

import re
from collections.abc import Sequence

import newton
import torch
import warp as wp
from isaaclab_newton.physics import configure_mpm_sleeping, configure_mpm_streaming, reset_mpm_solver_state

from isaaclab.envs import ManagerBasedRLEnv

from ...util.solver_setting import get_mpm_solver_cfg
from ...util.visualizer import CameraFollow, attach_particle_filter
from .env_cfg.physics_cfg import MPM_ENTRY, RIGID_ENTRY
from .streaming_mpm_env_cfg import G1MPMEnvCfg


class G1MPMEnv(ManagerBasedRLEnv):
    """G1 walking on a granular bed simulated by Newton's implicit MPM solver.

    The coupled solver exposes terrain contact forces through proxy feedback buffers.
    This environment reads each foot's feedback wrench once per environment step
    and derives contact flags and gait timers for the local MDP functions.
    """

    cfg: G1MPMEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.mpm_streaming = None
        """Newton's streaming particle pool, or None when every spawned particle stays in place."""
        # Activity colors need a sleeping preset; without one every particle is simply active.
        self._activity_colors_enabled = bool(
            cfg.mpm_activity_colors and cfg.sim.visualizer_cfgs and get_mpm_solver_cfg(cfg).sleeping is not None
        )
        if self._activity_colors_enabled:
            from ...util.particle_visualization import ActivityMPMObject

            cfg.scene.sand.class_type = ActivityMPMObject
        self._settled_bed = None
        super().__init__(cfg, render_mode, **kwargs)

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        if self.mpm_streaming is not None:
            self.mpm_streaming.check_status()
        # Rewards are computed from the pre-reset state, so the step on which a world diverges
        # carries a NaN reward even though `solver_diverged` already reset it. One NaN sample is
        # enough to destroy a policy update, so it is scrubbed here.
        torch.nan_to_num_(reward, nan=0.0, posinf=0.0, neginf=0.0)
        # Preserve pre-reset diagnostics and give the logger a fresh per-step snapshot.
        observation_metrics = extras.pop("_observation_metrics", None)
        if observation_metrics is not None:
            extras["log"] = {**extras.get("log", {}), **observation_metrics}
        return obs, reward, terminated, truncated, extras

    def load_managers(self) -> None:
        # MDP terms resolve their parameters against this state, so it must exist beforehand
        self._setup_mpm_runtime()
        self._setup_contact_state()
        super().load_managers()
        if self.mpm_streaming is not None:
            sand = self.cfg.scene.terrain.sand
            surface = None
            if self.sim.visualizers and get_mpm_solver_cfg(self.cfg).streaming.cache is not None:
                if sand.render_mode == "isosurface":
                    from isaaclab_visualizers.newton import CachedParticleSurface

                    surface = CachedParticleSurface(
                        self.mpm_streaming,
                        self.sim.backend_manager.backend,
                        self.scene.terrain.background_mesh,
                        self.cfg.scene.terrain,
                        resolution=sand.surface_resolution,
                        smoothing=sand.surface_smoothing,
                        cache_mb=sand.surface_cache_mb,
                    )
                elif sand.render_mode != "particles":
                    raise ValueError(f"Unknown sand render mode {sand.render_mode!r}; use 'isosurface' or 'particles'.")
            for visualizer in self.sim.visualizers:
                if surface is not None:
                    surface.attach(visualizer)
                else:
                    attach_particle_filter(visualizer, self.mpm_streaming, sand.show_boundary_particles)
                if getattr(visualizer.cfg.backend, "follow_robot", False):
                    backend = self.sim.backend_manager.backend
                    labels = list(backend.model.body_label)
                    root = next(i for i, label in enumerate(labels) if label.startswith("/World/envs/env_0/Robot/"))
                    visualizer.camera_follow = CameraFollow(backend, root, visualizer.cfg.eye, visualizer.cfg.lookat)
        if self.cfg.mpm_cache_initial_state:
            from ...util.settled_bed import SettledBed

            self._settled_bed = SettledBed(self)
        if self._activity_colors_enabled:
            from ...util.particle_visualization import ParticleActivityColors

            self._activity_colors = ParticleActivityColors(self)
            self.sim.add_render_callback("mpm_activity_colors", self._activity_colors.update)

    def _setup_mpm_runtime(self) -> None:
        """Apply the MPM solver's sleeping and streaming settings to Newton."""
        manager = self.sim.backend_manager
        backend = manager.backend
        mpm_cfg = get_mpm_solver_cfg(self.cfg)
        if backend.model.particle_count == 0:
            return
        solver = backend.solver.solver(MPM_ENTRY)
        configure_mpm_sleeping(manager, mpm_cfg.sleeping, solver=solver)
        if mpm_cfg.streaming is None:
            return
        surface = self.scene.terrain.background_mesh  # type: ignore
        self.mpm_streaming = configure_mpm_streaming(
            manager,
            mpm_cfg.streaming,
            surface_mesh=surface.mesh,
            world_origins=surface.env_origins,
            spawn_heights=surface.env_clone_origins[:, 2],
            solver=solver,
            state=backend.solver.entry_state(MPM_ENTRY, phase="input"),
            particle_spawn_cfg=self.cfg.scene.sand.spawn if mpm_cfg.streaming.cache is not None else None,
            dt=self.physics_dt,
        )

    """
    properties
    """

    @property
    def foot_count(self) -> int:
        """Number of feet tracked against the granular bed."""
        return self._foot_body_ids.shape[1]

    @property
    def foot_contact_force(self) -> torch.Tensor:
        """Granular reaction force in world frame [N], shape ``(num_envs, foot_count, 3)``.

        Nonfinite components are reported as zero; the solver's feedback buffer is unchanged.
        """
        self.update_contact_state()
        return self._foot_contact_force

    @property
    def foot_contact(self) -> torch.Tensor:
        """Contact flag per foot as ``float``, shape ``(num_envs, foot_count)``."""
        self.update_contact_state()
        return self._foot_contact.float()

    @property
    def foot_first_contact(self) -> torch.Tensor:
        """Whether a foot touched down this step as ``float``, shape ``(num_envs, foot_count)``."""
        self.update_contact_state()
        return self._foot_first_contact.float()

    @property
    def foot_air_time(self) -> torch.Tensor:
        """Time since each foot last left the bed [s], shape ``(num_envs, foot_count)``."""
        self.update_contact_state()
        return self._foot_air_time

    @property
    def foot_contact_time(self) -> torch.Tensor:
        """Time since each foot last touched the bed [s], shape ``(num_envs, foot_count)``."""
        self.update_contact_state()
        return self._foot_contact_time

    """
    Contact state accessors used by the MDP terms.
    """

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

    def reset_mpm_state(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        """Restore the bed and clear the solver history of the selected environments.

        Args:
            env_ids: Environment indices or a slice selecting environments to reset.
        """
        # Newton reset masks carry one trailing slot for global (world -1) entities.
        world_mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        world_mask[: self.num_envs][env_ids] = True
        if not world_mask.any():
            return
        if self._settled_bed is not None:
            self._settled_bed.restore(world_mask[: self.num_envs].nonzero().flatten())
            self._sand._invalidate_caches(pos=True, vel=True)
        else:
            particle_state = self._sand.data.default_particle_state_w.torch[env_ids].clone()
            if self.cfg.reset_particle_jitter > 0.0:
                jitter = (
                    2.0 * torch.rand(particle_state.shape[:2] + (3,), device=self.device) - 1.0
                ) * self.cfg.reset_particle_jitter
                particle_state[..., :3] += jitter
            particle_state[..., 3:] = 0.0
            self._sand.write_particle_state_to_sim_index(particle_state, env_ids=env_ids)

            # Clear the constitutive, contact and proxy-feedback history of exactly the reset worlds.
            reset_mpm_solver_state(
                self.sim.backend_manager.backend,
                world_mask=wp.from_torch(world_mask, dtype=wp.bool),
                flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE,
            )

        self._foot_air_time[env_ids] = 0.0
        self._foot_contact_time[env_ids] = 0.0
        self._foot_first_contact[env_ids] = False
        # the cleared feedback is not visible in the solver buffers until the next solver step
        self._contact_state_step = -1
        if self.mpm_streaming is not None:
            self.mpm_streaming.reset(wp.from_torch(world_mask, dtype=wp.bool))

    """
    Internal helpers.
    """

    def _setup_contact_state(self) -> None:
        """Resolve the proxy feedback buffer and allocate the per-foot caches."""
        self.sim.backend_manager.backend.model.particle_max_velocity = self.cfg.particle_max_velocity
        self._robot = self.scene["robot"]
        self._sand = self.scene["sand"]

        _, foot_names = self._robot.find_bodies(self.cfg.foot_body_expr, preserve_order=True)
        self._foot_body_ids = self._resolve_newton_body_ids(foot_names)
        self._coupling_force_body_ids = self._foot_body_ids
        self._coupling_forces = self._resolve_coupling_feedback_buffer()

        foot_shape = (self.num_envs, len(foot_names))
        self._foot_contact_force = torch.zeros((*foot_shape, 3), device=self.device)
        self._foot_contact = torch.zeros(foot_shape, dtype=torch.bool, device=self.device)
        self._foot_first_contact = torch.zeros(foot_shape, dtype=torch.bool, device=self.device)
        self._foot_air_time = torch.zeros(foot_shape, device=self.device)
        self._foot_contact_time = torch.zeros(foot_shape, device=self.device)

        self._contact_state_step = -1
        self._gait_timer_step = -1
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
        pattern = re.compile(r"^/World/envs/env_(\d+)/Robot/(?:.*/)?([^/]+)$")
        body_ids = torch.full((self.num_envs, len(foot_names)), -1, dtype=torch.long, device=self.device)
        name_to_column = {name: column for column, name in enumerate(foot_names)}
        for body_id, label in enumerate(self.sim.backend_manager.backend.model.body_label):
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

    def _resolve_coupling_feedback_buffer(self) -> wp.array:
        """Resolve the foot–sand interface wrench buffer and its body index mapping.

        Proxy forces use global Newton body indices and exclude external forces.
        """
        model = self.sim.backend_manager.backend.model
        if model.particle_count == 0:
            return wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
        solver = self.sim.backend_manager.backend.solver
        for mapping in getattr(solver, "_proxy_mappings", ()):
            if mapping.src_name == RIGID_ENTRY and mapping.dst_name == MPM_ENTRY:
                return mapping.coupling_forces
        raise RuntimeError(
            f"The coupled solver has no body proxy mapping from {RIGID_ENTRY!r} to {MPM_ENTRY!r}, so the granular"
            " reaction on the feet cannot be observed."
        )

    def _refresh_contact_forces(self) -> None:
        """Read feedback, replace nonfinite components with zero, and rebuild contact flags."""
        # the coupler stores the feedback as (force, torque); only the linear part is a contact force
        forces = wp.to_torch(self._coupling_forces, requires_grad=False)[:, :3]
        self._foot_contact_force = forces[self._coupling_force_body_ids].to(self.device)
        # Retain evidence of invalid feedback even if a reset clears it later in this step.
        nonfinite_fraction = (~torch.isfinite(self._foot_contact_force)).float().mean()
        metrics = self.extras.setdefault("_observation_metrics", {})
        key = "Metrics/mpm/contact_force_nonfinite_fraction"
        metrics[key] = torch.maximum(metrics.get(key, nonfinite_fraction), nonfinite_fraction)
        torch.nan_to_num_(self._foot_contact_force, nan=0.0, posinf=0.0, neginf=0.0)
        force_magnitude = torch.norm(self._foot_contact_force, dim=-1)
        self._foot_contact = force_magnitude > self.cfg.foot_contact_force_threshold

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        super()._reset_idx(env_ids)  # type: ignore
        self.reset_mpm_state(env_ids)
