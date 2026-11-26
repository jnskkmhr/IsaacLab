# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Streaming G1 environment on mixed rigid/sand terrain: only sand environments hold sand.

Foot contact comes from the sand and from MJWarp.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import mujoco_warp as mjw
import newton
import numpy as np
import torch
import warp as wp
from isaaclab_newton.assets import MPMObject
from isaaclab_newton.envs.mdp.events import randomize_mpm_material as randomize_newton_mpm_material
from isaaclab_newton.physics import reset_mpm_solver_state
from isaaclab_newton.physics.mpm_material_access import MPMMaterialAccess
from isaaclab_newton.physics.newton_builder_cfg import NewtonBuilderCfg

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.physics import PhysicsEvent

from ...util.solver_setting import get_mpm_solver_cfg
from .env_cfg.physics_cfg import RIGID_ENTRY
from .streaming_mpm_env import G1MPMEnv

if TYPE_CHECKING:
    from .streaming_mpm_mixed_env_cfg import G1MPMMixedEnvCfg


@wp.kernel
def sum_rigid_foot_forces(
    contact_count: wp.array[int],
    contact_world: wp.array[int],
    contact_geom: wp.array[wp.vec2i],
    geom_to_shape: wp.array2d[int],
    shape_to_foot: wp.array[int],
    shape_is_ground: wp.array[wp.bool],
    contact_wrench: wp.array[wp.spatial_vector],
    foot_force: wp.array[wp.vec3],
):
    """Sum the world-frame forces applied to each foot across its MJWarp ground contacts."""
    contact = wp.tid()
    if contact < contact_count[0]:
        world = contact_world[contact]
        geoms = contact_geom[contact]
        force = wp.spatial_top(contact_wrench[contact])
        for side in range(2):
            shape = geom_to_shape[world, geoms[side]]
            other_shape = geom_to_shape[world, geoms[1 - side]]
            if shape >= 0 and other_shape >= 0 and shape_is_ground[other_shape]:
                foot = shape_to_foot[shape]
                if foot >= 0:
                    sign = float(2 * side - 1)
                    wp.atomic_add(foot_force, foot, sign * force)


class MixedMPMObject(MPMObject):
    """Sand of the sand environments; the other environments hold none."""

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        model = self._backend_manager.backend.model
        self.env_ids = wp.to_torch(model.particle_world)[wp.to_torch(self._particle_offsets).long()].long()
        """Environment of each sand instance, shape ``(num_instances,)``."""
        self._instance_of_env = torch.full((model.world_count,), -1, dtype=torch.long, device=self.device)
        self._instance_of_env[self.env_ids] = torch.arange(self.num_instances, device=self.device)

    def instance_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        """Sand instances of the selected environments; environments without sand are skipped."""
        ids = self._instance_of_env[slice(None) if env_ids is None else env_ids]
        return ids[ids >= 0]

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        if env_mask is not None:
            env_ids = wp.to_torch(env_mask).nonzero().flatten()
        ids = self.instance_ids(env_ids)
        if ids.numel():
            super().reset(env_ids=ids)


class randomize_mpm_material(randomize_newton_mpm_material):
    """Sample the sand material of the sand environments."""

    def __init__(self, cfg: EventTermCfg, env: G1MPMMixedEnv) -> None:
        # The generic term requires sand in every environment. Its sample rows are the sand instances here.
        ManagerTermBase.__init__(self, cfg, env)
        self._materials = MPMMaterialAccess(env.sim.backend_manager)
        self._asset: MixedMPMObject = env.scene[cfg.params["asset_cfg"].name]
        self._particle_offsets = wp.to_torch(self._asset._particle_offsets).long()
        self._particle_indices = torch.arange(self._asset._particles_per_object, device=env.device)
        self._env_ids = torch.arange(self._asset.num_instances, device=env.device)
        self.material_parameters: dict[str, torch.Tensor] = {}

    def __call__(
        self,
        env: G1MPMMixedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        parameter_ranges: dict[str, tuple[float, float]],
        distribution: str = "uniform",
    ) -> None:
        ids = self._asset.instance_ids(env_ids)
        if not ids.numel():
            return
        super().__call__(env, ids, asset_cfg, parameter_ranges, distribution)
        if "density" in parameter_ranges and env.mpm_streaming is not None:
            # Keep the streaming pool's restored masses consistent with the sampled density.
            selected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            selected[self._asset.env_ids[ids]] = True
            density = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
            density[self._asset.env_ids] = self.material_parameters["density"].to(env.device, torch.float32)
            env.mpm_streaming.set_dynamic_particle_density(
                wp.from_torch(density, dtype=wp.float32), wp.from_torch(selected, dtype=wp.bool)
            )


class G1MPMMixedEnv(G1MPMEnv):
    """G1 on terrain whose columns alternate between rigid ground and streaming MPM sand.

    The sand reaction on the feet is read as in :class:`G1MPMEnv`. The rigid ground reaction is read from the
    MJWarp contacts and added to it, so the contact flags and gait timers also work on the rigid columns.
    """

    cfg: G1MPMMixedEnvCfg

    def _init_sim(self):
        # The sand keeps this distance from the feet; the MPM solver takes it when it is constructed.
        solver_cfg = get_mpm_solver_cfg(self.cfg)
        solver_cfg.collider_margins = dict.fromkeys(solver_cfg.collider_margins or (), self.cfg.mpm_contact_margin)
        # The model is finalized inside the simulation reset, so margins must be registered before it.
        callback = self.sim.backend_manager.register_callback(self._set_contact_margin, PhysicsEvent.MODEL_INIT)
        try:
            super()._init_sim()
        finally:
            callback.deregister()

    def _set_contact_margin(self, payload=None):
        """Set the margin of the feet's rigid contact before the solvers copy the shapes.

        The sand's margin on the feet is not a shape margin: it is ``mpm_contact_margin``, passed to the MPM solver.
        """
        mpm = self.scene.terrain.mpm_env_mask.cpu().tolist()
        builder = self.sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=self.cfg.sim.physics))
        if set(builder.particle_world) != {env_id for env_id, is_mpm in enumerate(mpm) if is_mpm}:
            raise ValueError("The environments that hold sand are not the ones the terrain placed on sand.")
        matched_envs = set()
        for shape_index, body_index in enumerate(builder.shape_body):
            if body_index < 0 or not re.fullmatch(
                self.cfg.foot_body_expr, builder.body_label[body_index].rsplit("/", 1)[-1]
            ):
                continue
            env_id = builder.body_world[body_index]
            if 0 <= env_id < self.num_envs:
                builder.shape_margin[shape_index] = self.cfg.rigid_contact_margin
                matched_envs.add(env_id)
        if len(matched_envs) != self.num_envs:
            raise ValueError("Could not find foot shapes in every environment when setting contact margins.")

    def _setup_contact_state(self) -> None:
        entry = self.sim.backend_manager.backend.solver._entries[RIGID_ENTRY]
        self._rigid_solver = entry.solver
        foot_names = self.scene["robot"].find_bodies(self.cfg.foot_body_expr, preserve_order=True)[1]
        foot_bodies = self._resolve_newton_body_ids(foot_names).cpu().numpy().flatten()
        body_to_foot = {int(body): index for index, body in enumerate(foot_bodies)}
        local_to_global = entry.body_local_to_global.numpy()
        shape_to_foot = np.full(entry.solver.model.shape_count, -1, dtype=np.int32)
        for shape, body in enumerate(entry.solver.model.shape_body.numpy()):
            if body >= 0:
                shape_to_foot[shape] = body_to_foot.get(int(local_to_global[body]), -1)
        self._shape_to_foot = wp.array(shape_to_foot, dtype=int, device=self.device)
        self._shape_is_ground = wp.array(
            [label.startswith("/World/ground/") for label in entry.solver.model.shape_label],
            dtype=wp.bool,
            device=self.device,
        )
        capacity = self._rigid_solver.mjw_data.contact.geom.shape[0]
        self._contact_indices = wp.array(np.arange(capacity), dtype=int, device=self.device)
        self._rigid_contact_wrench = wp.zeros(capacity, dtype=wp.spatial_vector, device=self.device)
        self._rigid_foot_force = wp.zeros(len(foot_bodies), dtype=wp.vec3, device=self.device)
        super()._setup_contact_state()

    def _refresh_contact_forces(self) -> None:
        super()._refresh_contact_forces()
        # Coupled solvers do not support ContactSensor yet; read the MJWarp entry.
        solver = self._rigid_solver
        mjw.contact_force(solver.mjw_model, solver.mjw_data, self._contact_indices, True, self._rigid_contact_wrench)
        self._rigid_foot_force.zero_()
        wp.launch(
            sum_rigid_foot_forces,
            dim=self._contact_indices.shape[0],
            inputs=[
                solver.mjw_data.nacon,
                solver.mjw_data.contact.worldid,
                solver.mjw_data.contact.geom,
                solver.mjc_geom_to_newton_shape,
                self._shape_to_foot,
                self._shape_is_ground,
                self._rigid_contact_wrench,
            ],
            outputs=[self._rigid_foot_force],
            device=self.device,
        )
        # MJWarp ground contacts: the rigid columns, and the support under the sand when a foot reaches it.
        rigid_force = wp.to_torch(self._rigid_foot_force).reshape(self.num_envs, self.foot_count, 3)
        self._foot_contact_force = self._foot_contact_force + torch.nan_to_num(
            rigid_force, nan=0.0, posinf=0.0, neginf=0.0
        )
        self._foot_contact = self._foot_contact_force.norm(dim=-1) > self.cfg.foot_contact_force_threshold

    def reset_mpm_state(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        """Restore the sand of the selected sand environments and clear the solver history of all selected ones.

        Args:
            env_ids: Environment indices or a slice selecting environments to reset.
        """
        # Newton reset masks carry one trailing slot for global (world -1) entities.
        world_mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        world_mask[: self.num_envs][env_ids] = True
        if not world_mask.any():
            return
        sand_ids = self._sand.instance_ids(env_ids)
        if sand_ids.numel():
            particle_state = self._sand.data.default_particle_state_w.torch[sand_ids].clone()
            if self.cfg.reset_particle_jitter > 0.0:
                jitter = (
                    2.0 * torch.rand(particle_state.shape[:2] + (3,), device=self.device) - 1.0
                ) * self.cfg.reset_particle_jitter
                particle_state[..., :3] += jitter
            particle_state[..., 3:] = 0.0
            self._sand.write_particle_state_to_sim_index(particle_state, env_ids=sand_ids)
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
