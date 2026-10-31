# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replace the native material step with Transolver and analytical foot reactions.

The native coupled solver still advances the actuated robot. No MPM step or
constitutive solve is called after installation. Initial MPM allocation is retained
for the existing task's particle state and reset interface.
"""

import newton
import numpy as np
import torch
import warp as wp

from particle_nerd.g1.data import (
    collider_state,
    compact_step,
    compact_step_batched,
    infer_collider_slots,
    resolve_active_k,
)
from particle_nerd.g1.forces import G1ProxyEstimator
from particle_nerd.transolver.features import material_features
from particle_nerd.transolver.model import ParticleTransolver


@wp.kernel
def scatter_wrench(
    local_feet: wp.array(dtype=int),
    mapping: wp.array(dtype=int),
    source: wp.array(dtype=wp.spatial_vector),
    target: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    target[mapping[local_feet[i]]] = source[i]


class LearnedMaterial:
    def __init__(
        self,
        raw,
        checkpoint,
        *,
        iterations=50,
        effective_mass_scale=10.0,
        contact_relaxation=0.1,
        contact_solver="jacobi",
        active_k=None,
    ):
        from isaaclab_newton.physics import NewtonManager

        self.num_envs = raw.num_envs
        self.raw, self.manager = raw, NewtonManager
        self.coupled = NewtonManager._solver
        self.entry = self.coupled._entries["sand"]
        self.mpm = self.entry.solver
        self.master = NewtonManager.get_model()
        self.device = str(self.master.device)
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if saved["config"].get("schema") != "g1_particle_nerd_v1":
            raise ValueError("Requires the G1-trained checkpoint")
        self.dt = saved["config"]["dt"]
        if not np.isclose(self.dt, 0.005):
            raise ValueError("This adapter requires a 200 Hz particle model")
        self.architecture = saved.get("architecture")
        if self.architecture != "particle_transolver_v1":
            raise ValueError(f"Expected particle_transolver_v1, got {self.architecture}")
        self.active_k = active_k
        self.model_config = saved["config"]["model"]
        self.model = ParticleTransolver(**self.model_config).to(self.device).eval()
        self.model.load_state_dict(saved["model"])
        self.support = saved["config"]["features"]["support"]
        pw, bw = self.master.particle_world.numpy(), self.master.body_world.numpy()
        self.pids = [
            torch.as_tensor(np.flatnonzero(pw == e), device=self.device, dtype=torch.long) for e in range(self.num_envs)
        ]
        self.particles_per_env = len(self.pids[0])
        if any(len(p) != self.particles_per_env for p in self.pids):
            raise ValueError("All G1 environments must have equal particle counts")
        global_feet_all = raw._foot_body_ids.cpu().numpy()
        mapping = self.entry.body_local_to_global.numpy()
        local_feet_all = []
        for e in range(self.num_envs):
            for gf in global_feet_all[e]:
                local_feet_all.append(int(np.flatnonzero(mapping == gf).item()))
        self.estimator = G1ProxyEstimator(
            self.mpm,
            local_feet_all,
            iterations=iterations,
            effective_mass_scale=effective_mass_scale,
            relaxation=contact_relaxation,
            contact_solver=contact_solver,
            particles_per_env=self.particles_per_env,
        )
        self.local_feet = wp.array(local_feet_all, dtype=int, device=self.master.device)
        self.local_foot_tensor = torch.tensor(local_feet_all, device=self.device, dtype=torch.long).view(
            self.num_envs, -1
        )
        self.global_foot_tensor = torch.tensor(global_feet_all, device=self.device, dtype=torch.long)
        self.wrench = self.estimator.wrench
        self.metadata = self.estimator.metadata
        sb, st = self.master.shape_body.numpy(), self.master.shape_type.numpy()

        def tensor(a):
            return torch.as_tensor(a, dtype=torch.float32, device=self.device)

        self.geometries = []
        self.foot_slots_list = []
        for e in range(self.num_envs):
            bids_e = set(np.flatnonzero(bw == e))
            gfeet_e = global_feet_all[e]
            eligible = set(gfeet_e) | {i for i in bids_e if "/MPM" in self.master.body_label[i]}
            ids = np.array([i for i, b in enumerate(sb) if b in eligible and st[i] == newton.GeoType.BOX])
            if len(ids) != 7:
                raise ValueError(f"Env {e}: Expected seven native collision boxes, got {len(ids)}")
            sizes_e = tensor(self.master.shape_scale.numpy()[ids])
            fslots_e = torch.tensor(
                [int(np.flatnonzero(sb[ids] == foot).item()) for foot in gfeet_e],
                device=self.device,
                dtype=torch.long,
            )
            fslots_e, floor_slot_e = infer_collider_slots(sizes_e, fslots_e)
            self.geometries.append(
                dict(
                    body_ids=torch.as_tensor(sb[ids], device=self.device, dtype=torch.long),
                    local_pose=tensor(self.master.shape_transform.numpy()[ids]),
                    sizes=sizes_e,
                    friction=tensor(self.master.shape_material_mu.numpy()[ids]),
                    com=tensor(self.master.body_com.numpy()),
                    kinds=torch.ones(7, dtype=torch.int32, device=self.device),
                    foot_slots=fslots_e,
                    floor_slot=floor_slot_e,
                )
            )
            self.foot_slots_list.append(fslots_e)
        self.geometry = self.geometries[0]
        self.foot_slots = self.foot_slots_list[0]
        from particle_nerd.g1 import MATERIAL_KEYS

        material = {k: getattr(raw.cfg.scene.sand.spawn.material, k) for k in MATERIAL_KEYS}
        self.material = tensor(
            material_features(
                dict(material=material, grid_spacing_m=0.04),
                float(self.mpm.model.particle_mass.numpy()[0]),
                float(self.mpm.model.particle_radius.numpy()[0]),
                self.dt,
            )
        )
        self.origins = raw.scene.env_origins.clone()
        self.origin = self.origins[0]
        self.body_world_tensor = torch.as_tensor(bw, device=self.device, dtype=torch.long).clamp_min(0)
        self.particle_world_tensor = torch.as_tensor(pw, device=self.device, dtype=torch.long).clamp_min(0)
        raw_active_k = active_k if active_k is not None else saved["config"].get("active_k")
        self.active_k = resolve_active_k(raw_active_k, self.particles_per_env, self.material)
        self.anchor_m = int(saved["config"].get("anchor_m", 512) or 512)
        self.last_active_counts = [self.particles_per_env] * self.num_envs
        self.history = None
        self.steps = 0
        self.original_coupled_step = self.coupled.step
        self.original_material_step = self.mpm.step
        self.original_harvest = self.mpm.coupling_harvest_proxy_wrenches
        self.coupled.step = self.step
        self.mpm.step = self.particle_step
        self.mpm.coupling_harvest_proxy_wrenches = self.harvest

    @torch.inference_mode()
    def step(self, state_in, state_out, control, contacts, dt):
        if not np.isclose(dt, self.dt):
            raise ValueError(f"Model dt {self.dt} != physics dt {dt}")
        self.body_pose = wp.to_torch(state_in.body_q).clone()
        self.body_pose[:, :3] -= self.origins[self.body_world_tensor]
        self.body_velocity = wp.to_torch(state_in.body_qd).clone()
        return self.original_coupled_step(state_in, state_out, control, contacts, dt)

    @torch.inference_mode()
    def particle_step(self, state_in, state_out, control, contacts, dt):
        del control, contacts
        self.estimator.capture(state_in, dt)
        particle_origins = self.origins[self.particle_world_tensor]
        x = wp.to_torch(state_in.particle_q) - particle_origins
        if self.history is None:
            v = wp.to_torch(state_in.particle_qd)
            self.history = torch.stack([x - k * dt * v for k in reversed(range(self.model_config["history"]))], 1)
        if self.num_envs == 1:
            poses, velocities = collider_state(self.body_pose, self.body_velocity, self.geometry)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                displacement, active_cnt = compact_step(
                    self.model,
                    self.history,
                    poses,
                    velocities,
                    self.geometry,
                    self.material,
                    dt,
                    support=self.support,
                    active_k=self.active_k,
                    anchor_m=self.anchor_m,
                    return_active_count=True,
                )
                self.last_active_counts = [active_cnt]
        else:
            poses_list, vels_list = [], []
            for e in range(self.num_envs):
                geom_e = self.geometries[e]
                p_e, v_e = collider_state(self.body_pose, self.body_velocity, geom_e)
                poses_list.append(p_e)
                vels_list.append(v_e)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                displacement, self.last_active_counts = compact_step_batched(
                    self.model,
                    self.history,
                    poses_list,
                    vels_list,
                    self.geometries,
                    self.material,
                    dt,
                    support=self.support,
                    active_k=self.active_k,
                    anchor_m=self.anchor_m,
                )
        predicted = x + displacement
        state_out.assign(state_in)
        wp.to_torch(state_out.particle_q).copy_(predicted + particle_origins)
        wp.to_torch(state_out.particle_qd).copy_(displacement / dt)
        self.history = torch.cat((self.history[:, 1:], predicted[:, None]), 1)
        self.estimator.evaluate(state_out)
        self.steps += 1

    def harvest(self, body_local_to_proxy_global, out_body_f, **kwargs):
        del kwargs
        out_body_f.zero_()
        wp.launch(
            scatter_wrench,
            len(self.local_feet),
            [self.local_feet, body_local_to_proxy_global, self.wrench, out_body_f],
            device=self.master.device,
        )

    def close(self):
        self.coupled.step = self.original_coupled_step
        self.mpm.step = self.original_material_step
        self.mpm.coupling_harvest_proxy_wrenches = self.original_harvest
