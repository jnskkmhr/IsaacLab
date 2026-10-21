# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replace the native material step with Transolver and analytical foot reactions.

The native coupled solver still advances the actuated robot. No MPM step or
constitutive solve is called after installation. Initial MPM allocation is retained
for the existing task's particle state and reset interface. Single environment.
"""

import newton
import numpy as np
import torch
import warp as wp

from particle_nerd.g1.data import collider_state
from particle_nerd.g1.forces import G1ProxyEstimator
from particle_nerd.gnn.features import material_features
from particle_nerd.transolver.features import features
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
    def __init__(self, raw, checkpoint, *, iterations=50, effective_mass_scale=10.0, contact_relaxation=0.1):
        from isaaclab_newton.physics import NewtonManager

        if raw.num_envs != 1:
            raise ValueError("G1 evaluation currently supports one environment")
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
        self.model = ParticleTransolver(**saved["config"]["model"]).to(self.device).eval()
        self.model.load_state_dict(saved["model"])
        self.support = saved["config"]["features"]["support"]
        global_feet = raw._foot_body_ids[0].cpu().numpy()
        mapping = self.entry.body_local_to_global.numpy()
        local_feet = [int(np.flatnonzero(mapping == i).item()) for i in global_feet]
        self.estimator = G1ProxyEstimator(
            self.mpm,
            local_feet,
            iterations=iterations,
            effective_mass_scale=effective_mass_scale,
            relaxation=contact_relaxation,
        )
        self.local_feet = wp.array(local_feet, dtype=int, device=self.master.device)
        sb, st = self.master.shape_body.numpy(), self.master.shape_type.numpy()
        eligible = set(global_feet) | {i for i, s in enumerate(self.master.body_label) if "/MPM" in s}
        ids = np.array([i for i, b in enumerate(sb) if b in eligible and st[i] == newton.GeoType.BOX])
        if len(ids) != 7:
            raise ValueError(f"Expected seven native collision boxes, got {len(ids)}")

        def tensor(a):
            return torch.as_tensor(a, dtype=torch.float32, device=self.device)

        self.geometry = dict(
            body_ids=torch.as_tensor(sb[ids], device=self.device, dtype=torch.long),
            local_pose=tensor(self.master.shape_transform.numpy()[ids]),
            sizes=tensor(self.master.shape_scale.numpy()[ids]),
            friction=tensor(self.master.shape_material_mu.numpy()[ids]),
            com=tensor(self.master.body_com.numpy()),
            kinds=torch.ones(7, dtype=torch.int32, device=self.device),
        )
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
        self.origin = raw.scene.env_origins[0].clone()
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
        # Capture geometry at n, before native robot forward dynamics advances it.
        self.body_pose = wp.to_torch(state_in.body_q).clone()
        self.body_pose[:, :3] -= self.origin
        self.body_velocity = wp.to_torch(state_in.body_qd).clone()
        return self.original_coupled_step(state_in, state_out, control, contacts, dt)

    @torch.inference_mode()
    def particle_step(self, state_in, state_out, control, contacts, dt):
        del control, contacts
        self.estimator.capture(state_in, dt)
        x = wp.to_torch(state_in.particle_q) - self.origin
        if self.history is None:
            v = wp.to_torch(state_in.particle_qd)
            # Startup from the available current state, with constant-velocity history.
            self.history = torch.stack([x - k * dt * v for k in reversed(range(self.model.config["history"]))], 1)
        poses, velocities = collider_state(self.body_pose, self.body_velocity, self.geometry)
        inputs = features(
            self.history,
            poses,
            velocities,
            self.geometry["kinds"],
            self.geometry["sizes"],
            self.geometry["friction"],
            self.material,
            dt,
            support=self.support,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            displacement = self.model(*inputs)
        predicted = x + displacement
        # Preserve unchanged body state and MPM state storage; never run its solver.
        state_out.assign(state_in)
        wp.to_torch(state_out.particle_q).copy_(predicted + self.origin)
        wp.to_torch(state_out.particle_qd).copy_(displacement / dt)
        self.history = torch.cat((self.history[:, 1:], predicted[:, None]), 1)
        self.estimator.evaluate(state_out)
        self.steps += 1

    def harvest(self, body_local_to_proxy_global, out_body_f, **kwargs):
        del kwargs
        out_body_f.zero_()
        wp.launch(
            scatter_wrench,
            len(self.estimator.body_ids),
            [self.local_feet, body_local_to_proxy_global, self.estimator.wrench, out_body_f],
            device=self.master.device,
        )

    def close(self):
        self.coupled.step = self.original_coupled_step
        self.mpm.step = self.original_material_step
        self.mpm.coupling_harvest_proxy_wrenches = self.original_harvest
