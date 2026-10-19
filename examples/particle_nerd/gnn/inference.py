# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPM-free particle inference: Warp state/geometry operations and a Torch GNN.

The caller supplies current rigid-body pose and velocity. This module predicts
particles; it does not prescribe or integrate the box, nor estimate its force.
Use evaluate.py to replay held-out rigid motion and measure particle rollout drift.
"""

import torch
import warp as wp

from particle_nerd.gnn.features import Features, material_features
from particle_nerd.gnn.model import ParticleGNN


@wp.kernel
def advance(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    history: wp.array2d(dtype=float),
    correction: wp.array2d(dtype=float),
    dt: float,
    overflow: wp.array(dtype=int),
):
    i = wp.tid()
    old_v = v[i]
    dp = wp.vec3(correction[i, 0], correction[i, 1], correction[i, 2])
    dv = wp.vec3(correction[i, 3], correction[i, 4], correction[i, 5])
    nx = x[i] + dt * (old_v + dp)
    nv = old_v + dv
    if overflow[0] != 0:
        # A capacity error must not silently truncate the physical neighborhood.
        nx = wp.vec3(wp.nan)
        nv = wp.vec3(wp.nan)
    x[i] = nx
    v[i] = nv
    for j in range(history.shape[1] - 3):
        history[i, j] = history[i, j + 3]
    for k in range(3):
        history[i, history.shape[1] - 3 + k] = nv[k]


class ParticlePredictor:
    def __init__(
        self,
        checkpoint,
        position,
        velocity,
        manifest,
        particle_mass,
        particle_radius,
        device="cuda:0",
        capacity_per_particle=96,
        compile_model=False,
    ):
        self.device = device
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model = ParticleGNN(**saved["config"]).to(device).eval()
        self.model.load_state_dict(saved["model"])
        self.forward = torch.compile(self.model, fullgraph=True, dynamic=False) if compile_model else self.model
        self.dt = self.model.config["dt"]
        self.x = torch.as_tensor(position, dtype=torch.float32, device=device).clone()
        self.v = torch.as_tensor(velocity, dtype=torch.float32, device=device).clone()
        self.history = self.v.repeat(1, self.model.config["history"])
        self.material = torch.as_tensor(
            material_features(manifest, particle_mass, particle_radius, self.dt),
            device=device,
        )
        self.mass = manifest["body_mass_kg"]
        self.features = Features(
            len(self.x),
            radius=self.model.config["radius"],
            history=self.model.config["history"],
            capacity_per_particle=capacity_per_particle,
            device=device,
        )

    @torch.no_grad()
    def step(self, body_pose, body_velocity):
        """Enqueue a 20 ms particle step. Inputs are CUDA tensors; no CPU transfers."""
        features = self.features.prepare(self.x, self.history, body_pose, body_velocity, self.material, self.mass)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            correction = self.forward(*features).contiguous()
        stream = wp.stream_from_torch(torch.cuda.current_stream(self.x.device))
        with wp.ScopedStream(stream):
            wp.launch(
                advance,
                len(self.x),
                [
                    wp.from_torch(self.x, dtype=wp.vec3),
                    wp.from_torch(self.v, dtype=wp.vec3),
                    wp.from_torch(self.history),
                    wp.from_torch(correction),
                    self.dt,
                    self.features.overflow,
                ],
            )
        return self.x, self.v

    def diagnostics(self):
        """Explicit host synchronization outside the simulation step."""
        return dict(
            graph_overflow=bool(wp.to_torch(self.features.overflow)[0].item()),
            finite=bool(torch.isfinite(self.x).all() and torch.isfinite(self.v).all()),
            edge_count=int(wp.to_torch(self.features.offsets)[-1].item()),
        )
