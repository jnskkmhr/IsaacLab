# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Adapted from Felix Meyer's paged-MPM settle_template (see NOTICE.md).

Uses the task's sampled positions and material rather than the example's lattice.
The mirrored frictionless-wall construction and full constitutive-state transfer
are retained. A template that fails the settling criterion is never called settled.
"""

from collections import OrderedDict, deque

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM

MATERIAL_KEYS = (
    "young_modulus",
    "poisson_ratio",
    "friction",
    "damping",
    "yield_pressure",
    "tensile_yield_ratio",
    "yield_stress",
    "hardening",
    "dilatancy",
    "viscosity",
    "hardening_rate",
    "softening_rate",
)


@wp.kernel
def guard_container(q: wp.array[wp.vec3], qd: wp.array[wp.vec3], size: float):
    """Felix's floor guard for particles outside the collider's finite search distance."""
    i = wp.tid()
    p, v = q[i], qd[i]
    for axis in range(2):
        if p[axis] < 0.0:
            p[axis] = 0.0
            v[axis] = wp.max(v[axis], 0.0)
        elif p[axis] > size:
            p[axis] = size
            v[axis] = wp.min(v[axis], 0.0)
    if p[2] < 0.0:
        p[2] = 0.0
        v[2] = wp.max(v[2], 0.0)
    q[i] = p
    qd[i] = v


def settle_template(positions, size, depth, material, mass, radius, plastic, voxel_size, cfg, device):
    """Relax a tile under gravity between frictionless mirror walls."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.add_ground_plane()
    wall = newton.ModelBuilder.ShapeConfig(mu=0.0)
    t, hz = 0.1, depth + 0.2
    for x in (-t, size + t):
        builder.add_shape_box(
            -1, xform=wp.transform((x, 0.5 * size, hz), wp.quat_identity()), hx=t, hy=size, hz=hz, cfg=wall
        )
    for y in (-t, size + t):
        builder.add_shape_box(
            -1, xform=wp.transform((0.5 * size, y, hz), wp.quat_identity()), hx=size, hy=t, hz=hz, cfg=wall
        )
    points = positions.copy()
    points[:, 2] = np.clip(points[:, 2] + depth, 0.0, depth)
    n = len(points)
    builder.add_particles(pos=points.tolist(), vel=[(0.0, 0.0, 0.0)] * n, mass=[mass] * n, radius=[radius] * n)
    model = builder.finalize(device=device)
    for name, value in material.items():
        getattr(model.mpm, name).fill_(value)
    state = model.state()
    state.mpm.particle_Jp.fill_(plastic)
    config = SolverImplicitMPM.Config()
    config.voxel_size = voxel_size
    config.strain_basis = "P0"
    config.transfer_scheme = "pic"
    config.max_iterations = 100
    config.tolerance = 1.0e-6
    config.critical_fraction = 0.0
    solver = SolverImplicitMPM(model, config=config)
    heights = deque(maxlen=cfg.settle_quiet_steps)
    quiet = 0
    for step in range(cfg.settle_max_steps):
        solver.step(state, state, control=None, contacts=None, dt=cfg.settle_dt)
        wp.launch(guard_container, dim=n, inputs=[state.particle_q, state.particle_qd, size], device=device)
        speed = float(np.linalg.norm(state.particle_qd.numpy(), axis=1).max())
        if not np.isfinite(speed):
            raise RuntimeError("MPM material template diverged during settling")
        heights.append(float(state.particle_q.numpy()[:, 2].mean()))
        quiet = quiet + 1 if speed < cfg.settle_speed else 0
        if quiet >= cfg.settle_quiet_steps and max(heights) - min(heights) <= cfg.settle_height_tolerance:
            break
    else:
        raise RuntimeError(f"MPM template did not settle in {cfg.settle_max_steps} steps (max speed {speed:.6g} m/s)")
    q = state.particle_q.numpy()
    if float(np.ptp(q[:, 2])) < 0.25 * depth:
        raise RuntimeError("MPM template collapsed against the floor instead of supporting its material depth")
    q[:, 2] -= depth
    return dict(
        q=q,
        elastic=state.mpm.particle_elastic_strain.numpy(),
        transform=state.mpm.particle_transform.numpy(),
        stress=state.mpm.particle_stress.numpy(),
        plastic=state.mpm.particle_Jp.numpy(),
        steps=step + 1,
        max_speed=speed,
        height_drift=max(heights) - min(heights),
    )


class SettledTemplates:
    """Bounded material-keyed templates, with one GPU template per simulation world."""

    def __init__(self, patch, positions, radius):
        self.patch, self.positions, self.radius = patch, positions, radius
        self.cache = OrderedDict()
        self.keys = [None] * patch.model.world_count
        self.stats = [None] * patch.model.world_count
        count = patch.model.world_count * len(positions)
        self.q = wp.empty(count, dtype=wp.vec3, device=patch.model.device)
        self.elastic = wp.empty(count, dtype=wp.mat33, device=patch.model.device)
        self.transform = wp.empty_like(self.elastic)
        self.stress = wp.empty_like(self.elastic)
        self.plastic = wp.empty(count, dtype=float, device=patch.model.device)

    def update(self, reset):
        """Rebuild only reset worlds whose material differs from the cached template."""
        patch, cfg = self.patch, self.patch.terrain.particle_cache
        selected = np.flatnonzero(reset)
        if not len(selected):
            return
        # Material can be randomized independently per world; the template key includes every coefficient.
        material_arrays = {k: getattr(patch.model.mpm, k).numpy() for k in MATERIAL_KEYS if hasattr(patch.model.mpm, k)}
        masses = patch.particle_dynamic_mass.numpy()
        plastic = patch.initial_particle_plastic_volume_ratio.numpy()
        worlds = patch.model.particle_world.numpy()
        for world in selected:
            world = int(world)
            slot = int(np.flatnonzero(worlds == world)[0])
            material = {k: float(v[slot] if len(v) > 1 else v[0]) for k, v in material_arrays.items()}
            key = (*material.items(), float(masses[slot]), float(plastic[slot]))
            if self.keys[world] == key:
                continue
            if key not in self.cache:
                with wp.ScopedDevice(patch.model.device):
                    self.cache[key] = settle_template(
                        self.positions,
                        cfg.tile_size,
                        patch.terrain.particle_depth,
                        material,
                        float(masses[slot]),
                        self.radius,
                        float(plastic[slot]),
                        patch.terrain.voxel_size,
                        cfg,
                        patch.model.device,
                    )
            self.cache.move_to_end(key)
            result = self.cache[key]
            begin, end = world * len(self.positions), (world + 1) * len(self.positions)
            for name in ("q", "elastic", "transform", "stress", "plastic"):
                getattr(self, name)[begin:end].assign(result[name])
            self.keys[world] = key
            self.stats[world] = {k: result[k] for k in ("steps", "max_speed", "height_drift")}
            while len(self.cache) > cfg.max_material_templates:
                self.cache.popitem(last=False)
