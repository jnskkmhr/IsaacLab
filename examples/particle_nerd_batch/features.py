# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Persistent feature buffers for all environments; supports active compaction."""

import torch
import warp as wp

from .sdf import surface as sdf_surface


@wp.kernel
def compute_activity_scores(
    history: wp.array3d(dtype=float),
    poses: wp.array3d(dtype=float),
    velocities: wp.array3d(dtype=float),
    kinds: wp.array(dtype=int),
    sizes: wp.array2d(dtype=float),
    material: wp.array2d(dtype=float),
    particles: int,
    dt: float,
    sdf: wp.array3d(dtype=float),
    sdf_origin: wp.vec3,
    sdf_spacing: float,
    scores: wp.array2d(dtype=float),
):
    i = wp.tid()
    env = i // particles
    local_idx = i % particles
    h = history.shape[1]
    x = wp.vec3(history[i, h - 1, 0], history[i, h - 1, 1], history[i, h - 1, 2])
    previous = wp.vec3(history[i, h - 2, 0], history[i, h - 2, 1], history[i, h - 2, 2])
    center = wp.vec3(poses[env, 0, 0], poses[env, 0, 1], poses[env, 0, 2])
    rotation = wp.quat(poses[env, 0, 3], poses[env, 0, 4], poses[env, 0, 5], poses[env, 0, 6])
    local = wp.quat_rotate_inv(rotation, x - center)
    normal = wp.vec3(0.0, 0.0, 1.0)
    distance = local[2]
    kind = kinds[0]
    if kind == 1:
        q = wp.vec3(wp.abs(local[0]) - sizes[0, 0], wp.abs(local[1]) - sizes[0, 1], wp.abs(local[2]) - sizes[0, 2])
        outside = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        length = wp.length(outside)
        distance = length + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
        axis = int(0)
        if q[1] > q[axis]:
            axis = 1
        if q[2] > q[axis]:
            axis = 2
        normal = wp.vec3(0.0)
        for a in range(3):
            sign = float(1.0)
            if local[a] < 0.0:
                sign = -1.0
            if length > 1.0e-8:
                normal[a] = sign * outside[a] / length
            elif a == axis:
                normal[a] = sign
    elif kind == 2:
        length = wp.length(local)
        distance = length - sizes[0, 0]
        if length > 1.0e-8:
            normal = local / length
    elif kind == 3:
        distance, normal = sdf_surface(local, sdf, sdf_origin, sdf_spacing)
    normal = wp.quat_rotate(rotation, normal)
    surface = x - distance * normal
    linear = wp.vec3(velocities[env, 0, 0], velocities[env, 0, 1], velocities[env, 0, 2])
    angular = wp.vec3(velocities[env, 0, 3], velocities[env, 0, 4], velocities[env, 0, 5])
    v_surf = linear + wp.cross(angular, surface - center)
    v_part = (x - previous) / dt
    approach = wp.max(0.0, wp.dot(normal, v_surf - v_part))
    speed = wp.length(v_part)
    # Material-property-dependent stress-cone reach R_mat(mu, E, rho, dilatancy, viscosity):
    # Higher internal friction (tan(phi)), stiffness, and dilatancy widen the Mohr-Coulomb
    # shear-band zone ahead of the collider; higher viscosity damps far-field propagation.
    mu_mat = material[env, 3]
    stiff_norm = material[env, 1]
    dilatancy = material[env, 8]
    visc_norm = material[env, 9]
    p_radius = 0.02 * material[env, 11]
    r_mat = (0.04 + 0.08 * mu_mat + 0.02 * stiff_norm + 0.03 * dilatancy) / (1.0 + 0.25 * visc_norm) + 2.0 * dt * approach
    gap = distance - p_radius
    # Negative score means physically active (within support radius 0.07 m or shearing above creep threshold).
    # Particles between 0.07 m and r_mat provide neural slice context without being artificially displaced.
    v_thresh = 0.015 * (1.0 + 0.5 * mu_mat)
    scores[env, local_idx] = (gap - 0.07) - 0.8 * wp.max(0.0, speed - v_thresh) - 0.1 * wp.max(0.0, r_mat - gap)


@wp.kernel
def particle_features(
    history: wp.array3d(dtype=float),
    material: wp.array2d(dtype=float),
    particles: int,
    dt: float,
    origin: wp.vec3,
    gravity: wp.vec3,
    nodes: wp.array2d(dtype=float),
):
    i = wp.tid()
    env = i // particles
    h = history.shape[1]
    for a in range(3):
        nodes[i, a] = history[i, h - 1, a] - origin[a]
    for t in range(h - 1):
        for a in range(3):
            nodes[i, 3 + t * 3 + a] = (history[i, t + 1, a] - history[i, t, a]) / dt
    for k in range(12):
        nodes[i, 3 * h + k] = material[env, k]
    nodes[i, 3 * h + 12] = dt / 0.02
    for a in range(3):
        nodes[i, 3 * h + 13 + a] = gravity[a] / 9.81


@wp.kernel
def particle_features_indexed(
    history: wp.array3d(dtype=float),
    selected: wp.array(dtype=int),
    material: wp.array2d(dtype=float),
    selected_per_env: int,
    dt: float,
    origin: wp.vec3,
    gravity: wp.vec3,
    nodes: wp.array2d(dtype=float),
    x_cur_out: wp.array2d(dtype=float),
    x_prev_out: wp.array2d(dtype=float),
):
    slot = wp.tid()
    env = slot // selected_per_env
    i = selected[slot]
    h = history.shape[1]
    for a in range(3):
        cur_a = history[i, h - 1, a]
        prev_a = history[i, h - 2, a]
        nodes[slot, a] = cur_a - origin[a]
        x_cur_out[slot, a] = cur_a
        x_prev_out[slot, a] = prev_a
    for t in range(h - 1):
        for a in range(3):
            nodes[slot, 3 + t * 3 + a] = (history[i, t + 1, a] - history[i, t, a]) / dt
    for k in range(12):
        nodes[slot, 3 * h + k] = material[env, k]
    nodes[slot, 3 * h + 12] = dt / 0.02
    for a in range(3):
        nodes[slot, 3 * h + 13 + a] = gravity[a] / 9.81


@wp.kernel
def collider_features(
    history: wp.array3d(dtype=float),
    poses: wp.array3d(dtype=float),
    velocities: wp.array3d(dtype=float),
    kinds: wp.array(dtype=int),
    sizes: wp.array2d(dtype=float),
    friction: wp.array(dtype=float),
    material: wp.array2d(dtype=float),
    particles: int,
    dt: float,
    support: float,
    sdf: wp.array3d(dtype=float),
    sdf_origin: wp.vec3,
    sdf_spacing: float,
    pairs: wp.array3d(dtype=float),
    influence: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    env = i // particles
    h = history.shape[1]
    x = wp.vec3(history[i, h - 1, 0], history[i, h - 1, 1], history[i, h - 1, 2])
    previous = wp.vec3(history[i, h - 2, 0], history[i, h - 2, 1], history[i, h - 2, 2])
    center = wp.vec3(poses[env, j, 0], poses[env, j, 1], poses[env, j, 2])
    rotation = wp.quat(poses[env, j, 3], poses[env, j, 4], poses[env, j, 5], poses[env, j, 6])
    local = wp.quat_rotate_inv(rotation, x - center)
    normal = wp.vec3(0.0, 0.0, 1.0)
    distance = local[2]
    if kinds[j] == 1:
        q = wp.vec3(wp.abs(local[0]) - sizes[j, 0], wp.abs(local[1]) - sizes[j, 1], wp.abs(local[2]) - sizes[j, 2])
        outside = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        length = wp.length(outside)
        distance = length + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
        axis = int(0)
        if q[1] > q[axis]:
            axis = 1
        if q[2] > q[axis]:
            axis = 2
        normal = wp.vec3(0.0)
        for a in range(3):
            sign = float(1.0)
            if local[a] < 0.0:
                sign = -1.0
            if length > 1.0e-8:
                normal[a] = sign * outside[a] / length
            elif a == axis:
                normal[a] = sign
    elif kinds[j] == 2:
        length = wp.length(local)
        distance = length - sizes[j, 0]
        if length > 1.0e-8:
            normal = local / length
    elif kinds[j] == 3:
        distance, normal = sdf_surface(local, sdf, sdf_origin, sdf_spacing)
    normal = wp.quat_rotate(rotation, normal)
    surface = x - distance * normal
    linear = wp.vec3(velocities[env, j, 0], velocities[env, j, 1], velocities[env, j, 2])
    angular = wp.vec3(velocities[env, j, 3], velocities[env, j, 4], velocities[env, j, 5])
    relative = linear + wp.cross(angular, surface - center) - (x - previous) / dt
    gap = distance - 0.02 * material[env, 11]
    pairs[i, j, 0] = wp.clamp(gap / support, -1.0, 1.0)
    for a in range(3):
        pairs[i, j, 1 + a] = normal[a]
        pairs[i, j, 4 + a] = relative[a]
    pairs[i, j, 7] = friction[j]
    influence[i, j] = wp.max(0.0, 1.0 - wp.max(gap, 0.0) / support)


@wp.kernel
def collider_features_indexed(
    history: wp.array3d(dtype=float),
    selected: wp.array(dtype=int),
    poses: wp.array3d(dtype=float),
    velocities: wp.array3d(dtype=float),
    kinds: wp.array(dtype=int),
    sizes: wp.array2d(dtype=float),
    friction: wp.array(dtype=float),
    material: wp.array2d(dtype=float),
    selected_per_env: int,
    dt: float,
    support: float,
    sdf: wp.array3d(dtype=float),
    sdf_origin: wp.vec3,
    sdf_spacing: float,
    pairs: wp.array3d(dtype=float),
    influence: wp.array2d(dtype=float),
):
    slot, j = wp.tid()
    env = slot // selected_per_env
    i = selected[slot]
    h = history.shape[1]
    x = wp.vec3(history[i, h - 1, 0], history[i, h - 1, 1], history[i, h - 1, 2])
    previous = wp.vec3(history[i, h - 2, 0], history[i, h - 2, 1], history[i, h - 2, 2])
    center = wp.vec3(poses[env, j, 0], poses[env, j, 1], poses[env, j, 2])
    rotation = wp.quat(poses[env, j, 3], poses[env, j, 4], poses[env, j, 5], poses[env, j, 6])
    local = wp.quat_rotate_inv(rotation, x - center)
    normal = wp.vec3(0.0, 0.0, 1.0)
    distance = local[2]
    if kinds[j] == 1:
        q = wp.vec3(wp.abs(local[0]) - sizes[j, 0], wp.abs(local[1]) - sizes[j, 1], wp.abs(local[2]) - sizes[j, 2])
        outside = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        length = wp.length(outside)
        distance = length + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
        axis = int(0)
        if q[1] > q[axis]:
            axis = 1
        if q[2] > q[axis]:
            axis = 2
        normal = wp.vec3(0.0)
        for a in range(3):
            sign = float(1.0)
            if local[a] < 0.0:
                sign = -1.0
            if length > 1.0e-8:
                normal[a] = sign * outside[a] / length
            elif a == axis:
                normal[a] = sign
    elif kinds[j] == 2:
        length = wp.length(local)
        distance = length - sizes[j, 0]
        if length > 1.0e-8:
            normal = local / length
    elif kinds[j] == 3:
        distance, normal = sdf_surface(local, sdf, sdf_origin, sdf_spacing)
    normal = wp.quat_rotate(rotation, normal)
    surface = x - distance * normal
    linear = wp.vec3(velocities[env, j, 0], velocities[env, j, 1], velocities[env, j, 2])
    angular = wp.vec3(velocities[env, j, 3], velocities[env, j, 4], velocities[env, j, 5])
    relative = linear + wp.cross(angular, surface - center) - (x - previous) / dt
    gap = distance - 0.02 * material[env, 11]
    pairs[slot, j, 0] = wp.clamp(gap / support, -1.0, 1.0)
    for a in range(3):
        pairs[slot, j, 1 + a] = normal[a]
        pairs[slot, j, 4 + a] = relative[a]
    pairs[slot, j, 7] = friction[j]
    influence[slot, j] = wp.max(0.0, 1.0 - wp.max(gap, 0.0) / support)


@wp.kernel
def pack_rigid_colliders(
    pose: wp.array(dtype=wp.transform),
    velocity: wp.array(dtype=wp.spatial_vector),
    poses: wp.array3d(dtype=float),
    velocities: wp.array3d(dtype=float),
):
    env = wp.tid()
    p = wp.transform_get_translation(pose[env])
    q = wp.transform_get_rotation(pose[env])
    for a in range(3):
        poses[env, 0, a] = p[a]
    for a in range(4):
        poses[env, 0, 3 + a] = q[a]
    for a in range(6):
        velocities[env, 0, a] = velocity[env][a]


class FeatureBuffers:
    """Build features once per batch into persistent NN input buffers."""

    def __init__(
        self,
        history,
        material,
        kinds,
        sizes,
        friction,
        sdf,
        dt,
        support=0.07,
        origin=(0, 0, 0),
        gravity=(0, 0, -9.81),
        active_particles=None,
        anchor_particles=64,
    ):
        self.environments, self.particles, h, _ = history.shape
        self.total, self.colliders = self.environments * self.particles, len(kinds)
        self.device, self.dt = str(history.device), dt
        self.history_torch = history
        self.history = wp.from_torch(history.reshape(self.total, h, 3))
        self.material = wp.from_torch(material)
        self.kinds, self.sizes, self.friction = (wp.from_torch(x) for x in (kinds, sizes, friction))
        self.sdf = (wp.from_torch(sdf[0]), wp.vec3(*sdf[1]), sdf[2])
        self.support, self.origin, self.gravity = support, wp.vec3(*origin), wp.vec3(*gravity)
        self.poses = wp.zeros((self.environments, self.colliders, 7), device=self.device)
        self.velocities = wp.zeros((self.environments, self.colliders, 6), device=self.device)
        wp.to_torch(self.poses)[..., 6] = 1

        self.compact = active_particles is not None and (
            active_particles == "auto" or int(active_particles) < self.particles
        )
        if self.compact:
            if active_particles == "auto":
                # Derive physical upper bound K_cap from material friction & bed density:
                # Mohr-Coulomb passive wedge volume scales with (1 + 2 * mu_mat)^2.
                mu_max = float(material[:, 3].max().item())
                stiff_max = float(material[:, 1].max().item())
                heuristic_frac = min(0.20, 0.045 * (1.0 + 1.5 * mu_max + 0.15 * stiff_max))
                min_k = min(256, max(16, self.particles // 2))
                max_k = max(min_k, self.particles - max(1, int(anchor_particles)))
                raw_k = min(max_k, max(min_k, int(round(self.particles * heuristic_frac / 16.0)) * 16))
                self.active_k = raw_k
                self.heuristic_mode = True
            else:
                self.active_k = int(active_particles)
                self.heuristic_mode = True
            self.anchor_m = min(int(anchor_particles), self.particles - self.active_k)
            self.selected_per_env = self.active_k + self.anchor_m
            self.selected_total = self.environments * self.selected_per_env
            self.scores = wp.empty((self.environments, self.particles), dtype=float, device=self.device)
            self.scores_torch = wp.to_torch(self.scores)
            bg_1d = torch.linspace(0, self.particles - 1, self.anchor_m, device=history.device).long()
            self.bg_local = bg_1d.unsqueeze(0).expand(self.environments, -1).contiguous()
            self.env_offsets = (
                torch.arange(self.environments, device=history.device, dtype=torch.int64)[:, None] * self.particles
            )
            self.active_global = torch.empty(
                (self.environments, self.active_k), dtype=torch.int32, device=history.device
            )
            self.active_mask = torch.ones(
                (self.environments, self.active_k), dtype=torch.float32, device=history.device
            )
            self.active_count = torch.zeros((self.environments,), dtype=torch.float32, device=history.device)
            self.selected_global = torch.empty(
                (self.environments, self.selected_per_env), dtype=torch.int32, device=history.device
            )
            self.multiplicity = torch.ones(
                (self.environments, self.selected_per_env), dtype=torch.float32, device=history.device
            )
            self.nodes = wp.empty((self.selected_total, 16 + 3 * h), device=self.device)
            self.pairs = wp.empty((self.selected_total, self.colliders, 8), device=self.device)
            self.influence = wp.empty((self.selected_total, self.colliders), device=self.device)
            self.x_cur_buf = wp.empty((self.selected_total, 3), device=self.device)
            self.x_prev_buf = wp.empty((self.selected_total, 3), device=self.device)
            self.x_cur = wp.to_torch(self.x_cur_buf).view(self.environments, self.selected_per_env, 3)[
                :, : self.active_k
            ]
            self.x_prev = wp.to_torch(self.x_prev_buf).view(self.environments, self.selected_per_env, 3)[
                :, : self.active_k
            ]
            self.inputs = (
                wp.to_torch(self.nodes).view(self.environments, self.selected_per_env, -1),
                wp.to_torch(self.pairs).view(self.environments, self.selected_per_env, self.colliders, 8),
                wp.to_torch(self.influence).view(self.environments, self.selected_per_env, self.colliders),
            )
        else:
            self.active_k = self.particles
            self.anchor_m = 0
            self.selected_per_env = self.particles
            self.selected_total = self.total
            self.multiplicity = None
            self.active_mask = None
            self.active_count = torch.full(
                (self.environments,), float(self.particles), dtype=torch.float32, device=history.device
            )
            self.nodes = wp.empty((self.total, 16 + 3 * h), device=self.device)
            self.pairs = wp.empty((self.total, self.colliders, 8), device=self.device)
            self.influence = wp.empty((self.total, self.colliders), device=self.device)
            self.inputs = (
                wp.to_torch(self.nodes).view(self.environments, self.particles, -1),
                wp.to_torch(self.pairs).view(self.environments, self.particles, self.colliders, 8),
                wp.to_torch(self.influence).view(self.environments, self.particles, self.colliders),
            )

    def build(self, body_pose, body_velocity):
        wp.launch(
            pack_rigid_colliders,
            self.environments,
            [body_pose, body_velocity, self.poses, self.velocities],
            device=self.device,
        )
        if self.compact:
            wp.launch(
                compute_activity_scores,
                self.total,
                [
                    self.history,
                    self.poses,
                    self.velocities,
                    self.kinds,
                    self.sizes,
                    self.material,
                    self.particles,
                    self.dt,
                    *self.sdf,
                    self.scores,
                ],
                device=self.device,
            )
            top_vals, topk_local = torch.topk(self.scores_torch, k=self.active_k, dim=-1, largest=False, sorted=False)
            # Only particles satisfying the physical proximity + material stress-cone criterion (score < 0)
            # are actively integrated; if fewer than 64 are within reach, keep the closest 64 so contact is ready.
            phys_mask = (top_vals < 0.0).float()
            min_active = (top_vals <= torch.topk(top_vals, k=min(64, self.active_k), dim=-1, largest=False).values.max(dim=-1, keepdim=True).values).float()
            act_mask = torch.maximum(phys_mask, min_active)
            self.active_mask.copy_(act_mask)
            n_act = act_mask.sum(dim=-1, keepdim=True)
            self.active_count.copy_(n_act.squeeze(-1))
            thresh = torch.minimum(top_vals.max(dim=-1, keepdim=True).values, torch.zeros_like(n_act))
            bg_scores = torch.gather(self.scores_torch, 1, self.bg_local)
            bg_valid = (bg_scores > thresh).float()
            # Inactive slots in topk_local plus the background anchors represent the (N - N_active) stationary bed
            inactive_top = 1.0 - act_mask
            total_bg_slots = bg_valid.sum(dim=-1, keepdim=True) + inactive_top.sum(dim=-1, keepdim=True)
            bg_weight = (float(self.particles) - n_act) / total_bg_slots.clamp_min(1.0)
            self.multiplicity[:, : self.active_k] = act_mask + inactive_top * bg_weight
            self.multiplicity[:, self.active_k :] = bg_valid * bg_weight
            active_g = (self.env_offsets + topk_local).to(torch.int32)
            bg_g = (self.env_offsets + self.bg_local).to(torch.int32)
            self.active_global.copy_(active_g)
            self.selected_global[:, : self.active_k].copy_(active_g)
            self.selected_global[:, self.active_k :].copy_(bg_g)
            sel_wp = wp.from_torch(self.selected_global.view(-1), dtype=wp.int32)
            wp.launch(
                particle_features_indexed,
                self.selected_total,
                [
                    self.history,
                    sel_wp,
                    self.material,
                    self.selected_per_env,
                    self.dt,
                    self.origin,
                    self.gravity,
                    self.nodes,
                    self.x_cur_buf,
                    self.x_prev_buf,
                ],
                device=self.device,
            )
            wp.launch(
                collider_features_indexed,
                (self.selected_total, self.colliders),
                [
                    self.history,
                    sel_wp,
                    self.poses,
                    self.velocities,
                    self.kinds,
                    self.sizes,
                    self.friction,
                    self.material,
                    self.selected_per_env,
                    self.dt,
                    self.support,
                    *self.sdf,
                    self.pairs,
                    self.influence,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                particle_features,
                self.total,
                [self.history, self.material, self.particles, self.dt, self.origin, self.gravity, self.nodes],
                device=self.device,
            )
            wp.launch(
                collider_features,
                (self.total, self.colliders),
                [
                    self.history,
                    self.poses,
                    self.velocities,
                    self.kinds,
                    self.sizes,
                    self.friction,
                    self.material,
                    self.particles,
                    self.dt,
                    self.support,
                    *self.sdf,
                    self.pairs,
                    self.influence,
                ],
                device=self.device,
            )
