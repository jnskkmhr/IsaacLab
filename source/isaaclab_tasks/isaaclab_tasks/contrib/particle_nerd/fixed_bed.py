# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-bed coordinate transforms matching the G1 example's neural terrain inputs."""

import math

import newton
import numpy as np
import torch
import warp as wp
from torch.nn import functional as F

from .network.geometry import boundary_boxes
from .network.inference import collider_state, multiply, rotate


class FixedBed:
    """Describe real flat retaining boxes or an intrinsic slope, wave or powered tray.

    This only transforms neural inputs and outputs. It neither steps a native
    material solver nor projects predicted particles back into the container.
    """

    def __init__(self, model, cfg, geometries):
        self.model, self.cfg = model, cfg
        if cfg.kind not in ("flat", "slope", "wave", "tilting"):
            raise ValueError(f"Unsupported fixed bed kind: {cfg.kind}")
        extent = np.asarray(cfg.upper) - np.asarray(cfg.lower)
        if not np.isfinite(extent).all() or (extent <= 0).any():
            raise ValueError("Fixed bed bounds must have positive finite extents")
        self.depth = float(extent[2])
        self.origins = torch.zeros((model.world_count, 3), device=str(model.device))
        self.tray_ids = []
        bodies, worlds = model.shape_body.numpy(), model.body_world.numpy()
        shape_types = model.shape_type.numpy()
        boundary_pose, boundary_size, boundary_mu = boundary_boxes(cfg.lower, cfg.upper, str(model.device))
        self.boundary_pose = boundary_pose
        for world, geometry in enumerate(geometries):
            geometry["sliding_window"] = max(extent[:2]) > 3.1 if cfg.sliding_window is None else cfg.sliding_window
            if cfg.kind == "flat":
                names = ("MPMBedFloor", "MPMBedWallFront", "MPMBedWallLeft", "MPMBedWallRight", "MPMApproachBank")
                ids = []
                for name in names:
                    found = [
                        i
                        for i, label in enumerate(model.body_label)
                        if worlds[i] == world and label.endswith("/" + name)
                    ]
                    if len(found) != 1:
                        raise ValueError(f"Expected one retaining body {name} in world {world}; got {found}")
                    ids += found
                shapes = [
                    int(np.flatnonzero((bodies == body) & (shape_types == int(newton.GeoType.BOX))).item())
                    for body in ids
                ]
                geometry["body_ids"] = torch.cat((geometry["body_ids"], torch.tensor(ids, device=self.origins.device)))
                for key, values in (
                    ("local_pose", model.shape_transform.numpy()[shapes]),
                    ("sizes", model.shape_scale.numpy()[shapes]),
                    ("friction", model.shape_material_mu.numpy()[shapes]),
                ):
                    geometry[key] = torch.cat((geometry[key][:2], torch.as_tensor(values, device=self.origins.device)))
            else:
                geometry["sizes"] = torch.cat((geometry["sizes"][:2], boundary_size))
                geometry["friction"] = torch.cat((geometry["friction"][:2], boundary_mu))
            if cfg.kind == "tilting":
                found = [
                    i
                    for i, label in enumerate(model.body_label)
                    if worlds[i] == world and label.endswith("/Testbed/tray")
                ]
                if len(found) != 1:
                    raise ValueError(f"Expected one tray proxy in world {world}; got {found}")
                self.tray_ids += found

    def bind_origins(self, origins):
        """Copy environment origins once before stepping, keeping CUDA graph storage stable."""
        self.origins.copy_(origins)

    def height_and_slope(self, x):
        length = self.cfg.upper[0] - self.cfg.lower[0]
        u = (x - self.cfg.lower[0]).clamp(0.0, length)
        if self.cfg.kind == "slope":
            return self.cfg.slope * u, torch.full_like(x, self.cfg.slope)
        if self.cfg.kind == "wave":
            omega = 2 * math.pi * self.cfg.wave_count / length
            return (
                0.5 * self.cfg.wave_height * (1 - torch.cos(omega * u)),
                0.5 * self.cfg.wave_height * omega * torch.sin(omega * u),
            )
        return torch.zeros_like(x), torch.zeros_like(x)

    def _tray(self, state, world):
        body = self.tray_ids[world]
        pose = wp.to_torch(state.body_q)[body]
        velocity = wp.to_torch(state.body_qd)[body]
        # Newton stores linear velocity at the COM; transforms require the body origin.
        com = rotate(pose[3:], wp.to_torch(self.model.body_com)[body])
        linear = velocity[:3] - torch.linalg.cross(velocity[3:], com, dim=-1)
        return pose[:3], pose[3:], linear, velocity[3:]

    def inputs(self, history, state, world, geometry):
        """Return history and collider kinematics in the bed's neural coordinate frame."""
        poses, velocities = collider_state(wp.to_torch(state.body_q), wp.to_torch(state.body_qd), geometry)
        hist = history.clone()
        if self.cfg.kind == "tilting":
            position, rotation, linear, angular = self._tray(state, world)
            inverse = torch.cat((-rotation[:3], rotation[3:]))
            hist = rotate(inverse.expand(*hist.shape[:-1], 4), hist - position)
            hist[..., 2] -= self.depth
            relative = poses[:, :3] - position
            poses[:, :3] = rotate(inverse.expand(len(poses), 4), relative)
            poses[:, 2] -= self.depth
            poses[:, 3:] = F.normalize(multiply(inverse.expand(len(poses), 4), poses[:, 3:]), dim=-1)
            at_shape = linear + torch.linalg.cross(angular.expand_as(relative), relative, dim=-1)
            velocities[:, :3] = rotate(inverse.expand(len(poses), 4), velocities[:, :3] - at_shape)
            velocities[:, 3:] = rotate(inverse.expand(len(poses), 4), velocities[:, 3:] - angular)
        else:
            hist -= self.origins[world]
            poses[:, :3] -= self.origins[world]
            if self.cfg.kind in ("slope", "wave"):
                height, _ = self.height_and_slope(hist[..., 0])
                hist[..., 2] -= height
                height, slope = self.height_and_slope(poses[:, 0])
                half_angle = 0.5 * torch.atan(slope)
                zeros = torch.zeros_like(half_angle)
                rotation = torch.stack((zeros, torch.sin(half_angle), zeros, torch.cos(half_angle)), dim=-1)
                poses[:, 2] -= height
                poses[:, 3:] = F.normalize(multiply(rotation, poses[:, 3:]), dim=-1)
                velocities[:, 2] -= slope * velocities[:, 0]
        if self.cfg.kind != "flat":
            poses = torch.cat((poses, self.boundary_pose))
            velocities = torch.cat((velocities, torch.zeros((5, 6), device=velocities.device)))
        return hist, poses, velocities

    def to_world(self, predicted, state, world, dt):
        """Convert a prediction to world space, advancing the prescribed tray frame by dt."""
        predicted = predicted.clone()
        if self.cfg.kind == "tilting":
            position, rotation, linear, angular = self._tray(state, world)
            delta = torch.cat((0.5 * dt * angular, torch.ones(1, device=angular.device)))
            next_rotation = F.normalize(multiply(delta, rotation), dim=-1)
            predicted[:, 2] += self.depth
            return position + dt * linear + rotate(next_rotation.expand(len(predicted), 4), predicted)
        if self.cfg.kind in ("slope", "wave"):
            height, _ = self.height_and_slope(predicted[:, 0])
            predicted[:, 2] += height
        return predicted + self.origins[world]
