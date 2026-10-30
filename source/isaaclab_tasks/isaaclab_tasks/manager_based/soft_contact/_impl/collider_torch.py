# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch

from .collider import ColliderCfg, PlaneColliderCfg, BoxColliderCfg, SphereColliderCfg

# Backward compatibility alias
IntruderGeometryCfg = PlaneColliderCfg


class ColliderTorch:
    """
    Pure-PyTorch collider that generates contact point positions and
    surface normals in body frame. Mirror of the warp-based ``Collider``
    in ``collider.py``.
    """

    def __init__(
        self,
        cfg: ColliderCfg,
        num_envs: int,
        num_bodies: int,
        device: torch.device | str,
    ) -> None:
        self._cfg = cfg
        self._num_envs = num_envs
        self._num_bodies = num_bodies
        self._device = device

        if isinstance(cfg, PlaneColliderCfg):
            self._setup_plane(cfg)
        elif isinstance(cfg, BoxColliderCfg):
            self._setup_box(cfg)
        elif isinstance(cfg, SphereColliderCfg):
            self._setup_sphere(cfg)
        else:
            raise ValueError(f"Unsupported collider config type: {type(cfg)}")

    # ------------------------------------------------------------------
    # Properties (read by the solver)
    # ------------------------------------------------------------------

    @property
    def contact_point_local(self) -> torch.Tensor:
        """Body-frame contact positions. Shape: (N, B, C, 3)."""
        return self._contact_point_local

    @property
    def normal_dir_local(self) -> torch.Tensor:
        """Body-frame surface normals. Shape: (N, B, C, 3)."""
        return self._normal_dir_local

    @property
    def num_contact_points(self) -> int:
        """Total number of contact points per body."""
        return self._num_contact_points

    @property
    def surface_area(self) -> float:
        """Total contact surface area (m^2)."""
        return self._surface_area

    @property
    def dA(self) -> torch.Tensor:
        """Per-contact-point area element. Shape: (C,)."""
        return self._dA

    # ------------------------------------------------------------------
    # Plane collider setup
    # ------------------------------------------------------------------

    def _setup_plane(self, cfg: PlaneColliderCfg) -> None:
        """
        Build contact point grid for a single planar face using PyTorch.

        Generates an (nx, ny) grid on the XY rectangle defined by
        contact_edge_x/y at z = -foot_depth. Normal = (0, 0, -1).
        """
        nx, ny = cfg.resolution
        self._num_contact_points = nx * ny

        lx = cfg.contact_edge_x[1] - cfg.contact_edge_x[0]
        ly = cfg.contact_edge_y[1] - cfg.contact_edge_y[0]
        self._surface_area = lx * ly

        foot_depth = cfg.contact_edge_z[1] - cfg.contact_edge_z[0]

        # Uniform dA for a single face
        dA_val = self._surface_area / self._num_contact_points
        self._dA = torch.full((self._num_contact_points,), dA_val, device=self._device)

        # Build 2D meshgrid
        xs = torch.linspace(cfg.contact_edge_x[0], cfg.contact_edge_x[1], nx, device=self._device)
        ys = torch.linspace(cfg.contact_edge_y[0], cfg.contact_edge_y[1], ny, device=self._device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        contact_point_offset = torch.stack(
            (
                grid_x.flatten(),
                grid_y.flatten(),
                -foot_depth * torch.ones(self._num_contact_points, device=self._device),
            ),
            dim=-1,
        )  # (C, 3)

        # Expand to (N, B, C, 3)
        self._contact_point_local = (
            contact_point_offset
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self._num_envs, self._num_bodies, -1, -1)
            .clone()
        )

        # Uniform downward normal
        self._normal_dir_local = (
            torch.tensor([0.0, 0.0, -1.0], device=self._device)
            .view(1, 1, 1, 3)
            .expand(self._num_envs, self._num_bodies, self._num_contact_points, -1)
            .clone()
        )

    # ------------------------------------------------------------------
    # Box collider setup
    # ------------------------------------------------------------------

    def _setup_box(self, cfg: BoxColliderCfg) -> None:
        """
        Build contact point grids for all 6 faces of a box using PyTorch.

        Each face has (nx, ny) points. Total = 6 * nx * ny.
        Face ordering: -Z, +Z, -Y, +Y, -X, +X.
        """
        nx, ny = cfg.resolution
        pts_per_face = nx * ny
        self._num_contact_points = 6 * pts_per_face

        x_min, x_max = cfg.contact_edge_x
        y_min, y_max = cfg.contact_edge_y
        z_min, z_max = cfg.contact_edge_z

        lx = x_max - x_min
        ly = y_max - y_min
        lz = z_max - z_min
        self._surface_area = 2.0 * (lx * ly + ly * lz + lz * lx)

        # Per-face dA
        face_areas = [lx * ly, lx * ly, lx * lz, lx * lz, ly * lz, ly * lz]
        dA_list = []
        for fa in face_areas:
            dA_list.extend([fa / pts_per_face] * pts_per_face)
        self._dA = torch.tensor(dA_list, dtype=torch.float32, device=self._device)

        # Parametric coordinates
        u = torch.linspace(0.0, 1.0, nx, device=self._device)
        v = torch.linspace(0.0, 1.0, ny, device=self._device)
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
        uf = grid_u.flatten()  # (pts_per_face,)
        vf = grid_v.flatten()  # (pts_per_face,)

        all_pts = []
        all_nrm = []

        # Face 0: -Z (bottom)
        all_pts.append(torch.stack([
            x_min + uf * lx, y_min + vf * ly,
            torch.full_like(uf, z_min),
        ], dim=-1))
        all_nrm.append(torch.tensor([0.0, 0.0, -1.0], device=self._device).expand(pts_per_face, -1))

        # Face 1: +Z (top)
        all_pts.append(torch.stack([
            x_min + uf * lx, y_min + vf * ly,
            torch.full_like(uf, z_max),
        ], dim=-1))
        all_nrm.append(torch.tensor([0.0, 0.0, 1.0], device=self._device).expand(pts_per_face, -1))

        # Face 2: -Y (front)
        all_pts.append(torch.stack([
            x_min + uf * lx, torch.full_like(uf, y_min),
            z_min + vf * lz,
        ], dim=-1))
        all_nrm.append(torch.tensor([0.0, -1.0, 0.0], device=self._device).expand(pts_per_face, -1))

        # Face 3: +Y (back)
        all_pts.append(torch.stack([
            x_min + uf * lx, torch.full_like(uf, y_max),
            z_min + vf * lz,
        ], dim=-1))
        all_nrm.append(torch.tensor([0.0, 1.0, 0.0], device=self._device).expand(pts_per_face, -1))

        # Face 4: -X (left)
        all_pts.append(torch.stack([
            torch.full_like(uf, x_min),
            y_min + uf * ly, z_min + vf * lz,
        ], dim=-1))
        all_nrm.append(torch.tensor([-1.0, 0.0, 0.0], device=self._device).expand(pts_per_face, -1))

        # Face 5: +X (right)
        all_pts.append(torch.stack([
            torch.full_like(uf, x_max),
            y_min + uf * ly, z_min + vf * lz,
        ], dim=-1))
        all_nrm.append(torch.tensor([1.0, 0.0, 0.0], device=self._device).expand(pts_per_face, -1))

        # Concatenate all faces (6*pts_per_face, 3)
        pts = torch.cat(all_pts, dim=0)
        nrm = torch.cat(all_nrm, dim=0)

        # Expand to (N, B, C, 3)
        self._contact_point_local = (
            pts.unsqueeze(0).unsqueeze(0)
            .expand(self._num_envs, self._num_bodies, -1, -1)
            .clone()
        )
        self._normal_dir_local = (
            nrm.unsqueeze(0).unsqueeze(0)
            .expand(self._num_envs, self._num_bodies, -1, -1)
            .clone()
        )

    # ------------------------------------------------------------------
    # Sphere collider setup
    # ------------------------------------------------------------------

    def _setup_sphere(self, cfg: SphereColliderCfg) -> None:
        """
        Build contact point grid for a sphere using spherical coordinates.

        Points sampled on (n_theta, n_phi) grid with half-step pole offset.
        """
        n_theta, n_phi = cfg.resolution
        self._num_contact_points = n_theta * n_phi
        self._surface_area = 4.0 * math.pi * cfg.radius ** 2

        # Uniform dA for sphere
        dA_val = self._surface_area / self._num_contact_points
        self._dA = torch.full((self._num_contact_points,), dA_val, device=self._device)

        center = torch.tensor(cfg.center, device=self._device)

        # theta in (0, pi), avoiding exact poles
        theta = math.pi * (torch.arange(n_theta, device=self._device, dtype=torch.float32) + 0.5) / n_theta
        phi = 2.0 * math.pi * torch.arange(n_phi, device=self._device, dtype=torch.float32) / n_phi

        grid_theta, grid_phi = torch.meshgrid(theta, phi, indexing="ij")
        grid_theta = grid_theta.flatten()
        grid_phi = grid_phi.flatten()

        # Outward unit normals
        nrm = torch.stack([
            torch.sin(grid_theta) * torch.cos(grid_phi),
            torch.sin(grid_theta) * torch.sin(grid_phi),
            torch.cos(grid_theta),
        ], dim=-1)  # (C, 3)

        pts = center + cfg.radius * nrm  # (C, 3)

        # Expand to (N, B, C, 3)
        self._contact_point_local = (
            pts.unsqueeze(0).unsqueeze(0)
            .expand(self._num_envs, self._num_bodies, -1, -1)
            .clone()
        )
        self._normal_dir_local = (
            nrm.unsqueeze(0).unsqueeze(0)
            .expand(self._num_envs, self._num_bodies, -1, -1)
            .clone()
        )
