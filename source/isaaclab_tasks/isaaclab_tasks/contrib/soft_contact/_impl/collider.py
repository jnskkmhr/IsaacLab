# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import torch
import warp as wp

from isaaclab.utils.configclass import configclass

from .kernels import (
    box_contact_points,
    cylinder_contact_points,
    expand_1d_vec3_to_3d_vec3,
    plane_contact_points,
    sphere_contact_points,
)

"""
Collider configuration classes.
"""


@configclass
class ColliderCfg:
    """Base collider configuration. Subclass per shape."""


@configclass
class PlaneColliderCfg(ColliderCfg):
    """
    Single planar face collider (e.g. foot sole).

    Contact points are sampled as an (nx, ny) grid on the XY plane
    at z = contact_edge_z[0] (bottom of the geometry).
    Surface normal points in -Z direction in body frame.
    """

    contact_edge_x: tuple[float, float] = MISSING  # type: ignore
    """(min, max) bounds in body-frame X (m)."""
    contact_edge_y: tuple[float, float] = MISSING  # type: ignore
    """(min, max) bounds in body-frame Y (m)."""
    contact_edge_z: tuple[float, float] = MISSING  # type: ignore
    """(min, max) bounds in body-frame Z (m). Depth = z[1] - z[0]."""
    resolution: tuple[int, int] = (5, 5)
    """(nx, ny) grid resolution on the face."""


@configclass
class BoxColliderCfg(ColliderCfg):
    """
    6-face box collider.

    Contact points are sampled as an (nx, ny) grid on each of the 6 faces.
    Total contact points = nx * ny * 6.
    """

    contact_edge_x: tuple[float, float] = MISSING  # type: ignore
    """(min, max) bounds in body-frame X (m)."""
    contact_edge_y: tuple[float, float] = MISSING  # type: ignore
    """(min, max) bounds in body-frame Y (m)."""
    contact_edge_z: tuple[float, float] = MISSING  # type: ignore
    """(min, max) bounds in body-frame Z (m)."""
    resolution: tuple[int, int] = (5, 5)
    """(nx, ny) grid resolution per face."""


@configclass
class SphereColliderCfg(ColliderCfg):
    """
    Sphere collider.

    Contact points are sampled using a spherical coordinate grid.
    Total contact points = n_theta * n_phi.
    """

    radius: float = MISSING  # type: ignore
    """Sphere radius (m)."""
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Center of the sphere in body frame (m)."""
    resolution: tuple[int, int] = (8, 8)
    """(n_theta, n_phi) angular grid resolution."""


@configclass
class CylinderColliderCfg(ColliderCfg):
    """
    Cylinder collider (lateral wall + 2 end caps).

    Contact points are sampled on an (n_a, n_phi) grid per surface:
    n_a axial slices on the lateral wall / radial rings on each cap,
    n_phi azimuthal samples. Total contact points = 3 * n_a * n_phi.
    """

    radius: float = MISSING  # type: ignore
    """Cylinder radius (m)."""
    height: float = MISSING  # type: ignore
    """Cylinder height along the axis (m)."""
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Cylinder axis direction in body frame (need not be normalized). Default: Z."""
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Center of the cylinder in body frame (m)."""
    resolution: tuple[int, int] = (4, 8)
    """(n_a, n_phi) grid resolution per surface."""


def _quat_from_z_to_axis(axis: tuple[float, float, float]) -> wp.quatf:
    """Shortest-arc rotation quaternion mapping +Z to the given axis direction."""
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-9:
        raise ValueError(f"Cylinder axis must be non-zero, got: {axis}")
    ax, ay, az = ax / norm, ay / norm, az / norm

    if az > 1.0 - 1e-8:
        return wp.quatf(0.0, 0.0, 0.0, 1.0)  # already +Z
    if az < -1.0 + 1e-8:
        return wp.quatf(1.0, 0.0, 0.0, 0.0)  # 180 deg about X

    # Rotation axis = Z x axis = (-ay, ax, 0), angle = acos(Z . axis)
    angle = math.acos(az)
    s = math.sin(0.5 * angle)
    rx, ry = -ay, ax
    r_norm = math.sqrt(rx * rx + ry * ry)
    return wp.quatf(s * rx / r_norm, s * ry / r_norm, 0.0, math.cos(0.5 * angle))


"""
Collider runtime class.
"""


class Collider:
    """
    Generates contact point positions and surface normals in body frame.

    Each collider shape dispatches to its own warp kernel for building
    the contact point grid. The solver reads the outputs via properties
    and does not need to know the collider shape.
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
        elif isinstance(cfg, CylinderColliderCfg):
            self._setup_cylinder(cfg)
        else:
            raise ValueError(f"Unsupported collider config type: {type(cfg)}")

    # ------------------------------------------------------------------
    # Properties (read by the solver)
    # ------------------------------------------------------------------

    @property
    def contact_point_local(self) -> wp.array:
        """Body-frame contact positions. Shape: (N, B, C) dtype=vec3f."""
        return self._contact_point_local

    @property
    def normal_dir_local(self) -> wp.array:
        """Body-frame surface normals. Shape: (N, B, C) dtype=vec3f."""
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
    def dA(self) -> wp.array:
        """Per-contact-point area element. Shape: (C,) dtype=float32."""
        return self._dA

    # ------------------------------------------------------------------
    # ConeDRFT grid point generation
    # TODO: refactor later
    # ------------------------------------------------------------------

    def get_cone_single_point(self) -> tuple[tuple[float, float, float], float]:
        """
        Get a single bottom-center contact point for ConeDRFT model.

        Returns:
            (center_point, r_h) where:
            - center_point: (x, y, z) tuple in body frame (bottom-center)
            - r_h: hydraulic radius (float)
        """
        if isinstance(self._cfg, (PlaneColliderCfg, BoxColliderCfg)):
            return self._cone_single_plane_or_box()
        elif isinstance(self._cfg, SphereColliderCfg):
            return self._cone_single_sphere()
        elif isinstance(self._cfg, CylinderColliderCfg):
            return self._cone_single_cylinder()
        else:
            raise ValueError(f"ConeDRFT single point not supported for collider type: {type(self._cfg)}")

    def _cone_single_plane_or_box(self) -> tuple[tuple[float, float, float], float]:
        """Get single point for PlaneColliderCfg or BoxColliderCfg."""
        cfg = self._cfg
        x_min, x_max = cfg.contact_edge_x
        y_min, y_max = cfg.contact_edge_y
        z_val = cfg.contact_edge_z[0]

        lx = x_max - x_min
        ly = y_max - y_min
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)

        r_h = (lx * ly) / (lx + ly)

        return (cx, cy, z_val), r_h

    def _cone_single_sphere(self) -> tuple[tuple[float, float, float], float]:
        """Get single point for SphereColliderCfg."""
        cfg = self._cfg
        cx, cy, cz = cfg.center
        r = cfg.radius

        # Bottom-center point
        point = (cx, cy, cz - r)
        r_h = r

        return point, r_h

    def _cone_single_cylinder(self) -> tuple[tuple[float, float, float], float]:
        """Get single point for CylinderColliderCfg. Requires a vertical axis."""
        cfg = self._cfg
        self._cylinder_vertical_axis_sign(cfg)
        cx, cy, cz = cfg.center

        # Bottom-cap center point; hydraulic radius of a circular
        # cross-section (2A/P) is the cylinder radius.
        point = (cx, cy, cz - 0.5 * cfg.height)
        r_h = cfg.radius

        return point, r_h

    def get_cone_grid_points(self) -> tuple[list[tuple[float, float, float]], float]:
        """
        Reuse collider contact points for ConeDRFT models.

        Extracts contact points from the existing collider grid and computes
        the per-point hydraulic radius based on actual point spacing.

        For BoxCollider: Uses only bottom face (-Z) contact points.
        For PlaneCollider: Uses all contact points (already bottom face).
        For SphereCollider: Uses all contact points.
        For CylinderCollider: Uses only bottom cap contact points (vertical axis only).

        Returns:
            (contact_points, r_h_per_point) where:
            - contact_points: list of (x, y, z) tuples in body frame
            - r_h_per_point: hydraulic radius for each contact point (float)
        """
        if isinstance(self._cfg, PlaneColliderCfg):
            return self._cone_grid_plane()
        elif isinstance(self._cfg, BoxColliderCfg):
            return self._cone_grid_box()
        elif isinstance(self._cfg, SphereColliderCfg):
            return self._cone_grid_sphere()
        elif isinstance(self._cfg, CylinderColliderCfg):
            return self._cone_grid_cylinder()
        else:
            raise ValueError(f"ConeDRFT grid not supported for collider type: {type(self._cfg)}")

    def _cone_grid_plane(self) -> tuple[list[tuple[float, float, float]], float]:
        """Reuse all contact points from plane collider."""
        cfg = self._cfg
        nx, ny = cfg.resolution

        # Extract contact points from collider (already in body frame)
        # Plane collider generates all points on the bottom face
        contact_points_np = self._contact_point_local[0, 0].numpy()  # (num_points, 3)
        contact_points = [tuple(pt) for pt in contact_points_np]

        # Compute hydraulic radius from point spacing
        lx = cfg.contact_edge_x[1] - cfg.contact_edge_x[0]
        ly = cfg.contact_edge_y[1] - cfg.contact_edge_y[0]
        cell_x = lx / max(nx - 1, 1) if nx > 1 else lx
        cell_y = ly / max(ny - 1, 1) if ny > 1 else ly
        r_h_per_point = (cell_x * cell_y) / (cell_x + cell_y)

        return contact_points, r_h_per_point

    def _cone_grid_box(self) -> tuple[list[tuple[float, float, float]], float]:
        """Reuse only bottom face contact points from box collider."""
        cfg = self._cfg
        nx, ny = cfg.resolution

        # Box collider face ordering: -Z, +Z, -Y, +Y, -X, +X
        # Bottom face is -Z (face index 0), spanning points [0:nx*ny]
        contact_points_np = self._contact_point_local[0, 0].numpy()  # (6*nx*ny, 3)
        bottom_face_points = contact_points_np[:nx * ny]  # Only -Z face
        contact_points = [tuple(pt) for pt in bottom_face_points]

        # Compute hydraulic radius from point spacing
        lx = cfg.contact_edge_x[1] - cfg.contact_edge_x[0]
        ly = cfg.contact_edge_y[1] - cfg.contact_edge_y[0]
        cell_x = lx / max(nx - 1, 1) if nx > 1 else lx
        cell_y = ly / max(ny - 1, 1) if ny > 1 else ly
        r_h_per_point = (cell_x * cell_y) / (cell_x + cell_y)

        return contact_points, r_h_per_point

    def _cone_grid_sphere(self) -> tuple[list[tuple[float, float, float]], float]:
        """Reuse all contact points from sphere collider."""
        cfg = self._cfg

        # Extract all contact points from sphere collider
        contact_points_np = self._contact_point_local[0, 0].numpy()  # (n_theta*n_phi, 3)
        contact_points = [tuple(pt) for pt in contact_points_np]

        # Compute hydraulic radius scaled by number of points
        radius = cfg.radius
        r_h_per_point = radius / math.sqrt(len(contact_points))

        return contact_points, r_h_per_point

    def _cone_grid_cylinder(self) -> tuple[list[tuple[float, float, float]], float]:
        """Reuse only bottom cap contact points from cylinder collider. Requires a vertical axis."""
        cfg = self._cfg
        axis_sign = self._cylinder_vertical_axis_sign(cfg)
        n_a, n_phi = cfg.resolution
        pts_per_surf = n_a * n_phi

        # Surface ordering: lateral, -Z cap, +Z cap (in the canonical frame).
        # With a downward-pointing axis, the canonical +Z cap is the bottom one.
        cap_idx = 1 if axis_sign > 0.0 else 2
        contact_points_np = self._contact_point_local[0, 0].numpy()  # (3*n_a*n_phi, 3)
        cap_points = contact_points_np[cap_idx * pts_per_surf : (cap_idx + 1) * pts_per_surf]
        contact_points = [tuple(pt) for pt in cap_points]

        # Compute hydraulic radius scaled by number of points
        r_h_per_point = cfg.radius / math.sqrt(len(contact_points))

        return contact_points, r_h_per_point

    @staticmethod
    def _cylinder_vertical_axis_sign(cfg: CylinderColliderCfg) -> float:
        """Return the sign of the cylinder's Z axis; raise if the axis is not vertical."""
        ax, ay, az = cfg.axis
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if abs(az) / norm < 1.0 - 1e-6:
            raise ValueError(
                f"ConeDRFT supports only vertical (Z-axis) cylinders, got axis: {cfg.axis}"
            )
        return az

    # ------------------------------------------------------------------
    # Plane collider setup
    # ------------------------------------------------------------------

    def _setup_plane(self, cfg: PlaneColliderCfg) -> None:
        """
        Build contact point grid for a single planar face.

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
        self._dA = wp.full(
            (self._num_contact_points,), dA_val, dtype=wp.float32, device=self._device
        )

        # Allocate warp arrays
        self._contact_point_local = wp.zeros(
            (self._num_envs, self._num_bodies, self._num_contact_points),
            dtype=wp.vec3f,
            device=self._device,
        )
        self._normal_dir_local = wp.zeros(
            (self._num_envs, self._num_bodies, self._num_contact_points),
            dtype=wp.vec3f,
            device=self._device,
        )

        # Flat 1D arrays for the kernel to write into
        contact_pts_flat = wp.zeros(
            (self._num_contact_points,), dtype=wp.vec3f, device=self._device
        )
        contact_nrm_flat = wp.zeros(
            (self._num_contact_points,), dtype=wp.vec3f, device=self._device
        )

        edge_x = wp.vec2f(
            (wp.float32(cfg.contact_edge_x[0]), wp.float32(cfg.contact_edge_x[1]))
        )
        edge_y = wp.vec2f(
            (wp.float32(cfg.contact_edge_y[0]), wp.float32(cfg.contact_edge_y[1]))
        )

        wp.launch(
            kernel=plane_contact_points,
            dim=(ny, nx),
            inputs=[edge_x, edge_y, -foot_depth, nx, ny, contact_pts_flat, contact_nrm_flat],
            device=self._device,
        )

        # Expand to (num_envs, num_bodies, num_contact_points)
        wp.launch(
            kernel=expand_1d_vec3_to_3d_vec3,
            dim=(self._num_envs, self._num_bodies, self._num_contact_points),
            inputs=[contact_pts_flat, self._contact_point_local],
            device=self._device,
        )
        wp.launch(
            kernel=expand_1d_vec3_to_3d_vec3,
            dim=(self._num_envs, self._num_bodies, self._num_contact_points),
            inputs=[contact_nrm_flat, self._normal_dir_local],
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Box collider setup
    # ------------------------------------------------------------------

    def _setup_box(self, cfg: BoxColliderCfg) -> None:
        """
        Build contact point grids for all 6 faces of a box.

        Each face has (nx, ny) points. Total = 6 * nx * ny.
        Normals point outward from each face.
        """
        nx, ny = cfg.resolution
        self._num_contact_points = 6 * nx * ny

        lx = cfg.contact_edge_x[1] - cfg.contact_edge_x[0]
        ly = cfg.contact_edge_y[1] - cfg.contact_edge_y[0]
        lz = cfg.contact_edge_z[1] - cfg.contact_edge_z[0]
        self._surface_area = 2.0 * (lx * ly + ly * lz + lz * lx)

        # Per-face areas
        face_areas = [
            lx * ly,   # -Z
            lx * ly,   # +Z
            lx * lz,   # -Y
            lx * lz,   # +Y
            ly * lz,   # -X
            ly * lz,   # +X
        ]
        dA_list = []
        pts_per_face = nx * ny
        for fa in face_areas:
            dA_list.extend([fa / pts_per_face] * pts_per_face)
        self._dA = wp.array(dA_list, dtype=wp.float32, device=self._device)

        # Allocate expanded warp arrays
        self._contact_point_local = wp.zeros(
            (self._num_envs, self._num_bodies, self._num_contact_points),
            dtype=wp.vec3f,
            device=self._device,
        )
        self._normal_dir_local = wp.zeros(
            (self._num_envs, self._num_bodies, self._num_contact_points),
            dtype=wp.vec3f,
            device=self._device,
        )

        # Flat 1D arrays for the kernel to write into
        contact_pts_flat = wp.zeros(
            (self._num_contact_points,), dtype=wp.vec3f, device=self._device
        )
        contact_nrm_flat = wp.zeros(
            (self._num_contact_points,), dtype=wp.vec3f, device=self._device
        )

        edge_x = wp.vec2f(
            (wp.float32(cfg.contact_edge_x[0]), wp.float32(cfg.contact_edge_x[1]))
        )
        edge_y = wp.vec2f(
            (wp.float32(cfg.contact_edge_y[0]), wp.float32(cfg.contact_edge_y[1]))
        )
        edge_z = wp.vec2f(
            (wp.float32(cfg.contact_edge_z[0]), wp.float32(cfg.contact_edge_z[1]))
        )

        wp.launch(
            kernel=box_contact_points,
            dim=(6, ny, nx),
            inputs=[edge_x, edge_y, edge_z, nx, ny, contact_pts_flat, contact_nrm_flat],
            device=self._device,
        )

        # Expand to (num_envs, num_bodies, num_contact_points)
        wp.launch(
            kernel=expand_1d_vec3_to_3d_vec3,
            dim=(self._num_envs, self._num_bodies, self._num_contact_points),
            inputs=[contact_pts_flat, self._contact_point_local],
            device=self._device,
        )
        wp.launch(
            kernel=expand_1d_vec3_to_3d_vec3,
            dim=(self._num_envs, self._num_bodies, self._num_contact_points),
            inputs=[contact_nrm_flat, self._normal_dir_local],
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Cylinder collider setup
    # ------------------------------------------------------------------

    def _setup_cylinder(self, cfg: CylinderColliderCfg) -> None:
        """
        Build contact point grids for a cylinder (lateral wall + 2 caps).

        Each surface has (n_a, n_phi) points. Total = 3 * n_a * n_phi.
        Normals point outward. The cylinder axis defaults to body-frame Z.
        """
        n_a, n_phi = cfg.resolution
        self._num_contact_points = 3 * n_a * n_phi

        lateral_area = 2.0 * math.pi * cfg.radius * cfg.height
        cap_area = math.pi * cfg.radius ** 2
        self._surface_area = lateral_area + 2.0 * cap_area

        # Per-point areas: uniform on the lateral wall; exact annulus areas
        # per radial ring on the caps (rings are equally spaced in radius,
        # so outer rings carry more area).
        dA_list = [lateral_area / (n_a * n_phi)] * (n_a * n_phi)
        cap_dA = []
        for i in range(n_a):
            ring_area = cap_area * ((i + 1) ** 2 - i ** 2) / (n_a ** 2)
            cap_dA.extend([ring_area / n_phi] * n_phi)
        dA_list.extend(cap_dA * 2)  # -Z and +Z caps
        self._dA = wp.array(dA_list, dtype=wp.float32, device=self._device)

        # Allocate expanded warp arrays
        self._contact_point_local = wp.zeros(
            (self._num_envs, self._num_bodies, self._num_contact_points),
            dtype=wp.vec3f,
            device=self._device,
        )
        self._normal_dir_local = wp.zeros(
            (self._num_envs, self._num_bodies, self._num_contact_points),
            dtype=wp.vec3f,
            device=self._device,
        )

        # Flat 1D arrays for the kernel
        contact_pts_flat = wp.zeros(
            (self._num_contact_points,), dtype=wp.vec3f, device=self._device
        )
        contact_nrm_flat = wp.zeros(
            (self._num_contact_points,), dtype=wp.vec3f, device=self._device
        )

        center = wp.vec3f(
            wp.float32(cfg.center[0]),
            wp.float32(cfg.center[1]),
            wp.float32(cfg.center[2]),
        )
        axis_rot = _quat_from_z_to_axis(cfg.axis)

        wp.launch(
            kernel=cylinder_contact_points,
            dim=(3, n_a, n_phi),
            inputs=[
                wp.float32(cfg.radius),
                wp.float32(cfg.height),
                center,
                axis_rot,
                n_a,
                n_phi,
                contact_pts_flat,
                contact_nrm_flat,
            ],
            device=self._device,
        )

        # Expand to (num_envs, num_bodies, num_contact_points)
        wp.launch(
            kernel=expand_1d_vec3_to_3d_vec3,
            dim=(self._num_envs, self._num_bodies, self._num_contact_points),
            inputs=[contact_pts_flat, self._contact_point_local],
            device=self._device,
        )
        wp.launch(
            kernel=expand_1d_vec3_to_3d_vec3,
            dim=(self._num_envs, self._num_bodies, self._num_contact_points),
            inputs=[contact_nrm_flat, self._normal_dir_local],
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Sphere collider setup
    # ------------------------------------------------------------------

    def _setup_sphere(self, cfg: SphereColliderCfg) -> None:
        """
        Build contact point grid for a sphere using spherical coordinates.

        Points are sampled on a (n_theta, n_phi) grid. Normals point
        radially outward.
        """
        n_theta, n_phi = cfg.resolution
        self._num_contact_points = n_theta * n_phi
        self._surface_area = 4.0 * math.pi * cfg.radius ** 2

        # Uniform dA for sphere
        dA_val = self._surface_area / self._num_contact_points
        self._dA = wp.full(
            (self._num_contact_points,), dA_val, dtype=wp.float32, device=self._device
        )

        # Allocate expanded warp arrays
        self._contact_point_local = wp.zeros(
            (self._num_envs, self._num_bodies, self._num_contact_points),
            dtype=wp.vec3f,
            device=self._device,
        )
        self._normal_dir_local = wp.zeros(
            (self._num_envs, self._num_bodies, self._num_contact_points),
            dtype=wp.vec3f,
            device=self._device,
        )

        # Flat 1D arrays for the kernel
        contact_pts_flat = wp.zeros(
            (self._num_contact_points,), dtype=wp.vec3f, device=self._device
        )
        contact_nrm_flat = wp.zeros(
            (self._num_contact_points,), dtype=wp.vec3f, device=self._device
        )

        center = wp.vec3f(
            wp.float32(cfg.center[0]),
            wp.float32(cfg.center[1]),
            wp.float32(cfg.center[2]),
        )

        wp.launch(
            kernel=sphere_contact_points,
            dim=(n_theta, n_phi),
            inputs=[
                wp.float32(cfg.radius),
                center,
                n_theta,
                n_phi,
                contact_pts_flat,
                contact_nrm_flat,
            ],
            device=self._device,
        )

        # Expand to (num_envs, num_bodies, num_contact_points)
        wp.launch(
            kernel=expand_1d_vec3_to_3d_vec3,
            dim=(self._num_envs, self._num_bodies, self._num_contact_points),
            inputs=[contact_pts_flat, self._contact_point_local],
            device=self._device,
        )
        wp.launch(
            kernel=expand_1d_vec3_to_3d_vec3,
            dim=(self._num_envs, self._num_bodies, self._num_contact_points),
            inputs=[contact_nrm_flat, self._normal_dir_local],
            device=self._device,
        )
