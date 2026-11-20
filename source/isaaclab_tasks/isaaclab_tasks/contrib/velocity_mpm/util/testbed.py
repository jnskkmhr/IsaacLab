# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed sand testbeds: terrain-conforming particle beds with retaining walls and a powered tilting tray."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import trimesh
import warp as wp
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMPointsCfg
from isaaclab_newton.sim.spawners.mpm.mpm import _generate_grid_particles, spawn_mpm_particles

from pxr import Gf, Sdf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from .terrain import WarpTerrainMesh

if TYPE_CHECKING:
    from pxr import Usd


def x_sloped_terrain(difficulty: float, cfg: MeshXSlopeTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Create a planar ramp rising along +X, with its negative-X entrance at Z = 0.

    Author the plane directly so height-field quantization and steep-slope vertex
    corrections cannot introduce steps or vertical faces. The slope is dz/dx.
    """
    length, width = cfg.size
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    if not math.isfinite(slope):
        raise ValueError("The terrain slope must be finite.")
    vertices = np.array(
        [(0.0, 0.0, 0.0), (length, 0.0, slope * length), (length, width, slope * length), (0.0, width, 0.0)],
        dtype=np.float64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    origin = np.array((0.5 * length, 0.5 * width, 0.5 * slope * length))
    return [mesh], origin


@configclass
class MeshXSlopeTerrainCfg(SubTerrainBaseCfg):
    """A single planar ramp along X, with no slope across Y."""

    function = x_sloped_terrain
    slope_range: tuple[float, float] = (0.0, 0.4)
    """Rise/run at difficulty 0 and 1. Negative values descend along +X."""


def x_dune_terrain(difficulty: float, cfg: MeshXDuneTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Create cosine-shaped dune ridges with a level, zero-height negative-X entrance."""
    length, width = cfg.size
    height = cfg.height_range[0] + difficulty * (cfg.height_range[1] - cfg.height_range[0])
    if not math.isfinite(height) or height < 0.0:
        raise ValueError("Dune height must be finite and non-negative [m].")
    if not isinstance(cfg.num_waves, int) or cfg.num_waves < 1:
        raise ValueError("num_waves must be a positive integer.")
    if not math.isfinite(cfg.horizontal_scale) or cfg.horizontal_scale <= 0.0:
        raise ValueError("horizontal_scale must be positive and finite [m].")
    count = max(math.ceil(length / cfg.horizontal_scale), 8 * cfg.num_waves) + 1
    x = np.linspace(0.0, length, count)
    z = 0.5 * height * (1.0 - np.cos(2.0 * math.pi * cfg.num_waves * x / length))
    vertices = np.column_stack((np.repeat(x, 2), np.tile([0.0, width], count), np.repeat(z, 2)))
    a = 2 * np.arange(count - 1)
    faces = np.concatenate((np.column_stack((a, a + 2, a + 3)), np.column_stack((a, a + 3, a + 1))))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    origin = np.array((length / 2, width / 2, np.interp(length / 2, x, z)))
    return [mesh], origin


@configclass
class MeshXDuneTerrainCfg(SubTerrainBaseCfg):
    """Smooth dune ridges along X, with no height-field quantization or slope correction."""

    function = x_dune_terrain
    height_range: tuple[float, float] = (0.1, 0.3)
    """Trough-to-crest height at difficulty 0 and 1 [m]."""
    num_waves: int = 2
    """Number of complete ridges over the bed length."""
    horizontal_scale: float = 0.05
    """Maximum separation between mesh vertices along X [m]."""


def terrain_height_from_entrance(terrain: SubTerrainBaseCfg, size: tuple[float, float], distance: float) -> float:
    """Return the generated surface height at a distance from the negative-X edge of the bed.

    Args:
        terrain: Sub-terrain whose shape does not depend on the difficulty.
        size: Bed length and width [m].
        distance: Distance from the negative-X edge along +X [m].

    Returns:
        The surface height above the negative-X edge [m].
    """
    meshes, _ = terrain.function(0.0, terrain.replace(size=size))
    vertices = meshes[0].vertices
    order = np.argsort(vertices[:, 0], kind="stable")
    return float(np.interp(distance, vertices[order, 0], vertices[order, 2]))


def generate_terrain_particles(
    cfg: TerrainMPMGridCfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, trimesh.Trimesh]:
    """Return terrain-conforming particles and their shared initial surface mesh.

    Grid Z coordinates are depths relative to the generated surface: upper[2]
    must be zero and lower[2] is the negative sand depth. Query the highest
    surface at each jittered particle XY, as in the moving-patch environment.
    Holes and overhang interiors are not filled; a missing height query is an error.
    """
    if cfg.upper[2] != 0.0:
        raise ValueError("Terrain particle upper[2] must be zero (the generated sand surface).")
    if cfg.terrain_generator.seed is None:
        raise ValueError("Set terrain_generator.seed so cloned terrain is reproducible.")
    positions, velocities, masses, radii = _generate_grid_particles(cfg)
    generator = cfg.terrain_generator.class_type(cfg.terrain_generator.copy(), device="cpu")
    surface_mesh = generator.terrain_mesh
    terrain_mesh = WarpTerrainMesh(surface_mesh.vertices, surface_mesh.faces, device="cpu")
    query_positions = wp.array(positions, dtype=wp.vec3, device="cpu")
    terrain_mesh.sample_surface_heights(query_positions)
    positions[:, 2] += query_positions.numpy()[:, 2]
    return positions, velocities, masses, radii, surface_mesh


@clone
def spawn_terrain_mpm(
    prim_path: str,
    cfg: TerrainMPMGridCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Spawn one fixed terrain bed; clone its particles and support geometry together.

    Both solvers use the same generated mesh lowered by the particle-layer depth.
    MPM's support is a kinematic rigid body; MJWarp's copy is static. Geometry is
    generated only during spawning, with no per-step height queries or recycling.
    """
    if not math.isfinite(cfg.wall_thickness) or cfg.wall_thickness <= 0.0:
        raise ValueError("wall_thickness must be positive and finite [m].")
    if not math.isfinite(cfg.wall_height) or cfg.wall_height < 0.0 or not math.isfinite(cfg.entrance_height):
        raise ValueError("wall_height must be non-negative and finite; entrance_height must be finite [m].")
    positions, velocities, masses, radii, surface_mesh = generate_terrain_particles(cfg)
    points_cfg = MPMPointsCfg(
        positions=positions,
        velocities=velocities,
        mass=masses,
        radius=radii,
        material=cfg.material,
        visible=cfg.visible,
        visual_color=cfg.visual_color,
        visual_material=cfg.visual_material,
        visual_update_frequency=cfg.visual_update_frequency,
    )
    root = spawn_mpm_particles(prim_path, points_cfg, translation=translation, orientation=orientation)
    cfg.spawn_path = prim_path
    support_mesh = surface_mesh.copy()
    support_mesh.apply_translation((0.0, 0.0, cfg.lower[2]))
    # Retain particles at the finite bed edges, down to the lowest support height.
    wall_bottom = float(support_mesh.bounds[0, 2]) - cfg.wall_thickness
    wall_top = float(surface_mesh.bounds[1, 2]) + cfg.wall_height
    # Planar ramps need a sloping rim, not a box as high as the far end everywhere.
    terrains = tuple(cfg.terrain_generator.sub_terrains.values())
    ramp = len(terrains) == 1 and isinstance(terrains[0], MeshXSlopeTerrainCfg)
    if ramp:
        # Ignore the generator's vertical border faces when identifying the ramp plane.
        top_faces = surface_mesh.faces[surface_mesh.face_normals[:, 2] > 0.0]
        ramp_vertices = surface_mesh.vertices[np.unique(top_faces)]
        slope, intercept = np.polyfit(ramp_vertices[:, 0], ramp_vertices[:, 2], 1)
    walls = []
    for axis in range(2):
        for side in range(2):
            top = cfg.entrance_height if axis == 0 and side == 0 else wall_top
            size = [cfg.upper[i] - cfg.lower[i] + 2 * cfg.wall_thickness for i in range(2)]
            center = [0.5 * (cfg.upper[i] + cfg.lower[i]) for i in range(2)]
            size[axis] = cfg.wall_thickness
            center[axis] = (
                cfg.lower[axis] - 0.5 * cfg.wall_thickness if side == 0 else cfg.upper[axis] + 0.5 * cfg.wall_thickness
            )
            if top <= wall_bottom:
                raise ValueError("The retaining wall top must be above the lowest support height.")
            wall = trimesh.creation.box(extents=(*size, top - wall_bottom))
            wall.apply_translation((*center, 0.5 * (top + wall_bottom)))
            if ramp and not (axis == 0 and side == 0):
                top_vertices = wall.vertices[:, 2] > wall_bottom + 0.5 * (top - wall_bottom)
                # Continue the plane through the wall thickness, keeping the rim linear in XZ.
                rim = slope * wall.vertices[top_vertices, 0] + intercept + cfg.wall_height
                if np.any(rim <= wall_bottom):
                    raise ValueError("The retaining wall top must be above the lowest support height.")
                wall.vertices[top_vertices, 2] = rim
            walls.append(wall)
    support_mesh = trimesh.util.concatenate([support_mesh, *walls])
    for name, rigid_props in (
        ("MPMBedFloor", sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)),
        ("RigidBedFloor", None),
    ):
        support_cfg = sim_utils.MeshFileCfg(
            mesh=sim_utils.MeshFileCfg.TrimeshObjectCfg(mesh=support_mesh),
            rigid_props=rigid_props,
            collision_props=[
                sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                NewtonCollisionCfg(contact_margin=cfg.support_contact_margin, contact_gap=0.0),
            ],
            mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"),
            physics_material=cfg.support_material,
            visible=name == "RigidBedFloor",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.37)),
        )
        support_cfg.func(f"{prim_path}/{name}", support_cfg)
    return root


@configclass
class TerrainMPMGridCfg(MPMGridCfg):
    """Fixed particle layer beneath a generated terrain surface, in the asset frame.

    Every environment clone receives the same mesh and initial particle positions.
    The generator's centered XY domain must cover the particle grid, including
    jitter. Sand depth is measured vertically, rather than along the surface normal.
    """

    func = spawn_terrain_mpm
    terrain_generator: TerrainGeneratorCfg = MISSING
    """Generator defining the initial sand surface and the support mesh shape."""
    support_contact_margin: float = 0.0
    """Contact margin of the generated support meshes [m]."""
    wall_thickness: float = 0.1
    """Thickness of retaining walls outside the particle XY bounds [m]."""
    wall_height: float = 0.1
    """Clearance above the local ramp surface, or the highest surface for other terrains [m]."""
    entrance_height: float = -0.05
    """Top of the negative-X retaining wall in the asset frame [m]."""
    support_material: RigidBodyMaterialBaseCfg = RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8)


# ----------------------------------------------------------------------------------------------
# Tilting sand testbed
# ----------------------------------------------------------------------------------------------


@sim_utils.clone
def spawn_tilting_testbed(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Author a fixed base, roll cradle, and pitch tray with five collision boxes."""
    stage = sim_utils.get_current_stage()
    root = sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    root.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    for name, mass, inertia in (
        ("base", 100.0, (100.0, 100.0, 100.0)),
        ("cradle", 100.0, (100.0, 100.0, 100.0)),
        ("tray", cfg.mass, (cfg.mass, 3.0 * cfg.mass, 3.0 * cfg.mass)),
    ):
        body = UsdGeom.Xform.Define(stage, f"{prim_path}/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mass_api = UsdPhysics.MassAPI.Apply(body)
        mass_api.CreateMassAttr(mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    fixed = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/base_fixed_joint")
    fixed.CreateBody1Rel().SetTargets([f"{prim_path}/base"])
    for name, parent, child, axis in (("roll", "base", "cradle", "X"), ("pitch", "cradle", "tray", "Y")):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/{name}")
        joint.CreateBody0Rel().SetTargets([f"{prim_path}/{parent}"])
        joint.CreateBody1Rel().SetTargets([f"{prim_path}/{child}"])
        joint.CreateAxisAttr(axis)
        joint.CreateLowerLimitAttr(-cfg.joint_limit_deg)
        joint.CreateUpperLimitAttr(cfg.joint_limit_deg)
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)

    length, width, depth = cfg.sand_bed_size
    wall = cfg.wall_thickness
    # The joint pivot is at the rigid supporting floor's upper surface.
    geometry = [
        ("floor", (length + 2 * wall, width + 2 * wall, cfg.floor_thickness), (0.0, 0.0, -cfg.floor_thickness / 2)),
        ("front", (wall, width + 2 * wall, depth), ((length + wall) / 2, 0.0, depth / 2)),
        ("back", (wall, width + 2 * wall, depth), (-(length + wall) / 2, 0.0, depth / 2)),
        ("left", (length, wall, depth), (0.0, (width + wall) / 2, depth / 2)),
        ("right", (length, wall, depth), (0.0, -(width + wall) / 2, depth / 2)),
    ]
    for name, size, position in geometry:
        box = sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.37), roughness=0.8),
        )
        box.func(f"{prim_path}/tray/{name}", box, translation=position)
    return root


@configclass
class TiltingTestbedSpawnerCfg(SpawnerCfg):
    """Geometry and mechanical limits for the powered tray."""

    func: Callable = spawn_tilting_testbed
    sand_bed_size: tuple[float, float, float] = MISSING
    """Interior sand-bed length, width, and depth [m], supplied by the scene config."""
    wall_thickness: float = MISSING
    """Tray wall thickness [m]."""
    floor_thickness: float = MISSING
    """Tray floor thickness [m]."""
    mass: float = 250.0
    """Tray mass [kg], excluding the simulated sand."""
    joint_limit_deg: float = 45.0
    """Symmetric mechanical joint limit [deg]."""


@configclass
class TestbedMotionCfg:
    """Smooth target motion; the joints, tray, and particles are integrated by physics."""

    roll_range: tuple[float, float] = (-math.radians(10), math.radians(10))
    pitch_range: tuple[float, float] = (-math.radians(10), math.radians(10))
    """Joint target ranges [rad], each containing the level orientation (zero)."""
    periods: tuple[float, float] = (12.0, 17.0)
    """Roll and pitch periods [s]."""
    settling_time: float = 2.0
    """Hold the tray level after reset [s]."""
    ramp_time: float = 3.0
    """Smoothly increase oscillation amplitude from zero [s]."""
    debug_vis: bool = True
    """Show the tray normal (blue) against the direction opposite gravity (green)."""
    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TestbedTilt",
        markers={
            "tray_normal": BLUE_ARROW_X_MARKER_CFG.markers["arrow"].replace(scale=(0.8, 0.12, 0.12)),
            "opposite_gravity": GREEN_ARROW_X_MARKER_CFG.markers["arrow"].replace(scale=(1.0, 0.08, 0.08)),
        },
    )

    def validate(self):
        for bounds in (self.roll_range, self.pitch_range):
            if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or not bounds[0] <= 0 <= bounds[1]:
                raise ValueError("Testbed angle ranges must be finite, ordered, and contain zero.")
        if len(self.periods) != 2 or any(not math.isfinite(value) or value <= 0 for value in self.periods):
            raise ValueError("Testbed periods must contain two positive finite durations.")
        if not math.isfinite(self.settling_time) or self.settling_time < 0:
            raise ValueError("Testbed settling_time must be finite and non-negative.")
        if not math.isfinite(self.ramp_time) or self.ramp_time <= 0:
            raise ValueError("Testbed ramp_time must be finite and positive.")
