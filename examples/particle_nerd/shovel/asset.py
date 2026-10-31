# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Preserve the reference shovel's geometry and inertia in a COM-centered asset."""

import argparse
import hashlib
import json
from pathlib import Path

import newton
import numpy as np
import warp as wp

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from particle_nerd.shovel.source_geometry import add_shovel

ASSET_DIR = Path(__file__).parent / "assets"


@wp.kernel
def union_distance(mesh: wp.uint64, origin: wp.vec3, spacing: float, values: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    p = origin + spacing * wp.vec3(float(i), float(j), float(k))
    q = wp.mesh_query_point_sign_parity(mesh, p, 5.0, 5, 0.1)
    if q.result:
        closest = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        values[i, j, k] = wp.min(values[i, j, k], q.sign * wp.length(p - closest))


def make_builder():
    builder = newton.ModelBuilder()
    add_shovel(builder)
    com = np.array(builder.body_com[0], dtype=np.float32)
    # Recenter all local geometry, preserving world geometry when the body pose
    # is shifted by the same rotated COM. Both solvers then use COM velocities.
    for i, t in enumerate(builder.shape_transform):
        builder.shape_transform[i] = wp.transform(
            wp.transform_get_translation(t) - wp.vec3(*com), wp.transform_get_rotation(t)
        )
    builder.body_com[0] = wp.vec3(0.0)
    builder.body_q[0] = wp.transform_identity()
    return builder, com


def build(output=ASSET_DIR, spacing=0.005):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    builder, original_com = make_builder()
    mass = float(builder.body_mass[0])
    inertia = np.asarray(builder.body_inertia[0], dtype=float).reshape(3, 3)
    shapes = []
    closed = []
    for i, kind in enumerate(builder.shape_type):
        t = builder.shape_transform[i]
        rotation = np.array(wp.quat_to_matrix(wp.transform_get_rotation(t))).reshape(3, 3)
        center = np.array(wp.transform_get_translation(t))
        visual = builder.shape_label[i] == "shovel_blade_visual"
        if kind == newton.GeoType.MESH:
            src = builder.shape_source[i]
            vertices = np.asarray(src.vertices) * np.array(builder.shape_scale[i])
            faces = np.asarray(src.indices).reshape(-1, 3)
        else:
            assert kind == newton.GeoType.CYLINDER
            radius, half = float(builder.shape_scale[i][0]), float(builder.shape_scale[i][1])
            n = 48
            theta = np.arange(n) * 2 * np.pi / n
            vertices = np.vstack(
                [
                    np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n, z)])
                    for z in [-half, half]
                ]
            )
            faces = []
            for a in range(n):
                b = (a + 1) % n
                faces.extend([(a, b, b + n), (a, b + n, a + n)])
            for a in range(1, n - 1):
                faces.extend([(0, a + 1, a), (n, n + a, n + a + 1)])
            faces = np.array(faces, dtype=np.int32)
        vertices = (vertices @ rotation.T + center).astype(np.float32)
        faces = faces.astype(np.int32)
        shapes.append(
            dict(
                name=builder.shape_label[i],
                vertices=vertices,
                faces=faces,
                visual=visual,
                color=np.array(builder.shape_color[i]).tolist(),
            )
        )
        if not visual:
            closed.append((vertices, faces))
    points = np.vstack([s["vertices"] for s in shapes])
    lower = points.min(0) - 0.15
    upper = points.max(0) + 0.15
    dims = tuple((np.ceil((upper - lower) / spacing).astype(int) + 1).tolist())
    wp.init()
    values = wp.full(dims, 5.0, dtype=float, device="cuda:0")
    meshes = []
    for vertices, faces in closed:
        mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device="cuda:0"),
            indices=wp.array(faces.ravel(), dtype=int, device="cuda:0"),
        )
        meshes.append(mesh)
        wp.launch(union_distance, dims, [mesh.id, wp.vec3(*lower), spacing, values], device="cuda:0")
    sdf = values.numpy()
    assert np.isfinite(sdf).all() and sdf.min() < 0
    np.savez_compressed(output / "shovel_sdf.npz", values=sdf, origin=lower, spacing=spacing)
    payload = {}
    for i, s in enumerate(shapes):
        payload[f"vertices_{i}"] = s["vertices"]
        payload[f"faces_{i}"] = s["faces"]
    np.savez_compressed(output / "shovel_meshes.npz", **payload)
    stage = Usd.Stage.CreateNew(str(output / "shovel.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Shovel")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mp = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mp.CreateMassAttr(mass)
    mp.CreateCenterOfMassAttr(Gf.Vec3f(0))
    eigen, basis = np.linalg.eigh(inertia)
    if np.linalg.det(basis) < 0:
        basis[:, 0] *= -1
    q = wp.quat_from_matrix(wp.mat33(*basis.flatten()))
    mp.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, eigen)))
    mp.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(float(q[0]), float(q[1]), float(q[2]))))
    for s in shapes:
        mesh = UsdGeom.Mesh.Define(stage, "/Shovel/" + s["name"])
        mesh.CreatePointsAttr(s["vertices"])
        mesh.CreateFaceVertexCountsAttr([3] * len(s["faces"]))
        mesh.CreateFaceVertexIndicesAttr(s["faces"].ravel())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*s["color"])])
        if not s["visual"]:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        if "collision_" in s["name"]:
            mesh.CreateVisibilityAttr("invisible")
    stage.GetRootLayer().Save()
    metadata = dict(
        geometry="shovel",
        body_mass_kg=mass,
        body_inertia=inertia.tolist(),
        body_com=[0, 0, 0],
        original_local_com=original_com.tolist(),
        body_size_m=np.ptp(points, axis=0).tolist(),
        blade_length_m=0.60,
        blade_width_m=0.32,
        blade_thickness_m=0.014,
        body_contact_friction=0.5,
        sdf_spacing_m=spacing,
        sdf_origin=lower.tolist(),
        sdf_shape=dims,
        shapes=[{k: v for k, v in s.items() if k not in ("vertices", "faces")} for s in shapes],
        source="newton/examples/mpm_moving_patch_beta/impl/shovel.py",
        source_sha256=hashlib.sha256(Path(__file__).with_name("source_geometry.py").read_bytes()).hexdigest(),
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "shapes"}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ASSET_DIR)
    p.add_argument("--spacing", type=float, default=0.005)
    a = p.parse_args()
    build(a.output, a.spacing)
