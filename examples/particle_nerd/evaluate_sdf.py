# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare analytic and sampled rigid SDFs using reference particle/body states.

This isolates geometry approximation from learned-rollout error. Reference MPM
forces are scoring targets only; no MPM solve is called during reconstruction.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from body_contact_reconstruction import reconstruct_body_contact
from geometry import SampledSDF, body_contacts
from learning_data import EpisodeStore
from scipy.spatial import ConvexHull


def triangulated_primitive(shape, size):
    """Known closed geometries for a mesh-to-SDF accuracy control."""
    half = np.asarray(size) / 2
    angle = np.arange(64) * 2 * np.pi / 64
    if shape == "box":
        vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    elif shape == "cylinder":
        ring = np.c_[np.cos(angle), np.sin(angle)] * half[0]
        vertices = np.concatenate([np.c_[ring, np.full(len(ring), z)] for z in (-half[2], half[2])])
    elif shape == "sphere":
        latitude = np.arange(1, 32) * np.pi / 32
        vertices = np.array(
            [[np.sin(p) * np.cos(a), np.sin(p) * np.sin(a), np.cos(p)] for p in latitude for a in angle]
        )
        vertices = np.concatenate((vertices, [[0, 0, -1], [0, 0, 1]])) * half[0]
    else:
        raise ValueError(shape)
    return vertices, ConvexHull(vertices).simplices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mesh", action="store_true", help="Sample triangulated shapes with Warp instead of analytic SDF formulas."
    )
    args = parser.parse_args()
    store = EpisodeStore(args.data, "validation")
    rows = []
    for i in range(0, len(store), 2):
        e = store[i]
        body = e["moving_body"]
        shape = e["body_shapes"][body]
        if args.mesh:
            vertices, triangles = triangulated_primitive(shape, e["body_size"][body])
            grids = {
                spacing: SampledSDF.from_mesh(vertices, triangles, spacing, padding=0.4, device="cuda:0")
                for spacing in (0.01, 0.005)
            }
        else:
            grids = {
                spacing: SampledSDF.from_primitive(shape, e["body_size"][body], spacing, padding=0.4)
                for spacing in (0.01, 0.005)
            }
        for t in range(0, len(e["particle_position"]) - 1, 10):
            x = e["particle_position"][t]
            pose = e["body_pose"][t, body]
            query = body_contacts(x, pose, e["body_size"][body], shape)
            forces = {}
            for spacing, sdf in {0.0: None, **grids}.items():
                result = reconstruct_body_contact(
                    x,
                    e["particle_velocity"][t + 1],
                    e["particle_mass"],
                    pose,
                    e["body_velocity"][t, body],
                    e["body_mass"][body],
                    e["body_inertia"][body],
                    e["body_size"][body],
                    dt=e["dt"],
                    gravity=e["gravity"],
                    body_shape=shape,
                    body_sdf=sdf,
                    contact_offset=0,
                    effective_mass_scale=10,
                    friction=float(e["body_friction"][body]),
                    extrapolate_velocity=True,
                    velocity_gradient_device=None,
                    stabilization=0.01,
                    max_iterations=2000,
                )
                forces[spacing] = result.body_wrench
                distance, normal, _ = body_contacts(x, pose, e["body_size"][body], shape, sdf)
                near = abs(query[0]) < 0.03
                rows.append(
                    dict(
                        episode=e["name"],
                        shape=shape,
                        step=t,
                        sdf_spacing_m=spacing,
                        distance_max_error_m=float(np.max(abs(distance - query[0]))),
                        near_surface_normal_rmse=float(
                            np.sqrt(np.mean(np.sum((normal[near] - query[1][near]) ** 2, axis=-1)))
                        )
                        if near.any()
                        else None,
                        force_error_N=(result.body_wrench[:3] - e["force_reference"][t, body, :3]).tolist(),
                        force_difference_from_analytic_N=(result.body_wrench[:3] - forces[0.0][:3]).tolist(),
                        converged=result.converged,
                    )
                )
    summary = []
    for shape in ("box", "sphere", "cylinder"):
        for spacing in (0.0, 0.01, 0.005):
            selected = [r for r in rows if r["shape"] == shape and r["sdf_spacing_m"] == spacing]
            summary.append(
                dict(
                    shape=shape,
                    sdf_spacing_m=spacing,
                    samples=len(selected),
                    force_vector_rmse_N=float(
                        np.sqrt(np.mean(np.sum(np.array([r["force_error_N"] for r in selected]) ** 2, axis=-1)))
                    ),
                    force_difference_from_analytic_rmse_N=float(
                        np.sqrt(
                            np.mean(
                                np.sum(
                                    np.array([r["force_difference_from_analytic_N"] for r in selected]) ** 2, axis=-1
                                )
                            )
                        )
                    ),
                    distance_max_error_m=max(r["distance_max_error_m"] for r in selected),
                    solver_failures=sum(not r["converged"] for r in selected),
                )
            )
    output = dict(
        evaluation="reference states; geometry approximation only",
        sampled_from="triangle_mesh" if args.mesh else "analytic_formula",
        summary=summary,
        samples=rows,
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
