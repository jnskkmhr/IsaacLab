# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check particle force balance after fixing the estimated dynamic-box contacts.

The remaining unknowns are compressive central particle-pair forces and static
body contacts. This tests compatibility with that explicit internal-force
assumption; a small residual would not establish a unique granular stress state.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from body_contact_reconstruction import reconstruct_body_contact
from evaluate_body_contact import load
from reconstruction import reconstruct_reactions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a balance report.")
    experiment = json.loads(args.experiment.read_text())
    if not experiment["complete"]:
        raise ValueError("Incomplete experiment.")
    parameters = experiment["selected_parameters"]
    manifest, episodes = load(args.dataset)
    body = manifest["moving_object_id"]
    gravity = np.array(manifest["gravity"])
    rows = []
    for name, e in episodes.items():
        fixed = np.arange(len(e["body_mass"])) != body
        for t in range(0, len(e["reference"]), args.stride):
            position = e["particle_position"][t + int(parameters["geometry"] == "end")]
            contact = reconstruct_body_contact(
                position,
                e["particle_velocity"][t + 1],
                e["particle_mass"],
                e["body_pose"][t, body],
                e["body_velocity"][t, body],
                e["body_mass"][body],
                e["body_inertia"][body],
                e["body_size"][body],
                dt=e["dt"],
                gravity=gravity,
                contact_offset=parameters["contact_offset"],
                effective_mass_scale=parameters["effective_mass_scale"],
                extrapolate_velocity=parameters.get("affine", False),
                stabilization=parameters.get("stabilization", 0),
                friction=float(e["body_friction"][body]) if parameters.get("frictional", False) else None,
            )
            known_force = contact.particle_impulse / e["dt"]
            weight_scale = np.linalg.norm(e["particle_mass"][:, None] * gravity)
            for regularization in (1e-4, 1e-8):
                result = reconstruct_reactions(
                    position,
                    e["particle_velocity"][t],
                    e["particle_velocity"][t + 1],
                    e["particle_mass"],
                    e["body_pose"][t, fixed],
                    e["body_size"][fixed],
                    e["body_friction"][fixed],
                    dt=e["dt"],
                    gravity=gravity,
                    regularization=regularization,
                    optimizer="osqp",
                    max_iterations=10000,
                    known_particle_force=known_force,
                )
                # Particle-pair forces cancel exactly in the summed equation.
                required_external = np.sum(
                    e["particle_mass"][:, None]
                    * ((e["particle_velocity"][t + 1] - e["particle_velocity"][t]) / e["dt"] - gravity),
                    axis=0,
                )
                estimated_external = known_force.sum(axis=0) - result.body_wrench[:, :3].sum(axis=0)
                rows.append(
                    {
                        "episode": name,
                        "step": t,
                        "regularization": regularization,
                        "particle_force_vector_rmse_N": float(
                            np.sqrt(np.mean(np.sum(result.particle_residual**2, axis=1)))
                        ),
                        "particle_residual_relative_to_weight": float(
                            np.linalg.norm(result.particle_residual) / weight_scale
                        ),
                        "aggregate_particle_balance_error_N": float(
                            np.linalg.norm(estimated_external - required_external)
                        ),
                        "static_body_force_vector_rmse_N": float(
                            np.sqrt(
                                np.mean(np.sum((result.body_wrench[:, :3] - e["reference"][t, fixed, :3]) ** 2, axis=1))
                            )
                        ),
                        "dynamic_box_action_reaction_error_N": float(
                            np.linalg.norm(known_force.sum(axis=0) + contact.body_wrench[:3])
                        ),
                        "converged": result.converged,
                        "seconds": result.solve_seconds,
                    }
                )
            print(json.dumps(rows[-1]), flush=True)
            args.output.write_text(json.dumps({"complete": False, "records": rows}, indent=2) + "\n")
    summary = {}
    for regularization in (1e-4, 1e-8):
        selected = [r for r in rows if r["regularization"] == regularization]
        summary[str(regularization)] = {
            "samples": len(selected),
            "solver_failures": sum(not r["converged"] for r in selected),
            **{
                key: float(np.mean([r[key] for r in selected]))
                for key in (
                    "particle_force_vector_rmse_N",
                    "particle_residual_relative_to_weight",
                    "aggregate_particle_balance_error_N",
                    "static_body_force_vector_rmse_N",
                    "dynamic_box_action_reaction_error_N",
                    "seconds",
                )
            },
        }
    args.output.write_text(
        json.dumps(
            {
                "complete": True,
                "parameters": parameters,
                "dataset": str(args.dataset),
                "internal_force_assumption": "compressive central particle-pair forces within 0.065 m",
                "static_contact_assumption": "0.04 m shell; four-ray Coulomb cone",
                "summary": summary,
                "records": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
