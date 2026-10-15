# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrate geometric/contact approximations, then score untouched dynamic episodes.

The test file is opened only after parameter selection. Material motion comes from
MPM recordings, standing in for future NeRD motion; this is not learned rollout.
Only the dynamic body's force is estimated. Static-body reactions are unresolved.
"""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path

import h5py
import numpy as np
from body_contact_reconstruction import reconstruct_body_contact

STEPS = [0, 10, 20, 30, 35, 40, 45, 50, 55, 60, 70, 90, 130, 199, 299, 399]


def load(path):
    episodes = {}
    with h5py.File(path) as f:
        manifest = json.loads(f.attrs["particle_nerd_manifest"])
        if not manifest["complete"] or manifest["schema_version"] < 2:
            raise ValueError("Complete dynamic episodes are required.")
        for name, e in f["data"].items():
            episodes[name] = {
                k: e["states/" + k][:].astype(float)
                for k in ("particle_position", "particle_velocity", "body_pose", "body_velocity")
            }
            episodes[name].update(
                {
                    k: e["constants/" + k][0].astype(float)
                    for k in ("particle_mass", "body_mass", "body_inertia", "body_size", "body_friction")
                }
            )
            episodes[name]["dt"] = float(e["constants/dt"][0])
            episodes[name]["reference"] = e["reference/body_impulse"][:].astype(float) / episodes[name]["dt"]
    return manifest, episodes


def evaluate(episodes, names, parameters, *, body, gravity, dense=False):
    rows = []
    for name in names:
        e = episodes[name]
        previous_impulse = np.zeros_like(e["particle_position"][0])
        history_weight = parameters.get("history_weight", 0.0)
        steps = range(len(e["reference"])) if dense or history_weight else STEPS
        for t in steps:
            result = reconstruct_body_contact(
                e["particle_position"][t + int(parameters["geometry"] == "end")],
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
                previous_particle_impulse=previous_impulse,
                history_weight=history_weight,
                friction=float(e["body_friction"][body]) if parameters.get("frictional", False) else None,
                max_iterations=2000 if parameters.get("frictional", False) else 10000,
            )
            previous_impulse = result.particle_impulse
            if not dense and t not in STEPS:
                continue
            rows.append(
                {
                    "episode": name,
                    "step": t,
                    "estimate": result.body_wrench.tolist(),
                    "reference": e["reference"][t, body].tolist(),
                    "converged": result.converged,
                    "seconds": result.solve_seconds,
                    "contacts": result.contacts,
                }
            )
    estimated = np.array([r["estimate"] for r in rows])
    reference = np.array([r["reference"] for r in rows])
    error = estimated - reference
    return {
        "samples": len(rows),
        "box_force_vector_rmse_N": float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
        "box_force_mse_N2": float(np.mean(error[:, :3] ** 2)),
        "box_torque_vector_rmse_Nm": float(np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1)))),
        "relative_force_L2": float(np.linalg.norm(error[:, :3]) / max(np.linalg.norm(reference[:, :3]), 1e-12)),
        "median_seconds": float(np.median([r["seconds"] for r in rows])),
        "solver_failures": sum(not r["converged"] for r in rows),
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("--frictional", action="store_true")
    parser.add_argument("--affine", action="store_true")
    parser.add_argument("--dense-fitting", action="store_true")
    parser.add_argument("--offsets", default="0,.005,.01,.015,.02,.03,.04")
    parser.add_argument("--mass-scales", default="1,3,10,30,100,1000")
    parser.add_argument("--stabilization-grid", default="0")
    parser.add_argument("--history-grid", default="0")
    parser.add_argument("--geometry", choices=("both", "start", "end"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing experiment.")
    manifest, episodes = load(args.calibration)
    names = sorted(episodes)
    fit_names = names[:4]
    validation_names = names[4:]
    report = {
        "complete": False,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "osqp", "numba")},
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("body_contact_reconstruction.py", "contact_iterations.py", "evaluate_body_contact.py")
        },
        "calibration_file_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        "fitting_episodes": fit_names,
        "validation_episodes": validation_names,
        "fixed_fitting_steps": "all transitions" if args.dense_fitting else STEPS,
        "input_contract": (
            "current rigid pose/velocity + predicted next material velocity; no future rigid state or reference force"
        ),
        "limitations": [
            "Material motion is the recorded MPM output, not NeRD predictions.",
            "Effective contact response, not a granular constitutive model.",
            "Only dynamic-body reactions are estimated; static-body reactions remain unresolved.",
            "Fitting uses force labels for effective scalar contact parameters; inference does not.",
        ],
        "trials": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    grids = [list(map(float, value.split(","))) for value in (args.offsets, args.mass_scales, args.stabilization_grid)]
    geometries = ["start", "end"] if args.geometry == "both" else [args.geometry]
    history_weights = list(map(float, args.history_grid.split(",")))
    for offset, scale, stabilization, geometry, history in itertools.product(*grids, geometries, history_weights):
        parameters = {
            "contact_offset": offset,
            "effective_mass_scale": scale,
            "geometry": geometry,
            "frictional": args.frictional,
            "affine": args.affine,
            "stabilization": stabilization,
            "history_weight": history,
        }
        scores, _ = evaluate(
            episodes,
            fit_names,
            parameters,
            body=manifest["moving_object_id"],
            gravity=manifest["gravity"],
            dense=args.dense_fitting,
        )
        report["trials"].append({"parameters": parameters, **scores})
        save()
        print(json.dumps(report["trials"][-1]), flush=True)
    candidates = [r for r in report["trials"] if r["solver_failures"] == 0]
    if not candidates:
        raise RuntimeError("No candidate converged on all fitting frames.")
    chosen = min(candidates, key=lambda r: r["box_force_vector_rmse_N"])
    report["selected_parameters"] = chosen["parameters"]
    report["fitting_metrics"] = chosen
    # No selection uses validation or test labels.
    report["validation_metrics"], report["validation_records"] = evaluate(
        episodes,
        validation_names,
        chosen["parameters"],
        body=manifest["moving_object_id"],
        gravity=manifest["gravity"],
        dense=True,
    )
    save()
    test_manifest, test_episodes = load(args.test)
    report["test_file_sha256"] = hashlib.sha256(args.test.read_bytes()).hexdigest()
    report["test_metrics"], report["test_records"] = evaluate(
        test_episodes,
        sorted(test_episodes),
        chosen["parameters"],
        body=test_manifest["moving_object_id"],
        gravity=test_manifest["gravity"],
        dense=True,
    )
    report["complete"] = True
    save()
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("trials", "validation_records", "test_records")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
