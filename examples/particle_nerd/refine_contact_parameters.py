# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Refine four scalar contact parameters using fitting episodes only.

This calibrates an explicit mechanical approximation; it does not train a force
network. Validation episodes are evaluated after optimization. No test file is
accepted by this script.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from evaluate_body_contact import evaluate, load
from scipy.optimize import minimize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("initial_experiment", type=Path)
    parser.add_argument("--max-evaluations", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an optimization report.")
    initial = json.loads(args.initial_experiment.read_text())["selected_parameters"]
    manifest, episodes = load(args.calibration)
    names = sorted(episodes)
    fit_names, validation_names = names[:4], names[4:]
    report = {
        "complete": False,
        "fitting_episodes": fit_names,
        "validation_episodes": validation_names,
        "optimizer": "bounded Nelder-Mead; finite evaluation budget, no global-optimality claim",
        "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        "initial_experiment_sha256": hashlib.sha256(args.initial_experiment.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "body_contact_reconstruction.py",
                "contact_iterations.py",
                "evaluate_body_contact.py",
                Path(__file__).name,
            )
        },
        "trials": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def decode(x):
        return {
            **initial,
            "effective_mass_scale": float(np.exp(x[0])),
            "stabilization": float(np.exp(x[1])),
            "history_weight": float(x[2]),
            "contact_offset": float(x[3] / 1000),
        }

    def objective(x):
        parameters = decode(x)
        metrics, _ = evaluate(
            episodes, fit_names, parameters, body=manifest["moving_object_id"], gravity=manifest["gravity"], dense=True
        )
        report["trials"].append({"parameters": parameters, **metrics})
        save()
        print(json.dumps({"trial": len(report["trials"]), **report["trials"][-1]}), flush=True)
        return metrics["box_force_vector_rmse_N"] + 1000 * metrics["solver_failures"]

    x0 = np.array(
        [
            np.log(initial["effective_mass_scale"]),
            np.log(initial["stabilization"]),
            initial["history_weight"],
            1000 * initial["contact_offset"],
        ]
    )
    simplex = np.tile(x0, (5, 1))
    for i, step in enumerate([0.15, 0.2, -0.08, 1.0]):
        simplex[i + 1, i] += step
    fit = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        bounds=[(np.log(0.3), np.log(30)), (np.log(0.001), np.log(0.3)), (0, 0.98), (0, 5)],
        options={
            "maxfev": args.max_evaluations,
            "adaptive": True,
            "initial_simplex": simplex,
            "xatol": 0.01,
            "fatol": 0.001,
        },
    )
    candidates = [r for r in report["trials"] if not r["solver_failures"]]
    if not candidates:
        raise RuntimeError("No candidate contact solves converged.")
    chosen = min(candidates, key=lambda r: r["box_force_vector_rmse_N"])
    report["selected_parameters"] = chosen["parameters"]
    report["fitting_metrics"] = chosen
    report["optimization_converged"] = bool(fit.success)
    report["optimization_status"] = str(fit.message)
    report["validation_metrics"], report["validation_records"] = evaluate(
        episodes,
        validation_names,
        chosen["parameters"],
        body=manifest["moving_object_id"],
        gravity=manifest["gravity"],
        dense=True,
    )
    report["complete"] = True
    save()
    print(
        json.dumps(
            {"selected": chosen, "validation": report["validation_metrics"], "optimization_status": str(fit.message)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
