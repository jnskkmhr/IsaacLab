# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a deadline-limited scalar contact study with independent final evaluation.

The study varies four parameters of the analytical contact approximation. It
compares mean force error with error at reference impact peaks. No force network
is trained. MPM recordings are read offline; MPM never runs in reconstruction.
Optional publication writes aggregate progress only to an existing slide repo.
"""

import argparse
import hashlib
import html
import json
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
from dataset import validate_dataset
from evaluate_body_contact import evaluate, load
from replay_body_motion import replay
from scipy.optimize import minimize
from score_frozen_contacts import summarize_episodes


def now():
    return datetime.now(timezone.utc)


class SearchDeadline(Exception):
    pass


class HardDeadline(Exception):
    pass


def impact_summary(episodes):
    values = list(episodes.values())
    error = np.array([e["estimated_force_at_reference_peak_N"] - e["reference_peak_upward_force_N"] for e in values])
    max_error = np.array([e["estimated_peak_upward_force_N"] - e["reference_peak_upward_force_N"] for e in values])
    return {
        "peak_at_reference_time_rmse_N": float(np.sqrt(np.mean(error**2))),
        "peak_at_reference_time_bias_N": float(np.mean(error)),
        "maximum_force_error_rmse_N": float(np.sqrt(np.mean(max_error**2))),
        "peak_timing_rmse_s": float(np.sqrt(np.mean([e["estimated_peak_time_error_s"] ** 2 for e in values]))),
    }


def score(episodes, manifest, parameters):
    metrics, rows = evaluate(
        episodes,
        sorted(episodes),
        parameters,
        body=manifest["moving_object_id"],
        gravity=manifest["gravity"],
        dense=True,
    )
    by_episode, confidence = summarize_episodes(rows, manifest["dt"])
    return {"metrics": metrics, "impact": impact_summary(by_episode), "confidence": confidence, "episodes": by_episode}


def score_replay(episodes, manifest, parameters, excluded, limit=None):
    names = sorted(episodes)
    if limit is not None:
        names = names[:limit]
    results = {}
    skipped = []
    for name in names:
        if name in excluded:
            skipped.append(name)
            continue
        result = replay(
            episodes[name], parameters, manifest["moving_object_id"], manifest["gravity"], reference=parameters is None
        )
        results[name] = {k: v for k, v in result.items() if k not in ("poses", "velocities", "forces")}
    if not results:
        return {"episodes": 0, "skipped_reference_rigid_contacts": skipped, "valid": False}
    return {
        "episodes": len(results),
        "skipped_reference_rigid_contacts": skipped,
        "valid": all(
            e["completed"] and e["valid_without_rigid_contact"] and not e["solver_failures"] for e in results.values()
        ),
        "invalid_episodes": sum(
            not e["completed"] or not e["valid_without_rigid_contact"] or e["solver_failures"] > 0
            for e in results.values()
        ),
        **{
            key: float(np.mean([e[key] for e in results.values()]))
            for key in (
                "position_rmse_m",
                "orientation_rmse_deg",
                "final_position_error_m",
                "final_orientation_error_deg",
            )
        },
        "episode_results": results,
    }


def direct_contact_episodes(path, body):
    with h5py.File(path) as f:
        return {
            name
            for name, e in f["data"].items()
            if np.any(np.linalg.norm(e["reference/rigid_contact_impulse"][:, body, :3], axis=1) > 1e-8)
        }


def public_model(name, result, split):
    data = {
        "name": name,
        "split": split,
        "force_vector_rmse_N": result["metrics"]["box_force_vector_rmse_N"],
        "force_component_mse_N2": result["metrics"]["box_force_mse_N2"],
        "torque_vector_rmse_Nm": result["metrics"]["box_torque_vector_rmse_Nm"],
        "peak_force_rmse_N": result["impact"]["peak_at_reference_time_rmse_N"],
        "samples": result["metrics"]["samples"],
        "solver_failures": result["metrics"]["solver_failures"],
    }
    if "replay" in result:
        data["replay"] = {
            k: v
            for k, v in result["replay"].items()
            if k not in ("episode_results", "skipped_reference_rigid_contacts")
        }
        data["replay"]["skipped_reference_rigid_contacts"] = len(result["replay"]["skipped_reference_rigid_contacts"])
    return data


def update_slide(site, models, state):
    """Update a pre-reviewed results panel and its PDF from aggregate measurements."""
    path = site / "index.html"
    text = path.read_text()
    begin = "<!-- OVERNIGHT_RESULTS_START -->"
    end = "<!-- OVERNIGHT_RESULTS_END -->"
    if begin not in text or end not in text:
        raise RuntimeError("The slide deck is missing the overnight results panel.")

    def number(value, scale=1):
        return "—" if value is None else f"{value * scale:.2f}"

    rows = []
    for model in models:
        motion = model.get("replay", {})
        name = model["name"]
        labels = {"baseline": "No impulse history", "initial_history": "Initial history model"}
        label = labels.get(name, name.replace("search_", "Candidate "))
        if name == state.get("best_mean"):
            label += " · lowest mean error"
        if name == state.get("best_peak"):
            label += " · lowest peak error"
        rows.append(
            "<tr><td>"
            + html.escape(label)
            + "</td><td>"
            + number(model["force_vector_rmse_N"])
            + "</td><td>"
            + number(model["peak_force_rmse_N"])
            + "</td><td>"
            + (number(motion.get("position_rmse_m"), 100) if motion.get("valid") else "Invalid")
            + "</td><td>"
            + (number(motion.get("orientation_rmse_deg")) if motion.get("valid") else "Invalid")
            + "</td></tr>"
        )
    split = models[0]["split"] if models else "Awaiting evaluation"
    count = state["test_episodes"] if state.get("test_results") else state["validation_episodes"]
    content = (
        '<p class="dem-intro">'
        + html.escape(split)
        + f": {count} episodes, 400 transitions each. "
        + "Parameters are selected using separate fitting episodes.<br>"
        + "Peak error is measured at the time of the reference upward-force peak.</p>"
        + '<table class="dem-table"><thead><tr><th>Model</th><th>Force RMSE [N]</th><th>Peak RMSE [N]</th>'
        + "<th>Position RMSE [cm]</th><th>Orientation RMSE [°]</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + '<div class="result-note">Force estimates use predicted material motion and contact-impulse history. '
        + "The measurements here use recorded particles; they do not establish a coupled NeRD rollout.</div>"
    )
    first = text.index(begin) + len(begin)
    last = text.index(end, first)
    path.write_text(text[:first] + "\n" + content + "\n" + text[last:])
    with tempfile.TemporaryDirectory(prefix="particle-nerd-study-print-") as profile:
        subprocess.run(
            [
                "/usr/bin/google-chrome",
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--user-data-dir=" + profile,
                "--no-pdf-header-footer",
                "--virtual-time-budget=2000",
                "--print-to-pdf=" + str(site / "particle_nerd_results.pdf"),
                path.as_uri(),
            ],
            check=True,
            timeout=60,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if not (site / "particle_nerd_results.pdf").read_bytes().startswith(b"%PDF"):
        raise RuntimeError("PDF rendering failed.")


def publish(state, site, force=False):
    """Publish only an allowlisted aggregate object; recordings stay local."""
    if site is None:
        return
    if not force and time.monotonic() - publish.last_time < 600:
        return
    publish.last_time = time.monotonic()
    try:
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=site, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=site, text=True).strip()
        if Path(top).resolve() != site.resolve() or branch != "main":
            raise RuntimeError("Publication requires the configured slide repository on main.")
        progress = json.loads((site / "progress.json").read_text())
        models = []
        if state.get("test_results"):
            models = [public_model(name, result, "Independent test") for name, result in state["test_results"].items()]
        else:
            selected = []
            for key in ("baseline", "initial_history", state.get("best_mean"), state.get("best_peak")):
                if key and key in state["candidates"] and key not in selected:
                    selected.append(key)
            models = [public_model(name, state["candidates"][name]["validation"], "Validation") for name in selected]
        previous_models = progress.get("overnight_study", {}).get("models", [])
        changed_files = ["progress.json"]
        if models and models != previous_models:
            update_slide(site, models, state)
            changed_files.extend(["index.html", "particle_nerd_results.pdf"])
        progress["updated_at"] = now().isoformat()
        progress["status"] = "Study complete" if state.get("complete") else "In progress"
        if state.get("error"):
            progress["status"] = "Study stopped; see status"
        progress["current_task"] = state["phase"]
        progress["overnight_study"] = {
            "started_at": state["started_at"],
            "deadline": state["deadline"],
            "phase": state["phase"],
            "calibration_episodes": state["calibration_episodes"],
            "validation_episodes": state["validation_episodes"],
            "test_episodes": state["test_episodes"],
            "trials": len(state["trials"]),
            "completed_searches": len(state["searches"]),
            "models": models,
            "peak_definition": "Upward particle-on-box force error at the time of each reference impact peak.",
            "limitation": (
                "Reference particle motion; replay holds particle trajectories fixed. "
                "No learned or fully coupled NeRD rollout."
            ),
            "error": state.get("error"),
        }
        (site / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        subprocess.run(["git", "add", "--", *changed_files], cwd=site, check=True, timeout=20)
        changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *changed_files], cwd=site).returncode
        if changed:
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"Update overnight contact study: {len(state['trials'])} trials",
                    "--",
                    *changed_files,
                ],
                cwd=site,
                check=True,
                timeout=30,
            )
        subprocess.run(["git", "push", "origin", "main"], cwd=site, check=True, timeout=60)
        state["last_publication"] = now().isoformat()
    except HardDeadline:
        raise
    except Exception as error:
        state["publication_error"] = str(error)
        print(f"Publication failed; study continues locally: {error}", flush=True)


publish.last_time = -float("inf")


def choose_models(state):
    baseline = state["candidates"]["baseline"]["validation"]
    candidates = []
    for name, candidate in state["candidates"].items():
        result = candidate["validation"]
        motion = result["replay"]
        safe = (
            not result["metrics"]["solver_failures"]
            and motion.get("valid", False)
            and motion["position_rmse_m"] <= baseline["replay"]["position_rmse_m"]
            and motion["orientation_rmse_deg"] <= baseline["replay"]["orientation_rmse_deg"]
        )
        if safe:
            candidates.append(name)
    if not candidates:
        state["selection_status"] = "No candidate has passed validation contact and replay checks yet."
        state["best_mean"] = None
        state["best_peak"] = None
        return False
    state["best_mean"] = min(
        candidates, key=lambda k: state["candidates"][k]["validation"]["metrics"]["box_force_vector_rmse_N"]
    )
    minimum = baseline["metrics"]["box_force_vector_rmse_N"]
    peak_candidates = [
        k for k in candidates if state["candidates"][k]["validation"]["metrics"]["box_force_vector_rmse_N"] <= minimum
    ]
    state["best_peak"] = (
        min(
            peak_candidates,
            key=lambda k: state["candidates"][k]["validation"]["impact"]["peak_at_reference_time_rmse_N"],
        )
        if peak_candidates
        else None
    )
    state["selection_status"] = "Validation candidates selected; independent test remains separate."
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--initial-experiment", type=Path, required=True)
    parser.add_argument("--baseline-experiment", type=Path, required=True)
    parser.add_argument("--deadline", required=True, help="ISO timestamp with UTC offset.")
    parser.add_argument("--max-searches", type=int, default=12)
    parser.add_argument("--max-evaluations", type=int, default=80)
    parser.add_argument("--publish-site", type=Path)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline).astimezone(timezone.utc)
    search_until = deadline - timedelta(minutes=45)
    if search_until <= now():
        parser.error("Insufficient time for fitting and independent evaluation.")
    output = args.data_dir / "study.json"
    if output.exists():
        parser.error("Refusing to overwrite a study.")
    state = {
        "complete": False,
        "started_at": now().isoformat(),
        "deadline": deadline.isoformat(),
        "phase": "Validating new reference recordings.",
        "trials": [],
        "searches": [],
        "candidates": {},
        "calibration_episodes": 0,
        "validation_episodes": 0,
        "test_episodes": 0,
        "selection_rule": (
            "Minimum validation mean error, plus minimum peak error among models whose mean force error "
            "does not exceed the memoryless baseline. Require no failed contact solves and valid replay; "
            "mean position and orientation errors must not exceed that baseline."
        ),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")
        },
        "process_id": os.getpid(),
    }

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output)

    def hard_stop(signum, frame):
        raise HardDeadline("Requested study deadline reached.")

    signal.signal(signal.SIGALRM, hard_stop)
    signal.signal(signal.SIGTERM, hard_stop)
    signal.alarm(max(1, int((deadline - now()).total_seconds())))
    try:
        for split in ("calibration", "validation", "test"):
            path = args.data_dir / (split + ".hdf5")
            check = validate_dataset(path)
            if check["box_relative_force_balance_residual"] > 1e-5 or check["relative_linear_momentum_residual"] > 1e-3:
                raise RuntimeError(f"{split}: reference momentum check failed: {check}")
            state[split + "_episodes"] = check["episodes"]
            state[split + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            state[split + "_reference_checks"] = check
        # Test trajectories are not opened for estimator scoring until selection finishes.
        calibration_manifest, calibration = load(args.data_dir / "calibration.hdf5")
        validation_manifest, validation = load(args.data_dir / "validation.hdf5")
        excluded = direct_contact_episodes(args.data_dir / "validation.hdf5", validation_manifest["moving_object_id"])
        initial = json.loads(args.initial_experiment.read_text())["selected_parameters"]
        baseline = json.loads(args.baseline_experiment.read_text())["selected_parameters"]
        for name, parameters in [("baseline", baseline), ("initial_history", initial)]:
            state["phase"] = f"Evaluating {name.replace('_', ' ')} on new validation episodes."
            save()
            publish(state, args.publish_site, True)
            result = score(validation, validation_manifest, parameters)
            result["replay"] = score_replay(validation, validation_manifest, parameters, excluded, limit=8)
            state["candidates"][name] = {"parameters": parameters, "validation": result}
            save()
        choose_models(state)
        publish(state, args.publish_site, True)
        rng = np.random.default_rng(41827)
        for search_index in range(args.max_searches):
            if now() >= search_until:
                break
            peak_weight = (0.0, 0.03, 0.1)[search_index % 3]
            start = (
                initial
                if search_index < 3
                else {
                    **initial,
                    "effective_mass_scale": float(np.exp(rng.uniform(np.log(1), np.log(15)))),
                    "stabilization": float(np.exp(rng.uniform(np.log(0.005), np.log(0.1)))),
                    "history_weight": float(rng.uniform(0.1, 0.9)),
                    "contact_offset": 0.0,
                }
            )
            name = f"search_{search_index + 1}"
            trial_start = len(state["trials"])
            state["phase"] = (
                f"Calibrating contact response: search {search_index + 1}, peak-error weight {peak_weight:g}."
            )
            save()
            publish(state, args.publish_site, True)

            def decode(x):
                return {
                    **initial,
                    "effective_mass_scale": float(np.exp(x[0])),
                    "stabilization": float(np.exp(x[1])),
                    "history_weight": float(x[2]),
                    "contact_offset": float(x[3] / 1000),
                }

            def objective(x):
                if now() >= search_until:
                    raise SearchDeadline()
                parameters = decode(x)
                result = score(calibration, calibration_manifest, parameters)
                loss = (
                    result["metrics"]["box_force_vector_rmse_N"] ** 2
                    + peak_weight * result["impact"]["peak_at_reference_time_rmse_N"] ** 2
                )
                if result["metrics"]["solver_failures"]:
                    loss += 1e6 * result["metrics"]["solver_failures"]
                trial = {
                    "search": name,
                    "peak_weight": peak_weight,
                    "parameters": parameters,
                    "loss": float(loss),
                    "metrics": result["metrics"],
                    "impact": result["impact"],
                    "finished_at": now().isoformat(),
                }
                state["trials"].append(trial)
                save()
                publish(state, args.publish_site)
                print(json.dumps({"trial": len(state["trials"]), **trial}), flush=True)
                return loss

            x0 = np.array(
                [
                    np.log(start["effective_mass_scale"]),
                    np.log(start["stabilization"]),
                    start["history_weight"],
                    1000 * start["contact_offset"],
                ]
            )
            simplex = np.tile(x0, (5, 1))
            for i, step in enumerate([0.2, 0.3, -0.08, 1.0]):
                simplex[i + 1, i] += step
            status = "Search deadline reached."
            try:
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
                status = str(fit.message)
            except SearchDeadline:
                pass
            trials = [t for t in state["trials"][trial_start:] if not t["metrics"]["solver_failures"]]
            if not trials:
                break
            best = min(trials, key=lambda t: t["loss"])
            state["phase"] = f"Validating completed contact search {search_index + 1}."
            save()
            publish(state, args.publish_site, True)
            result = score(validation, validation_manifest, best["parameters"])
            result["replay"] = score_replay(validation, validation_manifest, best["parameters"], excluded, limit=8)
            state["candidates"][name] = {"parameters": best["parameters"], "validation": result, "fitting": best}
            state["searches"].append(
                {
                    "name": name,
                    "peak_weight": peak_weight,
                    "trials": len(state["trials"]) - trial_start,
                    "status": status,
                }
            )
            choose_models(state)
            save()
            publish(state, args.publish_site, True)
        state["phase"] = "Evaluating frozen candidates on the independent test episodes."
        save()
        publish(state, args.publish_site, True)
        test_manifest, test = load(args.data_dir / "test.hdf5")
        excluded = direct_contact_episodes(args.data_dir / "test.hdf5", test_manifest["moving_object_id"])
        state["test_results"] = {}
        selected = list(
            dict.fromkeys(
                k for k in ["baseline", "initial_history", state.get("best_mean"), state.get("best_peak")] if k
            )
        )
        for name in selected:
            if (deadline - now()).total_seconds() < 180:
                raise HardDeadline("Insufficient time to complete another independent evaluation.")
            parameters = state["candidates"][name]["parameters"]
            result = score(test, test_manifest, parameters)
            result["replay"] = score_replay(test, test_manifest, parameters, excluded)
            state["test_results"][name] = result
            save()
            publish(state, args.publish_site, True)
        state["reference_replay"] = score_replay(test, test_manifest, None, excluded)
        state["complete"] = True
        state["finished_at"] = now().isoformat()
        state["phase"] = (
            "Timed contact study finished. Independent force, impact-peak, and box-motion results are available."
        )
        save()
        publish(state, args.publish_site, True)
    except (Exception, HardDeadline) as error:
        state["error"] = str(error)
        state["phase"] = "Study stopped: " + str(error)
        state["finished_at"] = now().isoformat()
        save()
        publish(state, args.publish_site, True)
        raise
    finally:
        signal.alarm(0)
        save()


if __name__ == "__main__":
    main()
