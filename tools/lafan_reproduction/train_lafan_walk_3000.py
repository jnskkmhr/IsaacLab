# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert CSV frames 121-301, then train 2048 Newton environments for 3000 updates."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXPECTED_COMMIT = "e1f1d6d2a805d88f37b38ea418462ab5a2766808"


def verified_motion(conversions, csv_path):
    reports = list(conversions.glob("*/conversion.json"))
    if len(reports) != 1:
        raise RuntimeError("Expected exactly one successful conversion report")
    report = json.loads(reports[0].read_text())
    expected = {
        "commit": EXPECTED_COMMIT,
        "frame_range_1_based": [121, 301],
        "source_fps": 30,
        "output_fps": 60,
        "frames": 360,
        "physics_steps": 0,
        "loader_checks_passed": True,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"Conversion validation failed for {key}: {report.get(key)!r}")
    if report.get("csv_sha256") != hashlib.sha256(csv_path.read_bytes()).hexdigest():
        raise RuntimeError("Source CSV checksum differs from conversion report")
    motion = Path(report["npz"]).resolve()
    if not motion.is_relative_to(conversions.resolve()) or not motion.is_file():
        raise RuntimeError("Converted motion is not in this run directory")
    if report.get("npz_sha256") != hashlib.sha256(motion.read_bytes()).hexdigest():
        raise RuntimeError("Converted NPZ checksum mismatch")
    return motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.home() / "IsaacLab")
    opts = parser.parse_args()
    repo = opts.repo.expanduser().resolve()
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != EXPECTED_COMMIT:
        raise RuntimeError(f"Expected {EXPECTED_COMMIT}, found {head}")
    if Path(sys.executable).absolute().parent != repo / ".venv/bin":
        raise RuntimeError("Run with this checkout's uv run --frozen --no-sync python")
    dirty = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--", "source", "scripts"], text=True)
    if dirty.strip():
        raise RuntimeError("Review local source changes before starting this run")
    csv_path = Path.home() / "datasets/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv"
    converter = Path(__file__).resolve().with_name("convert_lafan_newton.py")
    if not csv_path.is_file() or not converter.is_file():
        raise FileNotFoundError("CSV or companion convert_lafan_newton.py is missing")
    import fcntl

    with (Path.home() / ".lafan_walk_3000.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This walking training launcher is already running") from exc
        busy = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True
        ).strip()
        if busy:
            raise RuntimeError("GPU compute is already in use; check before starting another job:\n" + busy)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_name = f"walk4to10s_2048env_3000updates_{stamp}"
        output = Path.home() / "lafan_walk_runs" / run_name
        output.mkdir(parents=True, exist_ok=False)
        conversions = output / "conversion"
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1", HYDRA_FULL_ERROR="1")
        env.pop("PYTORCH_JIT", None)
        convert_cmd = [
            sys.executable,
            "-u",
            "-X",
            "faulthandler",
            str(converter),
            "--repo",
            str(repo),
            "--csv",
            str(csv_path),
            "--frame-range",
            "121",
            "301",
            "--output-root",
            str(conversions),
        ]
        print("RUN_DIRECTORY:", output, flush=True)
        print("CONVERT:", shlex.join(convert_cmd), flush=True)
        subprocess.run(convert_cmd, cwd=repo, env=env, check=True)
        motion = verified_motion(conversions, csv_path)
        # Check again after conversion in case a collaborator changed the checkout.
        current = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if current != head:
            raise RuntimeError("Checkout changed during conversion")
        train_cmd = [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "isaaclab",
            "train",
            "--rl_library",
            "rsl_rl",
            "--task",
            "IsaacContrib-Mimic-G1-29dof-v0",
            "--num_envs",
            "2048",
            "--max_iterations",
            "3000",
            "--run_name",
            run_name,
            "--viz",
            "none",
            "--logger",
            "tensorboard",
            "--seed",
            "42",
            "physics=newton_mjwarp",
            f"env.commands.motion.motion_file={json.dumps(str(motion))}",
            "env.commands.motion.stance_phase_ranges=[]",
            "env.video_recorders=[]",
            "agent.experiment_name=lafan_walk",
        ]
        manifest = dict(
            commit=head,
            motion=str(motion),
            frames=360,
            reference_seconds=6,
            num_envs=2048,
            iterations=3000,
            seed=42,
            run_name=run_name,
            conversion_command=convert_cmd,
            training_command=train_cmd,
            resume=False,
            policy_performance_evaluated=False,
        )
        (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("TRAIN:", shlex.join(train_cmd), flush=True)
        subprocess.run(train_cmd, cwd=repo, env=env, check=True)
        checkpoints = list((repo / "logs/rsl_rl/lafan_walk").glob(f"*_{run_name}/model_2999.pt"))
        if len(checkpoints) != 1:
            raise RuntimeError("Training returned but the expected model_2999.pt was not found")
        manifest["checkpoint"] = str(checkpoints[0].resolve())
        (output / "completed.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("LAFAN_3000_TRAINING_COMPLETED:", checkpoints[0], flush=True)
        print("RUN_MANIFEST:", output / "completed.json", flush=True)


if __name__ == "__main__":
    main()
