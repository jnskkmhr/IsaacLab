# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the latest walking run; optionally record its trained policy without assistance."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path


def show_log(home):
    logs = sorted(home.glob("lafan-walk-*.log"), key=lambda p: p.stat().st_mtime)
    if logs:
        print("LATEST_LAUNCH_LOG:", logs[-1], flush=True)
        with logs[-1].open(errors="replace") as handle:
            print("".join(deque(handle, maxlen=45)), flush=True)
    subprocess.run(["pgrep", "-af", "train_lafan_walk_3000.py|isaaclab train"], check=False)


def completed_run(home):
    runs = sorted(p for p in (home / "lafan_walk_runs").glob("walk4to10s_2048env_3000updates_*") if p.is_dir())
    if not runs:
        print("NO_RUN_DIRECTORY: no launch of the 3000-update pipeline was found", flush=True)
        show_log(home)
        return None
    run = runs[-1]
    print("LATEST_RUN:", run, flush=True)
    completed = run / "completed.json"
    if not completed.is_file():
        print("NOT_CONFIRMED_COMPLETE: may still be running, stopped, or failed", flush=True)
        checkpoints = sorted(
            (home / "IsaacLab/logs/rsl_rl/lafan_walk").glob(f"*_{run.name}/model_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        if checkpoints:
            print("LATEST_SAVED_CHECKPOINT:", checkpoints[-1], flush=True)
        show_log(home)
        return None
    data = json.loads(completed.read_text())
    checkpoint = Path(data["checkpoint"])
    motion = Path(data["motion"])
    if data.get("iterations") != 3000 or data.get("num_envs") != 2048:
        raise RuntimeError("Unexpected training configuration in completion report")
    if checkpoint.name != "model_2999.pt" or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("Completion report exists but the final checkpoint is missing or empty")
    reports = list((run / "conversion").glob("*/conversion.json"))
    if len(reports) != 1:
        raise RuntimeError("Missing or ambiguous conversion report")
    conversion = json.loads(reports[0].read_text())
    if not motion.is_file() or conversion.get("npz_sha256") != hashlib.sha256(motion.read_bytes()).hexdigest():
        raise RuntimeError("Training motion file is missing or its checksum changed")
    print("TRAINING_COMPLETED: 2048 environments, 3000 iterations", flush=True)
    print("CHECKPOINT:", checkpoint, flush=True)
    print("MOTION:", motion, flush=True)
    return run, data, checkpoint, motion


def record(home, result):
    run, data, checkpoint, motion = result
    repo = home / "IsaacLab"
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != data["commit"]:
        raise RuntimeError("Checkout differs from training commit; check compatibility before playback")
    if Path(sys.executable).absolute().parent != repo / ".venv/bin":
        raise RuntimeError("Use this checkout's uv run --frozen --no-sync python")
    dirty = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--", "source", "scripts"], text=True)
    if dirty.strip():
        raise RuntimeError("Review local source changes before playback")
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True
    ).strip()
    if busy:
        raise RuntimeError("GPU compute is currently in use; recording was not started:\n" + busy)
    video_dir = checkpoint.parent / "videos/play"
    before = {p.resolve(): (p.stat().st_mtime_ns, p.stat().st_size) for p in video_dir.rglob("*.mp4")}
    command = [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "isaaclab",
        "play",
        "--rl_library",
        "rsl_rl",
        "--task",
        "IsaacContrib-Mimic-G1-29dof-Play-v0",
        "--checkpoint",
        str(checkpoint),
        "--num_envs",
        "1",
        "--seed",
        "42",
        "--logger",
        "tensorboard",
        "--viz",
        "newton_gl",
        "--video",
        "--video_length",
        "600",
        "physics=newton_mjwarp",
        f"env.commands.motion.motion_file={json.dumps(str(motion))}",
        "env.commands.motion.stance_phase_ranges=[]",
        "env.events.assistive_wrench=null",
        "env.video_recorders=[]",
        "agent.experiment_name=lafan_walk",
    ]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
    env.pop("PYTORCH_JIT", None)
    # This fork's Newton GL viewer selects EGL headless rendering when DISPLAY
    # is absent. Its preset override parser cannot index visualizer_cfgs lists.
    env.pop("DISPLAY", None)
    print("RECORDING_POLICY: assistance disabled; standard PLAY-mode termination settings", flush=True)
    print("RUN:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=repo, env=env, check=True, timeout=600)
    candidates = [
        p
        for p in video_dir.rglob("*.mp4")
        if p.stat().st_size > 0 and before.get(p.resolve()) != (p.stat().st_mtime_ns, p.stat().st_size)
    ]
    if not candidates:
        raise RuntimeError(f"Playback returned but no new MP4 was found in {video_dir}")
    source = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    destination = home / "Videos/lafan-walk-policy-3000.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    info = dict(
        recorded_at=datetime.now().isoformat(),
        checkpoint=str(checkpoint),
        motion=str(motion),
        source_video=str(source),
        download_video=str(destination),
        assistive_wrench=False,
        mode="standard PLAY task",
        performance_evaluated=False,
        command=command,
    )
    (run / "policy_video.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("POLICY_VIDEO_READY:", destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="Record only after completion is verified")
    args = parser.parse_args()
    home = Path.home()
    result = completed_run(home)
    if result is None:
        sys.exit(2)
    if args.record:
        record(home, result)


if __name__ == "__main__":
    main()
