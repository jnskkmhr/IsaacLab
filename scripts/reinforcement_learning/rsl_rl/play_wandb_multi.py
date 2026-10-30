# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper around ``play_wandb.py`` that plays back multiple WandB runs sequentially.

This is a thin orchestration layer: it does **not** import Isaac Lab / Isaac Sim
(doing so would launch the full simulator just to read a config), it only figures
out which WandB run ids to play and then launches ``play_wandb.py`` once per run as a
subprocess, forwarding any unrecognized arguments through verbatim.

Run selection (exactly one of the following is required):

* ``--wandb_runs ID [ID ...]``
    Explicit list of WandB run ids (hashes) to play, in the given order. This
    path needs no WandB access from the wrapper -- each id is handed straight to
    ``play_wandb.py``, which resolves the entity/project from the task's agent config.

* ``--wandb_count N``
    Play the ``N`` most recently created runs for ``--task``.

* ``--wandb_indices I [I ...]``
    Pick runs by position in the recency-sorted list (0 = most recent). For
    example ``--wandb_indices 0 1 3 4`` plays the two most recent runs, skips the
    3rd, then includes the 4th and 5th. Runs are played in the order listed.

For ``--wandb_count`` / ``--wandb_indices`` the wrapper needs to know which WandB
project/entity to list from. These are read from the task's RSL-RL agent config
(``wandb_project`` / ``wandb_entity``) by launching Isaac Sim once via the
``_resolve_wandb_cfg.py`` helper -- so the only thing you provide is ``--task``.

All other arguments are passed through to ``play_wandb.py`` unchanged, e.g.::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_wandb_multi.py \
        --wandb_count 3 \
        --task <Your-Task-Play> --num_envs 4 --video --headless

Note: ``play_wandb.py`` runs forever in interactive (non-video) mode, so for unattended
batch playback pass ``--video`` (each run records one clip and exits) or close
each window to advance to the next run.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from itertools import islice

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAY_SCRIPT = os.path.join(SCRIPT_DIR, "play_wandb.py")
RESOLVER_SCRIPT = os.path.join(SCRIPT_DIR, "_resolve_wandb_cfg.py")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse wrapper arguments, returning (known_args, passthrough_args)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # disable prefix matching so unknown play_wandb.py flags are never partially
        # matched against the wrapper's own options.
        allow_abbrev=False,
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--wandb_runs",
        nargs="+",
        metavar="RUN_ID",
        help="Explicit list of WandB run ids (hashes) to play, in order.",
    )
    selection.add_argument(
        "--wandb_count",
        type=int,
        metavar="N",
        help="Play the N most recently created runs for --task.",
    )
    selection.add_argument(
        "--wandb_indices",
        nargs="+",
        type=int,
        metavar="IDX",
        help=(
            "Indices into the recency-sorted run list (0 = most recent). "
            "e.g. '0 1 3 4' -> two most recent, skip the 3rd, include the 4th and 5th."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help=(
            "Name of the task. Forwarded to play_wandb.py, and used to resolve the WandB "
            "project/entity for --wandb_count / --wandb_indices."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve and print the selected runs without launching playback.",
    )
    return parser.parse_known_args()


def resolve_wandb_cfg(task: str) -> tuple[str | None, str]:
    """Launch Isaac once (via the resolver helper) to read the task's WandB entity/project."""
    fd, out_path = tempfile.mkstemp(prefix="wandb_cfg_", suffix=".json")
    os.close(fd)
    try:
        cmd = [sys.executable, RESOLVER_SCRIPT, "--task", task, "--output", out_path]
        print(f"[play_wandb_multi] Resolving WandB project/entity for task '{task}' (launching Isaac once)...")
        # NOTE: Isaac Sim's simulation_app.close() forces the child's exit code to
        # 0 even on error, so returncode is unreliable. Trust the output file.
        subprocess.run(cmd)
        if os.path.getsize(out_path) == 0:
            raise SystemExit(
                f"Failed to resolve WandB config for task '{task}': resolver wrote no output "
                "(see the resolver traceback above)."
            )
        with open(out_path) as f:
            data = json.load(f)
        if "error" in data:
            raise SystemExit(f"Failed to resolve WandB config for task '{task}': {data['error']}")
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
    return data.get("entity"), data["project"]


def resolve_runs(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    """Resolve the selection arguments into an ordered list of (run_id, run_name)."""
    # Explicit ids: no WandB access required, names unknown.
    if args.wandb_runs is not None:
        return [(run_id, None) for run_id in args.wandb_runs]

    # Listing requires a project/entity, read from the task's agent config.
    entity, project = resolve_wandb_cfg(args.task)

    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    print(f"[play_wandb_multi] Listing runs from '{path}' (most recent first)...")
    # Newest first. `runs` is lazily paginated, so only materialize what we need.
    runs = api.runs(path=path, order="-created_at")

    if args.wandb_count is not None:
        if args.wandb_count <= 0:
            raise SystemExit("--wandb_count must be a positive integer.")
        selected = list(islice(runs, args.wandb_count))
        if not selected:
            raise SystemExit(f"No runs found in '{path}'.")
    else:
        if any(i < 0 for i in args.wandb_indices):
            raise SystemExit("--wandb_indices values must be non-negative (0 = most recent).")
        # Materialize just enough of the head to cover the largest requested index.
        head = list(islice(runs, max(args.wandb_indices) + 1))
        selected = []
        for i in args.wandb_indices:
            if i >= len(head):
                raise SystemExit(f"Index {i} is out of range; only {len(head)} run(s) available in '{path}'.")
            selected.append(head[i])

    return [(run.id, run.name) for run in selected]


def run_play(run_id: str, task: str, passthrough: list[str]) -> int:
    """Launch play_wandb.py for a single run id and return its exit code."""
    cmd = [sys.executable, PLAY_SCRIPT, "--task", task, "--wandb_run", run_id, *passthrough]
    print(f"[play_wandb_multi] Launching: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main() -> None:
    args, passthrough = parse_args()

    # play_wandb.py's --wandb_run is managed per-subprocess by this wrapper.
    if any(arg.startswith("--wandb_run") for arg in passthrough):
        raise SystemExit(
            "Do not pass --wandb_run to play_wandb_multi.py; it is set automatically per run. "
            "Use --wandb_runs / --wandb_count / --wandb_indices instead."
        )

    runs = resolve_runs(args)

    print(f"[play_wandb_multi] Resolved {len(runs)} run(s) to play:")
    for i, (run_id, name) in enumerate(runs):
        label = f"  ({name})" if name else ""
        print(f"  [{i}] {run_id}{label}")

    if args.dry_run:
        print("[play_wandb_multi] --dry_run set; not launching playback.")
        return

    results: list[tuple[str, int]] = []
    for i, (run_id, name) in enumerate(runs):
        header = f"{run_id}{f' ({name})' if name else ''}"
        print(f"\n[play_wandb_multi] === Run {i + 1}/{len(runs)}: {header} ===")
        return_code = run_play(run_id, args.task, passthrough)
        results.append((run_id, return_code))
        # Stop the whole batch on an interrupt (SIGINT/SIGTERM) rather than
        # silently rolling on to the next run.
        if return_code in (-signal.SIGINT, -signal.SIGTERM, 130):
            print("[play_wandb_multi] Interrupted; stopping remaining runs.")
            break
        if return_code != 0:
            print(f"[play_wandb_multi] Run {run_id} exited with code {return_code}.")

    print("\n[play_wandb_multi] Summary:")
    for run_id, return_code in results:
        status = "ok" if return_code == 0 else f"FAILED (exit {return_code})"
        print(f"  {run_id}: {status}")

    if any(rc != 0 for _, rc in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
