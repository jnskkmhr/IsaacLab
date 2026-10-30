"""Internal helper for ``play_multi.py``.

Launches Isaac Sim once, loads a task's RSL-RL agent config, and writes the
task's WandB ``entity``/``project`` to a JSON file. This exists as a separate
process because importing any Isaac Lab task config requires a running
simulation app (everything pulls in ``pxr``), and we don't want ``play_multi.py``
itself to be an Isaac Sim process.

Not meant to be called directly -- ``play_multi.py`` invokes it.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Resolve a task's WandB entity/project to a JSON file.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--output", type=str, required=True, help="Path to write the resolved JSON to.")
args_cli = parser.parse_args()

# launch omniverse app (headless -- we only read a config and exit)
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json

import gymnasium as gym  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def main():
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    project = getattr(agent_cfg, "wandb_project", None)
    if project is None:
        raise ValueError(f"Task '{args_cli.task}' agent config has no 'wandb_project'.")
    data = {"entity": getattr(agent_cfg, "wandb_entity", None), "project": project}
    with open(args_cli.output, "w") as f:
        json.dump(data, f)
    print(f"[_resolve_wandb_cfg] {args_cli.task} -> entity='{data['entity']}' project='{data['project']}'")


if __name__ == "__main__":
    # NOTE: simulation_app.close() forces the process exit code to 0 even when
    # main() raises, so the parent can't rely on the return code. Report failure
    # through the output file, which close() can't clobber.
    try:
        main()
    except Exception as e:
        with open(args_cli.output, "w") as f:
            json.dump({"error": f"{type(e).__name__}: {e}"}, f)
        raise
    finally:
        simulation_app.close()