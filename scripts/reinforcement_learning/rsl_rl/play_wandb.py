# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL, resolving the checkpoint from a WandB run.

This is a thin variant of ``play.py`` that adds a single new option, ``--wandb_run``, to
download a checkpoint from a WandB run instead of resolving one from the local ``logs/``
directory. All other behavior (video recording, JIT/ONNX export, the play loop) is identical
to ``play.py``.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL, resolving the checkpoint from WandB.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--wandb_run",
    type=str,
    default=None,
    help="WandB run id (hash) to resolve the checkpoint from. The run's entity/project are read from the "
    "task's agent config ('wandb_entity' / 'wandb_project'), overridable with --wandb_entity / --wandb_project. "
    "Cannot be combined with --checkpoint or --use_pretrained_checkpoint.",
)
parser.add_argument(
    "--wandb_entity",
    type=str,
    default=None,
    help="WandB entity to resolve --wandb_run from. Defaults to the task agent config's 'wandb_entity'.",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default=None,
    help="WandB project to resolve --wandb_run from. Defaults to the task agent config's 'wandb_project'.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

if args_cli.wandb_run is not None and (args_cli.use_pretrained_checkpoint or args_cli.checkpoint):
    raise ValueError("--wandb_run cannot be combined with --checkpoint or --use_pretrained_checkpoint.")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import re
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def download_checkpoint_from_wandb(
    log_root_path: str,
    wandb_project: str,
    wandb_run: str,
    wandb_entity: str | None = None,
    checkpoint: str = "model_.*.pt",
) -> str:
    """Download the checkpoint file for a WandB run and return its local path.

    The checkpoint is resolved among the run's uploaded files by matching :attr:`checkpoint`
    as a regex. If more than one file matches, the alphabetically-last one is selected (this
    matches :func:`isaaclab_tasks.utils.get_checkpoint_path`'s behavior for ``model_<iter>.pt``
    style names, since RSL-RL's WandB writer uploads checkpoints as plain run files, not
    artifacts).

    Args:
        log_root_path: Local directory under which downloaded runs are cached.
        wandb_project: The WandB project the run lives in.
        wandb_run: The WandB run id (hash) to download the checkpoint from.
        wandb_entity: The WandB entity (team/user) the project belongs to. If None, WandB
            resolves it from the local login (e.g. the default entity for the API key in use).
        checkpoint: Regex expression to match the checkpoint file name. Defaults to the most
            recent RSL-RL torch checkpoint.

    Returns:
        The local path to the downloaded checkpoint file.
    """
    import wandb

    api = wandb.Api()
    run_path = f"{wandb_entity}/{wandb_project}/{wandb_run}" if wandb_entity else f"{wandb_project}/{wandb_run}"
    run = api.run(run_path)

    checkpoint_files = [f.name for f in run.files() if re.match(checkpoint, os.path.basename(f.name))]
    if not checkpoint_files:
        raise ValueError(f"No checkpoints matching '{checkpoint}' found in WandB run '{run_path}'.")
    # sort alphabetically while ensuring that *_10 comes after *_9
    checkpoint_files.sort(key=lambda m: f"{m:0>15}")
    checkpoint_file = checkpoint_files[-1]

    download_dir = os.path.join(log_root_path, "wandb", f"{run.name or run.id}_{run.id}")
    os.makedirs(download_dir, exist_ok=True)
    run.file(checkpoint_file).download(root=download_dir, replace=True)

    return os.path.join(download_dir, checkpoint_file)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.wandb_run is not None:
        if args_cli.wandb_entity is not None:
            wandb_entity = args_cli.wandb_entity
        else:
            wandb_entity = getattr(agent_cfg, "wandb_entity", None)
        wandb_project = args_cli.wandb_project if args_cli.wandb_project is not None else agent_cfg.wandb_project
        print(f"[INFO] Loading experiment from WandB run: {args_cli.wandb_run}")
        resume_path = download_checkpoint_from_wandb(
            log_root_path, wandb_project, args_cli.wandb_run, wandb_entity, agent_cfg.load_checkpoint
        )
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
