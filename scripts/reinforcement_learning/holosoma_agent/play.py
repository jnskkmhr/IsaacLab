# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with holosoma agent.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with holosoma agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="holosoma_agent_cfg_entry_point",
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append Holosoma cli arguments
cli_args.add_holosoma_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import time

import gymnasium as gym
import torch
from holosoma_agent import (
    PPO,
    BaseAlgo,
    FastSACAgent,
)

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

# from isaaclab.utils.io import dump_yaml
from isaaclab_rl.holosoma_agent import (
    FastSACConfig,
    FastSACVecEnvWrapper,
    PPOConfig,
    VecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: PPOConfig | FastSACConfig):
    """Train with Holosoma agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_holosoma_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    # env_cfg.seed = agent_cfg.seed # TODO: add seed in agent cfg
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    device = env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "holosoma_agent", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint, ["models"])
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    print("resume path: ", resume_path)

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
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for holosoma agent
    if agent_cfg.algorithm == "FastSAC":
        env = FastSACVecEnvWrapper(env)
        algo_class = FastSACAgent
    elif agent_cfg.algorithm == "PPO":
        env = VecEnvWrapper(env)
        algo_class = PPO
    else:
        raise ValueError(f"Invalid RL algorithm: {agent_cfg.algorithm}")

    algo: BaseAlgo = algo_class(
        env=env,
        config=agent_cfg,  # TODO: check if we can parse config class to dataclass
        device=device,
        log_dir=log_dir,
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.load(resume_path)

    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        algo.load(resume_path)

    # obtain the trained policy for inference
    policy = algo.get_inference_policy()
    policy_wrapper = algo.actor_onnx_wrapper

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_wrapper, normalizer=None, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_wrapper, normalizer=None, path=export_model_dir, filename="policy.onnx")
    # policy_jit = torch.jit.load(
    #     "logs/holosoma_agent/g1_29dof_rigid/2026-02-26_18-33-07/models/exported/policy.pt", map_location="cuda:0"
    # )
    # policy_jit = torch.jit.load(
    #     "logs/holosoma_agent/g1_29dof_soft/2026-03-02_17-21-04/models/exported/policy.pt", map_location="cuda:0"
    # )

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
            # actions = policy_wrapper(obs["policy"])
            # actions = policy_jit(obs["policy"])
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # # reset recurrent states for episodes that have terminated
            # policy_wrapper.reset(dones)
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
