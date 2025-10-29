# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_speed", type=float, default=1.0, help="Speed of the recorded video.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--log", action="store_true", default=False, help="Enable logging.")
parser.add_argument("--max_trials", type=int, default=1, help="Number of trials to run.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import datetime
from pathlib import Path
import os
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.dict import print_dict

from logger import DictBenchmarkLogger

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env.metadata["render_fps"] = int((1/env.unwrapped.step_dt) * args_cli.video_speed)  # type: ignore
    
    # get logging directory
    log_dir = (Path(os.path.realpath('__file__')).parent / "logs" / f"zero/{args_cli.task}").as_posix()
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[INFO] Log directory: {log_dir}")
    
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

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    # prepare logging 
    if args_cli.log:
        os.makedirs(log_dir, exist_ok=True)
        max_episode_length = int(env_cfg.episode_length_s/(env_cfg.decimation*env_cfg.sim.dt))
        log_item = ["obs", "contact"]
        logger = DictBenchmarkLogger(
            log_dir, 
            "mpc", 
            num_envs=args_cli.num_envs, 
            max_trials=args_cli.max_trials, 
            max_episode_length=max_episode_length, 
            log_item=log_item,
            )
        
    # record episode runs
    max_trials = args_cli.max_trials
    episode_length_log = [0]*args_cli.num_envs
    episode_counter = [0]*args_cli.num_envs
    
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device) # type: ignore
            # apply actions
            obs, _, done, timeout, _ = env.step(actions)
            dones = done | timeout
            
            policy_obs = obs["policy"].clone()
            contact_obs = obs["contact"].clone()
            
            # get termination conditions
            termination_manager = env.unwrapped.termination_manager # type: ignore
            term_dones = termination_manager._term_dones
            base_too_low_idx = termination_manager._term_name_to_term_idx.get("base_too_low", None)
            time_out_idx = termination_manager._term_name_to_term_idx.get("time_out", None)
            base_too_low = term_dones[:, base_too_low_idx] if base_too_low_idx is not None else None
            time_out = term_dones[:, time_out_idx] if time_out_idx is not None else None
        
        # process data to log
        if args_cli.log:
            item_dict = {
                "obs": policy_obs.cpu().numpy(),
                "contact": contact_obs.cpu().numpy(),
            }
            logger.log(item_dict)
            for i in range(args_cli.num_envs):
                episode_length_log[i] += 1
            
        # process episode terminations
        dones_np = dones.cpu().numpy() # type: ignore
        # Check each agent's done flag
        for i, done_flag in enumerate(dones_np):
            if episode_counter[i] < max_trials: # only allow logging when episode counter is less than max trials to avoid memory overflow
                if done_flag == 1:
                    # # print(f"[INFO] Env {i}: Episode {episode_counter[i]} completed with episode length {episode_length_log[i]}.")
                    # if terrain_out_of_bounds[i]:
                    #     print(f"[INFO] Env {i}: Episode {episode_counter[i]} - Terrain out of bounds with length {episode_length_log[i]}")
                    # if bad_orientation[i]:
                    #     print(f"[INFO] Env {i}: Episode {episode_counter[i]} - Bad orientation with length {episode_length_log[i]}")
                    if base_too_low is not None:
                        if base_too_low[i]: 
                            print(f"[INFO] Env {i}: Episode {episode_counter[i]} - Base too low with length {episode_length_log[i]}")
                    if time_out is not None:
                        if time_out[i]: # type: ignore
                            print(f"[INFO] Env {i}: Episode {episode_counter[i]} - Time out {episode_length_log[i]}")

                    if args_cli.log:
                        logger.save_to_buffer(trial_id=episode_counter[i], env_idx=i)
                        logger.save_episode_length_to_buffer(trial_id=episode_counter[i], env_idx=i, episode_length=episode_length_log[i])
                    episode_length_log[i] = 0
                    episode_counter[i] += 1
        
        # Check if max trials reached
        if all(tc >=max_trials for tc in episode_counter):
            print("[INFO] Max trials reached for all environments.")
            break
        
    # save all the log
    if args_cli.log:
        logger.save()
        print(f"Saved logs in {logger.log_dir}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
