# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_speed", type=float, default=1.0, help="Speed of the recorded video.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
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

parser.add_argument("--log", action="store_true", default=False, help="Enable logging.")
parser.add_argument("--max_trials", type=int, default=1, help="Number of trials to run.")
parser.add_argument("--tag", type=str, default="rl_play", help="Tag to add to the log directory.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
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

import gymnasium as gym
import os
import time
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
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from logger import DictBenchmarkLogger

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg): # type: ignore
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

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
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env.metadata["render_fps"] = int((1/env.unwrapped.step_dt) * args_cli.video_speed)  # type: ignore

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

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic # type: ignore

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    
    
    # prepare logging 
    if args_cli.log:
        max_episode_length = int(env_cfg.episode_length_s/(env_cfg.decimation*env_cfg.sim.dt))
        log_item = ["obs", "contact"]
        logger = DictBenchmarkLogger(
            log_dir=log_dir, 
            tag=args_cli.tag,
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
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # process data to log
        if args_cli.log:
            log_obs = obs["logging"] # only log necessary data
            contact_obs = obs["contact"]
            item_dict = {
                "obs": log_obs.cpu().numpy(),
                "contact": contact_obs.cpu().numpy(),
            }
            logger.log(item_dict)

        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            
            # get termination conditions
            termination_manager = env.unwrapped.termination_manager
            term_dones = termination_manager._term_dones
            base_too_low_idx = termination_manager._term_name_to_term_idx.get("base_too_low", None)
            time_out_idx = termination_manager._term_name_to_term_idx.get("time_out", None)
            base_too_low = term_dones[:, base_too_low_idx] if base_too_low_idx is not None else None
            time_out = term_dones[:, time_out_idx] if time_out_idx is not None else None

        # if args_cli.video:
        #     timestep += 1
        #     # Exit the play loop after recording one video
        #     if timestep == args_cli.video_length:
        #         break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        # increment episode length 
        for i in range(args_cli.num_envs):
            episode_length_log[i] += 1
            
        # process episode terminations
        dones_np = dones.cpu().numpy()
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
    main() # type: ignore
    # close sim app
    simulation_app.close()