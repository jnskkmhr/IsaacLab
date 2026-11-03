# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay an exported native Agile G1 FlashSAC actor in the current Isaac Lab runtime.

The checkpoint must be Agile's deterministic TorchScript actor (255 inputs, 14 outputs),
with its learned action support included. Raw learner checkpoints are not accepted.
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.common import add_video_args, apply_video_recording, pre_launch_video_config, run_playback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--real-time", action="store_true")
add_video_args(parser, action="play")
add_launcher_args(parser)
args = parser.parse_args()

import gymnasium as gym
import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.env_cfg.action_cfg import CONTROLLED_JOINT_NAMES
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.env_cfg.physics_cfg import G1PhysicsCfg
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.env_cfg.scene_cfg import G1_29DOF_DELAYED_DC_MOTOR
from isaaclab_tasks.contrib.velocity.config.g1_29dof_rigid_agile.rough_env_cfg import G1RoughEnvCfg


def main():
    """Restore the native actor contract without changing the registered training task."""
    cfg = G1RoughEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        cfg.sim.device = args.device
    cfg.scene.robot = G1_29DOF_DELAYED_DC_MOTOR.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.sim.physics = G1PhysicsCfg().newton_mjwarp
    cfg.sim.physics.default_shape_cfg.margin = 0.032
    cfg.curriculum = None
    cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0

    # Native G1 leaves waist yaw at its default target, outside the 14 policy actions.
    joints = [name for name in CONTROLLED_JOINT_NAMES if name != "waist_yaw_joint"]
    cfg.actions.joint_pos.joint_names = joints
    cfg.actions.joint_pos.mirror_params = {"joint_names": joints}
    for term in (cfg.observations.policy.controlled_joint_pos, cfg.observations.policy.controlled_joint_vel):
        term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=joints, preserve_order=True)
        term.mirror_params = {"joint_names": joints}
    cfg.observations.policy.actions.mirror_params = {"joint_names": joints}
    cfg.observations.critic.actions.mirror_params = {"joint_names": joints}
    pre_launch_video_config(cfg, args)

    with launch_simulation(cfg, args):
        apply_video_recording(cfg, str(args.checkpoint.resolve().parent), args, subdir="play")
        env = gym.make("IsaacContrib-Velocity-Flat-G1-29dof-Rigid-Agile", cfg=cfg)
        raw = env.unwrapped
        try:
            actor = torch.jit.load(str(args.checkpoint), map_location=raw.device).eval()

            def update_camera(_=None):
                x, y, z = raw.scene["robot"].data.root_pos_w.torch[0].tolist()
                raw.sim.set_camera_view(eye=(x + 3.0, y - 3.0, z + 1.5), target=(x, y, z))

            # raw.sim.vis_marker_registry.add_callback("native_flash_sac_camera", update_camera)
            obs, _ = env.reset()
            if obs["policy"].shape != (raw.num_envs, 255) or raw.action_manager.total_action_dim != 14:
                raise ValueError("Native Agile G1 playback requires 255 policy inputs and 14 actions")

            def step():
                nonlocal obs
                actions = actor(obs["policy"])
                if actions.shape != (raw.num_envs, 14) or not torch.isfinite(actions).all():
                    raise ValueError("Actor must return finite actions with shape (num_envs, 14)")
                obs, _, _, _, _ = env.step(actions)

            run_playback(step, dt=raw.step_dt, args_cli=args, env_cfg=cfg)
        finally:
            raw.sim.vis_marker_registry.remove_callback("native_flash_sac_camera")
            env.close()


if __name__ == "__main__":
    main()
