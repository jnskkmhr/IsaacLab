# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out the two frozen teachers before training a pushing student."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager

from isaaclab.app import launch_simulation

from isaaclab_tasks.utils import resolve_task_config


def load_actor(path, device, input_size, output_size):
    """Load the verified mean-action MLP; fail on any missing layer or size mismatch."""
    state = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
    actor = torch.nn.Sequential(
        torch.nn.Linear(input_size, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, output_size),
    ).to(device)
    actor.load_state_dict({k.removeprefix("mlp."): v for k, v in state.items() if k.startswith("mlp.")}, strict=True)
    return actor.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("exports/particle_nerd_object_push/rollout"))
    parser.add_argument(
        "--active-k", type=int, default=None, help="Optional reduced particle budget for a smoke test only"
    )
    args = parser.parse_args()
    task = "IsaacContrib-Push-G1-29dof-ParticleNeRD-MovingPatch-ObjectPush-Play"
    cfg, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=[])
    cfg.scene.num_envs = args.num_envs
    cfg.commands.object_direction.resampling_time_range = (1000.0, 1000.0)
    cfg.sim.physics.solver_cfg.entries[1].solver_cfg.compile_model = False
    if args.active_k is not None:
        cfg.sim.physics.solver_cfg.entries[1].solver_cfg.active_k = args.active_k
    cfg.video_recorders = []
    if args.video:
        cfg.sim.visualizer_cfgs = cfg.sim.visualizer_cfgs[:1]
        cfg.sim.visualizer_cfgs[0].headless = True
        cfg.sim.visualizer_cfgs[0].window.size = (960, 640)
    else:
        cfg.sim.visualizer_cfgs = []
    args.output.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, {"visualizer": ["newton_gl"] if args.video else []}):
        env = gym.make(task, cfg=cfg).unwrapped
        writer = None
        rows = []
        try:
            obs, _ = env.reset(seed=42)
            c = env.command_manager.get_term("object_direction")
            if args.num_envs > 1:
                c.direction[1] = 0
            hand_ids = env.scene["robot"].find_bodies(".*(wrist_yaw|hand_.*)_link")[0]
            c._update_command()
            obs = env.observation_manager.compute(update_history=False)
            lower = load_actor(agent.networks.locomotion_teacher.pretrained_cfg["path"], env.device, 799, 15)
            upper = load_actor(agent.networks.manipulation_teacher.pretrained_cfg["path"], env.device, 76, 14)
            if args.video:
                import imageio.v2 as imageio

                writer = imageio.get_writer(args.output / "teachers.mp4", fps=25, codec="libx264", quality=7)
            with torch.inference_mode():
                for step in range(args.steps):
                    a = lower(torch.cat((obs["locomotion_teacher_obs"], obs["locomotion_teacher_command"]), dim=-1))
                    b = upper(obs["manipulation_teacher_obs"])
                    obs, reward, done, trunc, _ = env.step(torch.cat((a, b), dim=-1))
                    estimator = NewtonManager._solver._entries["sand"].solver.estimator
                    rows.append(
                        dict(
                            step=step,
                            robot=env.scene["robot"].data.root_pos_w.torch.cpu().tolist(),
                            object=env.scene["object"].data.root_pos_w.torch.cpu().tolist(),
                            hand_error=c.metrics["hand_target_error"].cpu().tolist(),
                            projection_error=c.metrics["workspace_projection_error"].cpu().tolist(),
                            hand_contact=env.object_contact_force[:, hand_ids].norm(dim=-1).amax(-1).cpu().tolist(),
                            object_contact=env.object_contact_force.norm(dim=-1).amax(-1).cpu().tolist(),
                            sand_wrench=wp.to_torch(estimator.wrench).cpu().tolist(),
                            done=(done | trunc).cpu().tolist(),
                        )
                    )
                    # Auto-reset samples a new command; keep the stationary control stationary.
                    if args.num_envs > 1 and (done[1] or trunc[1]):
                        c.direction[1] = 0
                        c._update_command()
                        obs = env.observation_manager.compute(update_history=False)
                    if writer is not None and step % 2 == 0:
                        writer.append_data(env.sim._visualizers[0].render_rgb_array())
                    if step % 50 == 0:
                        print("ROLLOUT", json.dumps(rows[-1]), flush=True)
        finally:
            if writer is not None:
                writer.close()
            (args.output / "measurements.json").write_text(json.dumps(rows, indent=2))
            env.close()


if __name__ == "__main__":
    main()
