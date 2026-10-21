# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run dynamic G1 with native MPM or particle NeRD and analytical foot forces."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from particle_nerd.g1 import DEFAULT_POLICY
from particle_nerd.g1.collect import configure


def main():
    from isaaclab.app import add_launcher_args, launch_simulation

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, help="G1 particle checkpoint; omit for MPM")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--schedule", type=Path)
    p.add_argument("--trial", type=int, default=1)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=120000)
    p.add_argument("--effective-mass-scale", type=float, default=10.0)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--headless", action="store_true")
    add_launcher_args(p)
    args = p.parse_args()
    torch.set_num_threads(2)
    import gymnasium as gym
    from isaaclab_newton.physics import NewtonManager
    from rsl_rl.models import MLPModel

    from isaaclab.utils.math import quat_from_euler_xyz

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from particle_nerd.g1.adapter import LearnedMaterial

    task = "IsaacContrib-Velocity-Sand-G1-29dof-MPM-Play"
    cfg = configure(parse_env_cfg(task, num_envs=1), 1)
    cfg.seed = args.seed
    cfg.sim.physics.use_cuda_graph = args.checkpoint is None
    cfg.sim.physics.load_visual_shapes = True
    cfg.episode_length_s = args.duration + 1
    for viz in cfg.sim.visualizer_cfgs:
        viz.headless = args.headless
    trial = dict(
        trial_id=args.trial,
        initial_x_m=0,
        initial_y_m=0,
        initial_yaw_rad=0,
        command_start=[0.15, 0, 0],
        command_end=[0.1, 0.05, 0.25],
        switch_time_s=2.5,
        mode="policy",
        action_noise_amplitude=0,
    )
    if args.schedule:
        trial = next(t for t in json.loads(args.schedule.read_text())["trials"] if t["trial_id"] == args.trial)
    args.output.mkdir(parents=True, exist_ok=True)
    saved = json.loads(args.policy.with_name("wandb_config.json").read_text())
    launcher = dict(vars(args), visualizer=[] if args.headless else (args.visualizer or ["newton_rtx"]))
    with launch_simulation(cfg, launcher):
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=saved["train_cfg"].get("clip_actions"))
        raw, adapter = env.unwrapped, None
        original_hook = raw.recorder_manager.record_post_physics_decimation_step
        try:
            NewtonManager.set_decimation(1)
            raw._physics_handles_decimation = False
            torch.manual_seed(args.seed)
            env.reset()
            robot = raw.scene["robot"]
            pose = robot.data.root_state_w.torch.clone()
            pose[0, :3] = raw.scene.env_origins[0] + torch.tensor(
                [trial["initial_x_m"], trial["initial_y_m"], 0.76], device=raw.device
            )
            yaw = torch.tensor([trial["initial_yaw_rad"]], device=raw.device)
            pose[0, 3:7] = quat_from_euler_xyz(yaw * 0, yaw * 0, yaw)[0]
            pose[0, 7:] = 0
            robot.write_root_state_to_sim(pose)
            raw.scene.write_data_to_sim()
            raw.sim.forward()
            raw.scene.update(dt=0)
            obs = raw.observation_manager.compute()
            actor_cfg = saved["train_cfg"]["actor"].copy()
            assert actor_cfg.pop("class_name") == "MLPModel"
            actor = MLPModel(
                obs, saved["train_cfg"]["obs_groups"], "actor", raw.action_manager.total_action_dim, **actor_cfg
            ).to(raw.device)
            actor.load_state_dict(
                torch.load(args.policy, map_location=raw.device, weights_only=True)["actor_state_dict"]
            )
            actor.eval()
            model = NewtonManager.get_model()
            if args.checkpoint:
                # Loading the neural model must not alter the policy/noise RNG sequence.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    adapter = LearnedMaterial(
                        raw, args.checkpoint, iterations=args.iterations, effective_mass_scale=args.effective_mass_scale
                    )
            feet = raw._foot_body_ids[0].cpu().numpy()
            trace, snapshots = [], []
            step = 0

            def record():
                state = NewtonManager.get_state_0()
                bq, q = state.body_q.numpy(), state.particle_q.numpy()
                wrench = raw._coupling_forces.numpy()[feet]
                if not all(np.isfinite(a).all() for a in (q, bq, wrench)):
                    raise FloatingPointError(f"Nonfinite state at {step}")
                trace.append(
                    dict(
                        time_s=step * 0.005,
                        foot_wrench=wrench,
                        root_pose=bq[0].copy(),
                        root_velocity=state.body_qd.numpy()[0].copy(),
                    )
                )
                if step % 8 == 0:
                    snapshots.append(dict(time_s=step * 0.005, body_pose=bq.copy(), particle_position=q.copy()))

            def hook():
                nonlocal step
                original_hook()
                step += 1
                record()

            raw.recorder_manager.record_post_physics_decimation_step = hook
            record()
            term = raw.command_manager.get_term("base_velocity")
            noise = torch.zeros(1, raw.action_manager.total_action_dim, device=raw.device)
            target = torch.zeros_like(noise)
            outcome, failure = "duration", None
            with torch.inference_mode():
                for k in range(round(args.duration / 0.02)):
                    command = trial["command_end"] if k * 0.02 >= trial["switch_time_s"] else trial["command_start"]
                    term.vel_command_b[:] = torch.tensor(command, device=raw.device)
                    term.is_standing_env[:] = False
                    obs["command"][:] = term.command
                    action = actor(obs)
                    if k % 10 == 0:
                        target.uniform_(-1, 1)
                    noise.lerp_(target, 0.2)
                    if trial["mode"] == "policy-noise":
                        action += trial["action_noise_amplitude"] * noise
                    elif trial["mode"] == "random":
                        action = trial["action_noise_amplitude"] * noise
                    elif trial["mode"] == "zero":
                        action.zero_()
                    try:
                        obs, _, done, _ = env.step(action)
                    except (FloatingPointError, RuntimeError) as error:
                        outcome, failure = "simulation_error", str(error)
                        break
                    if bool(done[0]):
                        outcome = "native_termination"
                        break
                    if k % 25 == 0:
                        print(
                            "EVALUATION",
                            json.dumps(dict(time_s=step * 0.005, learned_steps=adapter.steps if adapter else 0)),
                            flush=True,
                        )
            metadata = dict(
                trial=trial,
                seed=args.seed,
                outcome=outcome,
                duration_s=float(trace[-1]["time_s"]),
                failure=failure,
                backend="particle_nerd" if adapter else "mpm",
                checkpoint=str(args.checkpoint),
                particle_count=model.particle_count,
                body_labels=list(model.body_label),
                estimator=adapter.estimator.metadata if adapter else None,
                learned_steps=adapter.steps if adapter else 0,
                history_initialization="constant velocity extrapolation from initial state",
                physics_hz=200,
                policy_hz=50,
                snapshots_hz=25,
                use_cuda_graph=cfg.sim.physics.use_cuda_graph,
            )
            for name, records in [("trace", trace), ("snapshots", snapshots)]:
                np.savez(
                    args.output / f"{name}.npz", **{key: np.stack([r[key] for r in records]) for key in records[0]}
                )
            (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print("EVALUATION_COMPLETE", json.dumps(dict(outcome=outcome, duration_s=trace[-1]["time_s"])), flush=True)
            if failure:
                raise RuntimeError(f"Saved valid rollout prefix before simulation failure: {failure}")
        finally:
            raw.recorder_manager.record_post_physics_decimation_step = original_hook
            if adapter:
                adapter.close()
            env.close()


if __name__ == "__main__":
    main()
