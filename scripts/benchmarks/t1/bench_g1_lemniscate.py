# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T1 terrain-history run with G1: the robot walks a figure-eight so that it returns to its own footprints.

The trained locomotion policy receives forward and yaw velocity commands from a path-tracking controller: the
yaw rate is proportional to the bearing of a point a fixed distance ahead on the path, and the forward speed
drops as that bearing grows. The terrain modes and the sand layer are those of ``bench_probe.py``.

The robot's motion differs between terrain modes, so only terrain height maps and video are compared.

Example:
    uv run python scripts/benchmarks/t1/bench_g1_lemniscate.py --terrain stream_cache --checkpoint model.pt \
        --out logs/t1/g1_lemniscate_stream_cache.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_probe import (  # noqa: E402
    END_MARGIN,
    PRESETS,
    TASKS,
    GpuMonitor,
    _git_state,
    _sync,
    configure_particle_video,
    draw_sand_particles,
    height_map,
    lemniscate_path,
    planned_pool_tiles,
)

CONTROL_DT = 0.02


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--terrain", choices=sorted(TASKS), required=True)
    parser.add_argument("--sleep", choices=("off", "on"), default="off")
    parser.add_argument("--checkpoint", required=True, help="Locomotion policy checkpoint.")
    parser.add_argument("--lemniscate_a", type=float, default=2.4, help="Half-length of the figure-eight [m].")
    parser.add_argument("--laps", type=int, default=2)
    parser.add_argument("--speed", type=float, default=0.3, help="Forward speed on straight stretches [m/s].")
    parser.add_argument("--max_yaw_rate", type=float, default=0.6, help="Yaw-rate command limit [rad/s].")
    parser.add_argument("--yaw_gain", type=float, default=2.0, help="Yaw rate per bearing error [1/s].")
    parser.add_argument("--lookahead", type=float, default=0.4, help="Distance of the tracked point ahead [m].")
    parser.add_argument("--cache_slots", type=float, default=None, help="simulation_capacity_multiplier.")
    parser.add_argument("--cache_slack", type=float, default=0.3, help="Path deviation covered by the cache [m].")
    parser.add_argument(
        "--cache_unsettled_template",
        action="store_true",
        help="Skip the cache's template settling, so new tiles start from the sampled particles.",
    )
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--video_viz", choices=("rtx", "gl"), default="rtx", help="Visualizer used for the video.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args(argv)


def configure(env_cfg, args: argparse.Namespace, geometry: dict, yaw: float) -> dict:
    """Give the task the T1 terrain and sand layer, a fixed start pose and externally set commands."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.physics import MPMSleepingCfg, StrainActivationCfg, TrackedRegionActivationCfg
    from isaaclab_newton.sim.spawners.mpm import MPMGridCfg

    import isaaclab.terrains as terrain_gen
    from isaaclab.terrains import TerrainGeneratorCfg

    from isaaclab_tasks.contrib.velocity_mpm.config.g1_29dof.env_cfg import streaming_scene_cfg
    from isaaclab_tasks.contrib.velocity_mpm.util.solver_setting import get_mpm_solver_cfg

    sand = streaming_scene_cfg.G1MPMSceneCfg(num_envs=1, env_spacing=1.0).terrain.sand
    length_x, length_y = geometry["length_x"], geometry["length_y"]
    mpm = get_mpm_solver_cfg(env_cfg)
    mpm.voxel_size = sand.voxel_size
    if args.terrain == "full":
        # As in the box probe: no pre-settled bed snapshot, which fails for long beds.
        env_cfg.mpm_cache_initial_state = False
        robot = env_cfg.scene.robot
        env_cfg.scene = type(env_cfg.scene)(
            num_envs=1,
            env_spacing=max(length_x, length_y) + 2.0,
            replicate_physics=True,
            sand=MPMObjectCfg(
                prim_path="{ENV_REGEX_NS}/Sand",
                spawn=MPMGridCfg(
                    lower=(-length_x / 2.0, -length_y / 2.0, -sand.particle_depth),
                    upper=(length_x / 2.0, length_y / 2.0, 0.0),
                    voxel_size=sand.voxel_size,
                    particles_per_cell=sand.particles_per_cell,
                    particle_placement="cell_center",
                    jitter=sand.jitter,
                    material=sand.material,
                    visual_color=sand.visual_color,
                ),
            ),
        )
        # The scene places the robot near the bed's end; the figure starts at the bed's center.
        env_cfg.scene.robot = robot
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, sand.robot_spawn_height)
    else:
        terrain = env_cfg.scene.terrain
        terrain.sand = sand
        terrain.terrain_generator = TerrainGeneratorCfg(
            seed=42,
            size=(length_x, length_y),
            num_rows=1,
            num_cols=1,
            border_width=0.0,
            horizontal_scale=0.1,
            curriculum=False,
            sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg()},
        )
        env_cfg.scene.num_envs = 1
        if mpm.streaming.cache is not None and args.cache_slots is not None:
            mpm.streaming.cache.simulation_capacity_multiplier = args.cache_slots
    if args.sleep == "on":
        mpm.reuse_grid_topology = True
        mpm.sleeping = MPMSleepingCfg(
            strain=StrainActivationCfg(stress_filter_time=0.1),
            region=TrackedRegionActivationCfg(body_names="pelvis", size=(1.2, 1.2, 3.0)),
        )
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (yaw, yaw)}
    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (1.0e9, 1.0e9)
    command.heading_command = False
    command.rel_standing_envs = 0.0
    command.debug_vis = False
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.episode_length_s = 1.0e6
    env_cfg.sim.visualizer_cfgs = []
    env_cfg.video_recorders = []
    env_cfg.seed = args.seed
    env_cfg.validate()
    cache = None if mpm.streaming is None else mpm.streaming.cache
    fitted = None
    if cache is not None:
        from isaaclab_newton.sim.spawners.mpm.streaming import sample_particle_tile

        per_tile = len(sample_particle_tile(env_cfg.scene.sand.spawn)[0])
        shape = int(round((mpm.streaming.size[0] + 2.0 * mpm.streaming.margin) / mpm.streaming.tile_size))
        path = lemniscate_path(args.lemniscate_a, args.speed, args.laps)
        loaded, tiles = planned_pool_tiles(path, geometry, mpm.streaming.tile_size, shape, slack=args.cache_slack)
        records = (loaded - shape * shape) * per_tile + math.ceil(1.2 * (2 * shape - 1) * per_tile)
        fitted = {"particles_per_tile": per_tile, "loaded_tiles": loaded, "tiles": tiles, "records": records}
        cache.max_visited_tiles_per_env = tiles
        cache.max_cached_particles_per_env = records
        if args.cache_unsettled_template:
            # One settling step and no rest check: new tiles then start from the sampled particles, like full.
            cache.settle_time_s = 0.001
            cache.settle_speed = float("inf")
        env_cfg.validate()
    return {
        "terrain_size_per_env": [length_x, length_y],
        "sand": {
            "depth_m": sand.particle_depth,
            "voxel_m": sand.voxel_size,
            "particles_per_cell": sand.particles_per_cell,
            "material": streaming_scene_cfg.MATERIAL_NAME,
        },
        "cache": None
        if cache is None
        else {
            "max_cached_particles_per_env": cache.max_cached_particles_per_env,
            "max_visited_tiles_per_env": cache.max_visited_tiles_per_env,
            "simulation_capacity_multiplier": cache.simulation_capacity_multiplier,
            "fitted": fitted,
        },
        "sleeping": mpm.sleeping is not None,
    }


def run(args: argparse.Namespace, result: dict, monitor: GpuMonitor) -> None:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    a = args.lemniscate_a
    geometry = {"length_x": 2.0 * a + 2.0 * END_MARGIN, "length_y": a + 2.0 * END_MARGIN}
    # Dense path for the controller; the figure starts at the crossing, heading into the first lobe.
    path = lemniscate_path(a, 0.01 / CONTROL_DT, args.laps)  # one point per centimetre
    yaw0 = math.atan2(path[1, 1] - path[0, 1], path[1, 0] - path[0, 0])
    task = TASKS[args.terrain]
    preset = PRESETS[(args.terrain, "off")]
    result["task"], result["preset"] = task, preset
    env_cfg, agent_cfg = resolve_task_config(
        task, "rsl_rl_cfg_entry_point", overrides=["env.scene.num_envs=1", f"physics={preset}"]
    )
    result["geometry"] = configure(env_cfg, args, geometry, yaw0)
    nominal_steps = int(len(path) * 0.01 / (args.speed * CONTROL_DT))
    max_steps = int(2.0 * nominal_steps)
    launcher = None
    if args.video_dir:
        prefix = f"g1_{args.terrain}_sleep_{args.sleep}_lemniscate"
        launcher = configure_particle_video(
            env_cfg, args.terrain, geometry, args.video_dir, prefix, max_steps, args.video_viz
        )

    with launch_simulation(env_cfg, launcher), contextlib.ExitStack() as cleanup:
        env = gym.make(task, cfg=env_cfg)
        cleanup.callback(env.close)
        base_env = env.unwrapped
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(args.checkpoint, load_cfg={"actor": True})
        policy = runner.get_inference_policy(device=base_env.device)
        obs = wrapped.get_observations()
        if args.video_dir:
            draw_sand_particles(base_env)
        backend = base_env.sim.backend_manager.backend
        model = backend.model
        pool = base_env.mpm_streaming
        cache = getattr(pool, "cache", None)
        result["particle_count"] = int(model.particle_count)
        robot = base_env.scene["robot"]
        feet = robot.find_bodies(".*ankle_roll_link")[0]
        command = base_env.command_manager.get_term("base_velocity")
        env_origin = base_env.scene.env_origins[0].cpu().numpy()
        particle_world = model.particle_world.numpy()
        terrain_lower = env_origin[:2] - np.array([geometry["length_x"], geometry["length_y"]]) / 2.0
        terrain_shape = (int(round(geometry["length_x"] / 0.04)), int(round(geometry["length_y"] / 0.04)))
        world_path = path + env_origin[:2]
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
        lap_length = arc[-1] / args.laps

        def terrain_map() -> np.ndarray:
            """Top-of-sand height over the whole terrain, from the simulated and the cached particles."""
            live = (model.particle_flags.numpy() & 1) != 0
            q = backend.state_0.particle_q.numpy()[live & (particle_world == 0)]
            if cache is not None:
                records = [record for chain in cache.pages[0].values() for record in chain]
                if records:
                    q = np.concatenate((q, cache.saved.q.numpy()[records]))
            return height_map(q, terrain_lower, terrain_shape)

        usage = {"slots": 0, "records": 0, "tiles": 0}
        snapshots = {"start": terrain_map()}
        root_log, feet_log, command_log, step_ms, lap_end = [], [], [], [], {}
        index, resets, lap = 0, 0, 1
        result["stage"] = "run"
        with torch.inference_mode():
            for step in range(max_steps):
                root = robot.data.root_pos_w.torch[0].cpu().numpy()
                quat = robot.data.root_quat_w.torch[0].cpu().numpy()  # (x, y, z, w)
                yaw = math.atan2(
                    2.0 * (quat[3] * quat[2] + quat[0] * quat[1]), 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
                )
                # Progress only moves forward, within a short window, so the crossing cannot switch lobes.
                window = slice(index, min(index + 150, len(world_path)))
                index += int(np.argmin(np.linalg.norm(world_path[window] - root[:2], axis=1)))
                if arc[index] >= arc[-1] - 0.05:
                    break
                ahead = world_path[min(index + int(args.lookahead / 0.01), len(world_path) - 1)]
                bearing = math.atan2(ahead[1] - root[1], ahead[0] - root[0]) - yaw
                bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
                yaw_rate = float(np.clip(args.yaw_gain * bearing, -args.max_yaw_rate, args.max_yaw_rate))
                forward = args.speed * max(0.3, 1.0 - abs(bearing) / 1.0)
                command.vel_command_b[:] = torch.tensor([forward, 0.0, yaw_rate], device=base_env.device)
                _sync()
                t0 = time.perf_counter()
                obs, _, dones, _ = wrapped.step(policy(obs))
                _sync()
                step_ms.append((time.perf_counter() - t0) * 1.0e3)
                policy.reset(dones)
                resets += int(dones.sum())
                root_log.append(robot.data.root_pos_w.torch[0].cpu().numpy())
                feet_log.append(robot.data.body_pos_w.torch[0, feet].cpu().numpy())
                command_log.append([forward, yaw_rate, arc[index]])
                if cache is not None and step % 5 == 0:
                    used = (cache.slot_bump.numpy() - cache.slot_free_count.numpy()).max()
                    usage["slots"] = max(usage["slots"], int(used))
                    usage["records"] = max(usage["records"], max(cache.cached_particle_counts))
                    usage["tiles"] = max(usage["tiles"], int(cache.visited_count.numpy().max()))
                if arc[index] >= lap * lap_length - 0.05 and lap < args.laps:
                    snapshots[f"lap_{lap}"] = terrain_map()
                    lap_end[f"lap_{lap}"] = step + 1
                    lap += 1
        snapshots["end"] = terrain_map()
        if pool is not None:
            pool.check_status()
        result["steps_run"] = len(step_ms)
        result["nominal_steps"] = nominal_steps
        result["lap_end_steps"] = lap_end
        result["resets"] = resets
        result["completed"] = bool(arc[index] >= arc[-1] - 0.05)
        result["path_progress_m"] = float(arc[index])
        result["path_length_m"] = float(arc[-1])
        roots = np.asarray(root_log)
        deviation = np.array([np.linalg.norm(world_path - p[:2], axis=1).min() for p in roots[::10]])
        result["path_deviation_m"] = {"mean": float(deviation.mean()), "max": float(deviation.max())}
        result["step_ms_mean"] = float(np.mean(step_ms[25:]))
        if cache is not None:
            result["cache_usage"] = {
                "slots_peak": usage["slots"],
                "slot_capacity": int(cache.slot_capacity),
                "records_peak": usage["records"],
                "record_capacity": int(cache.record_capacity),
                "tiles_peak": usage["tiles"],
                "tile_capacity": int(cache.max_visited),
            }
        np.savez_compressed(
            os.path.splitext(args.out)[0] + ".npz",
            root=roots,
            feet=np.asarray(feet_log),
            command=np.asarray(command_log),
            path=world_path,
            env_origin=env_origin,
            terrain_lower=terrain_lower,
            **{f"height_terrain_{name}": grid for name, grid in snapshots.items()},
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result: dict = {
        "status": "started",
        "kind": "g1_lemniscate",
        "terrain": args.terrain,
        "sleep": args.sleep,
        "argv": vars(args),
        "stage": "init",
    }
    with GpuMonitor() as monitor:
        try:
            import torch

            result["gpu"] = torch.cuda.get_device_name(0)
            result["git"] = _git_state()
            run(args, result, monitor)
            result["status"] = "ok"
        except BaseException as exc:  # noqa: BLE001 - record any failure
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"[:2000]
            result["traceback"] = traceback.format_exc()[-4000:]
        finally:
            result["peak_process_MiB"] = monitor.process_peak
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    keys = (
        "status",
        "terrain",
        "steps_run",
        "nominal_steps",
        "resets",
        "completed",
        "path_progress_m",
        "path_deviation_m",
        "cache_usage",
        "error",
    )
    print(json.dumps({k: result.get(k) for k in keys}, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
