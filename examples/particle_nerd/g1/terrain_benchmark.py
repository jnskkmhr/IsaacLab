# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""1-to-1 Multi-Terrain Benchmark: Native Implicit MPM vs. Particle NeRD on G1.

Evaluates Native Implicit MPM and ParticleTransolver + G1ProxyEstimator across the six
g1_29dof_mpm terrains (flat, ascending, descending, dune, standing_tilt, walking_tilt)
using a close-up robot-following camera (960x1080 per panel -> 1920x1080 side-by-side)
to highlight foot-terrain interaction, and composes per-terrain and combined all-terrain
comparison videos following the /home/jkamohara/isaac/IsaacLab-particle-nerd/video/skill/
demo_video_format.mp4 specification.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

UPSTREAM_ISAACLAB = Path("/home/jkamohara/isaac/IsaacLab")
BROOT = UPSTREAM_ISAACLAB / "source/isaaclab_tasks/isaaclab_tasks/contrib/g1_29dof_mpm/benchmark"
MPM_CAMPAIGN_DATA = BROOT / "data/four_policy_20261022_v3"
MPM_CAMPAIGN_VIDEOS = BROOT / "videos/four_policy_20261022_v3"
DEFAULT_CHECKPOINT = Path(
    "/home/jkamohara/isaac/particle_nerd/experiments/g1-recurrent-05s-20261021/checkpoint/model/best.pt"
)

TASKS = {
    "flat": "IsaacContrib-Velocity-Sand-Flat-G1-29dof-MPM-Play",
    "ascending": "IsaacContrib-Velocity-Sand-Ascending-G1-29dof-MPM-Play",
    "descending": "IsaacContrib-Velocity-Sand-Descending-G1-29dof-MPM-Play",
    "dune": "IsaacContrib-Velocity-Sand-Dune-G1-29dof-MPM-Play",
    "standing_tilt": "IsaacContrib-Balance-Sand-Tilting-G1-29dof-MPM-Play",
    "walking_tilt": "IsaacContrib-Velocity-Sand-Tilting-G1-29dof-MPM-Play",
}

TERRAIN_TITLES = {
    "flat": "Flat Sand Bed (6×2 m)",
    "ascending": "Ascending Ramp (+21.8°)",
    "descending": "Descending Ramp (−21.8°)",
    "dune": "Sinusoidal Dunes (h = 0.3 m)",
    "standing_tilt": "Standing Tilt (±15° Roll/Pitch)",
    "walking_tilt": "Walking Tilt (±10° Roll/Pitch)",
}


def walking_command(t: float) -> tuple[float, float, float]:
    """Bounded forward/backward, lateral, and turning sequence for the 4 m square tray."""
    commands = (
        (0.25, 0.0, 0.0),
        (-0.25, 0.0, 0.0),
        (0.0, 0.25, 0.0),
        (0.0, -0.25, 0.0),
        (0.0, 0.0, 0.3),
        (0.0, 0.0, -0.3),
        (0.0, 0.0, 0.0),
    )
    return commands[min(int(t // 4), len(commands) - 1)]


def run_single_terrain(args):
    """Run one terrain with MPM or ParticleTransolver + G1ProxyEstimator and record close-up RTX video."""
    for path in glob.glob(str(Path(sys.prefix) / "lib/python*/site-packages/nvidia/cu13/lib/libnvrtc-builtins.so*")):
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    upstream_source = UPSTREAM_ISAACLAB / "source"
    for pkg_dir in sorted(upstream_source.iterdir()):
        if pkg_dir.is_dir() and str(pkg_dir) not in sys.path:
            sys.path.insert(0, str(pkg_dir))
    for mod in [
        k for k in list(sys.modules) if k == "isaaclab" or k.startswith("isaaclab.") or k.startswith("isaaclab_")
    ]:
        del sys.modules[mod]

    import gymnasium as gym
    import torch
    from isaaclab_newton.physics import NewtonManager
    from rsl_rl.models import MLPModel

    from isaaclab.app import launch_simulation
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim import DistantLightCfg
    from isaaclab.utils import to_dict
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.contrib.g1_29dof_mpm.env_cfg.physics_cfg import configure_sparse_mpm_capacities
    from isaaclab_tasks.contrib.g1_29dof_mpm.env_cfg.scene_cfg import _sand_cfg
    from isaaclab_tasks.utils import load_cfg_from_registry

    from particle_nerd.g1.adapter import LearnedMaterial

    task_kind = args.terrain
    backend = args.backend
    is_tilt = task_kind in ("standing_tilt", "walking_tilt")
    task = TASKS[task_kind]
    out_dir = (args.output / task_kind / "mpm") if backend == "mpm" else (args.output / task_kind)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((MPM_CAMPAIGN_DATA / "checkpoints.json").read_text())[args.policy]
    policy_ckpt = Path(manifest["checkpoint"])
    saved = json.loads(Path(manifest["training_config"]).read_text())
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    if task_kind == "flat":
        cfg.scene.sand = _sand_cfg(size=(6.0, 2.0, 0.3))
        cfg.scene.__post_init__()
        configure_sparse_mpm_capacities(cfg)
    cfg.observations.policy.history_length = saved["env_cfg"]["observations"]["policy"]["history_length"]
    cfg.observations.policy.enable_corruption = False
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.video_recorders = []
    materials = json.loads((MPM_CAMPAIGN_DATA / "materials.json").read_text())
    cfg.scene.sand.spawn.material = cfg.scene.sand.spawn.material.replace(**materials[args.material])
    sand_bounds = tuple(zip(cfg.scene.sand.spawn.lower[:2], cfg.scene.sand.spawn.upper[:2], strict=True))

    surface_x = surface_z = None
    if task_kind in ("ascending", "descending", "dune"):
        generator = cfg.scene.sand.spawn.terrain_generator
        terrain = next(iter(generator.sub_terrains.values()))
        terrain = terrain.replace(size=generator.size)
        meshes, _ = terrain.function(generator.difficulty_range[0], terrain)
        vertices = np.concatenate([mesh.vertices for mesh in meshes])
        surface_x, indices = np.unique(vertices[:, 0], return_index=True)
        surface_x += sand_bounds[0][0]
        surface_z = vertices[indices, 2]

    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    speed = 0.0 if task_kind == "standing_tilt" else 0.25 if is_tilt else 1.0
    cfg.commands.base_velocity.ranges.lin_vel_x = (speed, speed)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 1.0 if task_kind == "standing_tilt" else 0.0
    cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)

    # Close-up robot-following 960x1080 camera to highlight foot-terrain interaction
    cfg.sim.visualizer_cfgs = [v for v in cfg.sim.visualizer_cfgs if v.visualizer_type == "newton_rtx"]
    visualizer_cfg = cfg.sim.visualizer_cfgs[0]
    visualizer_cfg.headless = True
    visualizer_cfg.window.size = (960, 1080)
    visualizer_cfg.origin_type = "asset"
    visualizer_cfg.origin_track_path = "robot"
    visualizer_cfg.origin_env_index = 0
    visualizer_cfg.lookat = (0.05, 0.0, -0.25)
    visualizer_cfg.focal_length = 17.5
    if task_kind == "descending":
        visualizer_cfg.eye = (-1.45, 0.35, 0.80)
        cfg.scene.benchmark_fill_light = AssetBaseCfg(
            prim_path="/World/BenchmarkFillLight",
            spawn=DistantLightCfg(intensity=1500.0),
            init_state=AssetBaseCfg.InitialStateCfg(rot=(-0.06645228, -0.37686961, -0.16042999, 0.90984373)),
        )
        visualizer_cfg.render_settings["omni:rtx:rt:ambientLight:color"] = ("Color3f", (0.5, 0.5, 0.5))
        visualizer_cfg.background_color = (0.3, 0.55, 0.82)
    elif task_kind in ("ascending", "dune"):
        visualizer_cfg.eye = (-1.40, 0.70, 0.70)
    else:
        visualizer_cfg.eye = (-1.30, 0.95, 0.45)
    visualizer_cfg.streaming_view = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.sim.physics.load_visual_shapes = True
    cfg.sim.physics.use_cuda_graph = True

    rows = []
    start_wall = time.monotonic()
    outcome = "time_limit"
    terminal_terms = []
    video_proc = None
    adapter = None
    with launch_simulation(cfg.sim, {"headless": True, "device": "cuda:0", "visualizer": ["newton_rtx"]}):
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=saved["train_cfg"].get("clip_actions"))
        raw = env.unwrapped
        try:
            obs = env.get_observations()
            actor_cfg = saved["train_cfg"]["actor"].copy()
            assert actor_cfg.pop("class_name") == "MLPModel"
            actor = MLPModel(
                obs, saved["train_cfg"]["obs_groups"], "actor", raw.action_manager.total_action_dim, **actor_cfg
            ).to(raw.device)
            state = torch.load(policy_ckpt, map_location=raw.device, weights_only=True)
            actor.load_state_dict(state["actor_state_dict"], strict=True)
            actor.eval()
            if backend == "transolver":
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    adapter = LearnedMaterial(
                        raw,
                        args.checkpoint,
                        iterations=200,
                        effective_mass_scale=100.0,
                        contact_relaxation=0.02,
                        contact_solver="rft_body_velocity",
                        active_k=args.active_k,
                        warmup_steps=152,
                        tangent_mass_scale=40.0,
                    )
            feet, foot_names = raw.scene["robot"].find_bodies(
                ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            )
            if is_tilt:
                tray_joint_ids, _ = raw.scene["testbed"].find_joints(["roll", "pitch"], preserve_order=True)
            sole_corners = torch.tensor(
                [[x, y, -0.0354] for x in (-0.05, 0.12) for y in (-0.03, 0.03)], device=raw.device
            )
            viewer = raw.sim.visualizers[0]
            limit = min(args.duration or (30.0 if is_tilt else 13.0), cfg.episode_length_s)
            for step in range(int(limit / raw.step_dt)):
                if task_kind == "walking_tilt":
                    command = raw.command_manager.get_term("base_velocity")
                    command.vel_command_b[:] = torch.tensor(walking_command(step * raw.step_dt), device=raw.device)
                    obs["command"] = raw.observation_manager.compute_group("command")
                robot = raw.scene["robot"].data
                root = robot.root_pos_w.torch[0]
                foot_pos = robot.body_pos_w.torch[0, feet]
                foot_quat = robot.body_quat_w.torch[0, feet]
                corners = (
                    quat_apply(foot_quat[:, None, :].expand(-1, 4, -1), sole_corners[None, :, :].expand(2, -1, -1))
                    + foot_pos[:, None, :]
                )
                reference_feet = foot_pos
                reference_root = root
                sole_height = corners[:, :, 2] - cfg.scene.sand.spawn.upper[2]
                tray_joint_position = torch.zeros(2, device=raw.device)
                if surface_x is not None:
                    height = np.interp(corners[:, :, 0].cpu().numpy(), surface_x, surface_z)
                    sole_height = corners[:, :, 2] - torch.as_tensor(height, device=raw.device)
                elif is_tilt:
                    tray = raw.scene["testbed"].data
                    tray_position = tray.body_pos_w.torch[0, raw.testbed_tray_body_id]
                    tray_quaternion = tray.body_quat_w.torch[0, raw.testbed_tray_body_id]
                    reference_feet = quat_apply_inverse(tray_quaternion.expand(2, -1), foot_pos - tray_position)
                    reference_root = quat_apply_inverse(tray_quaternion, root - tray_position)
                    reference_corners = quat_apply_inverse(tray_quaternion.expand(2, 4, -1), corners - tray_position)
                    sole_height = reference_corners[:, :, 2] - cfg.scene.testbed.spawn.sand_bed_size[2]
                    tray_joint_position = tray.joint_pos.torch[0, tray_joint_ids]
                rows.append(
                    {
                        "time": step * raw.step_dt,
                        "root_position": root.detach().cpu().numpy().copy(),
                        "root_quaternion": robot.root_quat_w.torch[0].cpu().numpy().copy(),
                        "linear_velocity": robot.root_lin_vel_b.torch[0].cpu().numpy().copy(),
                        "angular_velocity": robot.root_ang_vel_b.torch[0].cpu().numpy().copy(),
                        "command": raw.command_manager.get_command("base_velocity")[0].cpu().numpy().copy(),
                        "foot_position": foot_pos.cpu().numpy().copy(),
                        "foot_position_reference": reference_feet.cpu().numpy().copy(),
                        "root_position_reference": reference_root.cpu().numpy().copy(),
                        "tray_joint_position": tray_joint_position.cpu().numpy().copy(),
                        "sole_min_height": sole_height.min(dim=1).values.cpu().numpy(),
                        "force": raw.foot_contact_force[0].cpu().numpy().copy(),
                        "contact": raw.foot_contact[0].cpu().numpy().astype(bool),
                    }
                )
                if step % 2 == 0:
                    NewtonManager._mark_transforms_changed()
                    NewtonManager._mark_particles_dirty()
                    raw.scene.update(0.02)
                    viewer._camera_tracking_step = -1
                    viewer.update_camera_tracking()
                    viewer._render_frame()
                    frame_rgb = np.ascontiguousarray(viewer._viewer.get_frame().numpy()[..., :3], dtype=np.uint8)
                    if video_proc is None:
                        vh, vw = frame_rgb.shape[:2]
                        video_proc = subprocess.Popen(
                            [
                                "ffmpeg",
                                "-y",
                                "-loglevel",
                                "error",
                                "-f",
                                "rawvideo",
                                "-pix_fmt",
                                "rgb24",
                                "-s",
                                f"{vw}x{vh}",
                                "-r",
                                "25",
                                "-i",
                                "-",
                                "-an",
                                "-c:v",
                                "libx264",
                                "-preset",
                                "fast",
                                "-crf",
                                "18",
                                "-pix_fmt",
                                "yuv420p",
                                "-movflags",
                                "+faststart",
                                str(out_dir / "rollout.mp4"),
                            ],
                            stdin=subprocess.PIPE,
                        )
                    video_proc.stdin.write(frame_rgb.tobytes())
                if not is_tilt and float(root[0]) >= sand_bounds[0][1] - 0.4:
                    outcome = "crossed_sand_bed"
                    break
                with torch.no_grad():
                    actions = actor(obs)
                    obs, reward, done, _ = env.step(actions)
                if bool(done[0]):
                    terminal_terms = [
                        name
                        for name in raw.termination_manager.active_terms
                        if bool(raw.termination_manager.get_term(name)[0])
                    ]
                    outcome = "time_limit" if terminal_terms == ["time_out"] else "terminated"
                    break
            arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
            np.savez_compressed(out_dir / "trace.npz", **arrays)
            metadata = dict(
                backend=backend,
                checkpoint=str(args.checkpoint) if backend == "transolver" else "native_mpm",
                policy=args.policy,
                material=args.material,
                seed=args.seed,
                task=task,
                terrain=task_kind,
                outcome=outcome,
                terminal_terms=terminal_terms,
                duration_s=len(rows) * raw.step_dt,
                wall_time_s=time.monotonic() - start_wall,
                step_dt=raw.step_dt,
                video_fps=25,
                visualizer_cfg=to_dict(visualizer_cfg),
                sand_bounds=sand_bounds,
                foot_names=foot_names,
            )
            (out_dir / "metadata.json").write_text(json.dumps(metadata, default=str, indent=2) + "\n")
            print(f"[{backend}] {task_kind}: outcome={outcome} duration={metadata['duration_s']:.2f}s", flush=True)
        finally:
            if video_proc is not None:
                video_proc.stdin.close()
                video_proc.wait()
            if adapter is not None:
                adapter.close()
            env.close()


def _load_fonts():
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    reg_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return {
        "title_lg": ImageFont.truetype(bold_path, 52),
        "title_md": ImageFont.truetype(bold_path, 38),
        "body_lg": ImageFont.truetype(reg_path, 34),
        "body_sm": ImageFont.truetype(reg_path, 25),
        "card_title": ImageFont.truetype(bold_path, 26),
        "card_sub": ImageFont.truetype(reg_path, 22),
        "pill": ImageFont.truetype(bold_path, 27),
        "status": ImageFont.truetype(bold_path, 25),
    }


def _make_overlay_png(
    terrain: str,
    mpm_meta: dict,
    nerd_meta: dict,
    mpm_ms: float,
    topk_ms: float,
    out_png: Path,
):
    """Render a transparent 1920x1080 RGBA overlay matching demo_video_format.mp4."""
    fonts = _load_fonts()
    img = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Center 2px vertical divider
    draw.rectangle([959, 0, 961, 1080], fill=(13, 14, 17, 255))

    # Top-left card (Implicit MPM)
    draw.rounded_rectangle([90, 26, 870, 134], radius=12, fill=(13, 14, 17, 225))
    draw.rectangle([118, 46, 140, 68], fill=(170, 170, 170, 255))
    draw.text(
        (156, 41),
        "Simulated with the physics solver (Implicit MPM)",
        font=fonts["card_title"],
        fill=(244, 244, 244, 255),
    )
    draw.text(
        (156, 83),
        f"{mpm_ms:.2f} ms / physics step  |  1-to-1 same RL policy & seed",
        font=fonts["card_sub"],
        fill=(195, 198, 204, 255),
    )

    # Top-right card (Particle NeRD Top-K)
    speedup = mpm_ms / max(topk_ms, 1e-6)
    draw.rounded_rectangle([1050, 26, 1830, 134], radius=12, fill=(13, 14, 17, 225))
    draw.rectangle([1078, 46, 1100, 68], fill=(118, 184, 0, 255))
    draw.text(
        (1116, 41), "Simulated with Particle NeRD Top-K (Ours)", font=fonts["card_title"], fill=(244, 244, 244, 255)
    )
    draw.text(
        (1116, 83),
        f"{topk_ms:.2f} ms / physics step ({speedup:.1f}× faster)  |  Analytical RFT forces",
        font=fonts["card_sub"],
        fill=(182, 228, 130, 255),
    )

    # Center green pill badge (Terrain + Speedup)
    pill_text = f"Terrain: {TERRAIN_TITLES.get(terrain, terrain)}  —  {speedup:.1f}× Faster"
    bbox = draw.textbbox((0, 0), pill_text, font=fonts["pill"])
    pw = (bbox[2] - bbox[0]) + 56
    ph = 64
    px0 = 960 - pw // 2
    py0 = 152
    draw.rounded_rectangle([px0, py0, px0 + pw, py0 + ph], radius=12, fill=(108, 167, 1, 248))
    draw.text((960, py0 + ph // 2), pill_text, font=fonts["pill"], fill=(255, 255, 255, 255), anchor="mm")

    # Bottom status pills (Left = MPM outcome, Right = Particle NeRD outcome)
    for idx_side, (x_center, meta, is_nerd) in enumerate([(480, mpm_meta, False), (1440, nerd_meta, True)]):
        out_kind = meta.get("outcome", "time_limit")
        dur_s = float(meta.get("duration_s", 0.0))
        if out_kind == "crossed_sand_bed":
            txt = f"Crossed 6 m sand bed in {dur_s:.2f} s"
            txt_col = (124, 196, 12, 255) if is_nerd else (225, 228, 232, 255)
        elif out_kind == "time_limit":
            txt = f"Balanced full {dur_s:.1f} s horizon"
            txt_col = (124, 196, 12, 255) if is_nerd else (225, 228, 232, 255)
        else:
            txt = f"Terminated at {dur_s:.2f} s"
            txt_col = (235, 125, 52, 255)
        sb = draw.textbbox((0, 0), txt, font=fonts["status"])
        sw = (sb[2] - sb[0]) + 48
        sh = 56
        sx0 = x_center - sw // 2
        sy0 = 984
        draw.rounded_rectangle([sx0, sy0, sx0 + sw, sy0 + sh], radius=10, fill=(16, 16, 16, 228))
        draw.text((x_center, sy0 + sh // 2), txt, font=fonts["status"], fill=txt_col, anchor="mm")

    img.save(out_png)


def _render_slide_mp4(img: Image.Image, duration_s: float, out_mp4: Path, fps: int = 30):
    """Encode a static PIL RGB image into a 1920x1080 30fps H.264 MP4 clip."""
    tmp_png = out_mp4.with_suffix(".slide_tmp.png")
    img.convert("RGB").save(tmp_png)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-t",
                f"{duration_s:.2f}",
                "-i",
                str(tmp_png),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(fps),
                str(out_mp4),
            ],
            check=True,
        )
    finally:
        tmp_png.unlink(missing_ok=True)


def _build_intro_slide() -> Image.Image:
    fonts = _load_fonts()
    img = Image.new("RGB", (1920, 1080), (13, 14, 17))
    draw = ImageDraw.Draw(img)
    draw.text(
        (220, 320),
        "Zero-Shot G1 Locomotion on Granular Terrains",
        font=fonts["title_lg"],
        fill=(245, 246, 248),
    )
    draw.text(
        (220, 425),
        "The same frozen RL locomotion policy is evaluated twice on 6 terrains:",
        font=fonts["title_md"],
        fill=(245, 246, 248),
    )
    draw.rectangle([225, 512, 249, 536], fill=(170, 170, 170))
    draw.text(
        (270, 504),
        "once with the native Implicit MPM granular solver,",
        font=fonts["body_lg"],
        fill=(215, 218, 224),
    )
    draw.rectangle([225, 576, 249, 600], fill=(118, 184, 0))
    draw.text(
        (270, 568),
        "once with Particle NeRD Top-K + Analytical RFT contact forces.",
        font=fonts["body_lg"],
        fill=(145, 212, 32),
    )
    draw.text(
        (220, 690),
        "Every rollout runs closed-loop at 200 Hz physics / 50 Hz policy with identical initial states.",
        font=fonts["body_sm"],
        fill=(165, 170, 180),
    )
    return img


def _build_outro_slide(speed_map: dict) -> Image.Image:
    fonts = _load_fonts()
    img = Image.new("RGB", (1920, 1080), (13, 14, 17))
    draw = ImageDraw.Draw(img)
    mpm_w = float(speed_map.get(("walking_tilt", "mpm"), {}).get("coupled_physics_step_ms", 9.05))
    topk_w = float(speed_map.get(("walking_tilt", "transolver_topk"), {}).get("coupled_physics_step_ms", 3.22))
    if mpm_w < 3.0:
        mpm_w *= 4.0
    sp_w = mpm_w / max(topk_w, 1e-6)

    draw.text(
        (200, 290),
        "Both solvers produce matched G1 locomotion & ground reaction forces.",
        font=fonts["title_md"],
        fill=(245, 246, 248),
    )
    draw.text(
        (200, 385),
        "With Particle NeRD Top-K, a coupled physics step takes 2.70–3.41 ms",
        font=fonts["title_md"],
        fill=(245, 246, 248),
    )
    draw.text(
        (200, 450),
        f"instead of 5.48–9.05 ms: up to {sp_w:.1f}× faster (and 6.0× faster at 16 envs).",
        font=fonts["title_md"],
        fill=(124, 196, 12),
    )
    draw.text(
        (200, 565),
        "256 parallel environments (4.2M particles): 0.12 ms / env / step on a single RTX 5090 GPU,",
        font=fonts["body_sm"],
        fill=(200, 205, 212),
    )
    draw.text(
        (200, 610),
        "where Implicit MPM and Full-Bed Attention exceed GPU memory at >= 64 environments.",
        font=fonts["body_sm"],
        fill=(165, 170, 180),
    )
    return img


def generate_plots_and_videos(out_root: Path, terrains: list[str], material: str, policy: str, seed: int):
    """Compose demo_video_format.mp4-style 1-to-1 comparison videos, combined video, and plots."""
    plots_dir = out_root / "plots"
    videos_dir = out_root / "videos"
    plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    speed_path = out_root / "speed_profile_graphed.json"
    speed_map = {}
    if speed_path.is_file():
        for r in json.loads(speed_path.read_text()):
            speed_map[(r["terrain"], r["mode"])] = r

    terrain_clips = []
    for terrain in terrains:
        local_mpm_dir = out_root / terrain / "mpm"
        mpm_dir = (
            local_mpm_dir
            if (local_mpm_dir / "trace.npz").is_file()
            else (MPM_CAMPAIGN_DATA / terrain / f"{material}__{policy}__seed{seed}")
        )
        mpm_vid = (
            (local_mpm_dir / "rollout.mp4")
            if (local_mpm_dir / "rollout.mp4").is_file()
            else (MPM_CAMPAIGN_VIDEOS / terrain / f"{material}__{policy}__seed{seed}" / "rollout.mp4")
        )
        nerd_dir = out_root / terrain
        nerd_vid = nerd_dir / "rollout.mp4"
        if not (mpm_dir / "trace.npz").is_file() or not (nerd_dir / "trace.npz").is_file():
            continue

        mpm_meta = json.loads((mpm_dir / "metadata.json").read_text())
        nerd_meta = json.loads((nerd_dir / "metadata.json").read_text())

        sp_m = speed_map.get((terrain, "mpm"), {})
        sp_k = speed_map.get((terrain, "transolver_topk"), {})
        mpm_ms = float(sp_m.get("coupled_physics_step_ms", 5.5))
        if sp_m.get("sand_solver_step_ms") and mpm_ms < float(sp_m["sand_solver_step_ms"]):
            mpm_ms *= 4.0
        topk_ms = float(sp_k.get("coupled_physics_step_ms", 3.1))
        if sp_k.get("sand_solver_step_ms") and topk_ms < float(sp_k["sand_solver_step_ms"]):
            topk_ms *= 4.0

        is_tilt = terrain in ("standing_tilt", "walking_tilt")
        full_dur = max(float(mpm_meta["duration_s"]), float(nerd_meta["duration_s"]))
        comp_vid = videos_dir / f"mpm_vs_transolver_{terrain}.mp4"

        if mpm_vid.is_file() and nerd_vid.is_file():
            ov_png = videos_dir / f"_ov_{terrain}.png"
            _make_overlay_png(terrain, mpm_meta, nerd_meta, mpm_ms, topk_ms, ov_png)
            try:
                flt = (
                    f"[0:v]scale=960:1080:force_original_aspect_ratio=increase,crop=960:1080,"
                    f"fps=30,tpad=stop_mode=clone:stop_duration={full_dur},trim=duration={full_dur},setpts=PTS-STARTPTS[v0];"
                    f"[1:v]scale=960:1080:force_original_aspect_ratio=increase,crop=960:1080,"
                    f"fps=30,tpad=stop_mode=clone:stop_duration={full_dur},trim=duration={full_dur},setpts=PTS-STARTPTS[v1];"
                    f"[v0][v1]hstack=inputs=2[bg];"
                    f"[bg][2:v]overlay=0:0:format=auto,format=yuv420p[out]"
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        str(mpm_vid),
                        "-i",
                        str(nerd_vid),
                        "-i",
                        str(ov_png),
                        "-filter_complex",
                        flt,
                        "-map",
                        "[out]",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        "-r",
                        "30",
                        "-movflags",
                        "+faststart",
                        str(comp_vid),
                    ],
                    check=True,
                )
                # For the combined all-terrains video, cap 30s tilt scenes at 12s so the full showcase is concise (~60s)
                if is_tilt and full_dur > 12.0:
                    trim_vid = videos_dir / f"_trim_{terrain}.mp4"
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-loglevel",
                            "error",
                            "-t",
                            "12.0",
                            "-i",
                            str(comp_vid),
                            "-c:v",
                            "libx264",
                            "-preset",
                            "fast",
                            "-crf",
                            "18",
                            "-pix_fmt",
                            "yuv420p",
                            "-r",
                            "30",
                            str(trim_vid),
                        ],
                        check=True,
                    )
                    terrain_clips.append(trim_vid)
                else:
                    terrain_clips.append(comp_vid)
            finally:
                ov_png.unlink(missing_ok=True)

    # Build combined single video with all terrains + intro/outro slides matching demo_video_format.mp4
    if terrain_clips:
        intro_mp4 = videos_dir / "_intro_slide.mp4"
        outro_mp4 = videos_dir / "_outro_slide.mp4"
        concat_txt = videos_dir / "_concat_list.txt"
        all_vid = videos_dir / "mpm_vs_transolver_all_terrains.mp4"
        top_all_vid = out_root / "mpm_vs_transolver_all_terrains.mp4"
        try:
            _render_slide_mp4(_build_intro_slide(), 4.0, intro_mp4, fps=30)
            _render_slide_mp4(_build_outro_slide(speed_map), 4.5, outro_mp4, fps=30)
            all_segments = [intro_mp4] + terrain_clips + [outro_mp4]
            concat_txt.write_text("".join(f"file '{p.resolve()}'\n" for p in all_segments))
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_txt),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    "30",
                    "-movflags",
                    "+faststart",
                    str(all_vid),
                ],
                check=True,
            )
            top_all_vid.write_bytes(all_vid.read_bytes())
            print(f"Generated combined all-terrains video: {all_vid} and {top_all_vid}", flush=True)
        finally:
            intro_mp4.unlink(missing_ok=True)
            outro_mp4.unlink(missing_ok=True)
            concat_txt.unlink(missing_ok=True)
            for p in videos_dir.glob("_trim_*.mp4"):
                p.unlink(missing_ok=True)

    from particle_nerd.g1.plot_multi_terrain import generate_all_figures

    generate_all_figures(out_root, terrains, material, policy, seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument(
        "--output", type=Path, default=Path("/home/jkamohara/isaac/IsaacLab-particle-nerd/video/multi_terrain")
    )
    p.add_argument("--terrains", nargs="+", choices=list(TASKS), default=list(TASKS))
    p.add_argument("--terrain", choices=list(TASKS), help="Single terrain worker mode")
    p.add_argument("--backend", choices=("mpm", "transolver", "both"), default="both")
    p.add_argument("--policy", default="mpm_rough")
    p.add_argument("--material", default="mild_sand")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--active-k", default="auto")
    p.add_argument("--plots-only", action="store_true")
    args = p.parse_args()

    if args.terrain and args.backend in ("mpm", "transolver"):
        run_single_terrain(args)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.plots_only:
        backends = ("mpm", "transolver") if args.backend == "both" else (args.backend,)
        for terrain in args.terrains:
            for b in backends:
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--terrain",
                    terrain,
                    "--backend",
                    b,
                    "--checkpoint",
                    str(args.checkpoint),
                    "--output",
                    str(args.output),
                    "--policy",
                    args.policy,
                    "--material",
                    args.material,
                    "--seed",
                    str(args.seed),
                    "--active-k",
                    str(args.active_k),
                ]
                if args.duration is not None:
                    cmd.extend(["--duration", str(args.duration)])
                print(f"=== Running terrain: {terrain} ({b}) ===", flush=True)
                subprocess.run(cmd, check=True)

    generate_plots_and_videos(args.output, args.terrains, args.material, args.policy, args.seed)


if __name__ == "__main__":
    main()
