# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a WBC motion from an npz file.

.. code-block:: bash

    uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/data/motions/play_motion_file.py \
        --motion_file source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/data/motions/data/claw/g1_TO_jump_forward.npz

Untagged motion files use the legacy PhysX joint order. Files with ``joint_names``
metadata are reordered by name for Newton. No Isaac Sim Kit installation is required.
Terrain stored in the NPZ is loaded automatically. Use ``--terrain_mode scene``
to ignore it, or ``--terrain_mode motion`` to require paired terrain.
Interactive playback follows the saved FPS and repeats by default. Pass ``--no-loop``
to hold the final pose; the transition from the last frame to the first is not blended.

Pass ``--video`` to save the replay to an mp4 instead of only viewing it. This works headless:

.. code-block:: bash

    uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/data/motions/play_motion_file.py \
        --motion_file .../leap_g1_retargeted.npz --video --headless
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Replay a WBC npz motion with Newton MJWarp and OpenGL.")
parser.add_argument("--motion_file", "-f", type=str, required=True, help="Path to the motion npz file.")
parser.add_argument(
    "--quaternion_order",
    choices=("wxyz", "xyzw"),
    default="wxyz",
    help="Order for untagged motion files; NPZ quaternion_order metadata takes precedence.",
)
parser.add_argument(
    "--root_body_index",
    type=int,
    default=None,
    help="Override the root body index; otherwise resolve its name from NPZ metadata (legacy default: 0).",
)
parser.add_argument(
    "--terrain_mode",
    choices=("auto", "motion", "scene"),
    default="auto",
    help="auto: load NPZ terrain when present; motion: require it; scene: use the default plane.",
)
parser.add_argument(
    "--loop",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Repeat the motion (default); --no-loop holds the final pose.",
)
parser.add_argument("--camera_distance", type=float, default=1.5, help="Camera distance from the robot.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the replay.")
parser.add_argument(
    "--video_length",
    type=int,
    default=0,
    help="Length of the recorded video (in frames). Defaults to one full pass of the motion.",
)
parser.add_argument("--video_folder", type=str, default="logs/mimic", help="Directory to write the recorded video to.")
parser.add_argument(
    "--video_resolution", type=int, nargs=2, default=(1280, 720), help="Width and height of the recorded video."
)

add_launcher_args(parser)
parser.add_argument("--headless", action="store_true", help="Render offscreen with Newton GL.")
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs.utils.video_recorder import VideoRecorder
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import convert_quat
from isaaclab.visualizers import WindowCfg

from isaaclab_tasks.contrib.mimic.utils import configure_scene_motion_paired_terrain, motion_files_use_terrain

from isaaclab_assets.robots.unitree import UNITREE_G1_29DOF_BOX_FOOT_CFG


@configclass
class ReplayNpzSceneCfg(InteractiveSceneCfg):
    """Scene used to view a saved WBC npz motion."""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    robot: ArticulationCfg = UNITREE_G1_29DOF_BOX_FOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# Legacy NPZ files store joints in PhysX articulation order.
LEGACY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


class NpzMotion:
    """Small loader for the npz files consumed by the WBC motion command."""

    def __init__(self, motion_file: str, device: str, quaternion_order: Literal["wxyz", "xyzw"] = "wxyz"):
        data = np.load(motion_file, allow_pickle=False)
        required_keys = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise KeyError(f"Motion file is missing required keys: {missing_keys}")

        self.joint_names = list(data["joint_names"]) if "joint_names" in data else list(LEGACY_JOINT_NAMES)
        self.body_names = list(data["body_names"]) if "body_names" in data else None
        self.motion_file = motion_file
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if not np.isfinite(self.fps) or self.fps <= 0:
            data.close()
            raise ValueError(f"Motion FPS must be finite and positive, got {self.fps}.")
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self.body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self.body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        if "quaternion_order" in data:
            quaternion_order = str(np.asarray(data["quaternion_order"]).item())
        if quaternion_order not in ("wxyz", "xyzw"):
            data.close()
            raise ValueError(f"Unsupported motion quaternion order: {quaternion_order!r}")
        if quaternion_order == "wxyz":
            self.body_quat_w = convert_quat(self.body_quat_w, to="xyzw")
        data.close()
        self.num_frames = self.joint_pos.shape[0]
        if self.num_frames == 0:
            raise ValueError("Motion must contain at least one frame.")
        self.current_idx = 0

        print(f"[INFO]: Motion loaded: {motion_file}")
        print(f"[INFO]: fps={self.fps}, frames={self.num_frames}, duration={(self.num_frames - 1) / self.fps:.3f}s")

    def next_frame(self, *, loop: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frame = (
            self.joint_pos[self.current_idx],
            self.joint_vel[self.current_idx],
            self.body_pos_w[self.current_idx],
            self.body_quat_w[self.current_idx],
        )
        if loop:
            self.current_idx = (self.current_idx + 1) % self.num_frames
        else:
            self.current_idx = min(self.current_idx + 1, self.num_frames - 1)
        return frame


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Stream the saved motion state into the robot and render it."""

    motion = NpzMotion(args_cli.motion_file, sim.device, args_cli.quaternion_order)
    robot = scene["robot"]

    print("[INFO]: Isaac Lab articulation joint index order:")
    for index, name in enumerate(robot.joint_names):
        print(f"  joint_pos[{index:02d}] = {name}")

    print("[INFO]: Isaac Lab articulation body index order:")
    for index, name in enumerate(robot.body_names):
        print(f"  body_pos_w[{index:02d}] = {name}")

    if motion.joint_pos.shape[1] != robot.num_joints:
        raise ValueError(
            f"Motion has {motion.joint_pos.shape[1]} joints, but robot has {robot.num_joints}. "
            "This viewer expects npz joint_pos/joint_vel in the robot joint order."
        )

    root_body_index = args_cli.root_body_index
    if root_body_index is None:
        root_body_index = motion.body_names.index(robot.body_names[0]) if motion.body_names is not None else 0
    if not 0 <= root_body_index < motion.body_pos_w.shape[1]:
        raise ValueError(
            f"root_body_index={root_body_index} is out of range for {motion.body_pos_w.shape[1]} saved bodies."
        )

    joint_indices = [motion.joint_names.index(name) for name in robot.joint_names]

    recorder = None
    if args_cli.video:
        video_length = args_cli.video_length if args_cli.video_length > 0 else motion.num_frames
        recorder_cfg = VideoRecorderCfg(
            source="visualizer:newton_gl",
            output_dir=args_cli.video_folder,
            video_length=video_length,
            video_interval=0,
            output_filename_prefix=Path(args_cli.motion_file).stem,
        )
        # Standalone replay supplies the context normally provided by the environment.
        recorder = VideoRecorder(recorder_cfg, SimpleNamespace(sim=sim, scene=scene, step_dt=1.0 / motion.fps))
        print(f"[INFO]: Recording {video_length} frames at {motion.fps} fps.")

    replay_steps = 0
    last_render_time = None
    try:
        while sim.is_running():
            joint_pos_frame, joint_vel_frame, body_pos_frame, body_quat_frame = motion.next_frame(loop=args_cli.loop)

            root_states = robot.data.default_root_pose.torch.clone()
            root_states[:, :3] = body_pos_frame[root_body_index]
            root_states[:, :3] += scene.env_origins
            root_states[:, 3:7] = body_quat_frame[root_body_index]
            robot.write_root_pose_to_sim_index(root_pose=root_states[:, :7])
            robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch)

            joint_pos = robot.data.default_joint_pos.torch.clone()
            joint_vel = robot.data.default_joint_vel.torch.clone()
            joint_pos[:] = joint_pos_frame[joint_indices]
            joint_vel[:] = joint_vel_frame[joint_indices]
            robot.write_joint_position_to_sim_index(position=joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=joint_vel)

            # aim the camera before rendering so the captured frame matches the pose being shown
            pos_lookat = root_states[0, :3].cpu().numpy()
            camera_offset = np.array([args_cli.camera_distance, args_cli.camera_distance, 0.2])
            sim.set_camera_view(pos_lookat + camera_offset, pos_lookat)

            # Pace the interactive window; headless exports use the recorded FPS.
            # Never skip reference frames or catch up in a burst after a slow render.
            if not args_cli.headless:
                if last_render_time is not None:
                    delay = last_render_time + 1.0 / motion.fps - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                last_render_time = time.perf_counter()
            sim.render()
            scene.update(sim.get_physics_dt())

            if recorder is not None:
                recorder.step()
                replay_steps += 1
                if replay_steps >= video_length:
                    return
    finally:
        if recorder is not None:
            recorder.close()


def main():
    with launch_simulation(cfg=NewtonCfg(solver_cfg=MJWarpSolverCfg()), launcher_args=args_cli) as physics_cfg:
        motion_fps = NpzMotion(args_cli.motion_file, "cpu", args_cli.quaternion_order).fps
        sim_cfg = sim_utils.SimulationCfg(
            device=args_cli.device,
            physics=physics_cfg,
            dt=1.0 / motion_fps,
            visualizer_cfgs=[
                NewtonGLVisualizerCfg(
                    window=WindowCfg(size=tuple(args_cli.video_resolution), fps=motion_fps),
                    headless=args_cli.headless,
                )
            ],
        )
        sim = SimulationContext(sim_cfg)
        scene_cfg = ReplayNpzSceneCfg(num_envs=1, env_spacing=2.0)
        if motion_files_use_terrain([args_cli.motion_file], args_cli.terrain_mode):
            configure_scene_motion_paired_terrain(scene_cfg, [args_cli.motion_file])
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        print("[INFO]: Setup complete.")
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
