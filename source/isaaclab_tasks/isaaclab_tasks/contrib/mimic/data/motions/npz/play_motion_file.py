# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a WBC motion from an npz file.

.. code-block:: bash

    uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/data/motions/npz/play_motion_file.py \
        --motion_file source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/data/motions/npz/g1_TO_jump_forward.npz

Untagged motion files use the legacy PhysX joint order. Files with ``joint_names``
metadata are reordered by name for Newton. No Isaac Sim Kit installation is required.

Pass ``--video`` to save the replay to an mp4 instead of only viewing it. This works headless:

.. code-block:: bash

    uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/data/motions/npz/play_motion_file.py \
        --motion_file .../leap_g1_retargeted.npz --video --headless
"""

from __future__ import annotations

import argparse
import os
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
    default=0,
    help="Body index in body_pos_w/body_quat_w to use as the articulation root pose.",
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

import imageio.v2 as imageio
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import convert_quat

from isaaclab_assets.robots.unitree import UNITREE_G1_29DOF_CFG


@configclass
class ReplayNpzSceneCfg(InteractiveSceneCfg):
    """Scene used to view a saved WBC npz motion."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


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
        self.motion_file = motion_file
        self.fps = int(np.asarray(data["fps"]).reshape(-1)[0])
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
        self.current_idx = 0

        print(f"[INFO]: Motion loaded: {motion_file}")
        print(f"[INFO]: fps={self.fps}, frames={self.num_frames}, duration={(self.num_frames - 1) / self.fps:.3f}s")

    def next_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frame = (
            self.joint_pos[self.current_idx],
            self.joint_vel[self.current_idx],
            self.body_pos_w[self.current_idx],
            self.body_quat_w[self.current_idx],
        )
        self.current_idx = (self.current_idx + 1) % self.num_frames
        return frame


class ViewportRecorder:
    """Record RGB frames from the Newton GL visualizer."""

    def __init__(self, sim: SimulationContext, video_path: str, fps: int):
        self.visualizer = next(viz for viz in sim.visualizers if viz.cfg.visualizer_type == "newton_gl")
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        self.video_path = video_path
        self.writer = imageio.get_writer(video_path, fps=fps, macro_block_size=16)
        self.num_frames = 0

    def capture(self) -> bool:
        """Append the current RGB frame, returning whether capture succeeded."""
        rgb = self.visualizer.render_rgb_array()
        if rgb is None or rgb.size == 0:
            return False
        self.writer.append_data(rgb[:, :, :3])
        self.num_frames += 1
        return True

    def close(self):
        self.writer.close()
        print(f"[INFO]: Saved {self.num_frames} frames to {os.path.abspath(self.video_path)}")


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

    if args_cli.root_body_index >= motion.body_pos_w.shape[1]:
        raise ValueError(
            f"root_body_index={args_cli.root_body_index} is out of range for {motion.body_pos_w.shape[1]} saved bodies."
        )

    joint_indices = [motion.joint_names.index(name) for name in robot.joint_names]

    recorder = None
    if args_cli.video:
        video_length = args_cli.video_length if args_cli.video_length > 0 else motion.num_frames
        video_name = os.path.splitext(os.path.basename(args_cli.motion_file))[0] + ".mp4"
        recorder = ViewportRecorder(sim, os.path.join(args_cli.video_folder, video_name), motion.fps)
        print(f"[INFO]: Recording {video_length} frames at {motion.fps} fps.")

    try:
        while sim.is_headless_or_exist_active_visualizer():
            joint_pos_frame, joint_vel_frame, body_pos_frame, body_quat_frame = motion.next_frame()

            root_states = robot.data.default_root_pose.torch.clone()
            root_states[:, :3] = body_pos_frame[args_cli.root_body_index]
            root_states[:, :2] += scene.env_origins[:, :2]
            root_states[:, 3:7] = body_quat_frame[args_cli.root_body_index]
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

            sim.render()
            scene.update(sim.get_physics_dt())

            if recorder is not None:
                if not recorder.capture():
                    # renderer not ready yet, so replay this frame again rather than dropping it
                    motion.current_idx = (motion.current_idx - 1) % motion.num_frames
                elif recorder.num_frames >= video_length:
                    return
    finally:
        if recorder is not None:
            recorder.close()


def main():
    with launch_simulation(cfg=NewtonCfg(solver_cfg=MJWarpSolverCfg()), launcher_args=args_cli) as physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(
            device=args_cli.device,
            physics=physics_cfg,
            dt=1.0 / NpzMotion(args_cli.motion_file, "cpu", args_cli.quaternion_order).fps,
            visualizer_cfgs=[
                NewtonGLVisualizerCfg(
                    window_width=args_cli.video_resolution[0],
                    window_height=args_cli.video_resolution[1],
                    headless=args_cli.headless,
                )
            ],
        )
        sim = SimulationContext(sim_cfg)
        scene_cfg = ReplayNpzSceneCfg(num_envs=1, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        print("[INFO]: Setup complete.")
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
