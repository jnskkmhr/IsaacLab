# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a torch-pickled retarget motion (dict of global_translation/global_rotation/dof_pos/...)
into the wbc motion npz format expected by MotionLoader (wbc/mdp/commands.py).

The retarget source may have fewer DOF/bodies than the target G1 29dof robot (e.g. missing
waist_roll/waist_pitch). Missing joints are held at the robot's default pose, and missing body
poses are filled in via real forward kinematics through the actual robot articulation (not a
hand-approximated stand-in), by writing the known root pose + joint angles into the simulated
robot and reading back body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w for every body.

.. code-block:: bash

    python convert_retarget_to_npz.py -f leap_g1_retargeted.npy --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert a retargeted motion (.npy torch pickle) to a wbc motion npz.")
parser.add_argument("--input_file", "-f", type=str, required=True, help="Path to the input retarget .npy file.")
parser.add_argument("--output_name", type=str, help="Path to the output npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.output_name:
    args_cli.output_name = args_cli.input_file.rsplit(".", 1)[0] + ".npz"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_slerp

from isaaclab_tasks.contrib.mimic.config.g1_29dof.env_cfg.scene_cfg import G1SceneCfg

ROBOT_CFG = G1SceneCfg().robot

# target 29-joint order used by g1_29dof(_gm) (matches action_cfg.py / convert_csv_to_npz.py)
TARGET_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# retarget dof_names (no "_joint" suffix, "_pitch" on knee/elbow) -> target joint name
DOF_NAME_MAP = {
    "left_knee_pitch": "left_knee_joint",
    "right_knee_pitch": "right_knee_joint",
    "left_elbow_pitch": "left_elbow_joint",
    "right_elbow_pitch": "right_elbow_joint",
}


def to_target_joint_name(dof_name: str) -> str:
    return DOF_NAME_MAP.get(dof_name, f"{dof_name}_joint")


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return a * (1 - blend) + b * blend


def slerp_batch(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(a)
    for i in range(a.shape[0]):
        out[i] = quat_slerp(a[i], b[i], blend[i])  # type: ignore
    return out


def resample(root_pos, root_quat, dof_pos, input_fps: int, output_fps: int):
    num_frames = root_pos.shape[0]
    duration = (num_frames - 1) / input_fps
    times = torch.arange(0, duration, 1.0 / output_fps)
    phase = times / duration
    idx0 = (phase * (num_frames - 1)).floor().long()
    idx1 = torch.minimum(idx0 + 1, torch.tensor(num_frames - 1))
    blend = (phase * (num_frames - 1) - idx0).unsqueeze(-1)
    return (
        lerp(root_pos[idx0], root_pos[idx1], blend),
        slerp_batch(root_quat[idx0], root_quat[idx1], blend.squeeze(-1)),
        lerp(dof_pos[idx0], dof_pos[idx1], blend),
    )


def run(sim: SimulationContext, scene: InteractiveScene):
    data = torch.load(args_cli.input_file, map_location="cpu", weights_only=False)
    input_fps = data["fps"]

    # root = index 0 body in the retarget (pelvis)
    root_pos_in = data["global_translation"][:, 0].to(torch.float32)
    root_quat_xyzw = data["global_rotation"][:, 0].to(torch.float32)
    root_quat_in = root_quat_xyzw  # The source and this backend both use xyzw.

    dof_names_in = [to_target_joint_name(n) for n in data["dof_names"]]
    dof_pos_in_raw = data["dof_pos"].to(torch.float32)
    # reorder/expand retarget dof_pos into TARGET_JOINT_NAMES order, default 0 for missing joints
    dof_pos_in = torch.zeros((dof_pos_in_raw.shape[0], len(TARGET_JOINT_NAMES)), dtype=torch.float32)
    missing = []
    for j, name in enumerate(TARGET_JOINT_NAMES):
        if name in dof_names_in:
            dof_pos_in[:, j] = dof_pos_in_raw[:, dof_names_in.index(name)]
        else:
            missing.append(name)
    print(f"[INFO] Joints missing from retarget (defaulted to 0): {missing}")

    root_pos, root_quat, dof_pos = resample(root_pos_in, root_quat_in, dof_pos_in, input_fps, args_cli.output_fps)
    num_frames = root_pos.shape[0]
    print(f"[INFO] Resampled {input_fps}fps -> {args_cli.output_fps}fps, {num_frames} frames")

    robot = scene["robot"]
    root_pos = root_pos.to(robot.device)
    root_quat = root_quat.to(robot.device)
    dof_pos = dof_pos.to(robot.device)
    robot_joint_indexes = robot.find_joints(TARGET_JOINT_NAMES, preserve_order=True)[0]

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    dt = 1.0 / args_cli.output_fps
    frame_idx = 0
    while simulation_app.is_running() and frame_idx < num_frames:
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = root_pos[frame_idx]
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = root_quat[frame_idx]
        # finite-difference root velocity (0 at endpoints)
        if 0 < frame_idx < num_frames - 1:
            root_states[:, 7:10] = (root_pos[frame_idx + 1] - root_pos[frame_idx - 1]) / (2 * dt)
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = dof_pos[frame_idx]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # kinematics only, no physics step
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())
        frame_idx += 1

    # finite-difference joint_vel from the logged joint_pos (smoother than zeroed default)
    jp = np.stack(log["joint_pos"], axis=0)
    log["joint_vel"] = np.gradient(jp, dt, axis=0).astype(np.float32)
    for k in ("joint_pos", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        log[k] = np.stack(log[k], axis=0)

    np.savez(
        args_cli.output_name,
        joint_names=np.array(robot.joint_names),
        body_names=np.array(robot.body_names),
        quaternion_order=np.array("xyzw"),
        **log,
    )
    print("[INFO]: Motion npz file saved to", args_cli.output_name)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    run(sim, scene)
    sys.exit(0)  # exit cleanly to avoid hanging in interactive mode


if __name__ == "__main__":
    main()
    simulation_app.close()
