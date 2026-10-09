# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert G1 LAFAN1 robot CSV with the repo loader and Newton FK. No training.

Run this script from a Newton-enabled source checkout.
Input is a 30 Hz G1 CSV; output is a named 60 Hz NPZ. The original converter's MotionLoader
and G1 joint list are reused, without executing its Isaac Sim AppLauncher.
The Newton adapter evaluates FK without stepping physics, and differentiates
the resulting body link poses to obtain world-space reference velocities.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

TASK = "IsaacContrib-Mimic-G1-29dof-v0"
CONVERTER = "source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/data/motions/csv/convert_csv_to_npz.py"


def validate_motion_arrays(data: Mapping[str, np.ndarray], tracked_bodies: Sequence[str] = ()) -> int:
    """Validate named G1 motion arrays before training or prescribed playback.

    Args:
        data: NPZ arrays with 29 joints, 60 Hz timing and xyzw orientations.
            Positions and linear velocities use [m] and [m/s]; joint positions
            and angular velocities use [rad] and [rad/s].
        tracked_bodies: Body names required by the consuming task.

    Returns:
        Number of reference frames.

    Raises:
        ValueError: Motion dimensions, names, timing or orientations are invalid.
    """
    names = data["joint_names"].tolist()
    bodies = data["body_names"].tolist()
    frames = data["joint_pos"].shape[0]
    if frames < 3 or len(names) != 29 or len(set(names)) != 29:
        raise ValueError("Invalid frame count or joint metadata")
    if len(bodies) != len(set(bodies)) or "pelvis" not in bodies:
        raise ValueError("Invalid body metadata")
    if set(tracked_bodies) - set(bodies):
        raise ValueError("Reference is missing bodies required by the training task")
    if float(np.asarray(data["fps"]).reshape(-1)[0]) != 60:
        raise ValueError("Expected reference frequency of 60 Hz")
    if np.asarray(data["quaternion_order"]).item() != "xyzw":
        raise ValueError("Expected explicitly tagged xyzw quaternions")
    shapes = {
        "joint_pos": (frames, 29),
        "joint_vel": (frames, 29),
        "body_pos_w": (frames, len(bodies), 3),
        "body_quat_w": (frames, len(bodies), 4),
        "body_lin_vel_w": (frames, len(bodies), 3),
        "body_ang_vel_w": (frames, len(bodies), 3),
    }
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid shape or nonfinite values: {key}")
    if not np.allclose(np.linalg.norm(data["body_quat_w"], axis=-1), 1, atol=1e-4):
        raise ValueError("Invalid body quaternion norms")
    return frames


def main():
    """Parse conversion paths and write a validated NPZ and provenance report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="G1 robot CSV at 30 Hz")
    parser.add_argument("--output_dir", type=Path, required=True, help="New directory for the NPZ and report")
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Optional 1-based inclusive source CSV frame range",
    )
    opts = parser.parse_args()
    opts.repo = Path(__file__).resolve().parents[6]
    opts.csv = opts.csv.expanduser().resolve()
    opts.output_dir = opts.output_dir.expanduser().resolve()
    if not opts.csv.is_file():
        raise FileNotFoundError(opts.csv)
    opts.source_commit = subprocess.check_output(["git", "-C", str(opts.repo), "rev-parse", "HEAD"], text=True).strip()
    os.chdir(opts.repo)
    opts.output_dir.mkdir(parents=True, exist_ok=False)
    print("LAFAN_OUTPUT_DIR:", opts.output_dir, flush=True)
    _convert_motion(opts)
    print("CONVERSION_ONLY_FINISHED: no training started; visual review still required", flush=True)


def _original_loader_parts(source: str) -> tuple[str, list[str]]:
    """Extract only the original loader class and its explicit G1 joint order."""
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MotionLoader"]
    runs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_simulator"]
    if len(classes) != 1 or len(runs) != 1:
        raise ValueError("Unexpected original converter structure")
    assignments = [
        n
        for n in runs[0].body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "joint_names" for t in n.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("Expected one explicit joint_names list")
    names = ast.literal_eval(assignments[0].value)
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("Expected 29 unique source joint names")
    return ast.get_source_segment(source, classes[0]), names


def _world_angular_velocity(quat: np.ndarray, dt: float) -> np.ndarray:
    """Differentiate xyzw orientations into world angular velocities [rad/s]."""
    q = quat / np.linalg.norm(quat, axis=-1, keepdims=True)

    def difference(nxt: np.ndarray, prev: np.ndarray, spacing: float) -> np.ndarray:
        vector = -nxt[..., 3:] * prev[..., :3] + prev[..., 3:] * nxt[..., :3]
        vector -= np.cross(nxt[..., :3], prev[..., :3])
        scalar = nxt[..., 3:] * prev[..., 3:] + np.sum(nxt[..., :3] * prev[..., :3], axis=-1, keepdims=True)
        sign = np.where(scalar < 0, -1, 1)
        vector, scalar = vector * sign, scalar * sign
        length = np.linalg.norm(vector, axis=-1, keepdims=True)
        angle = 2 * np.arctan2(length, scalar)
        scale = np.divide(angle, length, out=np.full_like(length, 2), where=length > 1e-12)
        return vector * scale / spacing

    omega = np.empty((*q.shape[:-1], 3), dtype=np.float64)
    omega[1:-1] = difference(q[2:], q[:-2], 2 * dt)
    omega[:1] = difference(q[1:2], q[:1], dt)
    omega[-1:] = difference(q[-1:], q[-2:-1], dt)
    return omega


def _convert_motion(opts: argparse.Namespace):
    """Convert the selected CSV interval using the task robot and Newton FK."""
    # Delay simulator imports until after CLI parsing so --help needs no GPU runtime.
    # Set Warp configuration before importing the task packages, as the training CLI does.
    print("LAFAN_STAGE: importing Newton dependencies", flush=True)
    import warp as wp

    wp.config.enable_backward = False
    import torch

    import isaaclab.sim
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import build_simulation_context
    from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.contrib.mimic.mdp.commands import MotionLoader as NpzLoader
    from isaaclab_tasks.utils import resolve_task_config

    for module in (isaaclab.sim, isaaclab_tasks):
        if not Path(module.__file__).resolve().is_relative_to(opts.repo):
            raise RuntimeError(f"Wrong editable installation: {module.__file__}")

    source = (opts.repo / CONVERTER).read_text(encoding="utf-8-sig")
    loader_source, source_names = _original_loader_parts(source)
    namespace = dict(
        np=np,
        torch=torch,
        axis_angle_from_quat=axis_angle_from_quat,
        quat_conjugate=quat_conjugate,
        quat_mul=quat_mul,
        quat_slerp=quat_slerp,
    )
    # Only this class is executed: never the original file's AppLauncher or main().
    exec(compile(loader_source, str(opts.repo / CONVERTER), "exec"), namespace)
    csv_data = np.loadtxt(opts.csv, delimiter=",", ndmin=2)
    if csv_data.ndim != 2 or csv_data.shape[1] != 36 or not np.isfinite(csv_data).all():
        raise ValueError("Expected finite G1 CSV with 36 columns")
    if opts.frame_range:
        start, end = opts.frame_range
        if not 1 <= start < end <= len(csv_data) or end - start < 2:
            raise ValueError("frame_range must select at least three source frames, 1-based inclusive")
    norms = np.linalg.norm(csv_data[:, 3:7], axis=-1)
    if not np.allclose(norms, 1, atol=1e-3):
        raise ValueError("CSV root quaternions are not normalized")
    digest = hashlib.sha256(opts.csv.read_bytes()).hexdigest()
    # CPU avoids thousands of tiny CUDA launches in the original SLERP loop.
    motion = namespace["MotionLoader"](str(opts.csv), 30, 60, torch.device("cpu"), opts.frame_range)
    if motion.output_frames < 3:
        raise ValueError("Motion is too short")
    print("LAFAN_STAGE: resolving current task robot configuration", flush=True)
    cfg, _ = resolve_task_config(TASK, None, overrides=["physics=newton_mjwarp"])
    if not np.isclose(cfg.sim.dt * cfg.decimation, 1 / 60):
        raise RuntimeError("Training control rate is not 60 Hz")
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.sim.visualizer_cfgs = []
    cfg.video_recorders = []
    # Conversion needs just the task's robot and its geometry, not contact sensors.
    cfg.scene.contact_forces = None
    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    launcher = parser.parse_args(["--device", "cuda:0", "--viz", "none"])
    report = dict(
        commit=opts.source_commit,
        csv=str(opts.csv),
        csv_sha256=digest,
        source_fps=30,
        output_fps=60,
        frame_range_1_based=opts.frame_range,
        frames=int(motion.output_frames),
        source_converter=CONVERTER,
        source_converter_sha256=hashlib.sha256(source.encode()).hexdigest(),
        reference_kind="Kinematic FK, no policy and no physics integration",
        max_joint_write_error_rad=0.0,
        max_root_write_error_m=0.0,
        max_root_rotation_error_rad=0.0,
        max_soft_limit_violation_rad=0.0,
    )

    def tensor(value):
        return value if isinstance(value, torch.Tensor) else value.torch

    positions, quaternions = [], []
    print("LAFAN_STAGE: creating one Newton robot for conversion", flush=True)
    with launch_simulation(cfg, launcher):
        if "newton" not in type(cfg.sim.physics).__module__.lower():
            raise RuntimeError("Conversion did not select Newton")
        with build_simulation_context(sim_cfg=cfg.sim) as sim:
            scene = InteractiveScene(cfg.scene)
            sim.reset()
            scene.reset()
            robot = scene["robot"]
            names, bodies = list(robot.joint_names), list(robot.body_names)
            if set(names) != set(source_names) or len(names) != 29:
                raise ValueError("Source and robot joints do not match")
            pelvis = bodies.index("pelvis")
            if pelvis != 0:
                raise ValueError("Expected pelvis root link")
            permutation = [source_names.index(n) for n in names]
            joints = motion.motion_dof_poss[:, permutation].numpy().copy()
            joint_vel = motion.motion_dof_vels[:, permutation].numpy().copy()
            roots = torch.cat(
                [
                    motion.motion_base_poss,
                    motion.motion_base_rots,
                    motion.motion_base_lin_vels,
                    motion.motion_base_ang_vels,
                ],
                dim=1,
            ).to(sim.device)
            roots[:, :3] += scene.env_origins[0]
            joint_gpu = torch.as_tensor(joints, device=sim.device)
            vel_gpu = torch.as_tensor(joint_vel, device=sim.device)
            count_before = sim.get_physics_step_count()
            with torch.inference_mode():
                for i in range(motion.output_frames):
                    # Only poses are needed for FK. Do not confuse root-link linear
                    # velocity from CSV with the simulator's root-COM velocity.
                    robot.write_root_link_pose_to_sim_index(root_pose=roots[i : i + 1, :7])
                    robot.write_joint_state_to_sim_index(position=joint_gpu[i : i + 1], velocity=vel_gpu[i : i + 1])
                    sim.forward()
                    robot.update(sim.get_physics_dt())
                    p = (tensor(robot.data.body_pos_w)[0] - scene.env_origins[0]).cpu().numpy().copy()
                    q = tensor(robot.data.body_quat_w)[0].cpu().numpy().copy()
                    actual_j = tensor(robot.data.joint_pos)[0].cpu().numpy()
                    joint_err = float(np.max(np.abs(actual_j - joints[i])))
                    root_err = float(np.max(np.abs(p[pelvis] - motion.motion_base_poss[i].numpy())))
                    ref_q = motion.motion_base_rots[i].numpy()
                    root_dot = abs(np.dot(q[pelvis], ref_q) / (np.linalg.norm(q[pelvis]) * np.linalg.norm(ref_q)))
                    angle_err = float(2 * np.arccos(np.clip(root_dot, 0, 1)))
                    if not np.isfinite([joint_err, root_err, angle_err]).all():
                        raise ValueError(f"Nonfinite state at frame {i}")
                    if joint_err > 1e-4 or root_err > 1e-4 or angle_err > 0.005:
                        raise ValueError(f"State write mismatch at frame {i}: {joint_err}, {root_err}, {angle_err}")
                    report["max_joint_write_error_rad"] = max(report["max_joint_write_error_rad"], joint_err)
                    report["max_root_write_error_m"] = max(report["max_root_write_error_m"], root_err)
                    report["max_root_rotation_error_rad"] = max(report["max_root_rotation_error_rad"], angle_err)
                    positions.append(p)
                    quaternions.append(q)
                    if i % 600 == 0:
                        print(f"LAFAN_FRAME {i}/{motion.output_frames}", flush=True)
                limits = tensor(robot.data.soft_joint_pos_limits)[0].cpu().numpy()
                violation = np.maximum(limits[:, 0] - joints, joints - limits[:, 1]).clip(min=0)
                report["max_soft_limit_violation_rad"] = float(violation.max())
                report["physics_steps"] = sim.get_physics_step_count() - count_before
                if report["physics_steps"] != 0:
                    raise RuntimeError("Unexpected physics step during conversion")
    pos = np.stack(positions).astype(np.float64)
    quat = np.stack(quaternions).astype(np.float64)
    arrays = dict(
        fps=np.array([60]),
        joint_names=np.array(names),
        body_names=np.array(bodies),
        quaternion_order=np.array("xyzw"),
        joint_pos=joints,
        joint_vel=joint_vel,
        body_pos_w=pos.astype(np.float32),
        body_quat_w=quat.astype(np.float32),
        body_lin_vel_w=np.gradient(pos, 1 / 60, axis=0).astype(np.float32),
        body_ang_vel_w=_world_angular_velocity(quat, 1 / 60).astype(np.float32),
    )
    validate_motion_arrays(arrays, cfg.commands.motion.body_names)
    destination = opts.output_dir / "motion_60fps.pending.npz"
    np.savez(destination, source_csv_sha256=np.array(digest), **arrays)
    # Exercise the actual training loader, including a deliberately reversed joint order.
    body_indexes = [bodies.index(n) for n in cfg.commands.motion.body_names]
    target_names = names[::-1]
    loaded = NpzLoader(
        str(destination), body_indexes, device="cpu", source_joint_names=names, target_joint_names=target_names
    )
    np.testing.assert_allclose(loaded.joint_pos.numpy(), joints[:, ::-1], atol=1e-6)
    np.testing.assert_allclose(loaded.joint_vel.numpy(), joint_vel[:, ::-1], atol=1e-6)
    np.testing.assert_allclose(loaded.body_pos_w.numpy(), arrays["body_pos_w"][:, body_indexes], atol=1e-6)
    np.testing.assert_allclose(loaded.body_quat_w.numpy(), arrays["body_quat_w"][:, body_indexes], atol=1e-6)
    if loaded.fps != 60 or loaded.time_step_total != motion.output_frames:
        raise ValueError("Training loader timing mismatch")
    if hashlib.sha256(opts.csv.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Source CSV changed during conversion")
    final_path = opts.output_dir / "motion_60fps.npz"
    destination.replace(final_path)
    destination = final_path
    report.update(
        body_velocity_method="Finite differences of Newton FK link poses, not logged simulator velocities",
        npz=str(destination),
        npz_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        loader_checks_passed=True,
        independent_fk_validation=False,
        visual_review_completed=False,
        stance_phase_ranges=[],
        training_started=False,
    )
    (opts.output_dir / "conversion.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("LAFAN_CONVERSION_AND_LOADER_CHECKS_OK", str(destination), flush=True)


if __name__ == "__main__":
    main()
