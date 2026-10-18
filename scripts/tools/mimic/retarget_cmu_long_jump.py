# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Retarget a CMU T-pose BVH long jump to G1 using Mink and GMR's BVH reader.

See the g1_29dof_agent README for pinned sources, data terms, and reproduction.
This tool needs optional ``mink``, ``daqp``, and ``loop-rate-limiters`` packages;
use a uv dependency overlay rather than changing the training environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


def load_model(xml_path: Path) -> mujoco.MjModel:
    """Load the robot XML with a floor for models whose contact pairs reference it."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((xml_path.parent / compiler.get("meshdir", ".")).resolve()))
    world = root.find("worldbody")
    if world.find(".//geom[@name='floor']") is None:
        ET.SubElement(world, "geom", name="floor", type="plane", size="10 10 0.1", rgba="0.8 0.8 0.8 1")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def foot_bottom(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Return the lowest point of the robot's foot collision boxes [m]."""
    bottoms = []
    for geom in range(model.ngeom):
        name = model.body(model.geom_bodyid[geom]).name
        if "ankle_roll" not in name or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        bottoms.append(data.geom_xpos[geom, 2] - np.abs(rotation[2]) @ model.geom_size[geom])
    if not bottoms:
        raise ValueError("Expected box-foot G1 collision geometry.")
    return float(min(bottoms))


def retarget(args: argparse.Namespace) -> None:
    """Retarget, smooth, ground-correct, and export the selected public motion."""
    # These dependencies are required only by this offline conversion tool.
    sys.path.insert(0, str(args.gmr_repo.resolve()))
    import mink
    from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
    from general_motion_retargeting.utils.lafan_vendor.utils import quat_fk

    bvh = read_bvh(str(args.bvh))
    human_quat, human_pos = quat_fk(bvh.quats, bvh.pos, bvh.parents)
    source_dt = float(re.search(r"Frame Time:\s*([\d.]+)", args.bvh.read_text()).group(1))
    source_fps = round(1.0 / source_dt)
    stride = round(source_fps / args.fps)
    if not np.isclose(source_fps / stride, args.fps):
        raise ValueError("Output FPS must divide the source frame rate.")
    # CMU Y-up/forward +Z to robot Z-up/forward +X. Calibration cancels local bone axes.
    basis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    human_pos = human_pos @ basis.T
    human_quat = (
        Rotation.from_matrix(basis) * Rotation.from_quat(human_quat.reshape(-1, 4), scalar_first=True)
    ).as_quat(scalar_first=True).reshape(human_quat.shape)
    model = load_model(args.robot_xml)
    data = mujoco.MjData(model)
    neutral = model.qpos0.copy()
    for name, angle in (("left_shoulder_roll_joint", np.pi / 2), ("right_shoulder_roll_joint", -np.pi / 2)):
        neutral[model.jnt_qposadr[model.joint(name).id]] = angle
    data.qpos[:] = neutral
    mujoco.mj_forward(model, data)
    pairs = {"pelvis": "Hips", "torso_link": "Spine1"}
    for side in ("left", "right"):
        human = side.title()
        pairs.update({
            f"{side}_hip_yaw_link": human + "UpLeg", f"{side}_knee_link": human + "Leg",
            f"{side}_ankle_roll_link": human + "Foot", f"{side}_shoulder_yaw_link": human + "Arm",
            f"{side}_elbow_link": human + "ForeArm", f"{side}_wrist_yaw_link": human + "Hand",
        })
    human_feet = [bvh.bones.index(name) for name in ("LeftFoot", "RightFoot")]
    robot_feet = [model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    hip = bvh.bones.index("Hips")
    scale = (data.xpos[model.body("pelvis").id, 2] - data.xpos[robot_feet, 2].mean()) / (
        human_pos[0, hip, 2] - human_pos[0, human_feet, 2].mean()
    )
    human_pos *= scale
    start = human_pos[1, hip].copy()
    direction = human_pos[-1, hip] - start
    heading = Rotation.from_euler("z", -np.arctan2(direction[1], direction[0]))
    calibration = {}
    tasks = []
    for robot, human in pairs.items():
        body = model.body(robot).id
        index = bvh.bones.index(human)
        source_rot = Rotation.from_quat(human_quat[0, index], scalar_first=True)
        target_rot = Rotation.from_quat(data.xquat[body], scalar_first=True)
        calibration[robot] = (
            index, source_rot.inv() * target_rot,
            data.xpos[body] - neutral[:3] - (human_pos[0, index] - human_pos[0, hip]),
        )
        tasks.append(mink.FrameTask(
            frame_name=robot, frame_type="body",
            position_cost=100 if "ankle" in robot or robot == "pelvis" else 10,
            orientation_cost=10 if robot in ("pelvis", "torso_link") else 2, lm_damping=1,
        ))
    configuration = mink.Configuration(model)
    configuration.update(neutral)
    posture = mink.PostureTask(model, cost=0.02)
    posture.set_target(neutral)
    limits = [mink.ConfigurationLimit(model)]
    frames = np.arange(args.start_frame, len(human_pos), stride)
    qposes = []
    origin = np.array([0.0, 0.0, neutral[2]])
    for count, frame in enumerate(frames):
        root_target = heading.apply(human_pos[frame, hip] - start + neutral[:3] - origin) + origin
        for task, (robot, (index, offset, pos_offset)) in zip(tasks, calibration.items()):
            rotation = heading * Rotation.from_quat(human_quat[frame, index], scalar_first=True) * offset
            position = heading.apply(human_pos[frame, index] - start + neutral[:3] + pos_offset - origin) + origin
            if "ankle_roll" in robot:
                # Human feet are too narrow for G1's pelvis and collision boxes.
                sign = 1 if robot.startswith("left") else -1
                position[1] = root_target[1] + sign * max(sign * (position[1] - root_target[1]), args.foot_spacing / 2)
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rotation.as_quat(scalar_first=True)), position))
        for _ in range(40 if count == 0 else 8):
            velocity = mink.solve_ik(configuration, tasks + [posture], 1 / args.fps, "daqp", damping=0.05, limits=limits)
            configuration.integrate_inplace(velocity, 1 / args.fps)
        qposes.append(configuration.q.copy())
        if count % 60 == 0:
            print(f"Retargeted {count}/{len(frames)} frames", flush=True)
    qposes = np.asarray(qposes)
    qposes[:, :3] = savgol_filter(qposes[:, :3], 9, 3, axis=0)
    qposes[:, 7:] = savgol_filter(qposes[:, 7:], 9, 3, axis=0)
    for joint in range(1, model.njnt):
        address = model.jnt_qposadr[joint]
        qposes[:, address] = np.clip(qposes[:, address], *model.jnt_range[joint])
    # The G1 task uses neutral wrists; preserve arm swing without tracking human hand twist.
    for joint in range(1, model.njnt):
        if "_wrist_" in model.joint(joint).name:
            qposes[:, model.jnt_qposadr[joint]] = 0.0
    corrections = []
    for pose in qposes:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        corrections.append(max(0.0, 0.003 - foot_bottom(model, data)))
    correction = np.maximum(savgol_filter(corrections, 9, 2), corrections)
    qposes[:, 2] += correction
    # A quiet final hold provides time to learn a stable landing, not just first contact.
    hold_frames = round(args.hold_seconds * args.fps)
    initial_hold_frames = round(args.initial_hold_seconds * args.fps)
    qposes = np.concatenate((
        np.repeat(qposes[:1], initial_hold_frames, axis=0),
        qposes,
        np.repeat(qposes[-1:], hold_frames, axis=0),
    ))
    joint_ids = list(range(1, model.njnt))
    joint_names = [model.joint(i).name for i in joint_ids]
    body_ids = [i for i in range(1, model.nbody) if not model.body(i).name.endswith("_contact_point")]
    body_names = [model.body(i).name for i in body_ids]
    joint_pos = qposes[:, model.jnt_qposadr[joint_ids]]
    joint_vel = np.gradient(joint_pos, 1 / args.fps, axis=0)
    body_pos = []
    body_quat = []
    body_lin_vel = []
    body_ang_vel = []
    qvel = np.zeros((len(qposes), model.nv))
    for i in range(len(qposes)):
        lo, hi = max(i - 1, 0), min(i + 1, len(qposes) - 1)
        mujoco.mj_differentiatePos(model, qvel[i], (hi - lo) / args.fps, qposes[lo], qposes[hi])
        data.qpos[:] = qposes[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        body_pos.append(data.xpos[body_ids].copy())
        body_quat.append(data.xquat[body_ids][:, [1, 2, 3, 0]].copy())
        velocities = []
        for body in body_ids:
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            velocities.append(velocity)
        velocities = np.array(velocities)
        body_ang_vel.append(velocities[:, :3])
        body_lin_vel.append(velocities[:, 3:])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, fps=args.fps, joint_names=np.array(joint_names), body_names=np.array(body_names),
        quaternion_order=np.array("xyzw"), joint_pos=joint_pos.astype(np.float32), joint_vel=joint_vel.astype(np.float32),
        body_pos_w=np.asarray(body_pos, dtype=np.float32), body_quat_w=np.asarray(body_quat, dtype=np.float32),
        body_lin_vel_w=np.asarray(body_lin_vel, dtype=np.float32), body_ang_vel_w=np.asarray(body_ang_vel, dtype=np.float32),
        qpos=qposes, source_frames=frames,
    )
    metadata = {
        "source": "CMU Graphics Lab Motion Capture Database, subject 83, trial 43 (long jump forward)",
        "source_url": "https://mocap.cs.cmu.edu/search.php?subjectnumber=83",
        "bvh_url": "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/083/83_43.bvh",
        "source_sha256": hashlib.sha256(args.bvh.read_bytes()).hexdigest(),
        "robot_xml_sha256": hashlib.sha256(args.robot_xml.read_bytes()).hexdigest(),
        "source_fps": source_fps, "fps": args.fps, "source_start_frame": args.start_frame,
        "hold_seconds": args.hold_seconds,
        "initial_hold_seconds": args.initial_hold_seconds, "foot_spacing_m": args.foot_spacing,
        "scale_m_per_source_unit": float(scale), "max_ground_correction_m": float(np.max(correction)),
        "root_distance_m": float(qposes[-1, 0] - qposes[0, 0]),
        "frames": len(qposes), "quaternion_order": "xyzw",
        "wrist_reference_position_rad": 0.0,
        "processing": "T-pose-calibrated IK, heading normalization, anatomical scaling, widened feet, Savitzky-Golay smoothing, floor correction, final hold",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--gmr_repo", type=Path, required=True)
    parser.add_argument("--robot_xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--start_frame", type=int, default=181)
    parser.add_argument("--hold_seconds", type=float, default=1.0, help="Final static hold duration [s].")
    parser.add_argument("--initial_hold_seconds", type=float, default=0.0, help="Initial static hold duration [s].")
    parser.add_argument("--foot_spacing", type=float, default=0.20)
    retarget(parser.parse_args())
