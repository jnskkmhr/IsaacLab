# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import already-retargeted robot exports; no human skeleton retargeting is performed."""

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def load_motion(
    path: Path,
    *,
    input_fps: float | None = None,
    output_fps: float | None = None,
    quaternion_order: str | None = None,
    rotation_format: str = "quaternion",
    joint_names: list[str] | None = None,
    body_names: list[str] | None = None,
    robot_xml: str | None = None,
    field_map: dict[str, str] | None = None,
    joint_name_map: dict[str, str] | None = None,
    default_joint_positions: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Convert an NPZ or retargeted root/joint export into Mimic's labeled NPZ schema.

    CSV/numeric NPY and ``frames`` dictionaries contain root XYZ, root quaternion,
    then joint positions (optionally followed by joint velocities). For packed frames,
    rotation_format can instead be ``rotvec`` or ``euler_xyz`` (radians).
    Dictionaries may instead contain ``root_pos``, ``root_rot``,
    and ``dof_pos``. Torch exports with ``global_translation``, ``global_rotation``,
    ``dof_pos``, and ``dof_names`` are supported too. Use field_map to map those
    canonical input keys to exporter-specific keys. Unlabeled columns require names.

    For root/joint-only inputs, robot_xml must describe the same robot as the task.
    Missing model joints require explicit default_joint_positions; they are never
    silently filled with zero. Quaternion order and frame rate must be declared in
    the file or parameters. Complete body-state NPZ files do not need robot_xml.
    """
    data = _read_export(path)
    if field_map:
        data = {**data, **{key: data[source] for key, source in field_map.items()}}
    fps_value = data.get("fps", input_fps)
    if fps_value is None:
        raise ValueError("Specify input_fps for an export without fps metadata.")
    fps = float(np.asarray(fps_value).item())
    target_fps = fps if output_fps is None else float(output_fps)
    if not np.isfinite([fps, target_fps]).all() or min(fps, target_fps) <= 0:
        raise ValueError("Input and output FPS must be finite and positive.")
    if rotation_format not in ("quaternion", "rotvec", "euler_xyz"):
        raise ValueError("rotation_format must be quaternion, rotvec, or euler_xyz.")
    if rotation_format != "quaternion":
        if "frames" not in data:
            raise ValueError("Non-quaternion rotation_format is only supported for packed frames.")
        frames = np.asarray(data["frames"], dtype=np.float64)
        if frames.ndim != 2 or frames.shape[1] < 7:
            raise ValueError("Packed frames must contain root XYZ, rotation, and joint positions.")
        rotation = (
            Rotation.from_rotvec(frames[:, 3:6])
            if rotation_format == "rotvec"
            else Rotation.from_euler("xyz", frames[:, 3:6])
        )
        data["frames"] = np.concatenate((frames[:, :3], rotation.as_quat(), frames[:, 6:]), axis=1)
        data["quaternion_order"] = "xyzw"
    order = data.get("quaternion_order", quaternion_order)
    if order is None or str(np.asarray(order).item()) not in ("xyzw", "wxyz"):
        raise ValueError("Specify quaternion_order='xyzw' or 'wxyz' for untagged exports.")
    order = str(np.asarray(order).item())
    names = data.get("joint_names", data.get("dof_names", joint_names))
    if names is None:
        raise ValueError("Specify joint_names in source column order for unlabeled exports.")
    names = [str(name) for name in names]
    if joint_name_map:
        names = [joint_name_map.get(name, name) for name in names]

    if "body_pos_w" in data:
        labels = data.get("body_names", body_names)
        if labels is None:
            raise ValueError("Specify body_names in source column order for unlabeled body arrays.")
        positions = np.asarray(data["body_pos_w"], dtype=np.float64)
        quaternions = _xyzw(data["body_quat_w"], order)
        joints = np.asarray(data["joint_pos"], dtype=np.float64)
        if fps != target_fps:
            positions, quaternions, joints = _resample(positions, quaternions, joints, fps, target_fps)
        motion = _motion_arrays(positions, quaternions, joints, target_fps, names, labels)
        # Preserve measured velocities when no resampling is needed.
        if fps == target_fps:
            for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
                if key in data:
                    motion[key] = np.asarray(data[key], dtype=np.float32)
        return motion

    if robot_xml is None:
        raise ValueError("Root/joint-only exports require robot_xml to compute body poses with forward kinematics.")
    if "frames" in data:
        frames = np.asarray(data["frames"], dtype=np.float64)
        if frames.ndim != 2 or frames.shape[1] not in (7 + len(names), 7 + 2 * len(names)):
            raise ValueError(
                "Frames must contain root XYZ, quaternion, joint positions, and optionally joint velocities."
            )
        positions = frames[:, :3]
        quaternions = _xyzw(frames[:, 3:7], order)
        joints = frames[:, 7 : 7 + len(names)]
    elif "global_translation" in data:
        positions = np.asarray(data["global_translation"])[:, 0]
        quaternions = _xyzw(np.asarray(data["global_rotation"])[:, 0], order)
        joints = np.asarray(data["dof_pos"])
    else:
        positions = np.asarray(data["root_pos"])
        quaternions = _xyzw(data["root_rot"], order)
        joints = np.asarray(data["dof_pos"])
    if positions.shape != (len(joints), 3) or quaternions.shape != (len(joints), 4):
        raise ValueError("Root position/quaternion arrays must have shapes (frames, 3)/(frames, 4).")
    if joints.ndim != 2 or joints.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("Each joint column must have a unique source joint name.")
    if not all(np.isfinite(value).all() for value in (positions, quaternions, joints)):
        raise ValueError("Root and joint arrays must contain finite values.")
    positions, quaternions, joints = _resample(positions, quaternions, joints, fps, target_fps)
    return _forward_kinematics(
        robot_xml, positions, quaternions, joints, names, target_fps, body_names, default_joint_positions or {}
    )


def _read_export(path: Path) -> dict:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return dict(data)
    if path.suffix == ".csv":
        return {"frames": np.loadtxt(path, delimiter=",", ndmin=2)}
    if path.suffix == ".npy":
        with path.open("rb") as stream:
            if stream.read(6) == b"\x93NUMPY":
                return {"frames": np.load(path, allow_pickle=False)}
    if path.suffix in (".pt", ".pth", ".npy"):
        # Torch is only needed for its own export format. Arbitrary pickle globals are disabled.
        import torch

        return torch.load(path, map_location="cpu", weights_only=True)
    if path.suffix == ".pkl":
        with path.open("rb") as stream:
            return _ArrayUnpickler(stream).load()
    raise ValueError(f"Unsupported retargeted export: {path.suffix}. Raw BVH must first be retargeted to G1.")


def _xyzw(value: np.ndarray, order: str) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape[-1] != 4 or not np.isfinite(quat).all():
        raise ValueError("Quaternions must be finite arrays with four components.")
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1.0, atol=1.0e-3):
        raise ValueError("Input quaternions must have unit length.")
    return np.roll(quat, -1, axis=-1) if order == "wxyz" else quat


def _resample(
    positions: np.ndarray,
    quaternions: np.ndarray,
    joints: np.ndarray,
    fps: float,
    output_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(joints) < 2:
        raise ValueError("A motion needs at least two frames.")
    old_times = np.arange(len(joints)) / fps
    times = np.arange(int(np.floor(old_times[-1] * output_fps + 1.0e-8)) + 1) / output_fps
    times = np.minimum(times, old_times[-1])
    if len(times) < 2:
        raise ValueError("Motion duration is too short for output_fps.")
    flat = positions.reshape(len(joints), -1)
    pos = np.stack([np.interp(times, old_times, col) for col in flat.T], axis=-1)
    angles = np.stack([np.interp(times, old_times, col) for col in joints.T], axis=-1)
    flat_quat = quaternions.reshape(len(joints), -1, 4)
    quat = np.stack(
        [Slerp(old_times, Rotation.from_quat(flat_quat[:, i]))(times).as_quat() for i in range(flat_quat.shape[1])],
        axis=1,
    )
    return pos.reshape((len(times), *positions.shape[1:])), quat.reshape((len(times), *quaternions.shape[1:])), angles


def _motion_arrays(
    positions: np.ndarray,
    quaternions: np.ndarray,
    joints: np.ndarray,
    fps: float,
    joint_names: list[str],
    body_names: list[str],
) -> dict[str, np.ndarray]:
    frames = len(joints)
    if frames < 2:
        raise ValueError("A motion needs at least two frames.")
    lo = np.maximum(np.arange(frames) - 1, 0)
    hi = np.minimum(np.arange(frames) + 1, frames - 1)
    before = Rotation.from_quat(quaternions[lo].reshape(-1, 4))
    after = Rotation.from_quat(quaternions[hi].reshape(-1, 4))
    angular = (after * before.inv()).as_rotvec().reshape(frames, -1, 3)
    angular /= ((hi - lo) / fps)[:, None, None]
    return {
        "fps": np.asarray(fps),
        "quaternion_order": np.asarray("xyzw"),
        "joint_names": np.asarray(joint_names, dtype=str),
        "body_names": np.asarray(body_names, dtype=str),
        "joint_pos": joints.astype(np.float32),
        "joint_vel": np.gradient(joints, 1 / fps, axis=0).astype(np.float32),
        "body_pos_w": positions.astype(np.float32),
        "body_quat_w": quaternions.astype(np.float32),
        "body_lin_vel_w": np.gradient(positions, 1 / fps, axis=0).astype(np.float32),
        "body_ang_vel_w": angular.astype(np.float32),
    }


def _forward_kinematics(
    robot_xml: str,
    positions: np.ndarray,
    quaternions: np.ndarray,
    joints: np.ndarray,
    names: list[str],
    fps: float,
    body_names: list[str] | None,
    defaults: dict[str, float],
) -> dict[str, np.ndarray]:
    # This optional dependency is needed only for root/joint exports, not canonical NPZ files.
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(Path(robot_xml).expanduser().resolve()))
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1 or model.jnt_qposadr[free[0]] != 0:
        raise ValueError("robot_xml must contain one floating root with its pose at qpos[:7].")
    joint_ids = [i for i in range(model.njnt) if i != free[0]]
    scalar_joint_types = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    if any(int(model.jnt_type[i]) not in scalar_joint_types for i in joint_ids):
        raise ValueError("Only single-DOF robot joints are supported.")
    target_names = [model.joint(i).name for i in joint_ids]
    unknown = set(names) - set(target_names)
    missing = set(target_names) - set(names) - defaults.keys()
    if unknown or missing:
        raise ValueError(f"Joint mapping mismatch: unknown={sorted(unknown)}, missing defaults={sorted(missing)}")
    mapped = np.stack(
        [
            joints[:, names.index(name)] if name in names else np.full(len(joints), defaults[name])
            for name in target_names
        ],
        axis=1,
    )
    if body_names is None:
        body_names = [
            model.body(i).name for i in range(1, model.nbody) if not model.body(i).name.endswith("_contact_point")
        ]
    body_ids = [model.body(name).id for name in body_names]
    data = mujoco.MjData(model)
    body_pos, body_quat = [], []
    for pos, quat, angle in zip(positions, quaternions, mapped):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = pos
        data.qpos[3:7] = np.roll(quat, 1)
        data.qpos[model.jnt_qposadr[joint_ids]] = angle
        mujoco.mj_forward(model, data)
        body_pos.append(data.xpos[body_ids].copy())
        body_quat.append(np.roll(data.xquat[body_ids], -1, axis=-1))
    return _motion_arrays(np.asarray(body_pos), np.asarray(body_quat), mapped, fps, target_names, body_names)


class _ArrayUnpickler(pickle.Unpickler):
    """Read primitive containers only; executable pickle globals are not accepted."""

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError("Pickle globals are unsupported; export arrays as NPZ or primitive lists instead.")
