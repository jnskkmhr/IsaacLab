# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline dataset terms and canonical NPZ storage for retargeted robot motions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .dataset_cfg import DatasetTermCfg

MOTION_DATA_DIR = Path(__file__).parent / "data"


def validate_motion(motion: dict[str, np.ndarray]) -> None:
    """Check the labeled, world-frame, xyzw motion contract consumed by Mimic.

    All positions are in meters, joint angles in radians, and velocities per second.
    Body positions and velocities refer to link origins, not centers of mass.
    """
    required = (
        "fps",
        "joint_names",
        "body_names",
        "quaternion_order",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
    missing = set(required) - motion.keys()
    if missing:
        raise ValueError(f"Missing motion fields: {sorted(missing)}")
    fps = np.asarray(motion["fps"])
    if fps.size != 1 or not np.isfinite(fps).all() or fps.item() <= 0:
        raise ValueError("fps must be a finite positive scalar.")
    if str(np.asarray(motion["quaternion_order"]).item()) != "xyzw":
        raise ValueError("Canonical motions must use xyzw quaternions.")
    for key in ("joint_names", "body_names"):
        names = np.asarray(motion[key])
        if names.ndim != 1 or names.dtype.kind not in "US" or not len(names):
            raise ValueError(f"{key} must be a nonempty array of strings.")
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError(f"{key} must contain unique, nonempty names.")
    frames = len(motion["joint_pos"])
    if frames < 2:
        raise ValueError("A motion needs at least two frames.")
    joints, bodies = len(motion["joint_names"]), len(motion["body_names"])
    shapes = {
        "joint_pos": (frames, joints),
        "joint_vel": (frames, joints),
        "body_pos_w": (frames, bodies, 3),
        "body_quat_w": (frames, bodies, 4),
        "body_lin_vel_w": (frames, bodies, 3),
        "body_ang_vel_w": (frames, bodies, 3),
    }
    for key, shape in shapes.items():
        value = np.asarray(motion[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{key} must have shape {shape} and finite values; got {value.shape}.")
    if not np.allclose(np.linalg.norm(motion["body_quat_w"], axis=-1), 1.0, atol=1.0e-3):
        raise ValueError("body_quat_w must contain unit quaternions.")


class DatasetTerm:
    """Load and validate one dataset's already-retargeted robot motion."""

    def __init__(self, cfg: DatasetTermCfg):
        self.cfg = cfg

    def load(self, path: str | Path) -> dict[str, np.ndarray]:
        """Import a motion using this term's configured schema."""
        motion = self.cfg.func(Path(path).expanduser(), **self.cfg.params)
        validate_motion(motion)
        return motion


class DatasetManager:
    """Dispatch configured import terms and save portable NPZ motion files."""

    def __init__(self, cfg: dict[str, DatasetTermCfg], output_dir: str | Path = MOTION_DATA_DIR):
        self.terms = {name: DatasetTerm(term) for name, term in cfg.items()}
        self.output_dir = Path(output_dir).expanduser()

    def load(self, dataset: str, path: str | Path) -> dict[str, np.ndarray]:
        """Load a motion through a named dataset term."""
        return self.terms[dataset].load(path)

    def save(self, motion: dict[str, np.ndarray], name: str, *, overwrite: bool = False) -> Path:
        """Write a validated motion beneath output_dir, creating dataset subdirectories.

        Args:
            motion: Canonical arrays and metadata.
            name: Relative NPZ filename, for example ``lafan/walking/motion_60fps.npz``.
            overwrite: Whether an existing output may be replaced.
        """
        validate_motion(motion)
        path = self._output_path(name, ".npz")
        with path.open("wb" if overwrite else "xb") as stream:
            np.savez_compressed(stream, **motion)
        return path

    def save_csv(
        self,
        motion: dict[str, np.ndarray],
        name: str,
        *,
        joint_names: list[str] | None = None,
        body_name: str | None = None,
        joint_positions_only: bool = False,
        overwrite: bool = False,
    ) -> Path:
        """Export a loaded motion as a headerless CSV.

        Full rows contain body XYZ, body quaternion (xyzw), joint positions, and
        joint velocities. Select the root body to produce a re-importable motion.
        Joint-only rows contain absolute joint positions in radians.

        Args:
            motion: Canonical arrays and metadata returned by load.
            name: Relative CSV filename beneath output_dir.
            joint_names: Joint selection and output order; None preserves source order.
            body_name: Body whose world pose is exported; required for full rows.
            joint_positions_only: Export only joint positions, omitting body pose and velocities.
            overwrite: Whether an existing output may be replaced.
        """
        validate_motion(motion)
        source_names = list(motion["joint_names"])
        names = source_names if joint_names is None else joint_names
        if not names or len(set(names)) != len(names):
            raise ValueError("joint_names must be nonempty and unique.")
        missing = set(names) - set(source_names)
        if missing:
            raise ValueError(f"Motion is missing requested joints: {sorted(missing)}")
        indices = [source_names.index(joint) for joint in names]
        positions = motion["joint_pos"][:, indices]
        if joint_positions_only:
            rows = positions
        else:
            bodies = list(motion["body_names"])
            if body_name not in bodies:
                raise ValueError("Full CSV export requires body_name to identify a body in the motion.")
            body_index = bodies.index(body_name)
            rows = np.concatenate(
                (
                    motion["body_pos_w"][:, body_index],
                    motion["body_quat_w"][:, body_index],
                    positions,
                    motion["joint_vel"][:, indices],
                ),
                axis=1,
            )
        path = self._output_path(name, ".csv")
        with path.open("w" if overwrite else "x") as stream:
            np.savetxt(stream, rows, delimiter=",")
        return path

    def _output_path(self, name: str, suffix: str) -> Path:
        path = (self.output_dir / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(self.output_dir.resolve()) or path.suffix != suffix:
            raise ValueError(f"name must be a relative {suffix} path inside output_dir.")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
