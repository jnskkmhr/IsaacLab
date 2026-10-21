# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read NPZ terrain metadata and assemble static patches in the motion coordinate frame."""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def motion_files_use_terrain(motion_files: Sequence[str], mode: Literal["auto", "motion", "scene"]) -> bool:
    """Select NPZ terrain or the configured scene terrain before scene creation.

    ``auto`` uses NPZ terrain when every clip supplies geometry or ground height,
    and keeps the scene terrain when none do. A mixed collection is rejected:
    supply explicit ``ground_height`` for flat clips to give every motion a patch.
    ``motion`` requires terrain in every clip. ``scene`` ignores NPZ terrain.
    """
    if mode not in ("auto", "motion", "scene"):
        raise ValueError(f"Unknown terrain_mode {mode!r}; choose auto, motion, or scene.")
    if mode == "scene":
        return False
    if isinstance(motion_files, (str, bytes)) or not motion_files:
        raise ValueError("motion_files must be a nonempty sequence of NPZ paths.")
    terrain_keys = {
        "terrain_position",
        "terrain_dimension",
        "terrain_quat",
        "terrain_vertices",
        "terrain_faces",
        "ground_height",
        "terrain_boxes",
        "terrain_box_quat",
    }
    missing = []
    for path in motion_files:
        with np.load(path, allow_pickle=False) as data:
            if terrain_keys.isdisjoint(data.files):
                missing.append(path)
    if missing and (mode == "motion" or len(missing) != len(motion_files)):
        raise ValueError(
            f"Every motion needs terrain metadata for paired loading. Missing terrain in: {missing}. "
            "Add ground_height to flat clips, or explicitly select terrain_mode='scene'."
        )
    return not missing


@dataclass
class MotionPairedTerrain:
    """Box/mesh geometry in the reference motion frame; lengths are in [m]."""

    terrain_position: np.ndarray
    """Box centers [m], shape [N, 3]."""
    terrain_dimension: np.ndarray
    """Full box dimensions along each box's local axes [m], shape [N, 3]."""
    terrain_quat: np.ndarray
    """Box orientations in xyzw order, shape [N, 4]."""
    terrain_vertices: np.ndarray
    """Mesh vertices [m], shape [V, 3]."""
    terrain_faces: np.ndarray
    """Triangle vertex indices, shape [F, 3]."""
    ground_height: float | None
    """Optional local ground top height [m]; None adds no ground."""

    def mesh(self) -> trimesh.Trimesh:
        """Build the supplied obstacle geometry, excluding optional ground."""
        meshes = []
        for position, dimension, quat in zip(self.terrain_position, self.terrain_dimension, self.terrain_quat):
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(quat).as_matrix()
            transform[:3, 3] = position
            meshes.append(trimesh.creation.box(extents=dimension, transform=transform))
        if len(self.terrain_faces):
            meshes.append(trimesh.Trimesh(self.terrain_vertices, self.terrain_faces, process=False))
        return trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()

    def digest(self) -> str:
        """Identify equal geometry in the same reference frame, including ground height."""
        digest = hashlib.sha256()
        for value in vars(self).values():
            array = np.asarray(value, dtype=np.float64) if value is not None else np.empty(0)
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()


def load_motion_paired_terrain(data: Mapping[str, np.ndarray]) -> MotionPairedTerrain:
    """Read canonical terrain fields, with compatibility for previously exported clips.

    Box quaternions are xyzw independently of the motion's ``quaternion_order``.
    Boxes and a triangle mesh may coexist. An explicit ``ground_height`` creates
    bounded ground under the patch; no implicit plane is added at zero height.
    """
    position = data.get("terrain_position")
    dimension = data.get("terrain_dimension")
    quat = data.get("terrain_quat")
    legacy = "terrain_boxes" in data or "terrain_box_quat" in data
    if legacy:
        if position is not None or dimension is not None or quat is not None:
            raise ValueError("Do not mix legacy and canonical box terrain fields.")
        warnings.warn(
            "Legacy terrain_boxes/terrain_box_quat: use terrain_position, terrain_dimension, terrain_quat.",
            FutureWarning,
            stacklevel=2,
        )
        boxes = _array(data.get("terrain_boxes", np.empty((0, 6))), "terrain_boxes", 6)
        position, dimension = boxes[:, :3], boxes[:, 3:]
        quat = data.get("terrain_box_quat")
    if (position is None) != (dimension is None):
        raise ValueError("terrain_position and terrain_dimension must be supplied together.")
    position = _array(position if position is not None else np.empty((0, 3)), "terrain_position", 3)
    dimension = _array(dimension if dimension is not None else np.empty((0, 3)), "terrain_dimension", 3)
    if dimension.shape != position.shape or (dimension <= 0).any():
        raise ValueError("terrain_dimension must match terrain_position and contain positive full dimensions.")
    if quat is None:
        quat = np.tile([0.0, 0.0, 0.0, 1.0], (len(position), 1))
    quat = _array(quat, "terrain_quat", 4)
    if len(quat) != len(position) or not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-3, rtol=0):
        raise ValueError("terrain_quat must provide one unit xyzw quaternion per box.")
    if ("terrain_vertices" in data) != ("terrain_faces" in data):
        raise ValueError("terrain_vertices and terrain_faces must be supplied together.")
    vertices = _array(data.get("terrain_vertices", np.empty((0, 3))), "terrain_vertices", 3)
    faces = _array(data.get("terrain_faces", np.empty((0, 3))), "terrain_faces", 3)
    if (faces != np.floor(faces)).any() or (faces < 0).any() or (faces >= len(vertices)).any():
        raise ValueError("terrain_faces must contain valid integer vertex indices.")
    if bool(len(vertices)) != bool(len(faces)):
        raise ValueError("A terrain mesh must contain both vertices and faces.")
    ground = data.get("ground_height")
    if ground is not None:
        ground = np.asarray(ground)
        if ground.size != 1 or not np.isfinite(ground).all():
            raise ValueError("ground_height must be a finite scalar.")
        ground = float(ground.reshape(-1)[0])
    if not len(position) and not len(faces) and ground is None:
        raise ValueError("Paired terrain requires box/mesh geometry or an explicit ground_height.")
    return MotionPairedTerrain(position, dimension, quat, vertices, faces.astype(np.int64), ground)


class MotionPairedTerrainGenerator:
    """One static patch per distinct terrain, with a translation for every motion.

    Patch bounds include all reference body positions and obstacle vertices, plus
    ``margin`` [m]. Patches are separated by ``spacing`` [m]. Equal geometry shares
    a patch; an optional NPZ ``terrain_id`` asserts that clips use equal geometry.
    """

    def __init__(self, motion_files: Sequence[str], margin: float = 1.0, spacing: float = 2.0):
        if not motion_files:
            raise ValueError("Set at least one motion file for paired terrain.")
        if not math.isfinite(margin) or not math.isfinite(spacing) or margin <= 0 or spacing <= 0:
            raise ValueError("Terrain margin and spacing must be finite and positive.")
        self.motion_files = tuple(str(Path(path).expanduser().resolve()) for path in motion_files)
        terrains, meshes, bounds, motion_ids = [], [], [], []
        by_digest, by_name = {}, {}
        for path in self.motion_files:
            with np.load(path, allow_pickle=False) as data:
                terrain = load_motion_paired_terrain(data)
                digest = terrain.digest()
                if "terrain_id" in data:
                    name = str(np.asarray(data["terrain_id"]).item())
                    if name in by_name and by_name[name] != digest:
                        raise ValueError(f"{path}: terrain_id {name!r} refers to different geometry.")
                    by_name[name] = digest
                body_pos = np.asarray(data["body_pos_w"])
                if (
                    body_pos.ndim != 3
                    or body_pos.shape[-1] != 3
                    or not body_pos.size
                    or not np.isfinite(body_pos).all()
                ):
                    raise ValueError(f"{path}: body_pos_w must be finite with shape [frames, bodies, 3].")
                points = body_pos.reshape(-1, 3)
            if digest not in by_digest:
                by_digest[digest] = len(terrains)
                terrains.append(terrain)
                mesh = terrain.mesh()
                meshes.append(mesh)
                bounds.append(np.array([points.min(0), points.max(0)]))
                if len(mesh.vertices):
                    bounds[-1][0] = np.minimum(bounds[-1][0], mesh.bounds[0])
                    bounds[-1][1] = np.maximum(bounds[-1][1], mesh.bounds[1])
            index = by_digest[digest]
            bounds[index][0] = np.minimum(bounds[index][0], points.min(0))
            bounds[index][1] = np.maximum(bounds[index][1], points.max(0))
            motion_ids.append(index)
        self.motion_paired_terrain_ids = np.asarray(motion_ids, dtype=np.int64)
        bounds = np.asarray(bounds)
        bounds[:, 0, :2] -= margin
        bounds[:, 1, :2] += margin
        pitch = (bounds[:, 1, :2] - bounds[:, 0, :2]).max(0) + spacing
        columns = math.ceil(math.sqrt(len(terrains)))
        self.terrain_origins = np.zeros((len(terrains), 3), dtype=np.float32)
        world_meshes = []
        for index, (terrain, mesh) in enumerate(zip(terrains, meshes)):
            center = bounds[index, :, :2].mean(0)
            self.terrain_origins[index, :2] = np.array([index // columns, index % columns]) * pitch - center
            if terrain.ground_height is not None:
                ground = trimesh.creation.box(extents=(*np.diff(bounds[index, :, :2], axis=0)[0], 0.2))
                ground.apply_translation((*center, terrain.ground_height - 0.1))
                mesh = trimesh.util.concatenate([mesh, ground])
            mesh.apply_translation(self.terrain_origins[index])
            world_meshes.append(mesh)
        self.terrain_mesh = trimesh.util.concatenate(world_meshes)
        self.motion_origins = self.terrain_origins[self.motion_paired_terrain_ids]


def _array(value: np.ndarray, name: str, columns: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape [N, {columns}].")
    return array
