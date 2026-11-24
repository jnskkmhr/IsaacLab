# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Particle NeRD dataset: which episodes exist, the file of one episode, and windows for training.

One episode is one file ``<root>/<split>/<terrain>/<category>/NNNN.hdf5`` of the format :data:`SCHEMA`.

- Episode definitions: :func:`episode` returns the full, reproducible description of one episode from its
  terrain, category and index. Six categories (four with a rigid object, two with sand only), three terrains, and
  the split follow from the index.
- Episode file: :func:`write_episode` and :func:`read_episode`. No compression; ``particles/pos`` is chunked along
  time.
- Windows: :class:`EpisodeWindows` reads the files of one split as recurrent windows, in the form the network
  takes.

What the network sees of the rigid object in a window:

- A kinematic object (lateral and vertical push) uses the state recorded after the step. That is the command the
  MPM step saw.
- A dynamic object (impact, rest) uses the state before the step, with the release velocity restored at the
  release step.
- Sand-only episodes (``rest_bed``, ``freefall``) have no body; a box parked far above the bed fills the slot. The
  particles of a free-fall episode are the bed's followed by the released block's.
"""

import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

DT = 0.005
SEED = 20261201
SHAPES = ("cube", "box", "sphere", "cylinder")
CUBE_SIZE = 0.18
"""Edge length of the cube at scale 1 [m]."""
BASE_OBJECT_SIZE = {
    "cube": (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
    "box": (0.24, 0.18, 0.12),
    "sphere": (0.20, 0.20, 0.20),
    "cylinder": (0.20, 0.20, 0.12),
}
"""Bounding size (x, y, z) of each object at scale 1 [m]; the sphere and cylinder diameter is the x size."""
BASE_OBJECT_MASS = 2.0
"""Mass at scale 1 [kg]; mass grows with the cube of the scale (constant density)."""
SIZE_SCALES = (0.75, 1.0, 1.3)
"""Small, medium and large objects: the linear scale applied to `BASE_OBJECT_SIZE`."""
SLOPE_DEG = 10.0
PER_MOTION = 29
"""Episodes per (terrain, motion): 21 train, 3 validation and 5 test."""
SPLIT_COUNTS = {"train": 21, "validation": 3, "test": 5}
PRODUCTION_TERRAINS = ("flat", "slope", "wave")

TERRAINS = {
    "flat": {"bed_size_m": (2.0, 1.2, 0.30), "surface": "flat"},
    "slope": {"bed_size_m": (2.0, 1.2, 0.30), "surface": "slope_p10 or slope_m10, drawn per episode"},
    "wave": {"bed_size_m": (2.0, 1.2, 0.30), "surface": "wave"},
    "deep_flat": {"bed_size_m": (1.0, 0.6, 0.60), "surface": "flat"},
}
CATEGORY_MOTIONS = {
    "lateral_push": ("push_x", "push_y", "circle_cw", "circle_ccw"),
    "vertical_push": ("push_down", "pull_up"),
    "impact": ("drop", "throw"),
    "rest": ("rest",),
    "freefall": ("drop", "toss"),
    "rest_bed": ("rest",),
}
OBJECT_CATEGORIES = ("lateral_push", "vertical_push", "impact", "rest")
SAND_CATEGORIES = ("freefall", "rest_bed")
DURATION_S = {"lateral_push": 6.0, "vertical_push": 6.0, "impact": 6.0, "rest": 10.0, "freefall": 6.0, "rest_bed": 10.0}
SETTLE_S = 2.0
"""Pre-settle time [s]: the bed is simulated without the object before recording starts (t = 0 is the settled bed)."""
MOTION_TYPE = {
    "lateral_push": "kinematic",
    "vertical_push": "kinematic",
    "impact": "dynamic",
    "rest": "dynamic",
    "freefall": "none",
    "rest_bed": "none",
}
"""How the object is driven: prescribed pose and velocity (one-way), a free rigid body (two-way), or no body."""
HOVER_S = 1.0
"""Objects stay at their start height until this time (held up by an anti-gravity wrench if dynamic)."""
EMBED_SPEED = 0.4
"""Speed [m/s] at which a pulled object is first pressed into the bed."""
START_Z = 0.30
"""Height of a kinematic object above the bed surface before it moves [m]."""


def surface_name(terrain: str, sign: int = 1, slope_deg: float = SLOPE_DEG) -> str:
    """Name of the bed surface of a terrain, as ``scene.surface_height`` takes it."""
    if terrain == "slope":
        return f"slope_{'p' if sign > 0 else 'm'}{slope_deg:g}"
    return "flat" if terrain == "deep_flat" else terrain


def object_geometry(shape: str, level: int) -> dict:
    """Size, half height and mass of an object at one of the three scale levels."""
    scale = SIZE_SCALES[level]
    size = [float(v * scale) for v in BASE_OBJECT_SIZE[shape]]
    return {
        "size_level": level,
        "scale": scale,
        "object_size_m": size,
        "half_height": size[2] / 2.0,
        "mass_kg": BASE_OBJECT_MASS * scale**3,
    }


def tray_boundary_boxes(size, wall_top: float = 0.20) -> list[dict]:
    """Tray around the bed (a floor and four walls), the same boxes as ``scene.configure_terrain`` builds."""
    length, width, depth = size
    height = depth + wall_top
    centre = (wall_top - depth) / 2
    return [
        {"name": "floor", "size_m": [length + 0.2, width + 0.2, 0.20], "pose": [0, 0, -depth - 0.10, 0, 0, 0, 1]},
        {"name": "wall_x0", "size_m": [0.10, width + 0.2, height], "pose": [-length / 2 - 0.05, 0, centre, 0, 0, 0, 1]},
        {"name": "wall_x1", "size_m": [0.10, width + 0.2, height], "pose": [length / 2 + 0.05, 0, centre, 0, 0, 0, 1]},
        {"name": "wall_y0", "size_m": [length, 0.10, height], "pose": [0, -width / 2 - 0.05, centre, 0, 0, 0, 1]},
        {"name": "wall_y1", "size_m": [length, 0.10, height], "pose": [0, width / 2 + 0.05, centre, 0, 0, 0, 1]},
    ]


def split_of(k: int) -> str:
    """Split of the ``k``-th episode of one (terrain, motion): 21 train, then 3 validation, then 5 test."""
    return "train" if k < SPLIT_COUNTS["train"] else "validation" if k < 24 else "test"


def episode(terrain: str, category: str, index: int) -> dict:
    """Return the full, reproducible description of one episode."""
    if terrain not in TERRAINS or category not in CATEGORY_MOTIONS:
        raise ValueError(f"Unknown terrain/category: {terrain}/{category}")
    seed = SEED + 100000 * list(TERRAINS).index(terrain) + 1000 * list(CATEGORY_MOTIONS).index(category) + index
    rng = np.random.default_rng(seed)
    motions = CATEGORY_MOTIONS[category]
    motion_index = index % len(motions)
    motion = motions[motion_index]
    k = index // len(motions)
    length, width, _ = TERRAINS[terrain]["bed_size_m"]
    record = {
        "id": f"{terrain}-{category}-{index:04d}",
        "terrain": terrain,
        "terrain_variant": surface_name(terrain, 1 if (k + motion_index) % 2 == 0 else -1),
        "category": category,
        "motion": motion,
        "index": index,
        "k": k,
        "split": split_of(k),
        "seed": seed,
        "particle_seed": seed + 100000,
        "duration_s": DURATION_S[category],
        "dt_s": DT,
        "motion_type": MOTION_TYPE[category],
        "settle_s": SETTLE_S,
        "shape": "bed" if category == "rest_bed" else None,
    }
    if category in OBJECT_CATEGORIES:
        # Cycle through all 12 (shape, size) combinations so that every split covers many of them.
        combo = (5 * k + 3 * motion_index + list(CATEGORY_MOTIONS).index(category)) % (len(SHAPES) * len(SIZE_SCALES))
        record["shape"] = SHAPES[combo % len(SHAPES)]
        record.update(object_geometry(record["shape"], combo // len(SHAPES)))
        record["yaw"] = float(rng.uniform(-np.pi, np.pi))
        record["start_xy"] = [float(rng.uniform(-0.125, 0.125) * length), float(rng.uniform(-0.125, 0.125) * width)]
    if category == "lateral_push":
        record["direction"] = int(rng.choice((-1, 1)))
        record["penetration"] = float(rng.uniform(0.02, 0.06))
        axis = width if motion == "push_y" else length
        record["travel"] = float(rng.uniform(0.33, 0.63) * axis)
        record["radius"] = float(min(rng.uniform(0.18, 0.28), 0.35 * min(length, width)))
        record["lateral_offset"] = float(rng.uniform(-0.1, 0.1) * (length if motion == "push_y" else width))
    elif category == "vertical_push":
        # Constant-velocity motion: push down into the bed, or embed the object and pull it out.
        record["speed"] = float(rng.uniform(0.10, 0.40) if motion == "push_down" else rng.uniform(0.15, 0.40))
        record["z_end"] = float(rng.uniform(-0.10, -0.03))
        record["hold_s"] = float(rng.uniform(0.3, 0.8))
    elif category == "impact":
        record["release_height"] = float(rng.uniform(0.18, 0.35))
        heading = float(rng.uniform(0.0, 2.0 * np.pi))
        speed_max = 1.6 * min(1.0, min(length, width) / 1.2)
        speed = float(rng.uniform(0.4, speed_max)) if motion == "throw" else 0.0
        spin = rng.uniform(-3.0, 3.0, size=2) * float(rng.random() < 0.5)
        record["heading"] = heading
        record["release_velocity"] = [
            speed * np.cos(heading),
            speed * np.sin(heading),
            float(rng.uniform(0.0, 0.5)) if motion == "throw" else 0.0,
            float(spin[0]),
            float(spin[1]),
            0.0,
        ]
        reach = 0.15 * min(length, width)
        record["start_xy"] = [
            -reach * np.cos(heading) * (motion == "throw"),
            -reach * np.sin(heading) * (motion == "throw"),
        ]
    elif category == "freefall":
        w_xy = float(rng.uniform(0.34, 0.52))
        kind = ("cube", "cylinder", "sphere")[(k + motion_index) % 3]
        heading = float(rng.uniform(0.0, 2.0 * np.pi))
        speed = float(rng.uniform(0.4, 1.6)) if motion == "toss" else 0.0
        record.update(
            shape=kind,
            block_size_m=[w_xy, w_xy, w_xy if kind == "sphere" else float(rng.uniform(0.36, 0.56))],
            release_height=float(rng.uniform(0.06, 0.30)),
            heading=heading,
            release_velocity=[
                speed * np.cos(heading),
                speed * np.sin(heading),
                0.25 * (motion == "toss"),
                0.0,
                0.0,
                0.0,
            ],
            start_xy=[-0.15 * np.cos(heading) * (motion == "toss"), -0.15 * np.sin(heading) * (motion == "toss")],
        )
    return record


def episodes(terrain: str, category: str, per_motion: int = PER_MOTION) -> list[dict]:
    """All episodes of one terrain and category: ``per_motion`` for every motion of the category."""
    return [episode(terrain, category, i) for i in range(per_motion * len(CATEGORY_MOTIONS[category]))]


def lateral_pose(t: float, record: dict) -> np.ndarray:
    """Prescribed horizontal pose [x, y, z, qx, qy, qz, qw] in the tray frame; z is relative to the bed surface.

    The object starts on the sand with its bottom ``penetration`` below the surface, stays still until
    ``HOVER_S`` and then moves horizontally at a constant height. There is no vertical motion.
    """
    yaw, motion = record["yaw"], record["motion"]
    z = record["half_height"] - record["penetration"]
    phase = np.clip((t - HOVER_S) / (record["duration_s"] - HOVER_S), 0.0, 1.0)
    direction, travel, offset = record["direction"], record["travel"], record["lateral_offset"]
    if motion == "push_x":
        x, y = direction * travel * (phase - 0.5), offset
    elif motion == "push_y":
        x, y = offset, direction * travel * (phase - 0.5)
    else:
        angle = (-1 if motion == "circle_cw" else 1) * 2.0 * np.pi * phase
        x, y = record["radius"] * (np.cos(angle) - 1.0), record["radius"] * np.sin(angle) + offset
    return np.array([x, y, z, 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=np.float32)


def vertical_pose(t: float, record: dict) -> np.ndarray:
    """Constant-velocity vertical motion; z is the object centre relative to the bed surface.

    ``push_down`` descends at ``speed`` until ``z_end`` and holds. ``pull_up`` first descends at ``EMBED_SPEED`` to
    ``z_end``, holds for ``hold_s`` and then rises at ``speed`` back to the start height.
    """
    x, y = record["start_xy"]
    yaw, speed, z_end = record["yaw"], record["speed"], record["z_end"]
    if record["motion"] == "push_down":
        z = max(z_end, START_Z - speed * max(0.0, t - HOVER_S))
    else:
        down_s = (START_Z - z_end) / EMBED_SPEED
        t_hold = HOVER_S + down_s
        t_rise = t_hold + record["hold_s"]
        if t < t_hold:
            z = max(z_end, START_Z - EMBED_SPEED * max(0.0, t - HOVER_S))
        elif t < t_rise:
            z = z_end
        else:
            z = min(START_Z, z_end + speed * (t - t_rise))
    return np.array([x, y, z, 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=np.float32)


SCHEMA = "particle-nerd-dataset-v1"


def code_commit() -> str:
    """Short hash of the repository commit, stored in every episode file."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_episode(
    path: Path, record: dict, positions: np.ndarray, body: dict | None, attrs: dict, extra: dict | None = None
) -> None:
    """Write one episode atomically.

    Args:
        positions: Particle positions [T, N, 3] (float32).
        body: None, or arrays `pose` [T, 7], `vel` [T, 6], `ext_wrench` [T-1, 6], `contact_wrench` [T-1, 6]
            (one rigid body), or the same with a body axis [T, B, ...] for articulated robots.
        extra: Optional groups of arrays, for example `joints` and `control` for robots.
        attrs: Extra attributes (material, spacing, size, ...).
    """
    steps = positions.shape[0]
    if not np.isfinite(positions).all():
        raise RuntimeError(f"Nonfinite particle positions in {record['id']}")
    partial = path.with_suffix(".partial.h5")
    with h5py.File(partial, "w") as f:
        f.attrs["schema_version"] = SCHEMA
        f.attrs["code_commit"] = code_commit()
        f.attrs["solver"] = "mpm"
        f.attrs["num_steps"] = steps - 1
        f.attrs["num_particles"] = positions.shape[1]
        f.attrs["num_bodies"] = 0 if body is None else (1 if body["pose"].ndim == 2 else body["pose"].shape[1])
        for key in ("id", "terrain", "terrain_variant", "category", "motion", "motion_type", "shape", "split"):
            f.attrs[key if key != "id" else "episode_id"] = str(record.get(key))
        for key in ("seed", "particle_seed"):
            f.attrs[key] = int(record[key])
        f.attrs["dt"] = float(record["dt_s"])
        f.attrs["duration_s"] = float(record["duration_s"])
        f.attrs["settle_s"] = float(record["settle_s"])
        f.attrs["params"] = json.dumps(record)
        for key, value in attrs.items():
            f.attrs[key] = json.dumps(value) if value is None or isinstance(value, (dict, list, tuple)) else value
        f.create_dataset("time", data=np.arange(steps, dtype=np.float32) * np.float32(record["dt_s"]))
        particles = f.create_group("particles")
        particles.create_dataset("pos", data=positions, chunks=(min(16, steps), positions.shape[1], 3))
        if body is not None:
            group = f.create_group("body")
            for name, array in body.items():
                group.create_dataset(name, data=np.asarray(array, dtype=np.float32))
        for group_name, arrays in (extra or {}).items():
            group = f.create_group(group_name)
            for name, array in arrays.items():
                group.create_dataset(name, data=np.asarray(array, dtype=np.float32))
        f.attrs["complete"] = True
    partial.rename(path)


def read_episode(path: Path, particles: bool = True) -> dict:
    """Return `attrs`, `time`, optional `pos` and `body` arrays of one episode."""
    with h5py.File(path, "r") as f:
        if not f.attrs.get("complete", False):
            raise RuntimeError(f"Incomplete episode: {path}")
        out = {"attrs": {k: f.attrs[k] for k in f.attrs}, "time": f["time"][:]}
        if particles:
            out["pos"] = f["particles/pos"][:]
        if "body" in f:
            out["body"] = {k: f["body"][k][:] for k in f["body"]}
    return out


DEFAULT_CATEGORIES = (*OBJECT_CATEGORIES, *SAND_CATEGORIES)
KIND = {"box": 1, "cube": 1, "sphere": 2, "cylinder": 4, "rest": 1, "sandblock": 1}
FAR_POSE = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
FAR_SIZE = (0.24, 0.18, 0.12)
CONTACT_START = {  # first step of a window that contains the interaction, per category / motion
    "lateral_push": 200,
    "vertical_push": 400,
    "impact": 200,
    "rest": 20,
    "freefall": 40,
    "rest_bed": 1000,
}
REST_START_S = {"rest": 0.5, "rest_bed": 0.0}
"""Earliest start [s] of a long resting-bed unroll."""
HOVER_STEPS = 200


class EpisodeWindows:
    """Recurrent windows from the episode files of one split, optionally restricted by category and terrain.

    ``records`` lists the episodes as (description, scene data, file). :meth:`sequence` returns one window:
    the three history frames, the reference positions of the following steps, and the collider input of
    every step.
    """

    def __init__(self, root, split, *, categories=DEFAULT_CATEGORIES, terrains=None):
        # Imported here because recording uses this module and does not need the network package.
        from isaaclab_tasks.contrib.particle_nerd.network.feature_builder import material_features

        self._material_features = material_features
        self.records = []
        for path in sorted(Path(root).rglob("*.hdf5")):
            with h5py.File(path, "r") as handle:
                a = handle.attrs
                if not a.get("complete", False) or str(a.get("schema_version", "")) != SCHEMA:
                    continue
                if str(a["split"]) != split or str(a["category"]) not in categories:
                    continue
                if terrains is not None and str(a["terrain"]) not in terrains:
                    continue
                params = json.loads(a["params"])
                category, material = str(a["category"]), json.loads(a["material"])
                size = json.loads(a["bed_size_m"]) if "bed_size_m" in a and str(a["bed_size_m"]) != "null" else None
                motion_type = str(a["motion_type"])
                num_block = int(a["num_block_particles"]) if "num_block_particles" in a else 0
                spacing = float(a["particle_spacing_m"])
                mass = float(a["particle_mass_kg"])
                object_size = (
                    json.loads(a["object_size_m"])
                    if "object_size_m" in a and str(a["object_size_m"]) != "null"
                    else list(FAR_SIZE)
                )
            shape = {"rest_bed": "rest", "freefall": "sandblock"}.get(category, params["shape"])
            record = {
                **params,
                "id": params["id"],
                "shape": shape,
                "mode": "prescribed" if motion_type == "kinematic" else "dynamic",
                "motion": params["motion"],
                "dt_s": float(params["dt_s"]),
                "rest_unroll": category in ("rest", "rest_bed"),
                "rest_start_s": REST_START_S.get(category, 0.0),
                "contact_start": CONTACT_START[category],
                "num_block_particles": num_block,
                "particle_mass": mass,
                "particle_radius": spacing / 2.0,
            }
            manifest = {
                "material": material,
                "boundary_boxes": tray_boundary_boxes(size or (2.0, 1.2, 0.30)),
                "body_size_m": object_size,
                "dt_s": float(params["dt_s"]),
            }
            self.records.append((record, manifest, path))
        if not self.records:
            raise ValueError(f"No {split} trajectories found in {root} for categories {categories}")
        ids = [r[0]["id"] for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate trajectory IDs in the dataset")

    def num_frames(self, index) -> int:
        with h5py.File(self.records[index][2], "r") as handle:
            return int(handle["particles/pos"].shape[0])

    def sequence(self, rng, horizon, device, *, index=None, start=None):
        record, manifest, path = self.records[int(rng.integers(len(self.records))) if index is None else index]
        dt = record["dt_s"]
        category = record["category"]
        with h5py.File(path, "r") as handle:
            frames = handle["particles/pos"].shape[0]
            final = frames - horizon - 1
            if start is None:
                start = int(rng.integers(final + 1))
            if start < 0 or start > final:
                raise ValueError("Requested window exceeds this trajectory")
            lower = max(0, start - 2)
            positions = handle["particles/pos"][lower : start + horizon + 1]
            if lower == 0 and start < 2:
                if category == "freefall" and record["motion"] == "toss":
                    # The block is released with a known velocity: encode it in the history (the bed stays at rest).
                    velocity = np.zeros((positions.shape[1], 3), dtype=np.float32)
                    velocity[positions.shape[1] - record["num_block_particles"] :] = np.asarray(
                        record["release_velocity"][:3], dtype=np.float32
                    )
                    padding = np.stack([positions[0] - m * dt * velocity for m in range(2 - start, 0, -1)])
                    padding = padding.astype(positions.dtype)
                else:
                    padding = np.repeat(positions[:1], 2 - start, axis=0)
                positions = np.concatenate((padding, positions))
            if "body" in handle:
                if record["mode"] == "prescribed":
                    shift = start + 1  # the command that the MPM step saw is the state recorded after the step
                    pose = handle["body/pose"][shift : shift + horizon]
                    vel = handle["body/vel"][shift : shift + horizon]
                else:
                    pose = handle["body/pose"][start : start + horizon]
                    vel = handle["body/vel"][start : start + horizon].copy()
                    if category == "impact" and start <= HOVER_STEPS < start + horizon:
                        vel[HOVER_STEPS - start] = np.asarray(record["release_velocity"], dtype=np.float32)
                body = np.concatenate((pose, vel), axis=-1)
            else:
                body = np.tile(np.concatenate((FAR_POSE, np.zeros(6, dtype=np.float32)))[None], (horizon, 1))

        def tensor(value):
            return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)

        boxes = manifest["boundary_boxes"]
        boundary_pose = tensor([w["pose"] for w in boxes])
        poses = torch.cat((tensor(body[:, None, :7]), boundary_pose[None].expand(horizon, -1, -1)), dim=1)
        velocities = torch.cat((tensor(body[:, None, 7:]), torch.zeros((horizon, 5, 6), device=device)), dim=1)
        kinds = torch.cat(
            (
                torch.tensor([KIND[record["shape"]]], device=device, dtype=torch.int32),
                torch.ones(5, device=device, dtype=torch.int32),
            )
        )
        sizes = tensor([manifest["body_size_m"], *[w["size_m"] for w in boxes]]) / 2
        material = self._material_features(manifest, record["particle_mass"], record["particle_radius"], dt)
        return {
            "history": tensor(positions[:3].transpose(1, 0, 2)).contiguous(),
            "targets": tensor(positions[3:]),
            "poses": poses.contiguous(),
            "velocities": velocities.contiguous(),
            "material": material.to(device),
            "kinds": kinds,
            "sizes": sizes.contiguous(),
            "friction": tensor([1.0] * 6),
            "dt": dt,
            "id": record["id"],
            "start": start,
            "shape": record["shape"],
            "mode": record["mode"],
        }
