# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode sampling for the terrain / category dataset. Light-weight: no simulator imports."""

import numpy as np

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
    """Particle-bed surface used by `core.terrain.surface_height`."""
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
    """Retaining tray around the bed (floor and four walls), identical to `core.terrain.configure_terrain`."""
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


def open_ground_boundary_boxes(depth: float = 0.30) -> list[dict]:
    """Open flat ground: a floor and four hidden walls 10 m away (free-fall sand blocks)."""
    names = ("wall_x0", "wall_x1", "wall_y0", "wall_y1")
    walls = [
        ([0.1, 4.0, 0.5], [-10, 0, 0]),
        ([0.1, 4.0, 0.5], [10, 0, 0]),
        ([4.0, 0.1, 0.5], [0, -10, 0]),
        ([4.0, 0.1, 0.5], [0, 10, 0]),
    ]
    return [{"name": "floor", "size_m": [4.0, 4.0, 0.2], "pose": [0, 0, -depth - 0.1, 0, 0, 0, 1]}] + [
        {"name": n, "size_m": s, "pose": [*p, 0, 0, 0, 1]} for n, (s, p) in zip(names, walls, strict=True)
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


ROBOT_BED_M = {
    "franka": (1.2, 1.0, 0.30),
    "g1": (3.6, 1.6, 0.30),
    "go2": (3.0, 1.2, 0.30),
}
"""Flat bed size (x, y, depth) per robot; every bed is 0.30 m deep, as in the object dataset."""
ROBOT_WALL_TOP_M = {"franka": 0.04, "g1": 0.20, "go2": 0.20}
ROBOT_DURATION_S = {"franka": 10.0, "g1": 6.0, "go2": 6.0}
ROBOT_MOTION = {
    "franka": "shovel_push",
    "g1": "rl_walk",
    "go2": "rl_walk",
}
ROBOTS = tuple(ROBOT_BED_M)


PRODUCTION_ROBOTS = ("franka", "g1")
ROBOT_SLOPE_DEG = {"franka": 10.0, "g1": 5.0, "go2": 5.0}
"""Slope angle per robot: the G1's 3.6 m bed would run out of sand depth at 10 degrees."""


def robot_episode(robot: str, index: int, terrain: str = "flat") -> dict:
    """Return the reproducible description of one robot episode (0.30 m bed, pre-settled)."""
    if robot not in ROBOTS or terrain not in PRODUCTION_TERRAINS:
        raise ValueError(f"Unknown robot/terrain: {robot}/{terrain}")
    seed = SEED + 10_000_000 + 100_000 * PRODUCTION_TERRAINS.index(terrain) + 1000 * ROBOTS.index(robot) + index
    rng = np.random.default_rng(seed)
    record = {
        "id": f"{terrain}-robot-{robot}-{index:04d}",
        "terrain": terrain,
        "terrain_variant": surface_name(terrain, 1 if index % 2 == 0 else -1, ROBOT_SLOPE_DEG[robot]),
        "category": "robot",
        "robot": robot,
        "motion": ROBOT_MOTION[robot],
        "motion_type": "dynamic",
        "index": index,
        "k": index,
        "split": split_of(index),
        "seed": seed,
        "particle_seed": seed + 100000,
        "duration_s": ROBOT_DURATION_S[robot],
        "dt_s": DT,
        "settle_s": SETTLE_S,
        "shape": robot,
    }
    if robot == "franka":
        # Three long arc sweeps around the base at different radii, with the shovel lifted between them.
        radii = rng.permutation([0.42, 0.52, 0.62])
        record.update(
            theta_start=float(rng.uniform(-0.52, -0.46)),
            theta_span=float(rng.uniform(0.94, 1.02)),
            pass_radius=[float(r) for r in radii],
            height=float(rng.uniform(-0.01, 0.02)),
        )
    elif robot == "g1":
        record.update(forward_speed=float(rng.uniform(0.25, 0.45)))
    elif robot == "go2":
        record.update(
            command=[float(rng.uniform(0.25, 0.55)), float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.4, 0.4))]
        )
    return record
