# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproducible object trajectories; splits are assigned to complete episodes."""

import numpy as np

SHAPES = ("box", "sphere", "cylinder")
ALL_SHAPES = ("box", "sphere", "cylinder", "freefall")
DT = 0.005
DURATION = 6.0
SEED = 20261110
SIZES = {
    "box": (0.24, 0.18, 0.12),
    "sphere": (0.20, 0.20, 0.20),
    "cylinder": (0.20, 0.20, 0.12),
    "freefall": (0.48, 0.48, 0.48),
}


def freefall_episodes():
    """Return 400 dynamic 6-second free-fall sand block/column drop episodes on open flat ground."""
    records = []
    shape_index = 3
    for index in range(400):
        seed = SEED + 1000 * shape_index + index
        rng = np.random.default_rng(seed)
        w_xy = float(rng.uniform(0.34, 0.52))
        h_z = float(rng.uniform(0.36, 0.56))
        col_kind = "cylinder" if (index % 2 == 1) else "box"
        records.append(
            {
                "id": f"freefall-{index:04d}",
                "shape": "freefall",
                "column_shape": col_kind,
                "block_size_m": [w_xy, w_xy, h_z],
                "mode": "dynamic",
                "motion": "drop",
                "split": "validation" if index % 10 == 8 else "test" if index % 10 == 9 else "train",
                "seed": seed,
                "particle_seed": seed + 100000,
                "duration_s": DURATION,
                "dt_s": DT,
                "terrain": "flat",
                "yaw": 0.0,
                "direction": 1,
                "y": 0.0,
                "travel": 0.0,
                "depth": 0.0,
                "radius": 0.5 * w_xy,
                "entry_s": 0.0,
                "release_height": float(rng.uniform(0.06, 0.30)),
                "speed": 0.0,
                "spin": 0.0,
            }
        )
    return records


def freefall_toss_episodes(count: int = 240):
    """Return free-fall sand block/column/sphere episodes released with and without a horizontal velocity.

    These supplement the vertical drops of ``freefall_episodes``. A quarter are dropped from rest and the rest are
    tossed with a random horizontal speed (and the fixed upward component of ``release_velocity``), so the
    training set contains free flight with nonzero horizontal momentum.
    """
    records = []
    shape_index = 4
    kinds = ("box", "cylinder", "sphere")
    for index in range(count):
        seed = SEED + 1000 * shape_index + index
        rng = np.random.default_rng(seed)
        w_xy = float(rng.uniform(0.34, 0.52))
        kind = kinds[index % 3]
        h_z = w_xy if kind == "sphere" else float(rng.uniform(0.36, 0.56))
        throw = index % 4 != 0
        records.append(
            {
                "id": f"fftoss-{index:04d}",
                "shape": "freefall",
                "column_shape": kind,
                "block_size_m": [w_xy, w_xy, h_z],
                "mode": "dynamic",
                "motion": "throw" if throw else "drop",
                "split": "validation" if index % 10 == 8 else "test" if index % 10 == 9 else "train",
                "seed": seed,
                "particle_seed": seed + 100000,
                "duration_s": DURATION,
                "dt_s": DT,
                "terrain": "flat",
                "yaw": 0.0,
                "direction": 1,
                "y": 0.0,
                "travel": 0.0,
                "depth": 0.0,
                "radius": 0.5 * w_xy,
                "entry_s": 0.0,
                "release_height": float(rng.uniform(0.06, 0.30)),
                "speed": float(rng.uniform(0.4, 1.6)) if throw else 0.0,
                "spin": 0.0,
            }
        )
    return records


def support_episodes(shape: str, count: int = 20, duration_s: float = 20.0):
    """Return long drop-and-support episodes of one free object resting on the bed.

    The object is released at 1 s from a random height, position and yaw, and the trajectory runs on long
    enough for the object and the bed under it to come to rest. Long unrolls train on the resting part.
    """
    records = []
    shape_index = SHAPES.index(shape)
    for index in range(count):
        seed = SEED + 1000 * (6 + shape_index) + index
        rng = np.random.default_rng(seed)
        records.append(
            {
                "id": f"{shape}-support-{index:04d}",
                "shape": shape,
                "mode": "dynamic",
                "motion": "drop",
                "split": "validation" if index % 10 == 8 else "test" if index % 10 == 9 else "train",
                "seed": seed,
                "particle_seed": seed + 100000,
                "duration_s": float(duration_s),
                "dt_s": DT,
                "terrain": "flat",
                "yaw": float(rng.uniform(-np.pi, np.pi)),
                "direction": int(rng.choice((-1, 1))),
                "y": float(rng.uniform(-0.12, 0.12)),
                "travel": float(rng.uniform(0.65, 1.25)),
                "depth": float(rng.uniform(-0.06, 0.01)),
                "radius": float(rng.uniform(0.18, 0.28)),
                "entry_s": float(rng.uniform(0.7, 1.3)),
                "release_height": float(rng.uniform(0.18, 0.35)),
                "speed": float(rng.uniform(0.2, 0.65)),
                "spin": float(rng.uniform(-3.0, 3.0)),
                "rest_unroll": True,
                "rest_start_s": 2.5,
            }
        )
    return records


def rest_episodes(count: int = 48, duration_s: float = 20.0):
    """Return long resting sand-bed episodes in the finite tray (no object), differing in particle jitter.

    The bed settles from its jittered initial lattice and then stays at rest, so the late part of every
    trajectory shows the equilibrium that a long rollout must reproduce.
    """
    records = []
    for index in range(count):
        seed = SEED + 1000 * 5 + index
        records.append(
            {
                "id": f"rest-{index:04d}",
                "shape": "rest",
                "column_shape": "bed",
                "block_size_m": [0.4, 0.4, 0.4],
                "mode": "dynamic",
                "motion": "rest",
                "split": "validation" if index % 10 == 8 else "test" if index % 10 == 9 else "train",
                "seed": seed,
                "particle_seed": seed + 100000,
                "duration_s": float(duration_s),
                "dt_s": DT,
                "terrain": "flat",
                "yaw": 0.0,
                "direction": 1,
                "y": 0.0,
                "travel": 0.0,
                "depth": 0.0,
                "radius": 0.0,
                "entry_s": 0.0,
                "release_height": 0.0,
                "speed": 0.0,
                "spin": 0.0,
                "rest_unroll": True,
                "rest_start_s": 0.0,
            }
        )
    return records


def episodes(include_freefall: bool = False):
    """Return 600 object episodes (plus 400 free-fall sand episodes when include_freefall=True)."""
    records = []
    for shape_index, shape in enumerate(SHAPES):
        for index in range(200):
            seed = SEED + 1000 * shape_index + index
            rng = np.random.default_rng(seed)
            prescribed = index < 120
            motion = (
                ("horizontal", "vertical", "circle", "hold")[index // 30]
                if prescribed
                else ("drop", "slide", "throw", "tumble")[(index - 120) // 20]
            )
            records.append(
                {
                    "id": f"{shape}-{index:04d}",
                    "shape": shape,
                    "mode": "prescribed" if prescribed else "dynamic",
                    "motion": motion,
                    "split": "validation" if index % 10 == 8 else "test" if index % 10 == 9 else "train",
                    "seed": seed,
                    "particle_seed": seed + 100000,
                    "duration_s": DURATION,
                    "dt_s": DT,
                    "terrain": "flat",
                    "yaw": float(rng.uniform(-np.pi, np.pi)),
                    "direction": int(rng.choice((-1, 1))),
                    "y": float(rng.uniform(-0.12, 0.12)),
                    "travel": float(rng.uniform(0.65, 1.25)),
                    "depth": float(rng.uniform(-0.06, 0.01)),
                    "radius": float(rng.uniform(0.18, 0.28)),
                    "entry_s": float(rng.uniform(0.7, 1.3)),
                    "release_height": float(rng.uniform(0.18, 0.35)),
                    "speed": float(rng.uniform(0.2, 0.65)),
                    "spin": float(rng.uniform(-3.0, 3.0)),
                }
            )
    if include_freefall:
        records.extend(freefall_episodes())
    return records


def pose(t, record):
    """Prescribed collider pose in the tray frame (xyz, xyzw)."""
    direction, travel = record["direction"], record["travel"]
    yaw = record["yaw"]
    x = -direction * travel / 2 if record["motion"] == "horizontal" else 0.0
    y = record["y"]
    u = np.clip((t - 1.0) / record["entry_s"], 0.0, 1.0)
    blend = u * u * (3.0 - 2.0 * u)
    target = record["depth"] + (0.06 if record["shape"] == "sphere" else 0.0)
    z = (1 - blend) * 0.30 + blend * target
    phase = np.clip((t - 1.0 - record["entry_s"]) / (DURATION - 1.0 - record["entry_s"]), 0.0, 1.0)
    if record["motion"] == "horizontal":
        x += direction * travel * phase
    elif record["motion"] == "circle":
        angle = direction * 2.0 * np.pi * phase
        x = record["radius"] * (np.cos(angle) - 1.0)
        y += record["radius"] * np.sin(angle)
    elif record["motion"] == "vertical":
        z -= 0.06 * np.sin(np.pi * phase) ** 2
    return np.array([x, y, z, 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=np.float32)


def command(t, record):
    """Analytically prescribed pose with a forward-difference linear velocity."""
    current = pose(t, record)
    velocity = (pose(t + DT, record)[:3] - current[:3]) / DT
    return np.concatenate((current, velocity, np.zeros(3, dtype=np.float32)))


def release_velocity(record):
    """Initial free-body velocities; no pose overwrites occur after release."""
    velocity = np.zeros(6, dtype=np.float32)
    if record["motion"] in ("slide", "throw", "tumble"):
        velocity[0] = record["direction"] * record["speed"]
    if record["motion"] == "throw":
        velocity[2] = 0.25
    if record["motion"] == "tumble":
        velocity[4] = record["spin"]
    return velocity
