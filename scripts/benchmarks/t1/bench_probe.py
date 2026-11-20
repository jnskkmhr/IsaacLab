# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""T1 probe benchmark: a PD-driven box on a gantry moving through MPM sand, in six terrain modes.

The probe replaces the robot of the unified G1 MPM tasks. It is built entirely by overriding the configuration of
those tasks, so the terrain machinery (fixed bed, streaming pool, tile cache, sleeping), the coupled MJWarp/MPM
solver and the sand settings are the ones used elsewhere. Only the rigid body and the MDP terms differ.

Modes are ``--terrain`` times ``--sleep``:

* ``full``         -- fixed bed covering the whole terrain (``IsaacContrib-Velocity-G1-29dof-MPM-Flat-Play``).
* ``stream``       -- streaming pool without a tile cache (``IsaacContrib-Velocity-G1-29dof-MPM-Play``).
* ``stream_cache`` -- streaming pool with a tile cache (same task).

``--sleep on`` applies strain activation inside a region tracked around the box, in every terrain mode.

The terrain of one environment is ``--length_x`` by ``--length_y``, centered on the environment origin. Sand
depth, voxel size, sampling and material are those of the streaming task's sand layer in every mode.

Trajectories (target joint positions at the control rate, absolute metres):

* ``rest``       -- hover above the sand.
* ``indent``     -- press the box down at constant speed, hold, retract.
* ``drag``       -- lower to ``--drag_depth``, then translate +X from one end margin to the other.
* ``outback``    -- the same drag, then back along the same line.
* ``lemniscate`` -- lower, then follow a figure-eight at constant path speed for ``--laps`` laps.

Measured per control step: box force from the sand (Fx, Fy, Fz, world frame), box pose, step time; terrain
height maps at the end of chosen phases. Peak VRAM and GPU utilization are polled from ``nvidia-smi``.

Example:
    uv run python scripts/benchmarks/t1/bench_probe.py --terrain stream_cache --sleep on --trajectory drag \
        --length_x 6.4 --out logs/t1/probe_stream_cache_sleep_drag_x6.4.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import MISSING

import numpy as np

TASKS = {
    "full": "IsaacContrib-Velocity-G1-29dof-MPM-Flat-Play",
    "stream": "IsaacContrib-Velocity-G1-29dof-MPM-Play",
    "stream_cache": "IsaacContrib-Velocity-G1-29dof-MPM-Play",
}
PRESETS = {
    ("full", "off"): "solver_cfg_all_particle",
    # The fixed bed has no preset with strain activation inside a region; it is set on top of this one.
    ("full", "on"): "solver_cfg_all_particle",
    ("stream", "off"): "solver_cfg_all_active_no_cache",
    ("stream", "on"): "solver_cfg_no_cache",
    ("stream_cache", "off"): "solver_cfg_all_active_cache",
    ("stream_cache", "on"): "solver_cfg_cache",
}
BOX_BODY = "box"
"""Leaf name of the gantry link that the streaming pool and the activity region follow."""
BOX_PATH = r"/World/envs/env_[^/]+/Robot/links/box"
END_MARGIN = 0.8
"""Distance kept between the box and the terrain's X or Y end [m]: half of the pool's total width."""
HOVER = 0.15
"""Box-center height above the undisturbed surface when hovering [m]."""
CONTROL_DT = 0.02
REGION_SIZE = (1.2, 1.2, 3.0)
"""Activity region around the box [m], as in the streaming presets."""


# --------------------------------------------------------------------------------------
# GPU memory and utilization
# --------------------------------------------------------------------------------------
class GpuMonitor:
    """Poll ``nvidia-smi`` for this process's and the device's memory [MiB] and the device utilization [%]."""

    def __init__(self, period_s: float = 0.25):
        self.period_s = period_s
        self.pid = os.getpid()
        self.process_peak = 0.0
        self.device_peak = 0.0
        self.samples = 0
        self.utilization: list[tuple[float, float]] = []
        """``(time.perf_counter(), utilization.gpu)`` of every sample."""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _query(args: list[str]) -> list[str]:
        out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]

    def sample(self) -> tuple[float, float]:
        """Return ``(process_MiB, device_MiB)`` now."""
        proc = 0.0
        for line in self._query(["--query-compute-apps=pid,used_memory"]):
            pid, used = (item.strip() for item in line.split(","))
            if int(pid) == self.pid:
                proc = float(used)
        used, utilization = (float(v) for v in self._query(["--query-gpu=memory.used,utilization.gpu"])[0].split(","))
        self.utilization.append((time.perf_counter(), utilization))
        self.process_peak = max(self.process_peak, proc)
        self.device_peak = max(self.device_peak, used)
        self.samples += 1
        return proc, used

    def window(self, start: float, end: float) -> dict:
        """Utilization statistics of the samples taken between two ``time.perf_counter()`` values."""
        values = np.array([u for t, u in self.utilization if start <= t <= end])
        if values.size == 0:
            return {"n": 0}
        return {"mean": float(values.mean()), "max": float(values.max()), "n": int(values.size)}

    def _run(self):
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                self.sample()
            self._stop.wait(self.period_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5.0)


def _sync():
    import torch
    import warp as wp

    torch.cuda.synchronize()
    wp.synchronize()


def _git_state() -> dict:
    """Commit and dirty flag of IsaacLab and of the Newton checkout it runs against."""
    if os.environ.get("T1_COMMIT"):  # set by a remote runner; the synced tree has no usable .git
        return {"commit": os.environ["T1_COMMIT"], "dirty": os.environ.get("T1_DIRTY") == "1"}

    def state(path):
        def run(*cmd):
            return subprocess.run(cmd, capture_output=True, text=True, cwd=path).stdout.strip()

        return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}

    import newton

    return {**state(os.path.dirname(__file__)), "newton": state(os.path.dirname(newton.__file__))}


# --------------------------------------------------------------------------------------
# Gantry: fixed base -> X slide -> Y slide -> Z slide carrying the box
# --------------------------------------------------------------------------------------
def make_gantry_cfg(args: argparse.Namespace, start: tuple[float, float]):
    """Return the articulation config of the gantry; defined lazily because it needs the simulation modules."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.utils import configclass

    @sim_utils.clone
    def spawn_gantry(prim_path, cfg, translation=None, orientation=None, **kwargs):
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics

        stage = sim_utils.get_current_stage()
        root = sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
        UsdPhysics.ArticulationRootAPI.Apply(root)
        root.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        bx, by, bz = cfg.box_size
        box_inertia = tuple(cfg.box_mass * s / 12.0 for s in (by * by + bz * bz, bx * bx + bz * bz, bx * bx + by * by))
        UsdGeom.Xform.Define(stage, f"{prim_path}/links")
        for name, mass, inertia in (
            ("base", 100.0, (10.0, 10.0, 10.0)),
            ("carriage_x", cfg.carriage_mass, (1.0, 1.0, 1.0)),
            ("carriage_y", cfg.carriage_mass, (1.0, 1.0, 1.0)),
            (BOX_BODY, cfg.box_mass, box_inertia),
        ):
            body = UsdGeom.Xform.Define(stage, f"{prim_path}/links/{name}").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            mass_api = UsdPhysics.MassAPI.Apply(body)
            mass_api.CreateMassAttr(mass)
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        fixed = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/base_fixed_joint")
        fixed.CreateBody1Rel().SetTargets([f"{prim_path}/links/base"])
        for name, parent, child, axis, lower, upper in (
            ("slide_x", "base", "carriage_x", "X", -cfg.travel_x, cfg.travel_x),
            ("slide_y", "carriage_x", "carriage_y", "Y", -cfg.travel_y, cfg.travel_y),
            ("slide_z", "carriage_y", BOX_BODY, "Z", -cfg.travel_z, cfg.travel_z),
        ):
            joint = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/{name}")
            joint.CreateBody0Rel().SetTargets([f"{prim_path}/links/{parent}"])
            joint.CreateBody1Rel().SetTargets([f"{prim_path}/links/{child}"])
            joint.CreateAxisAttr(axis)
            joint.CreateLowerLimitAttr(lower)
            joint.CreateUpperLimitAttr(upper)
            joint.CreateLocalPos0Attr(Gf.Vec3f(0.0))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(0.0)
        shape = sim_utils.CuboidCfg(
            size=cfg.box_size,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.35, 0.7), roughness=0.6),
        )
        shape.func(f"{prim_path}/links/{BOX_BODY}/shape", shape)
        return root

    @configclass
    class GantrySpawnerCfg(SpawnerCfg):
        func: Callable = spawn_gantry
        box_size: tuple[float, float, float] = MISSING
        box_mass: float = MISSING
        carriage_mass: float = 10.0
        travel_x: float = 500.0
        travel_y: float = 500.0
        travel_z: float = 1.0

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=GantrySpawnerCfg(box_size=tuple(args.box_size), box_mass=args.box_mass),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), joint_pos={"slide_x": start[0], "slide_y": start[1], "slide_z": HOVER}
        ),
        actuators={
            "gantry": ImplicitActuatorCfg(
                joint_names_expr=["slide_x", "slide_y", "slide_z"],
                stiffness=args.stiffness,
                damping=args.damping,
                joint_effort_limit=1.0e5,
                joint_velocity_limit=10.0,
            )
        },
    )


# --------------------------------------------------------------------------------------
# Trajectories
# --------------------------------------------------------------------------------------
def _ramp(a: float, b: float, seconds: float) -> np.ndarray:
    n = max(int(round(seconds / CONTROL_DT)), 1)
    return np.linspace(a, b, n + 1)[1:]


def lemniscate_path(a: float, speed: float, laps: int) -> np.ndarray:
    """Points ``(n, 2)`` [m] of ``x = a sin(t), y = (a / 2) sin(2 t)`` at constant path speed, one per control step."""
    t = np.linspace(0.0, 2.0 * np.pi * laps, 20000 * laps + 1)
    xy = np.stack((a * np.sin(t), 0.5 * a * np.sin(2.0 * t)), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    samples = np.arange(1, int(round(arc[-1] / (speed * CONTROL_DT))) + 1) * speed * CONTROL_DT
    return np.stack((np.interp(samples, arc, xy[:, 0]), np.interp(samples, arc, xy[:, 1])), axis=1)


def trajectory_geometry(args: argparse.Namespace) -> dict:
    """Terrain size of one environment and the box start position, both in the environment frame [m]."""
    if args.trajectory == "lemniscate":
        a = args.lemniscate_a
        return {"length_x": 2.0 * a + 2.0 * END_MARGIN, "length_y": a + 2.0 * END_MARGIN, "start": (0.0, 0.0)}
    if args.trajectory in ("drag", "outback"):
        return {
            "length_x": args.length_x,
            "length_y": args.length_y,
            "start": (-args.length_x / 2.0 + END_MARGIN, 0.0),
            "travel": args.length_x - 2.0 * END_MARGIN,
        }
    return {"length_x": args.length_x, "length_y": args.length_y, "start": (0.0, 0.0)}


def planned_pool_tiles(
    points: np.ndarray, geometry: dict, tile: float, shape: int, slack: float = 0.05
) -> tuple[int, int]:
    """Tiles the streaming pool loads while its anchor follows ``points`` (``(n, 2)`` [m]), and tiles it touches.

    The pool center is the anchor position snapped to the tile size and clamped so that the pool stays on the
    terrain, as the solver does. ``slack`` also counts the tiles reached if the anchor is off its target by that
    much in any direction, which covers the gantry's tracking error. Sand leaving the pool can land in the tiles
    bordering it, which the cache also has to remember, so the second count includes a one-tile border.
    """
    half = np.array([geometry["length_x"], geometry["length_y"]]) / 2.0
    lower = np.ceil(-half / tile - 1.0e-9).astype(int)
    upper = np.floor(half / tile + 1.0e-9).astype(int) - shape
    loaded = set()
    for dx, dy in ((0.0, 0.0), (slack, 0.0), (-slack, 0.0), (0.0, slack), (0.0, -slack)):
        centers = np.floor((points + (dx, dy)) / tile + 0.5).astype(int)
        begins = np.unique(np.clip(centers - shape // 2, lower, np.maximum(lower, upper)), axis=0)
        for bx, by in begins:
            loaded.update((bx + i, by + j) for i in range(shape) for j in range(shape))
    touched = {(x + i, y + j) for x, y in loaded for i in (-1, 0, 1) for j in (-1, 0, 1)}
    return len(loaded), len(touched)


SNAPSHOT_PHASES = ("presettle", "out", "turn", "back", "press", "hold", "end")
"""Phases whose last step saves a terrain height map; phases named ``snap_*`` or ``lap_*`` do too."""


def build_trajectory(args: argparse.Namespace, geometry: dict, surface_z: float) -> tuple[np.ndarray, dict]:
    """Return target ``(T, 3)`` joint positions [m] and the end step of each named phase.

    Depths are measured from ``surface_z``, the settled sand surface height measured in the same run, so that
    the box meets the sand at the same relative depth in every terrain mode.
    """
    half_height = args.box_size[2] / 2.0
    hover_z = surface_z + HOVER
    z_hold = surface_z - args.drag_depth + half_height  # underside is drag_depth below the surface
    x0, y0 = geometry["start"]
    seg: list[np.ndarray] = []
    marks: dict[str, int] = {}

    def add(name: str, xs, ys, zs) -> None:
        n = max(np.size(xs), np.size(ys), np.size(zs))
        arr = np.stack([np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in (xs, ys, zs)], axis=1)
        marks[name] = sum(len(s) for s in seg) + n
        seg.append(arr)

    if args.trajectory == "rest":
        for second in range(1, args.rest_seconds + 1):
            add(f"snap_{second}s", [x0] * 50, [y0], [hover_z])
    elif args.trajectory == "indent":
        z_press = surface_z - args.indent_depth + half_height
        add("press", [x0], [y0], _ramp(hover_z, z_press, (hover_z - z_press) / args.speed))
        add("hold", [x0] * 50, [y0], [z_press])
        add("retract", [x0], [y0], _ramp(z_press, hover_z, (hover_z - z_press) / args.speed))
    else:
        add("lower", [x0], [y0], _ramp(hover_z, z_hold, (hover_z - z_hold) / args.lower_speed))
        add("settle_in", [x0] * 25, [y0], [z_hold])
        if args.trajectory == "lemniscate":
            path = lemniscate_path(args.lemniscate_a, args.speed, args.laps)
            per_lap = len(path) // args.laps
            for lap in range(args.laps):
                part = path[lap * per_lap : (lap + 1) * per_lap if lap < args.laps - 1 else len(path)]
                add(f"lap_{lap + 1}", x0 + part[:, 0], y0 + part[:, 1], [z_hold])
        else:
            travel = geometry["travel"]
            out = _ramp(x0, x0 + travel, travel / args.speed)
            add("out", out, [y0], [z_hold])
            if args.trajectory == "outback":
                add("turn", [out[-1]] * 25, [y0], [z_hold])
                add("back", _ramp(out[-1], x0, travel / args.speed), [y0], [z_hold])
        add("end", [seg[-1][-1, 0]] * 25, [seg[-1][-1, 1]], [z_hold])
    return np.concatenate(seg, axis=0), marks


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--terrain", choices=sorted(TASKS), required=True)
    parser.add_argument("--sleep", choices=("off", "on"), default="off")
    parser.add_argument("--trajectory", choices=("rest", "indent", "drag", "outback", "lemniscate"), default="drag")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--length_x", type=float, default=6.4, help="Terrain length of one environment [m].")
    parser.add_argument("--length_y", type=float, default=1.6, help="Terrain width of one environment [m].")
    parser.add_argument("--speed", type=float, default=1.0, help="Translation speed, or press speed for indent [m/s].")
    parser.add_argument("--lower_speed", type=float, default=0.5, help="Speed of lowering the box into the sand [m/s].")
    parser.add_argument("--drag_depth", type=float, default=0.06, help="Box underside depth below the surface [m].")
    parser.add_argument("--indent_depth", type=float, default=0.10, help="Press depth of the indentation [m].")
    parser.add_argument("--lemniscate_a", type=float, default=2.4, help="Half-length of the figure-eight [m].")
    parser.add_argument("--laps", type=int, default=2)
    parser.add_argument("--presettle", type=int, default=100, help="Hover steps before the trajectory (sand settles).")
    parser.add_argument("--rest_seconds", type=int, default=6, help="Length of the 'rest' trajectory [s].")
    parser.add_argument("--box_size", type=float, nargs=3, default=(0.2, 0.2, 0.1), metavar=("LX", "LY", "LZ"))
    parser.add_argument("--box_mass", type=float, default=10.0)
    parser.add_argument("--stiffness", type=float, default=1.0e5, help="Gantry PD stiffness [N/m].")
    parser.add_argument("--damping", type=float, default=4.0e3, help="Gantry PD damping [N s/m].")
    parser.add_argument("--depth", type=float, default=None, help="Sand depth [m]; default: the streaming task's.")
    parser.add_argument("--voxel", type=float, default=None, help="Voxel size [m]; default: the streaming task's.")
    parser.add_argument("--material", type=str, default=None, help="Material preset; default: the streaming task's.")
    parser.add_argument(
        "--cache_capacity", type=int, default=None, help="max_cached_particles_per_env; default: fitted to the path."
    )
    parser.add_argument(
        "--cache_tiles", type=int, default=None, help="max_visited_tiles_per_env; default: fitted to the path."
    )
    parser.add_argument("--cache_slots", type=float, default=None, help="simulation_capacity_multiplier.")
    parser.add_argument(
        "--cache_margin", type=float, default=0.2, help="Extra sand allowed in the strips saved by one update."
    )
    parser.add_argument(
        "--cache_unsettled_template",
        action="store_true",
        help="Skip the cache's template settling, so new tiles start from the sampled particles.",
    )
    parser.add_argument("--max_steps", type=int, default=0, help="Stop the trajectory after this many steps (0: no).")
    parser.add_argument("--video_dir", type=str, default=None, help="Record a video of the run into this directory.")
    parser.add_argument("--video_viz", choices=("rtx", "gl"), default="rtx", help="Visualizer used for the video.")
    parser.add_argument("--no_graph", action="store_true", help="Run the physics step without CUDA graph capture.")
    parser.add_argument(
        "--grid_type", choices=("sparse", "dense", "fixed"), default=None, help="MPM grid type; default: the task's."
    )
    parser.add_argument("--active_cells", type=int, default=None, help="Sparse-grid cell reserve per environment.")
    parser.add_argument("--leaf_nodes", type=int, default=None, help="Sparse-grid leaf-node reserve per environment.")
    parser.add_argument("--lower_nodes", type=int, default=None, help="Sparse-grid lower-node reserve per environment.")
    parser.add_argument("--warmup", type=int, default=25, help="Control steps excluded from the timing statistics.")
    parser.add_argument("--profile_physics", action="store_true", help="Time the physics step separately.")
    parser.add_argument("--monitor_period", type=float, default=0.25, help="nvidia-smi sampling interval [s].")
    parser.add_argument("--heightmap_half", type=float, default=0.6, help="Half-size of the height window [m].")
    parser.add_argument("--terrain_map", action="store_true", help="Also save height maps of the whole terrain.")
    parser.add_argument("--activity_every", type=int, default=50, help="Steps between sleep-state samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args(argv)


def configure_probe(env_cfg, args: argparse.Namespace, geometry: dict) -> dict:
    """Replace the G1 pieces of a unified MPM task configuration by the gantry probe; return what was set."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.physics import MPMSleepingCfg, StrainActivationCfg, TrackedRegionActivationCfg
    from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

    import isaaclab.envs.mdp as mdp
    import isaaclab.terrains as terrain_gen
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.terrains import TerrainGeneratorCfg
    from isaaclab.utils import configclass

    from isaaclab_tasks.contrib.velocity_mpm.config.g1_29dof.env_cfg import streaming_scene_cfg
    from isaaclab_tasks.contrib.velocity_mpm.util.solver_setting import get_mpm_solver_cfg

    # One sand layer for every mode: the streaming task's, with the command-line overrides.
    sand = streaming_scene_cfg.G1MPMSceneCfg(num_envs=1, env_spacing=1.0).terrain.sand
    sand.robot_spawn_height = 0.0
    if args.depth is not None:
        sand.particle_depth = args.depth
    if args.voxel is not None:
        sand.voxel_size = args.voxel
    material_name = args.material or streaming_scene_cfg.MATERIAL_NAME
    sand.material = MPMParticleMaterialCfg(**streaming_scene_cfg.MATERIAL_PRESETS[material_name]._asdict())

    length_x, length_y = geometry["length_x"], geometry["length_y"]
    n_side = math.ceil(math.sqrt(args.num_envs))
    mpm = get_mpm_solver_cfg(env_cfg)
    mpm.voxel_size = sand.voxel_size
    if args.grid_type is not None:
        mpm.grid_type = args.grid_type
    gantry = make_gantry_cfg(args, geometry["start"])

    if args.terrain == "full":
        spacing = max(length_x, length_y) + 2.0
        grid = MPMGridCfg(
            lower=(-length_x / 2.0, -length_y / 2.0, -sand.particle_depth),
            upper=(length_x / 2.0, length_y / 2.0, 0.0),
            voxel_size=sand.voxel_size,
            particles_per_cell=sand.particles_per_cell,
            particle_placement="cell_center",
            jitter=sand.jitter,
            material=sand.material,
            visual_color=sand.visual_color,
        )
        # The fixed-bed task settles its bed for one second at creation and restores that snapshot on reset.
        # That settling run goes non-finite for long beds (12.8 x 1.6 m), so every full run starts from the
        # sampled particles instead and settles during the hover that precedes contact.
        env_cfg.mpm_cache_initial_state = False
        # A new scene instance derives the floor and walls from the particle bounds.
        env_cfg.scene = type(env_cfg.scene)(
            num_envs=args.num_envs,
            env_spacing=spacing,
            replicate_physics=True,
            sand=MPMObjectCfg(prim_path="{ENV_REGEX_NS}/Sand", spawn=grid),
        )
    else:
        spacing = max(length_x, length_y)
        terrain = env_cfg.scene.terrain
        terrain.sand = sand
        terrain.terrain_generator = TerrainGeneratorCfg(
            seed=42,
            size=(
                n_side * spacing if args.num_envs > 1 else length_x,
                n_side * spacing if args.num_envs > 1 else length_y,
            ),
            num_rows=1,
            num_cols=1,
            border_width=0.0,
            horizontal_scale=0.1,
            curriculum=False,
            sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg()},
        )
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.scene.env_spacing = spacing
        env_cfg.scene.height_scanner = None
        streaming = mpm.streaming
        streaming.body_names = BOX_BODY
        if streaming.cache is not None and args.cache_slots is not None:
            streaming.cache.simulation_capacity_multiplier = args.cache_slots
    env_cfg.scene.robot = gantry

    if args.sleep == "on":
        # One rule in every terrain mode: the streaming presets' strain activation inside a tracked region.
        mpm.reuse_grid_topology = True
        mpm.sleeping = MPMSleepingCfg(
            strain=StrainActivationCfg(stress_filter_time=0.1),
            region=TrackedRegionActivationCfg(body_names=BOX_BODY, size=REGION_SIZE),
        )
    elif mpm.sleeping is not None:
        raise RuntimeError("The sleep-off preset has sleeping configured.")

    @configclass
    class ProbePolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProbeObservationsCfg:
        policy: ProbePolicyCfg = ProbePolicyCfg()

    @configclass
    class ProbeActionsCfg:
        joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["slide_x", "slide_y", "slide_z"], scale=1.0, use_default_offset=False
        )

    @configclass
    class ProbeRewardsCfg:
        alive = RewTerm(func=mdp.is_alive, weight=0.0)

    @configclass
    class ProbeTerminationsCfg:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

    @configclass
    class ProbeEventsCfg:
        reset_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
        )

    @configclass
    class EmptyCfg:
        pass

    env_cfg.observations = ProbeObservationsCfg()
    env_cfg.actions = ProbeActionsCfg()
    env_cfg.rewards = ProbeRewardsCfg()
    env_cfg.terminations = ProbeTerminationsCfg()
    env_cfg.events = ProbeEventsCfg()
    env_cfg.commands = EmptyCfg()
    env_cfg.curriculum = EmptyCfg()
    env_cfg.foot_body_expr = BOX_BODY
    env_cfg.foot_contact_force_threshold = 1.0
    env_cfg.episode_length_s = 1.0e6
    env_cfg.sim.visualizer_cfgs = []
    env_cfg.video_recorders = []
    env_cfg.seed = args.seed
    # Hand only the box to the particle solver, in every mode, so the modes differ in terrain handling only.
    for proxy in env_cfg.sim.physics.solver_cfg.proxies:
        proxy.bodies = [BOX_PATH]
    if args.no_graph:
        env_cfg.sim.physics.use_cuda_graph = False
    # The tasks reserve 8,192 sparse-grid leaf nodes (8 voxels a side) per environment, far more than a pool
    # needs and fewer than a long bed needs. Reserve what the simulated region's bounding box needs instead: the
    # bed for full, the pool for streaming, with one leaf of headroom above the surface and 25% on top.
    leaf = 8.0 * sand.voxel_size
    region = (length_x, length_y) if args.terrain == "full" else tuple(2.0 * END_MARGIN for _ in range(2))
    extents = (*region, sand.particle_depth + leaf)
    env_cfg.mpm_leaf_node_count_per_world = math.ceil(1.25 * math.prod(math.ceil(e / leaf) + 1 for e in extents))
    lower = 2 * math.prod(math.ceil(e / (16.0 * leaf)) + 1 for e in extents)
    env_cfg.mpm_lower_node_count_per_world = min(max(64, lower), env_cfg.mpm_leaf_node_count_per_world)
    if args.active_cells is not None:
        env_cfg.mpm_active_cell_count_per_world = args.active_cells
    if args.leaf_nodes is not None:
        env_cfg.mpm_leaf_node_count_per_world = args.leaf_nodes
    if args.lower_nodes is not None:
        env_cfg.mpm_lower_node_count_per_world = args.lower_nodes
    env_cfg.validate()
    cache = None if mpm.streaming is None else mpm.streaming.cache
    fitted = None
    if cache is not None:
        # Size the cache to this run: the tiles its path loads, and the sand of those tiles that is not in the pool.
        from isaaclab_newton.sim.spawners.mpm.streaming import sample_particle_tile

        per_tile = len(sample_particle_tile(env_cfg.scene.sand.spawn)[0])
        shape = int(round((mpm.streaming.size[0] + 2.0 * mpm.streaming.margin) / mpm.streaming.tile_size))
        targets, _ = build_trajectory(args, geometry, 0.0)
        loaded, tiles = planned_pool_tiles(targets[:, :2], geometry, mpm.streaming.tile_size, shape)
        # Besides the sand outside the pool, one update saves the strips leaving the pool before it restores the
        # strips entering it: up to a row and a column of tiles, with the sand that piled up in them.
        reserve = math.ceil((1.0 + args.cache_margin) * (2 * shape - 1) * per_tile)
        records = (loaded - shape * shape) * per_tile + reserve
        fitted = {"particles_per_tile": per_tile, "loaded_tiles": loaded, "tiles": tiles, "records": records}
        cache.max_visited_tiles_per_env = tiles if args.cache_tiles is None else args.cache_tiles
        cache.max_cached_particles_per_env = records if args.cache_capacity is None else args.cache_capacity
        if args.cache_unsettled_template:
            # One settling step and no rest check: new tiles then start from the sampled particles, like full.
            cache.settle_time_s = 0.001
            cache.settle_speed = float("inf")
        # The spawner copied the multiplier when the pool was derived; derive it again with the final values.
        env_cfg.validate()
    return {
        "terrain_size_per_env": [length_x, length_y],
        "env_spacing": spacing,
        "box_start": list(geometry["start"]),
        "travel_m": geometry.get("travel"),
        "sand": {
            "depth_m": sand.particle_depth,
            "voxel_m": sand.voxel_size,
            "particles_per_cell": sand.particles_per_cell,
            "jitter_m": sand.jitter,
            "material": material_name,
        },
        "streaming": None
        if mpm.streaming is None
        else {
            "window": list(mpm.streaming.size),
            "margin": mpm.streaming.margin,
            "tile": mpm.streaming.tile_size,
            "body": mpm.streaming.body_names,
        },
        "cache": None
        if cache is None
        else {
            "max_cached_particles_per_env": cache.max_cached_particles_per_env,
            "max_visited_tiles_per_env": cache.max_visited_tiles_per_env,
            "simulation_capacity_multiplier": cache.simulation_capacity_multiplier,
            "fitted": fitted,
        },
        "sleeping": None
        if mpm.sleeping is None
        else {
            "strain": mpm.sleeping.strain is not None,
            "region_body": mpm.sleeping.region.body_names,
            "region_size": list(mpm.sleeping.region.size),
            "keep_material_motion": mpm.sleeping.region.keep_material_motion,
        },
        "grid_reserve_per_env": {
            "active_cells": mpm.max_active_cell_count // args.num_envs,
            "leaf_nodes": mpm.max_leaf_node_count // args.num_envs,
            "lower_nodes": mpm.max_lower_node_count // args.num_envs,
        },
        "initial_state": {
            "settled_bed": bool(env_cfg.mpm_cache_initial_state),
            "settle_time_s": env_cfg.mpm_settle_time if env_cfg.mpm_cache_initial_state else None,
        },
    }


# --------------------------------------------------------------------------------------
# Video: sand particles only, in every terrain mode
# --------------------------------------------------------------------------------------
PARTICLE_COLOR = (0.7, 0.6, 0.4)
"""Color of every drawn sand particle; the streaming task's particle view uses it."""


def configure_particle_video(
    env_cfg, terrain: str, geometry: dict, video_dir: str, prefix: str, steps: int, viz: str = "rtx"
) -> dict:
    """Record the run, hiding ground and walls so that only sand particles and bodies are drawn.

    These settings are for the benchmark videos only. :func:`draw_sand_particles` then completes the sand drawing
    so that all three terrain modes show the sand they hold.

    Returns:
        The launcher settings that select the visualizer.
    """
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.visualizers import VisualizerCfg

    from isaaclab_tasks.contrib.velocity_mpm.util.visualizer import MovingPatchGLBackendCfg, MovingPatchRTXBackendCfg

    if terrain == "full":
        for name in ("bed_floor", "bed_wall_front", "bed_wall_back", "bed_wall_left", "bed_wall_right"):
            asset = getattr(env_cfg.scene, name, None)
            if asset is not None:
                asset.spawn.visible = False
    else:
        env_cfg.scene.terrain.disable_visual = True
        env_cfg.scene.terrain.sand.render_mode = "particles"
        env_cfg.validate()
    span = max(geometry["length_x"], 1.6 * geometry["length_y"])
    env_cfg.sim.visualizer_cfgs = [
        VisualizerCfg(
            # These viewer classes call a hook before every frame, which the sand drawing uses.
            backend=MovingPatchRTXBackendCfg() if viz == "rtx" else MovingPatchGLBackendCfg(show_particles=True),
            eye=(0.0, -0.45 * span - 0.3, 0.5 * span + 0.2),
            lookat=(0.0, -0.04 * span, 0.0),
            headless=True,
            streaming_view=False,
        )
    ]
    env_cfg.video_recorders = [
        VideoRecorderCfg(
            source=f"visualizer:newton_{viz}",
            output_dir=video_dir,
            video_length=steps,
            video_interval=10**9,
            output_filename_prefix=prefix,
        )
    ]
    return {"visualizer": [f"newton_{viz}"]}


OCTAHEDRON_VERTICES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32)
OCTAHEDRON_FACES = np.array(
    [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], dtype=np.int32
)


def draw_saved_sand_rtx(base_env, visualizer) -> None:
    """In the RTX view, also draw the sand a tile cache holds outside its pool.

    The RTX view draws the particles of the model, which for a streaming pool are its simulated window and its
    kinematic ring. It cannot draw another point cloud, so the saved sand is submitted as a mesh of one small
    octahedron per particle, with sphere normals, whenever the cache changes.
    """
    import warp as wp

    cache = getattr(base_env.mpm_streaming, "cache", None)
    if cache is None:
        return
    radius = 1.15 * float(cache.radius)
    color = tuple(base_env.cfg.scene.terrain.sand.visual_color)
    prepare_frame = visualizer._prepare_frame
    drawn = {"revision": -1}

    def prepare_frame_with_saved_sand():
        revision = int(cache.revision.numpy()[0])
        if revision != drawn["revision"]:
            drawn["revision"] = revision
            records = [r for world in range(cache.world_count) for chain in cache.pages[world].values() for r in chain]
            if records:
                centers = cache.saved.q.numpy()[records]
                points = (centers[:, None, :] + radius * OCTAHEDRON_VERTICES[None]).reshape(-1, 3)
                normals = np.broadcast_to(OCTAHEDRON_VERTICES[None], (len(centers), 6, 3)).reshape(-1, 3)
                indices = (OCTAHEDRON_FACES[None] + 6 * np.arange(len(centers), dtype=np.int32)[:, None, None]).ravel()
                device = cache.saved.q.device
                visualizer.log_mesh(
                    "/t1/saved_sand",
                    wp.array(points, dtype=wp.vec3, device=device),
                    wp.array(indices, dtype=wp.int32, device=device),
                    normals=wp.array(np.ascontiguousarray(normals), dtype=wp.vec3, device=device),
                    color=color,
                    roughness=1.0,
                    dynamic=True,
                )
        prepare_frame()

    visualizer._prepare_frame = prepare_frame_with_saved_sand


def draw_sand_particles(base_env) -> None:
    """Draw all the sand a terrain mode holds as particles, before every frame. For the benchmark videos only.

    That is every particle of a fixed bed; for a streaming pool its simulated window and its kinematic ring; and
    with a tile cache also the sand saved outside the pool. In the GL view the task's own particle view, which
    shows only the simulated window of a streaming pool, is replaced by one height-shaded point cloud.
    """
    import warp as wp

    if not base_env.sim.visualizers:
        return
    visualizer = base_env.sim.visualizers[0]
    if hasattr(visualizer, "_renderers"):
        draw_saved_sand_rtx(base_env, visualizer)
        return
    viewer = visualizer._viewer
    backend = base_env.sim.backend_manager.backend
    model = backend.model
    cache = getattr(base_env.mpm_streaming, "cache", None)
    radius = float(model.particle_radius.numpy()[0])
    viewer._log_particles = lambda state: None
    prepare_frame = visualizer._prepare_frame
    drawn = {"revision": -1, "saved": np.zeros((0, 3), dtype=np.float32), "level": 0.0, "frames": 0}

    def prepare_frame_with_sand():
        live = (model.particle_flags.numpy() & 1) != 0
        points = backend.state_0.particle_q.numpy()[live]
        if cache is not None:
            revision = int(cache.revision.numpy()[0])
            if revision != drawn["revision"]:
                drawn["revision"] = revision
                records = [
                    r for world in range(cache.world_count) for chain in cache.pages[world].values() for r in chain
                ]
                drawn["saved"] = cache.saved.q.numpy()[records] if records else np.zeros((0, 3), dtype=np.float32)
            points = np.concatenate((points, drawn["saved"]))
        points = points[np.isfinite(points).all(axis=1)]
        if drawn["frames"] < 110:
            # The undisturbed surface, followed while the sand settles during the first two seconds. A low
            # percentile of the surface heights ignores a pool's kinematic ring, which does not settle.
            lower = points[:, :2].min(axis=0)
            cells = np.ceil((points[:, :2].max(axis=0) - lower) / 0.04).astype(int) + 1
            drawn["level"] = float(np.nanpercentile(height_map(points, lower, tuple(cells)), 35.0))
        drawn["frames"] += 1
        # Darker below the undisturbed surface, lighter above it, so that tracks and berms show without shadows.
        shade = np.clip(1.0 + 0.6 * (points[:, 2] - drawn["level"]) / 0.07, 0.35, 1.2)
        colors = np.clip(shade[:, None] * np.asarray(PARTICLE_COLOR), 0.0, 1.0).astype(np.float32)
        device = model.device
        viewer.log_points(
            "/t1/sand",
            points=wp.array(points, dtype=wp.vec3, device=device),
            radii=wp.full(len(points), radius, dtype=float, device=device),
            colors=wp.array(colors, dtype=wp.vec3, device=device),
        )
        prepare_frame()

    visualizer._prepare_frame = prepare_frame_with_sand


# --------------------------------------------------------------------------------------
# Terrain height map from particle positions
# --------------------------------------------------------------------------------------
def height_map(particle_q: np.ndarray, lower: np.ndarray, shape: tuple[int, int], cell: float = 0.04) -> np.ndarray:
    """Top-of-particles height [m] on a grid of ``shape`` cells from ``lower``; NaN where no particle is present."""
    grid = np.full(shape, -np.inf)
    ij = np.floor((particle_q[:, :2] - lower) / cell).astype(int)
    keep = (ij >= 0).all(axis=1) & (ij < np.asarray(shape)).all(axis=1)
    np.maximum.at(grid, (ij[keep, 0], ij[keep, 1]), particle_q[keep, 2])
    grid[~np.isfinite(grid)] = np.nan
    return grid


# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------
def run_probe(args: argparse.Namespace, result: dict, monitor: GpuMonitor) -> None:
    import gymnasium as gym
    import torch

    from isaaclab.app import launch_simulation

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    geometry = trajectory_geometry(args)
    if geometry["length_x"] < 2.0 * END_MARGIN or geometry["length_y"] < 2.0 * END_MARGIN:
        raise ValueError("The terrain must be at least as large as the pool's total size (1.6 m).")
    task = TASKS[args.terrain]
    preset = PRESETS[(args.terrain, args.sleep)]
    result["task"], result["preset"] = task, preset
    env_cfg, _ = resolve_task_config(
        task, "rsl_rl_cfg_entry_point", overrides=[f"env.scene.num_envs={args.num_envs}", f"physics={preset}"]
    )
    result["geometry"] = configure_probe(env_cfg, args, geometry)
    launcher = None
    if args.video_dir:
        n_total = (0 if args.trajectory == "rest" else args.presettle) + len(build_trajectory(args, geometry, 0.0)[0])
        prefix = f"probe_{args.terrain}_sleep_{args.sleep}_{args.trajectory}"
        launcher = configure_particle_video(
            env_cfg, args.terrain, geometry, args.video_dir, prefix, n_total, args.video_viz
        )
    # Samples taken so far saw no work of this process on the GPU.
    idle = [u for _, u in monitor.utilization]
    result["gpu_utilization_idle_percent"] = {"mean": float(np.mean(idle)), "n": len(idle)} if idle else {"n": 0}

    with launch_simulation(env_cfg, launcher), contextlib.ExitStack() as cleanup:
        result["baseline_process_MiB"], result["baseline_device_MiB"] = monitor.sample()
        result["stage"] = "create_env"
        env = gym.make(task, cfg=env_cfg)
        cleanup.callback(env.close)
        base_env = env.unwrapped
        backend = base_env.sim.backend_manager.backend
        model = backend.model
        solver = backend.solver.solver("sand")
        pool = base_env.mpm_streaming
        cache = getattr(pool, "cache", None)
        result["particle_count"] = int(model.particle_count)
        result["after_create_process_MiB"], result["after_create_device_MiB"] = monitor.sample()
        result["step_dt"] = float(base_env.step_dt)
        if abs(base_env.step_dt - CONTROL_DT) > 1e-6:
            raise RuntimeError(f"control dt {base_env.step_dt} differs from the trajectory dt {CONTROL_DT}")

        physics_ms = [0.0]
        if args.profile_physics:
            original_step = base_env.sim.step

            def timed_step(*a, **k):
                _sync()
                t0 = time.perf_counter()
                out = original_step(*a, **k)
                _sync()
                physics_ms[0] += (time.perf_counter() - t0) * 1.0e3
                return out

            base_env.sim.step = timed_step

        if args.video_dir:
            draw_sand_particles(base_env)
        robot = base_env.scene["robot"]
        env_origin = base_env.scene.env_origins[0].cpu().numpy()
        env.reset()
        step_ms, phys_ms, force, joint, target_log, wall = [], [], [], [], [], []
        activity_log: list[list[int]] = []
        snapshots: dict[str, np.ndarray] = {}
        marks: dict[str, int] = {}
        particle_world = model.particle_world.numpy()
        actions = torch.zeros((args.num_envs, 3), device=base_env.device)
        surface = {"z": 0.0}
        half_cells = int(round(args.heightmap_half / 0.04))
        terrain_lower = env_origin[:2] - np.array([geometry["length_x"], geometry["length_y"]]) / 2.0
        terrain_shape = (int(round(geometry["length_x"] / 0.04)), int(round(geometry["length_y"] / 0.04)))

        def sand_particles(with_saved: bool) -> np.ndarray:
            """Positions of environment 0's sand: the simulated particles, and optionally the cached ones."""
            live = (model.particle_flags.numpy() & 1) != 0
            q = backend.state_0.particle_q.numpy()[live & (particle_world == 0)]
            if with_saved and cache is not None:
                records = [record for chain in cache.pages[0].values() for record in chain]
                if records:
                    q = np.concatenate((q, cache.saved.q.numpy()[records]))
            return q

        def activity_counts() -> list[int]:
            """Asleep, support and awake particles among the simulated (not kinematic, not spare) ones."""
            activity = solver.particle_activity
            if activity is None:
                return []
            selected = (model.particle_flags.numpy() & 1) != 0
            if pool is not None:
                selected &= pool.particle_is_dynamic.numpy() != 0
            return np.bincount(activity.numpy()[selected], minlength=3).tolist()

        usage = {"slots": 0, "records": 0, "tiles": 0}

        def track_cache() -> None:
            """Keep the peak number of simulation slots, saved records and visited tiles of any environment."""
            usage["slots"] = max(usage["slots"], int((cache.slot_bump.numpy() - cache.slot_free_count.numpy()).max()))
            usage["records"] = max(usage["records"], max(cache.cached_particle_counts))
            usage["tiles"] = max(usage["tiles"], int(cache.visited_count.numpy().max()))

        def run_phase(targets: np.ndarray, phase_end: dict[str, int]) -> None:
            """Step the environment through ``targets``; save height maps at the end of the snapshot phases."""
            offset = len(step_ms)
            snapshot_at = {
                offset + end - 1: name
                for name, end in phase_end.items()
                if name in SNAPSHOT_PHASES or name.startswith(("snap_", "lap_"))
            }
            for name, end in phase_end.items():
                marks[name] = offset + end
            for i, tgt in enumerate(targets):
                actions[:] = torch.as_tensor(tgt, dtype=torch.float32, device=base_env.device)
                physics_ms[0] = 0.0
                _sync()
                t0 = time.perf_counter()
                env.step(actions)
                _sync()
                t1 = time.perf_counter()
                step_ms.append((t1 - t0) * 1.0e3)
                wall.append(t1)
                phys_ms.append(physics_ms[0])
                force.append(base_env.foot_contact_force[0, 0].cpu().numpy())
                jp = robot.data.joint_pos.torch[0].cpu().numpy()
                joint.append(jp)
                target_log.append(tgt)
                if cache is not None and len(step_ms) % 5 == 0:
                    track_cache()
                if args.activity_every and len(step_ms) % args.activity_every == 0:
                    counts = activity_counts()
                    if counts:
                        activity_log.append([len(step_ms), *counts])
                name = snapshot_at.get(offset + i)
                if name is not None:
                    center = env_origin[:2] + np.array([jp[0], jp[1]])
                    q = sand_particles(with_saved=False)
                    snapshots[name] = height_map(q, center - args.heightmap_half, (2 * half_cells, 2 * half_cells))
                    surface["z"] = float(np.nanmedian(height_map(q, center - 0.3, (15, 15))))
                    if args.terrain_map:
                        snapshots[f"terrain_{name}"] = height_map(
                            sand_particles(with_saved=True), terrain_lower, terrain_shape
                        )

        result["stage"] = "run"
        start = np.array([*geometry["start"], HOVER])
        with torch.inference_mode():
            n_presettle = 0 if args.trajectory == "rest" else args.presettle
            if n_presettle:
                run_phase(np.tile(start, (n_presettle, 1)), {"presettle": n_presettle})
            targets, phase_end = build_trajectory(args, geometry, surface["z"])
            if args.max_steps and len(targets) > args.max_steps:
                targets = targets[: args.max_steps]
                phase_end = {name: end for name, end in phase_end.items() if end <= args.max_steps}
            result["surface_z_after_presettle_m"] = surface["z"]
            result["trajectory_steps"] = int(len(targets))
            run_phase(targets, phase_end)
        if pool is not None:
            pool.check_status()
        result["phase_end_steps"] = marks
        result["steps_run"] = len(step_ms)
        result["final_peak_process_MiB"], result["final_device_MiB"] = monitor.sample()
        ms = np.asarray(step_ms[args.warmup :])
        result["step_ms"] = {
            "mean": float(ms.mean()),
            "median": float(np.median(ms)),
            "p95": float(np.percentile(ms, 95)),
            "p99": float(np.percentile(ms, 99)),
            "max": float(ms.max()),
            "n": int(ms.size),
        }
        result["step_ms_per_env_mean"] = float(ms.mean() / args.num_envs)
        # Same window as the step time: from the end of the warm-up to the last step.
        result["gpu_utilization_percent"] = monitor.window(wall[args.warmup - 1], wall[-1])
        result["gpu_utilization_period_s"] = args.monitor_period
        if args.profile_physics:
            ph = np.asarray(phys_ms[args.warmup :])
            result["physics_ms"] = {
                "mean": float(ph.mean()),
                "median": float(np.median(ph)),
                "p95": float(np.percentile(ph, 95)),
            }
            result["physics_ms_per_env_mean"] = float(ph.mean() / args.num_envs)
        if activity_log:
            result["activity_asleep_support_awake_last"] = activity_log[-1][1:]
        if cache is not None:
            track_cache()
            result["cache_usage"] = {
                "slots_peak": usage["slots"],
                "slot_capacity": int(cache.slot_capacity),
                "records_peak": usage["records"],
                "record_capacity": int(cache.record_capacity),
                "tiles_peak": usage["tiles"],
                "tile_capacity": int(cache.max_visited),
            }
            result["cache_saved_records_env0"] = int(cache.cached_particle_counts[0])
            result["cache_visited_tiles_env0"] = len(cache.visited[0])
        np.savez_compressed(
            os.path.splitext(args.out)[0] + ".npz",
            step_ms=np.asarray(step_ms),
            physics_ms=np.asarray(phys_ms),
            force_world=np.asarray(force),
            joint_pos=np.asarray(joint),
            target=np.asarray(target_log),
            activity=np.asarray(activity_log),
            env_origin=env_origin,
            box_size=np.asarray(args.box_size),
            terrain_lower=terrain_lower,
            **{f"height_{k}": v for k, v in snapshots.items()},
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result: dict = {
        "status": "started",
        "kind": "probe",
        "terrain": args.terrain,
        "sleep": args.sleep,
        "trajectory": args.trajectory,
        "num_envs": args.num_envs,
        "speed": args.speed,
        "drag_depth": args.drag_depth,
        "box_size": list(args.box_size),
        "box_mass": args.box_mass,
        "gains": [args.stiffness, args.damping],
        "warmup_steps": args.warmup,
        "seed": args.seed,
        "argv": vars(args),
        "stage": "init",
    }
    with GpuMonitor(args.monitor_period) as monitor:
        try:
            import torch

            result["gpu"] = torch.cuda.get_device_name(0)
            result["git"] = _git_state()
            run_probe(args, result, monitor)
            result["status"] = "ok"
        except BaseException as exc:  # noqa: BLE001 - record any failure, including out-of-memory
            message = f"{type(exc).__name__}: {exc}"
            lowered = message.lower()
            result["status"] = "oom" if ("out of memory" in lowered or "failed to allocate" in lowered) else "error"
            result["error"] = message[:2000]
            result["traceback"] = traceback.format_exc()[-4000:]
        finally:
            result["peak_process_MiB"] = monitor.process_peak
            result["peak_device_MiB"] = monitor.device_peak
            result["memory_samples"] = monitor.samples
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    keys = (
        "status",
        "terrain",
        "sleep",
        "trajectory",
        "num_envs",
        "particle_count",
        "step_ms",
        "gpu_utilization_percent",
        "peak_process_MiB",
        "activity_asleep_support_awake_last",
        "error",
    )
    print(json.dumps({k: result.get(k) for k in keys}, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
