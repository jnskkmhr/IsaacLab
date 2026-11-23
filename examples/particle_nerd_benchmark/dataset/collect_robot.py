# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect closed-loop robot episodes (Franka shovel push, G1 and Go2 walking) on a 0.30 m flat bed.

Every bed is pre-settled with a sand-only MPM run before the robot is loaded, so recording starts on settled sand.
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation

from .episodes import (
    DT,
    PER_MOTION,
    PRODUCTION_TERRAINS,
    ROBOT_BED_M,
    ROBOT_WALL_TOP_M,
    ROBOTS,
    robot_episode,
    tray_boundary_boxes,
)
from .hdf5_io import write_episode
from .presettle import MATERIAL, simulate

G1_CHECKPOINT = Path(
    "/home/jkamohara/isaac/particle_nerd/Particle_NeRD_benchmark/benchmark_environment/checkpoints/jqur5jhm/model_19999.pt"
)
GO2_CHECKPOINT_DIR = Path("/home/jkamohara/isaac/IsaacLab-particle-nerd/logs/rsl_rl/unitree_go2_rough")


def settled_bed(record: dict, size, wall_top: float, device: str) -> np.ndarray:
    """Return the bed after ``settle_s`` of sand-only MPM, from the episode's jittered lattice."""
    from ..core.terrain import sand_cfg

    points = np.asarray(
        sand_cfg(record["terrain_variant"], size, seed=record["particle_seed"]).spawn.positions, dtype=np.float32
    )
    boxes = tray_boundary_boxes(size, wall_top)
    return simulate(points, np.zeros_like(points), boxes, 0, device, round(record["settle_s"] / DT))[0]


class Franka:
    """Franka with a welded open-box shovel, driven by joint targets from analytic IK along three long arc sweeps.

    Each pass sweeps the shovel along an arc around the robot's base at a fixed radius, with the tool yawed so its open
    front faces the direction of travel, then lifts, returns above the bed and lowers at the next radius. Sweeping
    around the base keeps the wrist roll nearly constant, so the base joint does the work.
    """

    base_offset = np.array([0.72, 0.0, 0.05], dtype=np.float32)
    approach_s = 0.8
    lift_height = 0.12
    home = (-0.5303, 0.7779, 0.0045, -2.3577, -0.1167, 2.8139, 0.3748)

    def __init__(self, env, record, args):
        self.record = record

    def _surface(self, x_base: float) -> float:
        """Bed surface height at a base-frame x (the base sits at x = -0.72 in the tray frame)."""
        from ..core.terrain import surface_height

        return float(surface_height(x_base - self.base_offset[0], self.record["terrain_variant"]))

    def pose(self, t):
        """Shovel target in the base frame: (x, y, z) of the end-effector point and the hand yaw [rad]."""
        r = self.record
        radii, n = r["pass_radius"], len(r["pass_radius"])
        th0, th1 = r["theta_start"], r["theta_start"] + r["theta_span"]
        z_dig, z_up = r["height"], r["height"] + self.lift_height

        def point(radius, theta, z):
            x = radius * np.cos(theta)
            return (x, radius * np.sin(theta), z + self._surface(x), theta + np.pi / 2)

        if t < self.approach_s:
            return point(radii[0], th0, z_up + (z_dig - z_up) * self._smooth(t / self.approach_s))
        segment = (r["duration_s"] - self.approach_s) / n
        i = min(int((t - self.approach_s) // segment), n - 1)
        u = (t - self.approach_s - i * segment) / segment
        r_next = radii[i + 1] if i + 1 < n else radii[i]
        if u < 0.66:
            return point(radii[i], th0 + (th1 - th0) * self._smooth(u / 0.66), z_dig)
        if u < 0.74:
            return point(radii[i], th1, z_dig + (z_up - z_dig) * self._smooth((u - 0.66) / 0.08))
        if u < 0.92:
            w = self._smooth((u - 0.74) / 0.18)
            return point(radii[i] + (r_next - radii[i]) * w, th1 + (th0 - th1) * w, z_up)
        return point(r_next, th0, z_up + (z_dig - z_up) * self._smooth((u - 0.92) / 0.08))

    @staticmethod
    def _quat(yaw):
        """Hand pointing down, yawed about world z (w, x, y, z)."""
        return (0.0, np.cos(yaw / 2), np.sin(yaw / 2), 0.0)

    def reset(self, env):
        """Start above the bed at the first waypoint; the generic home pose would put the shovel deep in the sand."""
        from .franka_ik import solve

        x, y, z, yaw = self.pose(0.0)
        arm, position_error, _ = solve(np.array([x, y, z - self.base_offset[2]]), self._quat(yaw), self.home)
        if position_error > 2e-3:
            raise RuntimeError(f"Start pose is unreachable ({position_error:.3f} m)")
        self.q = arm
        robot = env.scene["robot"]
        q = torch.zeros((1, robot.num_joints), device=env.device)
        for index, name in enumerate(robot.joint_names):
            if name.startswith("panda_joint"):
                q[0, index] = float(arm[int(name[len("panda_joint") :]) - 1])
        robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q))

    @staticmethod
    def _smooth(u):
        u = min(1.0, max(0.0, u))
        return u * u * (3.0 - 2.0 * u)

    def action(self, env, obs, t):
        """Joint position targets from analytic IK of the planned shovel pose, warm-started from the last solution."""
        from .franka_ik import solve

        x, y, z, yaw = self.pose(t)
        self.q, _, _ = solve(np.array([x, y, z - self.base_offset[2]]), self._quat(yaw), self.q, prior_weight=0.05)
        return torch.as_tensor(self.q[None], dtype=torch.float32, device=env.device)


class G1Policy:
    """G1 walking with the saved MPM-trained RL policy (forward command from the episode)."""

    def __init__(self, env, record, args):
        from rsl_rl.models import MLPModel

        checkpoint = Path(args.checkpoint or G1_CHECKPOINT)
        saved = json.loads((checkpoint.parent / "training_config.json").read_text())
        actor_cfg = saved["train_cfg"]["actor"].copy()
        if actor_cfg.pop("class_name") != "MLPModel":
            raise ValueError("Unsupported G1 actor class")
        obs, _ = env.reset()
        self.actor = MLPModel(
            obs, saved["train_cfg"]["obs_groups"], "actor", env.action_manager.total_action_dim, **actor_cfg
        ).to(env.device)
        weights = torch.load(checkpoint, map_location=env.device, weights_only=True)
        self.actor.load_state_dict(weights["actor_state_dict"], strict=True)
        self.actor.eval()
        self.policy = str(checkpoint)
        env.cfg.forward_speed = record["forward_speed"]
        self.variant = record["terrain_variant"]

    def reset(self, env):
        """Stand the robot on the terrain: lift its root by the bed surface height at its start position."""
        from ..core.terrain import surface_height

        robot = env.scene["robot"]
        pose = robot.data.root_pose_w.torch.clone()
        offset = float(surface_height(float(pose[0, 0]) - float(env.scene.env_origins[0, 0]), self.variant))
        if abs(offset) > 1e-6:
            pose[:, 2] += offset
            robot.write_root_pose_to_sim_index(root_pose=pose)

    def action(self, env, obs, t):
        return self.actor(obs)


class Go2Policy:
    """Go2 walking with the Isaac Lab flat velocity policy (RSL-RL actor) tracking a constant command."""

    def __init__(self, env, record, args):
        import yaml
        from rsl_rl.models import MLPModel

        checkpoint = Path(args.checkpoint) if args.checkpoint else sorted(GO2_CHECKPOINT_DIR.glob("*/model_*.pt"))[-1]
        saved = yaml.unsafe_load((checkpoint.parent / "params" / "agent.yaml").read_text())
        actor_cfg = dict(saved["actor"])
        if actor_cfg.pop("class_name") != "MLPModel":
            raise ValueError("Unsupported Go2 actor class")
        env.cfg.go2_command = torch.tensor([record["command"]], dtype=torch.float32)
        obs, _ = env.reset()
        self.actor = MLPModel(obs, saved["obs_groups"], "actor", env.action_manager.total_action_dim, **actor_cfg).to(
            env.device
        )
        weights = torch.load(checkpoint, map_location=env.device, weights_only=True)
        self.actor.load_state_dict(weights["actor_state_dict"], strict=True)
        self.actor.eval()
        self.policy = str(checkpoint)

    def action(self, env, obs, t):
        return self.actor(obs)


CONTROLLERS = {"franka": Franka, "g1": G1Policy, "go2": Go2Policy}


CAMERA = {
    "franka": ((1.65, -2.0, 1.55), (0.0, 0.0, -0.05)),
    "g1": ((3.2, -4.4, 3.0), (0.0, 0.0, 0.35)),
    "go2": ((0.2, -3.0, 1.5), (0.0, 0.0, 0.1)),
}


def _visualizer_cfg(robot: str):
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    from isaaclab.visualizers import WindowCfg

    eye, lookat = CAMERA[robot]
    cfg = NewtonGLVisualizerCfg(
        headless=True,
        show_particles=True,
        particle_color=(0.72, 0.59, 0.36),
        eye=eye,
        lookat=lookat,
        window=WindowCfg(size=(1280, 720)),
        streaming_view=False,
    )
    cfg.focal_length = 24.0
    return cfg


def make_env(robot: str, args, variant: str = "flat"):

    from ..core.case_config import make_config
    from ..core.terrain import configure_terrain

    if robot == "go2":
        from .go2_env import Go2PolicyEnvCfg

        cfg = Go2PolicyEnvCfg()
        configure_terrain(cfg.scene, variant, ROBOT_BED_M[robot], wall_top=ROBOT_WALL_TOP_M[robot])
        cfg.scene.robot.init_state.pos = (-1.0, 0.0, 0.36)
        cfg.sim.device = args.device
        cfg.sim.visualizer_cfgs = [_visualizer_cfg(robot)] if args.video else []
        return cfg
    if robot == "franka":
        from .franka_env import ShovelFrankaEnvCfg

        cfg = ShovelFrankaEnvCfg()
        configure_terrain(cfg.scene, variant, ROBOT_BED_M[robot], wall_top=ROBOT_WALL_TOP_M[robot])
        cfg.sim.device = args.device
        cfg.sim.visualizer_cfgs = [_visualizer_cfg(robot)] if args.video else []
        return cfg
    cfg = make_config("g1", "flat")
    configure_terrain(cfg.scene, variant, ROBOT_BED_M[robot], wall_top=ROBOT_WALL_TOP_M[robot])
    cfg.sim.device = args.device
    cfg.sim.visualizer_cfgs = [_visualizer_cfg(robot)] if args.video else []
    return cfg


class Capture:
    """Per-physics-step recording of particles, bodies, joints and the MPM reaction on every body."""

    def __init__(self, env, steps):
        from isaaclab_newton.physics import NewtonManager

        self.env = env
        self.manager = NewtonManager
        self.model = NewtonManager.get_model()
        self.entry = NewtonManager._solver._entries["sand"]
        self.solver = self.entry.solver
        self.mapping = wp.to_torch(self.entry.body_local_to_global).long()
        n, b = self.model.particle_count, self.model.body_count
        self.robot = env.scene["robot"] if "robot" in env.scene.articulations else None
        j = self.robot.num_joints if self.robot is not None else 0
        self.pos = np.empty((steps + 1, n, 3), dtype=np.float32)
        self.pose = np.empty((steps + 1, b, 7), dtype=np.float32)
        self.vel = np.empty((steps + 1, b, 6), dtype=np.float32)
        self.joint_pos = np.empty((steps + 1, j), dtype=np.float32)
        self.joint_vel = np.empty((steps + 1, j), dtype=np.float32)
        self.contact = np.zeros((steps, b, 6), dtype=np.float32)
        self.action = np.zeros((steps, env.action_manager.total_action_dim), dtype=np.float32)

    def record(self, index):
        from isaaclab.utils.math import quat_apply

        state = self.manager.get_state_0()
        self.pos[index] = state.particle_q.numpy()
        self.pose[index] = state.body_q.numpy()
        self.vel[index] = state.body_qd.numpy()
        if self.robot is not None:
            self.joint_pos[index] = self.robot.data.joint_pos.torch[0].cpu().numpy()
            self.joint_vel[index] = self.robot.data.joint_vel.torch[0].cpu().numpy()
        if index == 0:
            return
        self.action[index - 1] = self.env.action_manager.action[0].cpu().numpy()
        impulses, positions, colliders = self.solver.collect_collider_impulses(self.entry.state_0)
        ids = wp.to_torch(colliders).long()
        keep = ids >= 0
        local = wp.to_torch(self.solver.collider_body_index)[ids[keep]].long()
        impulse = wp.to_torch(impulses)[keep]
        position = wp.to_torch(positions)[keep]
        keep = local >= 0
        local, impulse, position = local[keep], impulse[keep], position[keep]
        q = wp.to_torch(self.entry.state_0.body_q)[local]
        centre = q[:, :3] + quat_apply(q[:, 3:], wp.to_torch(self.solver.model.body_com)[local])
        wrench = torch.cat((impulse, torch.cross(position - centre, impulse, dim=-1)), dim=-1) / DT
        out = torch.zeros((self.model.body_count, 6), device=self.env.device)
        out.index_add_(0, self.mapping[local], wrench)
        self.contact[index - 1] = out.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=ROBOTS, required=True)
    parser.add_argument("--terrain", choices=PRODUCTION_TERRAINS, default="flat")
    parser.add_argument("--variant", help="Bed surface variant (for example slope_p10); default: the only one")
    parser.add_argument("--per-motion", type=int, default=PER_MOTION, help="Episodes (test run: 1)")
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--duration", type=float, help="Override the duration (smoke check only)")
    parser.add_argument("--checkpoint", type=Path, help="Policy checkpoint (G1: needs training_config.json beside it)")
    parser.add_argument("--video", type=Path, help="Also record an MP4 preview of the first episode here")
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    if args.video:
        args.visualizer = ["newton_gl"]

    records = [robot_episode(args.robot, i, args.terrain) for i in range(args.per_motion)]
    if args.indices:
        records = [r for r in records if r["index"] in args.indices]
    if args.variant:
        records = [r for r in records if r["terrain_variant"] == args.variant]
    variants = {r["terrain_variant"] for r in records}
    if len(variants) > 1:
        raise ValueError(f"Mixed bed variants {sorted(variants)}; pass --variant")
    variant = variants.pop() if variants else "flat"
    todo = []
    for r in records:
        if args.duration is not None:
            r["duration_s"], r["smoke_check"] = args.duration, True
        target = args.output / r["split"] / args.terrain / "robot" / args.robot / f"{r['index']:04d}.hdf5"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            with h5py.File(target, "r") as existing:
                if existing.attrs.get("complete", False):
                    continue
        todo.append((r, target))
    if not todo:
        print("Nothing to do.", flush=True)
        return

    # Pre-settle every bed with sand-only MPM before the simulation app starts.
    wp.init()
    device = args.device if str(args.device).startswith("cuda") else "cuda:0"
    size, wall_top = ROBOT_BED_M[args.robot], ROBOT_WALL_TOP_M[args.robot]
    settled = {}
    for r, _ in todo:
        start = time.perf_counter()
        settled[r["id"]] = settled_bed(r, size, wall_top, device)
        print(f"settled {r['id']} in {time.perf_counter() - start:.1f} s", flush=True)

    import newton
    from isaaclab_newton.physics import NewtonMPMManager

    cfg = make_env(args.robot, args, variant)
    from ..core.terrain import DENSITY, PARTICLE_JITTER, SPACING, VOXEL_SIZE

    scene_json = args.output / "terrain" / f"{args.terrain}_robot_{args.robot}.json"
    scene_json.parent.mkdir(parents=True, exist_ok=True)
    if not scene_json.exists():
        scene_json.write_text(
            json.dumps(
                {
                    "robot": args.robot,
                    "terrain": args.terrain,
                    "bed_size_m": list(size),
                    "tray_wall_top_m": wall_top,
                    "boundary_boxes": tray_boundary_boxes(size, wall_top),
                    "particle_spacing_m": SPACING,
                    "voxel_size_m": VOXEL_SIZE,
                    "jitter_max_m": PARTICLE_JITTER,
                    "density": DENSITY,
                    "material": MATERIAL,
                    "physics_dt_s": cfg.sim.dt,
                    "control_decimation": cfg.decimation,
                    "gravity": [0.0, 0.0, -9.81],
                },
                indent=2,
            )
        )

    with launch_simulation(cfg, args):
        from isaaclab.envs import ManagerBasedRLEnv

        env = ManagerBasedRLEnv(cfg)
        try:
            with torch.inference_mode():
                for record, target in todo:
                    steps = round(record["duration_s"] / DT)
                    controller = CONTROLLERS[args.robot](env, record, args)
                    obs, _ = env.reset(seed=record["seed"])
                    sand = env.scene["sand"]
                    nodal = sand.data.default_nodal_state_w.torch.clone()
                    positions = settled[record["id"]]
                    if len(positions) != nodal.shape[1]:
                        raise RuntimeError(f"Settled bed has {len(positions)} particles, scene has {nodal.shape[1]}")
                    nodal[0, :, :3] = torch.as_tensor(positions, device=env.device) + env.scene.env_origins[0]
                    nodal[0, :, 3:6] = 0.0
                    sand.write_nodal_state_to_sim_index(nodal)
                    env.sim.forward()
                    if hasattr(controller, "reset"):
                        controller.reset(env)
                    env.sim.forward()
                    NewtonMPMManager.reset_solver_state(flags=newton.StateFlags.BODY | newton.StateFlags.PARTICLE)
                    capture = Capture(env, steps)
                    capture.record(0)
                    index = 0
                    original_update = env.scene.update

                    def record_physics(dt):
                        nonlocal index
                        original_update(dt)
                        index += 1
                        capture.record(index)

                    env.scene.update = record_physics
                    start = time.perf_counter()
                    encoder = None
                    try:
                        for control_step in range(steps // cfg.decimation):
                            t = control_step * cfg.decimation * DT
                            action = controller.action(env, obs, t)
                            obs, _, _, _, _ = env.step(action)
                            if args.video and record is todo[0][0] and index % 4 == 0:
                                frame = env.sim._visualizers[0].render_rgb_array()
                                if frame is None:
                                    raise RuntimeError("Visualizer did not produce a frame")
                                if encoder is None:
                                    import subprocess

                                    args.video.parent.mkdir(parents=True, exist_ok=True)
                                    h, w = frame.shape[:2]
                                    encoder = subprocess.Popen(
                                        [
                                            "ffmpeg",
                                            "-nostdin",
                                            "-v",
                                            "error",
                                            "-y",
                                            "-f",
                                            "rawvideo",
                                            "-pix_fmt",
                                            "rgb24",
                                            "-s",
                                            f"{w}x{h}",
                                            "-r",
                                            "50",
                                            "-i",
                                            "-",
                                            "-an",
                                            "-c:v",
                                            "libx264",
                                            "-crf",
                                            "22",
                                            "-pix_fmt",
                                            "yuv420p",
                                            "-movflags",
                                            "+faststart",
                                            str(args.video),
                                        ],
                                        stdin=subprocess.PIPE,
                                    )
                                encoder.stdin.write(frame[:, :, :3].astype(np.uint8).tobytes())
                    finally:
                        env.scene.update = original_update
                        if encoder is not None:
                            encoder.stdin.close()
                            encoder.wait()
                    if index != steps:
                        raise RuntimeError(f"Recorded {index} of {steps} steps")
                    robot = env.scene["robot"]
                    body = {"pose": capture.pose, "vel": capture.vel, "contact_wrench": capture.contact}
                    write_episode(
                        target,
                        record,
                        capture.pos,
                        body,
                        {
                            "particle_spacing_m": SPACING,
                            "voxel_size_m": VOXEL_SIZE,
                            "particle_mass_kg": DENSITY * SPACING**3,
                            "material": MATERIAL,
                            "bed_size_m": list(size),
                            "body_names": list(capture.model.body_label),
                            "joint_names": list(robot.joint_names),
                            "control_decimation": cfg.decimation,
                            "policy": getattr(controller, "policy", None),
                            "pose_columns": "xyz, quaternion xyzw",
                            "vel_columns": "linear xyz, angular xyz (world frame)",
                            "contact_wrench_note": "MPM sand-on-body wrench per body per step (impulse/dt)",
                            "action_note": "applied policy or controller action at each physics step",
                            "collection_seconds": time.perf_counter() - start,
                        },
                        extra={
                            "joints": {"pos": capture.joint_pos, "vel": capture.joint_vel},
                            "control": {"action": capture.action},
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "episode": record["id"],
                                "steps": steps,
                                "seconds": round(time.perf_counter() - start, 1),
                                "bytes": target.stat().st_size,
                            }
                        ),
                        flush=True,
                    )
        finally:
            env.close()


if __name__ == "__main__":
    main()
