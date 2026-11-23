# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train, evaluate, plot, and render 3D benchmark showcases for WorldParticle vs. Particle NeRD vs. MPM."""

import argparse
import importlib
import importlib.machinery
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyglet

pyglet.options["headless"] = True
import newton
import newton.viewer
import numpy as np
import torch
import warp as wp
from PIL import Image, ImageDraw, ImageFont

# Private package loaders to avoid triggering eager Gym registration in isaaclab_tasks.__init__
_CONTRIB = Path(__file__).resolve().parents[2]
for _pkg_name, _dir_name in (
    ("_wp_nerd", "particle_nerd"),
    ("_wp_world", "particle_world"),
):
    if _pkg_name not in sys.modules:
        _mod = types.ModuleType(_pkg_name)
        _mod.__path__ = [str(_CONTRIB / _dir_name)]
        _mod.__spec__ = importlib.machinery.ModuleSpec(_pkg_name, loader=None, is_package=True)
        _mod.__spec__.submodule_search_locations = _mod.__path__
        sys.modules[_pkg_name] = _mod

ParticleTransolver = importlib.import_module("_wp_nerd.network.model").ParticleTransolver
WorldParticleNetwork = importlib.import_module("_wp_world.network.model").WorldParticleNetwork
cconv_warp = importlib.import_module("_wp_world.network.cconv_warp")
features_differentiable = cconv_warp.features_differentiable
material_features = cconv_warp.material_features
BenchmarkForceReconstructor = importlib.import_module("_wp_world.showcase.contact_forces").BenchmarkForceReconstructor

COLOR_MPM = "#193c64"
COLOR_NERD = "#d95f02"
COLOR_WORLD = "#1b9e77"


def load_benchmark_case(run_dir: Path, device: str = "cuda:0") -> dict:
    """Load a 200 Hz MPM benchmark run directory into GPU tensors and numpy reference arrays."""
    meta = json.loads((run_dir / "metadata.json").read_text())
    h5_path = run_dir / "trajectory.h5"
    with h5py.File(h5_path, "r") as f:
        pos_np = f["particle_position"][:].astype(np.float32)  # (1201, N, 3)
        vel_np = f["particle_velocity"][:].astype(np.float32)  # (1201, N, 3)
        mass_np = f["particle_mass"][:].astype(np.float32)  # (N,)
        radius_np = f["particle_radius"][:].astype(np.float32)  # (N,)
        body_pose_np = f["body_pose"][:].astype(np.float32)  # (1201, 1, 7)
        body_vel_np = f["body_velocity"][:].astype(np.float32)  # (1201, 1, 6)
        force_np = f["material_force"][:, 0, :].astype(np.float32)  # (1200, 3)
        shape_type = f["shape_type"][:].astype(np.int32)
        shape_scale = f["shape_scale"][:].astype(np.float32)
        shape_transform = f["shape_transform"][:].astype(np.float32)

    dt = float(meta["physics_dt_s"])
    # Shape 0 is the interacting collider; Shapes 1..5 are the 5 static tray boxes
    obj_geo = int(shape_type[0])
    shape_name = "sphere" if obj_geo == int(newton.GeoType.SPHERE) else "box"
    obj_kind = 2 if shape_name == "sphere" else 1
    kinds = torch.tensor([obj_kind, 1, 1, 1, 1, 1], dtype=torch.int32, device=device)

    sizes = torch.as_tensor(shape_scale[:6], dtype=torch.float32, device=device).contiguous()
    wall_poses = torch.as_tensor(shape_transform[1:6], dtype=torch.float32, device=device)
    horizon = len(pos_np) - 1
    obj_poses = torch.as_tensor(body_pose_np[:horizon, 0:1, :], dtype=torch.float32, device=device)
    poses = torch.cat([obj_poses, wall_poses.unsqueeze(0).expand(horizon, -1, -1)], dim=1).contiguous()

    obj_vels = torch.as_tensor(body_vel_np[:horizon, 0:1, :], dtype=torch.float32, device=device)
    wall_vels = torch.zeros((horizon, 5, 6), dtype=torch.float32, device=device)
    velocities = torch.cat([obj_vels, wall_vels], dim=1).contiguous()

    friction = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    mat_vec = material_features(meta, float(mass_np[0]), float(radius_np[0]), dt=dt)
    material = torch.as_tensor(mat_vec, dtype=torch.float32, device=device).contiguous()

    spawn_cfg = meta["interacting_asset"]["spawn"]
    if "size" in spawn_cfg:
        body_size_m = tuple(float(x) for x in spawn_cfg["size"])
    else:
        r = float(spawn_cfg["radius"])
        body_size_m = (2.0 * r, 2.0 * r, 2.0 * r)

    return {
        "name": run_dir.name,
        "meta": meta,
        "dt": dt,
        "shape": shape_name,
        "body_size_m": body_size_m,
        "pos_np": pos_np,
        "vel_np": vel_np,
        "mass_np": mass_np,
        "radius_np": radius_np,
        "body_pose_np": body_pose_np,
        "body_vel_np": body_vel_np,
        "force_np": force_np,
        "wall_poses_np": shape_transform[1:6],
        "wall_sizes_np": shape_scale[1:6],
        "pos_gpu": torch.as_tensor(pos_np, dtype=torch.float32, device=device),
        "poses": poses,
        "velocities": velocities,
        "kinds": kinds,
        "sizes": sizes,
        "friction": friction,
        "material": material,
    }


def extract_window(case: dict, start: int, horizon: int) -> dict:
    """Slice a recurrent training window [start, start + horizon] from a loaded benchmark case."""
    pos_gpu = case["pos_gpu"]
    if start >= 2:
        hist = pos_gpu[start - 2 : start + 1].permute(1, 0, 2).contiguous()
    elif start == 1:
        hist = torch.stack([pos_gpu[0], pos_gpu[0], pos_gpu[1]], dim=1).contiguous()
    else:
        hist = torch.stack([pos_gpu[0], pos_gpu[0], pos_gpu[0]], dim=1).contiguous()
    targets = pos_gpu[start + 1 : start + 1 + horizon]
    return {
        "history": hist,
        "targets": targets,
        "poses": case["poses"][start : start + horizon],
        "velocities": case["velocities"][start : start + horizon],
        "kinds": case["kinds"],
        "sizes": case["sizes"],
        "friction": case["friction"],
        "material": case["material"],
        "dt": case["dt"],
    }


def rollout_world_particle(model: WorldParticleNetwork, sample: dict) -> torch.Tensor:
    """Autoregressively roll out WorldParticleNetwork over `sample`."""
    history = sample["history"]
    predicted = []
    for pose, velocity in zip(sample["poses"], sample["velocities"], strict=True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            current = model.step_from_history(
                history,
                pose,
                velocity,
                sample["kinds"],
                sample["sizes"],
                sample["friction"],
                sample["material"],
                dt=sample["dt"],
            )
        predicted.append(current)
        history = torch.cat((history[:, 1:], current[:, None]), dim=1).contiguous()
    return torch.stack(predicted)


def rollout_transolver(model: ParticleTransolver, sample: dict) -> torch.Tensor:
    """Autoregressively roll out baseline ParticleTransolver over `sample`."""
    history = sample["history"]
    predicted = []
    for pose, velocity in zip(sample["poses"], sample["velocities"], strict=True):
        inputs = features_differentiable(
            history,
            pose,
            velocity,
            sample["kinds"],
            sample["sizes"],
            sample["friction"],
            sample["material"],
            dt=sample["dt"],
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            displacement = model(*inputs)
        current = history[:, -1] + displacement
        predicted.append(current)
        history = torch.cat((history[:, 1:], current[:, None]), dim=1).contiguous()
    return torch.stack(predicted)


def balanced_velocity_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    initial: torch.Tensor,
    poses: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Equal weight for near-collider, moving, and stationary reference particle groups + velocity consistency."""
    moving = torch.linalg.vector_norm(target - initial[None], dim=-1) > 0.0005
    obj_center = poses[:, 0:1, :3]  # [H, 1, 3]
    near_col = torch.linalg.vector_norm(target - obj_center, dim=-1) < 0.18
    stationary = (~moving) & (~near_col)

    pos_err = ((predicted - target) / dt).square().sum(-1)
    groups = (near_col, moving, stationary)
    counts = torch.stack([g.sum() for g in groups])
    sums = torch.stack([(pos_err * g).sum() for g in groups])
    loss_pos = (sums / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)

    pred_seq = torch.cat([initial[None], predicted], dim=0)
    targ_seq = torch.cat([initial[None], target], dim=0)
    vel_pred = (pred_seq[1:] - pred_seq[:-1]) / dt
    vel_targ = (targ_seq[1:] - targ_seq[:-1]) / dt
    vel_err = (vel_pred - vel_targ).square().sum(-1)
    vel_sums = torch.stack([(vel_err * g).sum() for g in groups])
    loss_vel = (vel_sums / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)
    return 0.5 * loss_pos + 0.5 * loss_vel


def body_distance_np(
    points: np.ndarray, state: np.ndarray, shape: str, size_m: tuple[float, float, float]
) -> np.ndarray:
    """Signed distance from particle centers to the interacting body's surface."""
    relative = points - state[:3]
    q, w = -state[3:6], state[6]
    local = relative + 2.0 * np.cross(q, np.cross(q, relative) + w * relative)
    half = np.asarray(size_m) / 2.0
    if shape == "sphere":
        return np.linalg.norm(local, axis=1) - half[0]
    distance = np.abs(local) - half
    return np.linalg.norm(np.maximum(distance, 0.0), axis=1) + np.minimum(distance.max(axis=1), 0.0)


def below_body_mass_np(
    points: np.ndarray,
    state: np.ndarray,
    shape: str,
    size_m: tuple[float, float, float],
    mass: np.ndarray,
    depth: float = 0.10,
) -> float:
    """Sand mass in the horizontal bounding rectangle 0 to 0.10 m below the body's lowest point."""
    relative = points - state[:3]
    half = np.asarray(size_m) / 2.0
    q, w = state[3:6], state[6]
    axes = np.eye(3) + 2.0 * np.cross(q, np.cross(q, np.eye(3)) + w * np.eye(3))
    extent = np.full(3, half[0]) if shape == "sphere" else np.abs(axes).T @ half
    footprint = (np.abs(relative[:, 0]) <= extent[0]) & (np.abs(relative[:, 1]) <= extent[1])
    keep = footprint & (relative[:, 2] < -extent[2]) & (relative[:, 2] >= -extent[2] - depth) & (points[:, 2] >= -0.30)
    return float(mass[keep].sum())


def particle_rmse_mm(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        ref, pred = ref[mask], pred[mask]
    if len(ref) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.sum((pred - ref) ** 2, axis=-1))) * 1000.0)


@torch.no_grad()
def evaluate_full_rollout(model, rollout_fn, case: dict) -> tuple[np.ndarray, float]:
    """Run a 1,200-step (6.0 s) open-loop recurrent rollout from t=0 and measure inference latency."""
    model.eval()
    steps = len(case["pos_np"]) - 1
    sample = extract_window(case, start=0, horizon=steps)
    history = sample["history"]
    positions = [history[:, -1].cpu().numpy()]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step_idx in range(steps):
        step_sample = {
            "history": history,
            "poses": sample["poses"][step_idx : step_idx + 1],
            "velocities": sample["velocities"][step_idx : step_idx + 1],
            "kinds": sample["kinds"],
            "sizes": sample["sizes"],
            "friction": sample["friction"],
            "material": sample["material"],
            "dt": sample["dt"],
        }
        next_pos = rollout_fn(model, step_sample)[0]
        positions.append(next_pos.cpu().numpy())
        history = torch.cat((history[:, 1:], next_pos[:, None]), dim=1).contiguous()
    torch.cuda.synchronize()
    ms_per_step = (time.perf_counter() - t0) * 1000.0 / steps
    return np.stack(positions), ms_per_step


def compute_case_metrics(case: dict, pred_pos: np.ndarray, force_rec: BenchmarkForceReconstructor) -> dict:
    """Compute per-frame particle errors, sub-body mass, inside-body penetration mass, and 200 Hz contact forces."""
    ref_pos = case["pos_np"]
    mass = case["mass_np"]
    dt = case["dt"]
    body_pose = case["body_pose_np"]
    body_vel = case["body_vel_np"]
    shape = case["shape"]
    size_m = case["body_size_m"]
    steps = len(pred_pos) - 1

    times = np.arange(1, steps + 1) * dt
    global_rmse = np.zeros(steps, dtype=np.float32)
    near_rmse = np.zeros(steps, dtype=np.float32)
    moving_rmse = np.zeros(steps, dtype=np.float32)
    ref_below_kg = np.zeros(steps, dtype=np.float32)
    pred_below_kg = np.zeros(steps, dtype=np.float32)
    ref_inside_kg = np.zeros(steps, dtype=np.float32)
    pred_inside_kg = np.zeros(steps, dtype=np.float32)
    ref_ke_j = np.zeros(steps, dtype=np.float32)
    pred_ke_j = np.zeros(steps, dtype=np.float32)

    for f in range(1, steps + 1):
        rv = (ref_pos[f] - ref_pos[f - 1]) / dt
        pv = (pred_pos[f] - pred_pos[f - 1]) / dt
        bstate = body_pose[f, 0]
        dist_ref = body_distance_np(ref_pos[f], bstate, shape, size_m)
        dist_pred = body_distance_np(pred_pos[f], bstate, shape, size_m)
        near_mask = np.abs(dist_ref) <= 0.05
        mov_mask = np.linalg.norm(rv, axis=1) > 0.01

        idx = f - 1
        global_rmse[idx] = particle_rmse_mm(ref_pos[f], pred_pos[f])
        near_rmse[idx] = particle_rmse_mm(ref_pos[f], pred_pos[f], near_mask)
        moving_rmse[idx] = particle_rmse_mm(ref_pos[f], pred_pos[f], mov_mask)

        if f % 10 == 0 or f == 1 or f == steps:
            rb = below_body_mass_np(ref_pos[f], bstate, shape, size_m, mass)
            pb = below_body_mass_np(pred_pos[f], bstate, shape, size_m, mass)
            ri = float(mass[dist_ref < -0.008].sum())
            pi = float(mass[dist_pred < -0.008].sum())
        ref_below_kg[idx] = rb
        pred_below_kg[idx] = pb
        ref_inside_kg[idx] = ri
        pred_inside_kg[idx] = pi
        ref_ke_j[idx] = float(0.5 * np.sum(mass * np.sum(rv**2, axis=1)))
        pred_ke_j[idx] = float(0.5 * np.sum(mass * np.sum(pv**2, axis=1)))

    pred_forces = force_rec.estimate_trajectory(pred_pos, body_pose, body_vel)
    ref_forces = case["force_np"][:steps]
    force_diff = pred_forces - ref_forces
    force_rmse_vec = float(np.sqrt(np.mean(np.sum(force_diff**2, axis=-1))))
    force_rmse_xyz = np.sqrt(np.mean(force_diff**2, axis=0)).tolist()

    return {
        "time_s": times,
        "global_rmse_mm": global_rmse,
        "near_rmse_mm": near_rmse,
        "moving_rmse_mm": moving_rmse,
        "ref_below_kg": ref_below_kg,
        "pred_below_kg": pred_below_kg,
        "ref_inside_kg": ref_inside_kg,
        "pred_inside_kg": pred_inside_kg,
        "ref_ke_j": ref_ke_j,
        "pred_ke_j": pred_ke_j,
        "ref_forces": ref_forces,
        "pred_forces": pred_forces,
        "summary": {
            "mean_global_rmse_mm": float(np.mean(global_rmse)),
            "final_global_rmse_mm": float(global_rmse[-1]),
            "mean_near_contact_rmse_mm": float(np.mean(near_rmse[near_rmse > 0])) if np.any(near_rmse > 0) else 0.0,
            "mean_moving_rmse_mm": float(np.mean(moving_rmse[moving_rmse > 0])) if np.any(moving_rmse > 0) else 0.0,
            "force_vector_rmse_n": force_rmse_vec,
            "force_component_rmse_n": force_rmse_xyz,
            "mean_inside_body_mass_kg": float(np.mean(pred_inside_kg)),
        },
    }


def plot_case_comparison(case: dict, nerd_metrics: dict, world_metrics: dict, out_dir: Path) -> None:
    """Save publication-quality comparison figures for a single benchmark scenario."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "mathtext.fontset": "stix",
        }
    )
    title = case["name"].replace("-", " ").title()
    t = world_metrics["time_s"]

    # 1. 6-panel Particle & Field Evaluation Figure
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))
    fig.suptitle(f"{title} — 6.0 s Open-Loop Recurrent Rollout (200 Hz, N = {len(case['mass_np']):,})", fontsize=15)
    fig.subplots_adjust(hspace=0.35, wspace=0.28, top=0.90)

    axes[0, 0].plot(t, nerd_metrics["global_rmse_mm"], color=COLOR_NERD, label="Particle NeRD (Transolver)", lw=1.6)
    axes[0, 0].plot(
        t, world_metrics["global_rmse_mm"], color=COLOR_WORLD, label="WorldParticle (Warp CConv + ToMe)", lw=1.8
    )
    axes[0, 0].set(title="(a) Global Particle Position RMSE", xlabel="Time (s)", ylabel="RMSE (mm)")
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].plot(t, nerd_metrics["near_rmse_mm"], color=COLOR_NERD, label="Particle NeRD", lw=1.6)
    axes[0, 1].plot(t, world_metrics["near_rmse_mm"], color=COLOR_WORLD, label="WorldParticle", lw=1.8)
    axes[0, 1].set(title="(b) Near-Collider (<5 cm) RMSE", xlabel="Time (s)", ylabel="RMSE (mm)")
    axes[0, 1].legend(fontsize=9)

    axes[0, 2].plot(t, nerd_metrics["moving_rmse_mm"], color=COLOR_NERD, label="Particle NeRD", lw=1.6)
    axes[0, 2].plot(t, world_metrics["moving_rmse_mm"], color=COLOR_WORLD, label="WorldParticle", lw=1.8)
    axes[0, 2].set(title="(c) Moving-Particle (>1 cm/s) RMSE", xlabel="Time (s)", ylabel="RMSE (mm)")
    axes[0, 2].legend(fontsize=9)

    axes[1, 0].plot(t, world_metrics["ref_below_kg"], color=COLOR_MPM, label="Implicit MPM", lw=1.8)
    axes[1, 0].plot(t, nerd_metrics["pred_below_kg"], color=COLOR_NERD, label="Particle NeRD", lw=1.5, ls="--")
    axes[1, 0].plot(t, world_metrics["pred_below_kg"], color=COLOR_WORLD, label="WorldParticle", lw=1.8)
    axes[1, 0].set(title="(d) Sand Mass Beneath Body (0–10 cm)", xlabel="Time (s)", ylabel="Mass (kg)")
    axes[1, 0].legend(fontsize=9)

    axes[1, 1].plot(t, world_metrics["ref_inside_kg"], color=COLOR_MPM, label="Implicit MPM", lw=1.8)
    axes[1, 1].plot(t, nerd_metrics["pred_inside_kg"], color=COLOR_NERD, label="Particle NeRD", lw=1.5, ls="--")
    axes[1, 1].plot(t, world_metrics["pred_inside_kg"], color=COLOR_WORLD, label="WorldParticle", lw=1.8)
    axes[1, 1].set(title="(e) Penetrating Mass Inside Collider", xlabel="Time (s)", ylabel="Mass (kg)")
    axes[1, 1].legend(fontsize=9)

    axes[1, 2].plot(t, world_metrics["ref_ke_j"], color=COLOR_MPM, label="Implicit MPM", lw=1.8)
    axes[1, 2].plot(t, nerd_metrics["pred_ke_j"], color=COLOR_NERD, label="Particle NeRD", lw=1.5, ls="--")
    axes[1, 2].plot(t, world_metrics["pred_ke_j"], color=COLOR_WORLD, label="WorldParticle", lw=1.8)
    axes[1, 2].set(title="(f) Sand Particle Kinetic Energy", xlabel="Time (s)", ylabel="Kinetic Energy (J)")
    axes[1, 2].legend(fontsize=9)

    for ax in axes.flat:
        ax.grid(alpha=0.18)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"particle_evaluation.{ext}", dpi=260)
    plt.close(fig)

    # 2. 3-panel Material-on-Object Contact Force Comparison (Fx, Fy, Fz)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    fig.suptitle(f"{title} — 200 Hz Material-on-Object Contact Force Reconstruction", fontsize=14)
    fig.subplots_adjust(wspace=0.26, top=0.83)
    for comp, axis_name in enumerate(("x", "y", "z")):
        ax = axes[comp]
        ax.plot(t, world_metrics["ref_forces"][:, comp], color=COLOR_MPM, label="Implicit MPM", lw=1.4, alpha=0.9)
        ax.plot(t, nerd_metrics["pred_forces"][:, comp], color=COLOR_NERD, label="Particle NeRD", lw=1.2, alpha=0.8)
        ax.plot(t, world_metrics["pred_forces"][:, comp], color=COLOR_WORLD, label="WorldParticle", lw=1.4, alpha=0.95)
        ax.set(
            title=f"({chr(97 + comp)}) World $F_{axis_name}$",
            xlabel="Simulation Time (s)",
            ylabel=f"$F_{axis_name}$ (N)",
        )
        ax.grid(alpha=0.18)
    axes[0].legend(fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"forces_comparison.{ext}", dpi=260)
    plt.close(fig)


def render_case_videos(
    case: dict,
    nerd_pos: np.ndarray,
    world_pos: np.ndarray,
    out_dir: Path,
    viewer: newton.viewer.ViewerGL,
    width: int = 640,
    height: int = 480,
) -> None:
    """Render 50 fps side-by-side 3D comparison videos and filmstrips using a shared `ViewerGL`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0, is_kinematic=True, label="InteractingObject")
    half = np.asarray(case["body_size_m"]) / 2.0
    color = (0.12, 0.32, 0.25)
    if case["shape"] == "box":
        builder.add_shape_box(body, hx=float(half[0]), hy=float(half[1]), hz=float(half[2]), color=color)
    else:
        builder.add_shape_sphere(body, radius=float(half[0]), color=color)

    for w_idx, (w_pose, w_half) in enumerate(zip(case["wall_poses_np"], case["wall_sizes_np"], strict=True)):
        builder.add_shape_box(
            -1,
            xform=wp.transform(w_pose[:3], w_pose[3:]),
            hx=float(w_half[0]),
            hy=float(w_half[1]),
            hz=float(w_half[2]),
            color=(0.25, 0.30, 0.33),
            label=f"wall_{w_idx}",
        )

    x0 = case["pos_np"][0]
    builder.add_particles(pos=x0, vel=np.zeros_like(x0), mass=case["mass_np"], radius=case["radius_np"])
    model = builder.finalize(device="cuda:0")
    state = model.state()

    header = 36
    viewer.set_model(model)
    viewer.show_particles = True
    viewer.set_camera(wp.vec3(1.65, -2.0, 1.55))
    viewer.camera.look_at((0.0, 0.0, -0.05))
    viewer.camera.fov = 50.0
    colors = wp.full(len(x0), value=wp.vec3(0.72, 0.59, 0.36), dtype=wp.vec3, device="cuda:0")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)

    frame_indices = list(range(4, len(world_pos), 4))
    poster_frame = frame_indices[min(len(frame_indices) - 1, int(0.82 * len(frame_indices)))]
    strip_frames = [frame_indices[int(q * (len(frame_indices) - 1))] for q in (0.25, 0.50, 0.75, 1.0)]
    strip_canvases = []

    video_3way = out_dir / "comparison_3way.mp4"
    video_2way = out_dir / "comparison.mp4"

    enc_3way = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{3 * width}x{height + header}",
            "-r",
            "50",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_3way),
        ],
        stdin=subprocess.PIPE,
    )
    enc_2way = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{2 * width}x{height + header}",
            "-r",
            "50",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_2way),
        ],
        stdin=subprocess.PIPE,
    )
    sources = (
        (case["pos_np"], "Implicit MPM (Reference)"),
        (nerd_pos, "Particle NeRD (Transolver)"),
        (world_pos, "WorldParticle (Warp CConv + 3D-RoPE ToMe)"),
    )
    try:
        for frame in frame_indices:
            t_sec = frame * case["dt"]
            canvas3 = Image.new("RGB", (3 * width, height + header), "white")
            canvas2 = Image.new("RGB", (2 * width, height + header), "white")
            draw3 = ImageDraw.Draw(canvas3)
            draw2 = ImageDraw.Draw(canvas2)
            state.body_q.assign(case["body_pose_np"][frame, 0:1, :7])

            rendered_imgs = []
            for pts, label in sources:
                state.particle_q.assign(pts[frame])
                viewer.begin_frame(t_sec)
                viewer.log_state(state)
                viewer.log_points("/model/particles", state.particle_q, radii=model.particle_radius, colors=colors)
                viewer.end_frame()
                pixels = viewer.get_frame(render_ui=False).numpy()
                rendered_imgs.append((Image.fromarray(pixels), label))

            for col, (img, label) in enumerate(rendered_imgs):
                canvas3.paste(img, (col * width, header))
                draw3.text(
                    (col * width + width / 2, header / 2),
                    f"{label}  (t = {t_sec:.2f} s)",
                    font=font,
                    fill="#1c2a32",
                    anchor="mm",
                )

            for col, idx in enumerate((0, 2)):
                img, label = rendered_imgs[idx]
                canvas2.paste(img, (col * width, header))
                draw2.text(
                    (col * width + width / 2, header / 2),
                    f"{label}  (t = {t_sec:.2f} s)",
                    font=font,
                    fill="#1c2a32",
                    anchor="mm",
                )

            if frame == poster_frame:
                canvas3.save(out_dir / "comparison_poster.png")
            if frame in strip_frames:
                strip_canvases.append(canvas3.copy())
            enc_3way.stdin.write(np.asarray(canvas3).tobytes())
            enc_2way.stdin.write(np.asarray(canvas2).tobytes())

        enc_3way.stdin.close()
        enc_2way.stdin.close()
        enc_3way.wait()
        enc_2way.wait()
        if strip_canvases:
            row_h = height + header
            film = Image.new("RGB", (3 * width, len(strip_canvases) * row_h), "white")
            for r_i, c_img in enumerate(strip_canvases):
                film.paste(c_img, (0, r_i * row_h))
            film.save(out_dir / "filmstrip.png")
    finally:
        for enc in (enc_3way, enc_2way):
            if enc.poll() is None:
                enc.stdin.close()
                enc.wait()


def plot_overall_dashboard(cases: list[dict], all_nerd: dict, all_world: dict, out_dir: Path) -> None:
    """Create a 4x3 multi-scenario overview figure across all benchmark showcase tasks."""
    fig, axes = plt.subplots(len(cases), 3, figsize=(16.0, 3.6 * len(cases)))
    if len(cases) == 1:
        axes = axes[None, :]
    fig.suptitle(
        "MIT CSAIL WorldParticle (Warp CConv + 3D-RoPE ToMe) vs. Particle NeRD vs. Implicit MPM",
        fontsize=16,
    )
    fig.subplots_adjust(hspace=0.38, wspace=0.25, top=0.93)

    for row, case in enumerate(cases):
        name = case["name"]
        title = name.replace("-", " ").title()
        nm = all_nerd[name]
        wm = all_world[name]
        t = wm["time_s"]

        axes[row, 0].plot(t, nm["global_rmse_mm"], color=COLOR_NERD, label="Particle NeRD", lw=1.6)
        axes[row, 0].plot(t, wm["global_rmse_mm"], color=COLOR_WORLD, label="WorldParticle", lw=1.8)
        axes[row, 0].set(title=f"{title} — Global RMSE (mm)", xlabel="Time (s)", ylabel="RMSE (mm)")
        axes[row, 0].legend(fontsize=8)

        axes[row, 1].plot(t, nm["near_rmse_mm"], color=COLOR_NERD, label="Particle NeRD", lw=1.6)
        axes[row, 1].plot(t, wm["near_rmse_mm"], color=COLOR_WORLD, label="WorldParticle", lw=1.8)
        axes[row, 1].set(title=f"{title} — Near-Collider (<5 cm) RMSE", xlabel="Time (s)", ylabel="RMSE (mm)")
        axes[row, 1].legend(fontsize=8)

        # Plot primary contact force component (Fz for vertical-flat, Fx for horizontal/rolling/circle)
        f_idx = 2 if "vertical" in name else 0
        f_lbl = "z" if f_idx == 2 else "x"
        axes[row, 2].plot(t, wm["ref_forces"][:, f_idx], color=COLOR_MPM, label="Implicit MPM", lw=1.4)
        axes[row, 2].plot(t, nm["pred_forces"][:, f_idx], color=COLOR_NERD, label="Particle NeRD", lw=1.2, alpha=0.8)
        axes[row, 2].plot(t, wm["pred_forces"][:, f_idx], color=COLOR_WORLD, label="WorldParticle", lw=1.5)
        axes[row, 2].set(title=f"{title} — Contact Force $F_{f_lbl}$ (N)", xlabel="Time (s)", ylabel=f"$F_{f_lbl}$ (N)")
        axes[row, 2].legend(fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.18)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"showcase_overview.{ext}", dpi=260)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-runs",
        type=Path,
        default=Path("Particle_NeRD_benchmark/benchmark_environment/runs"),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["horizontal-flat", "vertical-flat", "circle-flat", "rolling-flat"],
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("video/benchmarks/particle_world_showcase"),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and load existing checkpoints from --output",
    )
    args = parser.parse_args()

    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    np.random.seed(42)
    wp.init()
    wp.set_device(device)

    args.output.mkdir(parents=True, exist_ok=True)
    cases = [load_benchmark_case(args.benchmark_runs / name, device=device) for name in args.cases]
    print(f"Loaded {len(cases)} benchmark cases: {[c['name'] for c in cases]}", flush=True)

    world_model = WorldParticleNetwork(
        d_model=128,
        cconv_dim=64,
        heads=4,
        slot_attn_iters=3,
        num_decoder_layers=3,
        pre_merge_k=16,
        k_partite=2,
        velhead=True,
    ).to(device)
    nerd_model = ParticleTransolver(hidden=128, heads=4, slices=64, layers=4, history=3).to(device)

    for model_name, model, rollout_fn in (
        ("particle_world", world_model, rollout_world_particle),
        ("particle_nerd", nerd_model, rollout_transolver),
    ):
        ckpt_path = args.output / f"{model_name}_best.pt"
        if args.eval_only and ckpt_path.is_file():
            saved = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(saved["model"])
            print(f"Loaded existing checkpoint {ckpt_path}", flush=True)
            continue
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        rng = np.random.default_rng(42)
        horizons = (4, 8, 12, 16, 20)
        t_start = time.perf_counter()
        model.train()
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"\n=== Training {model_name} ({n_params:,} params) for {args.steps} steps ===",
            flush=True,
        )
        for step in range(1, args.steps + 1):
            stage = min(len(horizons) - 1, (step - 1) * len(horizons) // args.steps)
            horizon = horizons[stage]
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * (step - 1) / args.steps)))
            for g in optimizer.param_groups:
                g["lr"] = lr

            case = cases[(step - 1) % len(cases)]
            max_start = len(case["pos_np"]) - horizon - 1
            # Sample both settling/entry (0..400) and active contact/pushing (350..max_start)
            if rng.random() < 0.75:
                start_idx = int(rng.integers(200, max_start + 1))
            else:
                start_idx = int(rng.integers(0, max_start + 1))
            sample = extract_window(case, start=start_idx, horizon=horizon)

            optimizer.zero_grad(set_to_none=True)
            pred = rollout_fn(model, sample)
            loss = balanced_velocity_loss(
                pred, sample["targets"], sample["history"][:, -1], sample["poses"], sample["dt"]
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 50 == 0 or step == 1 or step == args.steps:
                print(
                    json.dumps(
                        {
                            "model": model_name,
                            "step": step,
                            "case": case["name"],
                            "horizon": horizon,
                            "loss": round(float(loss.detach()), 6),
                            "grad_norm": round(float(grad_norm), 4),
                            "elapsed_s": round(time.perf_counter() - t_start, 1),
                        }
                    ),
                    flush=True,
                )

        torch.save(
            {
                "architecture": "world_particle_v1" if model_name == "particle_world" else "particle_transolver_v1",
                "model": model.state_dict(),
                "config": {
                    "schema": "world_particle_v1" if model_name == "particle_world" else "object_particle_nerd_v1",
                    "model": model.config,
                    "dt": 0.005,
                    "features": {"support": 0.07},
                },
            },
            ckpt_path,
        )

    # Full 1,200-step (6.0 s) Open-Loop Evaluation, Plots, and 3D Video Rendering
    all_nerd_metrics = {}
    all_world_metrics = {}
    summary_table = {}

    viewer = newton.viewer.ViewerGL(width=640, height=480, headless=True, vsync=False)
    try:
        for case in cases:
            name = case["name"]
            case_out = args.output / name
            print(f"\n=== Evaluating 1,200-step (6.0 s) open-loop rollout on {name} ===", flush=True)
            force_rec = BenchmarkForceReconstructor(
                case["mass_np"],
                case["body_size_m"],
                case["shape"],
                dt=case["dt"],
                device=device,
            )
            nerd_pos, nerd_ms = evaluate_full_rollout(nerd_model, rollout_transolver, case)
            world_pos, world_ms = evaluate_full_rollout(world_model, rollout_world_particle, case)

            nerd_m = compute_case_metrics(case, nerd_pos, force_rec)
            world_m = compute_case_metrics(case, world_pos, force_rec)
            nerd_m["summary"]["step_time_ms"] = nerd_ms
            world_m["summary"]["step_time_ms"] = world_ms

            all_nerd_metrics[name] = nerd_m
            all_world_metrics[name] = world_m
            summary_table[name] = {
                "particle_nerd": nerd_m["summary"],
                "particle_world": world_m["summary"],
            }
            print(
                json.dumps(
                    {
                        "case": name,
                        "particle_nerd_rmse_mm": round(nerd_m["summary"]["mean_global_rmse_mm"], 2),
                        "particle_world_rmse_mm": round(world_m["summary"]["mean_global_rmse_mm"], 2),
                        "particle_nerd_near_rmse_mm": round(nerd_m["summary"]["mean_near_contact_rmse_mm"], 2),
                        "particle_world_near_rmse_mm": round(world_m["summary"]["mean_near_contact_rmse_mm"], 2),
                        "particle_nerd_force_rmse_n": round(nerd_m["summary"]["force_vector_rmse_n"], 2),
                        "particle_world_force_rmse_n": round(world_m["summary"]["force_vector_rmse_n"], 2),
                        "particle_world_step_ms": round(world_ms, 2),
                    }
                ),
                flush=True,
            )
            plot_case_comparison(case, nerd_m, world_m, case_out)
            render_case_videos(case, nerd_pos, world_pos, case_out, viewer=viewer)
    finally:
        viewer.close()

    plot_overall_dashboard(cases, all_nerd_metrics, all_world_metrics, args.output)
    (args.output / "summary.json").write_text(json.dumps(summary_table, indent=2))
    print(f"\nShowcase complete! Results written to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
