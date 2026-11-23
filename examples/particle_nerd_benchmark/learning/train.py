# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train one shared Particle Transolver or WorldParticle model from native 200 Hz object trajectories."""

import argparse
import hashlib
import inspect
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import warp as wp
from torch.nn.parallel import DistributedDataParallel

from .data import Episodes
from .data_v1 import DEFAULT_CATEGORIES, EpisodesV1
from .network import (
    CONTACT_SETTINGS,
    OBJECT_PIPELINE,
    OBJECT_SCHEMA,
    PARTICLE_NERD_ARCHITECTURE,
    PARTICLE_NERD_SOURCES,
    ObjectParticleStepper,
    ParticleNeRDNetwork,
    ParticleTransolver,
    WorldParticleNetwork,
    features,
    features_world_particle,
    nerd_source_hashes,
)


def _unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def rollout(model, sample, return_vel=False, return_history=False):
    """Advance predicted particle history without reference-state replacement."""
    raw_model = _unwrap_model(model)
    is_world_particle = isinstance(raw_model, (WorldParticleNetwork, ParticleNeRDNetwork))
    use_vel_state = isinstance(raw_model, ParticleNeRDNetwork) or (
        is_world_particle and bool(raw_model.config.get("translation_invariant", False))
    )
    history = sample["history"]
    stepper = (
        None
        if is_world_particle
        else ObjectParticleStepper(
            sample["kinds"], sample["sizes"], sample["friction"], sample["material"], sample["dt"]
        )
    )
    predicted = []
    predicted_vels = []
    for pose, velocity in zip(sample["poses"], sample["velocities"], strict=True):
        if is_world_particle:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if use_vel_state or return_vel:
                    current, vel_next = raw_model.step_from_history(
                        history,
                        pose.contiguous(),
                        velocity.contiguous(),
                        sample["kinds"],
                        sample["sizes"],
                        sample["friction"],
                        sample["material"],
                        dt=sample["dt"],
                        return_vel=True,
                    )
                    predicted_vels.append(vel_next)
                else:
                    current = raw_model.step_from_history(
                        history,
                        pose.contiguous(),
                        velocity.contiguous(),
                        sample["kinds"],
                        sample["sizes"],
                        sample["friction"],
                        sample["material"],
                        dt=sample["dt"],
                    )
        else:
            if sample["collider_input"] != OBJECT_PIPELINE["collider_input"]:
                raise ValueError("NeRD training requires pre-rigid collider inputs")
            current = stepper.step(model, history, pose, velocity)
        predicted.append(current)
        if use_vel_state:
            # Propagate explicit velhead state v_{t+1} into the history buffer
            prev_synth = current - sample["dt"] * vel_next
            history = torch.stack((history[:, 1], prev_synth, current), dim=1).contiguous()
        else:
            history = torch.cat((history[:, 1:], current[:, None]), dim=1).contiguous()
    if return_history:
        return torch.stack(predicted), torch.stack(predicted_vels), history
    if return_vel:
        return torch.stack(predicted), torch.stack(predicted_vels)
    return torch.stack(predicted)


def rest_unroll(model, dataset, rest_indices, rng, device, steps, segment):
    """Unroll a long resting-bed window with truncated gradients and accumulate their gradients in place.

    The model's own predicted state is carried from one segment to the next (detached), so later segments start
    from the drifted state that the model itself produced. Each segment is compared with the reference and its
    gradient is accumulated before the graph is freed, so memory matches one short window.
    """
    idx = int(rng.choice(rest_indices))
    length = dataset.num_frames(idx)
    record = dataset.records[idx][0]
    earliest = round(float(record.get("rest_start_s", 0.0)) / float(record["dt_s"]))
    start = int(rng.integers(min(earliest, length - steps - 1), length - steps))
    sample = dataset.sequence(rng, steps, device, index=idx, start=start)
    history, dt, count = sample["history"], sample["dt"], steps // segment
    total = 0.0
    for k in range(count):
        a, b = k * segment, (k + 1) * segment
        part = dict(sample)
        part["history"] = history
        part["poses"], part["velocities"], part["targets"] = (
            sample["poses"][a:b],
            sample["velocities"][a:b],
            sample["targets"][a:b],
        )
        reference_start = sample["history"][:, -1] if k == 0 else sample["targets"][a - 1]
        pred, pred_vel, history = rollout(model, part, return_vel=True, return_history=True)
        loss = world_particle_loss(pred, pred_vel, part["targets"], reference_start, dt) / count
        loss.backward()
        total += float(loss.detach())
        history = history.detach()
    return torch.tensor(total, device=device)


def balanced_loss(predicted, target, initial, dt):
    """Equal weight for nonempty moving and stationary reference particle groups."""
    moving = torch.linalg.vector_norm(target - initial[None], dim=-1) > 0.0005
    error = ((predicted - target) / dt).square().sum(-1)
    counts = torch.stack((moving.sum(), (~moving).sum()))
    sums = torch.stack(((error * moving).sum(), (error * ~moving).sum()))
    return (sums / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)


def world_particle_loss(predicted, predicted_vel, target, initial, dt):
    """Upstream WorldParticle position + velocity + stationary-equilibrium loss."""
    loss_pos = balanced_loss(predicted, target, initial, dt)
    targ_seq = torch.cat((initial[None], target), dim=0)
    vel_targ = (targ_seq[1:] - targ_seq[:-1]) / dt
    vel_err = (predicted_vel - vel_targ).square().sum(-1)
    moving = torch.linalg.vector_norm(target - initial[None], dim=-1) > 0.0005
    counts = torch.stack((moving.sum(), (~moving).sum()))
    vel_sums = torch.stack(((vel_err * moving).sum(), (vel_err * ~moving).sum()))
    loss_vel = (vel_sums / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)

    # Explicit step-to-step displacement consistency + zero-drift penalty on at-rest particles
    pred_seq = torch.cat((initial[None], predicted), dim=0)
    fd_vel_pred = (pred_seq[1:] - pred_seq[:-1]) / dt
    at_rest = torch.linalg.vector_norm(vel_targ, dim=-1) < 0.001
    rest_drift = ((fd_vel_pred.square().sum(-1)) * at_rest).sum() / at_rest.sum().clamp_min(1)
    return 0.6 * loss_pos + 0.25 * loss_vel + 0.15 * rest_drift


def freefall_macro_loss(predicted, predicted_vel, target, initial, dt, z_floor=-0.30, p_radius=0.016):
    """Cylindrical (r_i, z_i) + joint (r, z) Sliced-Wasserstein profile + floor & rest equilibrium loss."""
    pred = predicted.float()
    pred_vel = predicted_vel.float()
    targ = target.float()
    init = initial.float()
    base = world_particle_loss(pred, pred_vel, targ, init, dt)

    com_pred = pred.mean(dim=1)
    com_targ = targ.mean(dim=1)
    v_com_pred = pred_vel.mean(dim=1)
    targ_seq = torch.cat((init[None], targ), dim=0)
    vel_targ = (targ_seq[1:] - targ_seq[:-1]) / dt
    v_com_targ = vel_targ.mean(dim=1)
    loss_com = ((com_pred - com_targ) / dt).square().sum(dim=-1).mean() + (v_com_pred - v_com_targ).square().sum(
        dim=-1
    ).mean()

    # Per-particle cylindrical (r_i, z_i) loss (avoids Cartesian Jensen shrinkage toward r = 0)
    r_pred = torch.sqrt(((pred[:, :, :2] - com_pred[:, None, :2]).square().sum(dim=-1)) + 1e-6)
    r_targ = torch.sqrt(((targ[:, :, :2] - com_targ[:, None, :2]).square().sum(dim=-1)) + 1e-6)
    loss_cyl_pos = ((r_pred - r_targ) / dt).square().mean() + ((pred[:, :, 2] - targ[:, :, 2]) / dt).square().mean()

    pred_seq = torch.cat((init[None], pred), dim=0)
    r_pred_seq = torch.sqrt(
        ((pred_seq[:, :, :2] - pred_seq[:, :, :2].mean(dim=1, keepdim=True)).square().sum(dim=-1)) + 1e-6
    )
    r_targ_seq = torch.sqrt(
        ((targ_seq[:, :, :2] - targ_seq[:, :, :2].mean(dim=1, keepdim=True)).square().sum(dim=-1)) + 1e-6
    )
    vr_pred = (r_pred_seq[1:] - r_pred_seq[:-1]) / dt
    vr_targ = (r_targ_seq[1:] - r_targ_seq[:-1]) / dt
    vz_pred = (pred_seq[1:, :, 2] - pred_seq[:-1, :, 2]) / dt
    vz_targ = vel_targ[:, :, 2]
    loss_cyl_vel = (vr_pred - vr_targ).square().mean() + (vz_pred - vz_targ).square().mean()

    # Joint (r, z) 2D Sliced-Wasserstein profile across 16 projection angles in [0, pi)
    angles = torch.linspace(0.0, torch.pi * 15.0 / 16.0, 16, device=pred.device, dtype=torch.float32)
    ca = torch.cos(angles).view(1, 1, 16)
    sa = torch.sin(angles).view(1, 1, 16)
    proj_pred = r_pred.unsqueeze(-1) * ca + pred[:, :, 2:3] * sa
    proj_targ = r_targ.unsqueeze(-1) * ca + targ[:, :, 2:3] * sa
    proj_sort_pred = torch.sort(proj_pred, dim=1)[0]
    proj_sort_targ = torch.sort(proj_targ, dim=1)[0]
    n_pts = pred.shape[1]
    q_weights = (1.0 + 3.0 * torch.linspace(0.0, 1.0, n_pts, device=pred.device).pow(2)).view(1, n_pts, 1)
    loss_profile = (((proj_sort_pred - proj_sort_targ) / dt).square() * q_weights).mean()

    # Settled rest-equilibrium penalty (L2 + L1 Charbonnier for finite-time Coulomb dry-friction arrest)
    fd_vel_pred = (pred_seq[1:] - pred_seq[:-1]) / dt
    at_rest = (vel_targ.norm(dim=-1) < 0.015).float()
    v_sq = fd_vel_pred.square().sum(-1) + pred_vel.square().sum(-1)
    v_l1 = torch.sqrt(v_sq + 1e-6)
    loss_rest = ((v_sq + 0.5 * v_l1) * at_rest).sum() / at_rest.sum().clamp_min(1.0)

    # 3D Chamfer distance + short-range inter-particle anti-clumping repulsion on sampled steps
    t_len = pred.shape[0]
    s_idx = torch.unique(torch.linspace(0, t_len - 1, min(t_len, 6), device=pred.device).round().long())
    p_sub = pred[s_idx]  # [K, N, 3]
    t_sub = targ[s_idx]  # [K, N, 3]
    d_pt = torch.cdist(p_sub, t_sub)  # [K, N, N]
    loss_chamfer = 0.5 * ((d_pt.amin(dim=-1) / dt).square().mean() + (d_pt.amin(dim=-2) / dt).square().mean())

    d_pp = torch.cdist(p_sub, p_sub)
    eye_mask = torch.eye(n_pts, device=pred.device, dtype=torch.bool).unsqueeze(0)
    d_pp = d_pp.masked_fill(eye_mask, 1.0)
    overlap = torch.relu(0.022 - d_pp)
    loss_repel = (overlap / dt).square().sum(dim=-1).mean()

    z_min_allowed = z_floor + 0.25 * p_radius
    floor_pen = torch.relu(z_min_allowed - pred[:, :, 2])
    loss_floor = (floor_pen / dt).square().mean()
    return (
        0.15 * base
        + 1.5 * loss_com
        + 1.0 * loss_cyl_pos
        + 0.6 * loss_cyl_vel
        + 2.5 * loss_profile
        + 3.0 * loss_chamfer
        + 1.5 * loss_repel
        + 2.5 * loss_rest
        + 4.0 * loss_floor
    )


@torch.no_grad()
def validate(model, dataset, device, horizon, count):
    """Fixed windows, averaged per trajectory; no training trajectory is sampled."""
    model.eval()
    rng = np.random.default_rng(9042)
    metrics = []
    groups = {}
    for index, (record, _, _) in enumerate(dataset.records):
        groups.setdefault((record["shape"], record["mode"]), []).append(index)
    # Spread fixed validation windows across shape, actuation mode and motion.
    selected = []
    per_group = max(1, count // len(groups))
    for indices in groups.values():
        selected.extend(indices[i] for i in np.linspace(0, len(indices) - 1, min(per_group, len(indices))).astype(int))
    for index in selected:
        record = dataset.records[index][0]
        contact_start = record.get(
            "contact_start",
            {"horizontal": 600, "circle": 600, "vertical": 400, "hold": 1000}.get(record["motion"], 200),
        )
        per_episode = []
        # Every selected trajectory contributes initial settling and actual body
        # interaction. Dynamic objects are released at 1 s; prescribed entries
        # end by 2.3 s. Never validate a horizontal push only before it starts.
        for start in (0, contact_start):
            sample = dataset.sequence(rng, horizon, device, index=index, start=start)
            pred = rollout(model, sample)
            initial = sample["history"][:, -1]
            per_episode.append(
                [
                    float(balanced_loss(pred, sample["targets"], initial, sample["dt"])),
                    float((pred - sample["targets"]).square().sum(-1).mean().sqrt() * 1000),
                    float((initial[None] - sample["targets"]).square().sum(-1).mean().sqrt() * 1000),
                ]
            )
        metrics.append(np.mean(per_episode, axis=0))
    values = np.mean(metrics, axis=0)
    model.train()
    return dict(
        validation_loss=float(values[0]),
        position_rmse_mm=float(values[1]),
        frozen_baseline_rmse_mm=float(values[2]),
        validation_trajectories=len(metrics),
        validation_windows=2 * len(metrics),
        horizon=horizon,
    )


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arch",
        choices=("particle_transolver", "world_particle", "particle_nerd_local"),
        default="particle_transolver",
        help="Neural particle architecture to train",
    )
    parser.add_argument("--resume-from", type=Path, default=None, help="Optional checkpoint path to resume from")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument(
        "--horizon", type=int, default=100, help="Maximum recurrent window; 100 native steps correspond to 0.5 seconds"
    )
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-count", type=int, default=24)
    parser.add_argument(
        "--validation-horizon",
        type=int,
        default=200,
        help="Recurrent validation steps; 200 steps correspond to one second",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--far-support-scale", type=float, default=5.0, help="Long-range tanh gap multiplier relative to support"
    )
    parser.add_argument(
        "--far-gap-weight", type=float, default=0.3, help="Weight on long-range tanh gap term in [0, 1]"
    )
    parser.add_argument(
        "--far-influence-weight", type=float, default=0.15, help="Weight on long-range sech^2 influence tail in [0, 1]"
    )
    parser.add_argument("--smoke", action="store_true", help="Short overfit check on local training episodes only")
    parser.add_argument(
        "--update-mode",
        choices=("gated",),
        default="gated",
        help="Particle update rule of particle_nerd_local; only the gated rule exists (kept for existing job files)",
    )
    parser.add_argument(
        "--finetune-from",
        type=Path,
        default=None,
        help="Initialize weights from a checkpoint (step and optimizer restart)",
    )
    parser.add_argument(
        "--rest-fraction", type=float, default=0.0, help="Share of training steps that unroll a long resting bed"
    )
    parser.add_argument("--rest-steps", type=int, default=1000, help="Native steps in one long resting-bed unroll")
    parser.add_argument("--rest-segment", type=int, default=100, help="Truncation length of each gradient segment")
    parser.add_argument(
        "--rest-windows",
        action="store_true",
        help="Also sample resting episodes (object resting on the bed, bare bed) as ordinary training windows",
    )
    parser.add_argument(
        "--freefall-fraction", type=float, default=0.40, help="Share of training windows drawn from free-fall episodes"
    )
    parser.add_argument(
        "--data-format",
        choices=("legacy", "v1"),
        default="legacy",
        help="legacy: manifest.json + one .h5 per trajectory; v1: the terrain / category dataset (attributes only)",
    )
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES), help="v1 categories to train on")
    parser.add_argument("--terrains", nargs="+", default=None, help="v1 terrains to train on (default: all)")
    args = parser.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42)
    np.random.seed(42)
    wp.init()
    wp.set_device(device)
    torch.set_num_threads(4)
    if world > 1:
        # Rank zero evaluates the full validation set while other ranks wait.
        dist.init_process_group("nccl", device_id=torch.device(device), timeout=timedelta(minutes=30))
    is_world = args.arch in ("world_particle", "particle_nerd_local")
    collider_input = "material_step" if is_world else OBJECT_PIPELINE["collider_input"]
    if args.data_format == "v1":

        def load(split):
            return EpisodesV1(args.data, split, categories=tuple(args.categories), terrains=args.terrains)

    else:

        def load(split):
            return Episodes(args.data, split, collider_input=collider_input)

    train = load("train")
    validation = train if args.smoke else load("validation")
    if args.arch == "particle_nerd_local":
        model = ParticleNeRDNetwork(
            far_support_scale=args.far_support_scale,
            far_gap_weight=args.far_gap_weight,
            far_influence_weight=args.far_influence_weight,
        ).to(device)
        source_objs = PARTICLE_NERD_SOURCES
        arch_tag = PARTICLE_NERD_ARCHITECTURE
    elif args.arch == "world_particle":
        model = WorldParticleNetwork(
            translation_invariant=True,
            far_support_scale=args.far_support_scale,
            far_gap_weight=args.far_gap_weight,
            far_influence_weight=args.far_influence_weight,
            use_actual_velocity=True,
        ).to(device)
        source_objs = (features_world_particle, WorldParticleNetwork)
        arch_tag = "world_particle_v1"
    else:
        model = ParticleTransolver().to(device)
        source_objs = (features, ParticleTransolver, ObjectParticleStepper)
        arch_tag = "particle_transolver_v1"
    distributed = model if (world == 1 or is_world) else DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_step = 1
    best = float("inf")
    if args.finetune_from is not None:
        source = torch.load(args.finetune_from, map_location=device, weights_only=False)["model"]
        if isinstance(model, ParticleNeRDNetwork):
            # Drops the weights of removed modules and fails on any other mismatch
            model.load_checkpoint_state(source)
            missing, unexpected = [], []
        else:
            missing, unexpected = model.load_state_dict(source, strict=False)
        if rank == 0:
            print(
                f"Initialized from {args.finetune_from} (missing={len(missing)}, unexpected={len(unexpected)})",
                flush=True,
            )
    if args.resume_from is not None and args.resume_from.is_file():
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        missing, _ = model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt and not missing:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0)) + 1
        best = float("inf") if missing else float(ckpt.get("validation", {}).get("validation_loss", float("inf")))
        if rank == 0:
            print(f"Resumed from {args.resume_from} at step {start_step - 1} (new keys={len(missing)})", flush=True)
    if world > 1 and is_world:
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
    rng = np.random.default_rng(42 + rank + 1000 * (start_step - 1))
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / "last.pt").exists():
            raise ValueError("Output already has a checkpoint; use a new directory")
        (args.output / "config.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "architecture": arch_tag,
                    "model": model.config,
                    "world_size": world,
                    "train_trajectories": len(train.records),
                    "validation_trajectories": len(validation.records),
                    "data_manifest_sha256": {
                        str(p.relative_to(args.data)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(args.data.rglob("manifest.json"))
                    },
                    "network_source_sha256": nerd_source_hashes()
                    if not is_world
                    else {
                        str(Path(inspect.getfile(obj)).name): hashlib.sha256(
                            Path(inspect.getfile(obj)).read_bytes()
                        ).hexdigest()
                        for obj in source_objs
                    },
                    "horizon_curriculum": (
                        "Six equal stages: 4, 8, 16, 32, 64, 128 steps, each capped at the requested horizon"
                    ),
                    "learning_rate_schedule": "Cosine from initial learning rate to 10% of initial",
                    "validation_windows": (
                        "Two windows per selected trajectory: initial settling, plus "
                        "3 s horizontal/circle, 2 s vertical, 5 s hold, or 1 s dynamic release"
                    ),
                    "active_selection": None if is_world else OBJECT_PIPELINE["active_k"],
                    "particle_projection": False if is_world else OBJECT_PIPELINE["particle_projection"],
                    "displacement_gate": not is_world,
                    "pipeline": None if is_world else OBJECT_PIPELINE,
                    "contact_settings": None if is_world else CONTACT_SETTINGS,
                    "loss": "equal nonempty moving/stationary group means; reference displacement threshold 0.5 mm",
                    "code_hashes": {
                        str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in Path(__file__).parents[1].rglob("*.py")
                    },
                },
                indent=2,
                default=str,
            )
        )
    start_time = time.perf_counter()
    if rank == 0 and args.resume_from is not None and args.resume_from.is_file():
        ckpt_init = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        ckpt_init["model"] = {k: v.cpu() for k, v in model.state_dict().items()}
        ckpt_init["model_config"] = model.config
        torch.save(ckpt_init, args.output / "last.pt")
        torch.save(ckpt_init, args.output / "best.pt")
    log = (args.output / "metrics.jsonl").open("a") if rank == 0 else None
    try:
        wp_horizons = (8, 16, 32, 64, 100, 100)
        # Resting episodes are only seen in the long unrolls unless --rest-windows also samples them as windows.
        obj_indices = [
            i
            for i, (r, _, _) in enumerate(train.records)
            if r["shape"] != "freefall" and (args.rest_windows or (r["shape"] != "rest" and not r.get("rest_unroll")))
        ]
        ff_indices = [i for i, (r, _, _) in enumerate(train.records) if r["shape"] == "freefall"]
        rest_indices = [i for i, (r, _, _) in enumerate(train.records) if r.get("rest_unroll")]
        if rank == 0:
            resting = len(set(obj_indices) & set(rest_indices))
            print(
                f"Window episodes: {len(obj_indices)} ({resting} resting), free-fall pool: {len(ff_indices)}, "
                f"long-unroll pool: {len(rest_indices)}",
                flush=True,
            )
        for step in range(start_step, args.steps + 1):
            stage = min(5, 6 * (step - 1) // args.steps)
            horizon = min(args.horizon, wp_horizons[stage] if is_world else 4 * 2**stage)
            learning_rate = args.lr * (0.1 + 0.9 * (1 + np.cos(np.pi * (step - 1) / args.steps)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            if rest_indices and is_world and rng.random() < args.rest_fraction:
                optimizer.zero_grad(set_to_none=True)
                loss = rest_unroll(distributed, train, rest_indices, rng, device, args.rest_steps, args.rest_segment)
            else:
                if ff_indices and obj_indices and rng.random() < args.freefall_fraction:
                    idx = int(rng.choice(ff_indices))
                    u_win = rng.random()
                    if u_win < 0.35:
                        start_win = int(rng.integers(0, 45))
                    elif u_win < 0.75:
                        start_win = int(rng.integers(45, 220))
                    else:
                        start_win = None
                    sample = train.sequence(rng, horizon, device, index=idx, start=start_win)
                else:
                    idx = int(rng.choice(obj_indices)) if obj_indices else None
                    sample = train.sequence(rng, horizon, device, index=idx)
                optimizer.zero_grad(set_to_none=True)
                if is_world:
                    pred, pred_vel = rollout(distributed, sample, return_vel=True)
                    if sample["shape"] == "freefall":
                        loss = freefall_macro_loss(
                            pred, pred_vel, sample["targets"], sample["history"][:, -1], sample["dt"]
                        )
                    else:
                        loss = world_particle_loss(
                            pred, pred_vel, sample["targets"], sample["history"][:, -1], sample["dt"]
                        )
                else:
                    pred = rollout(distributed, sample)
                    loss = balanced_loss(pred, sample["targets"], sample["history"][:, -1], sample["dt"])
                loss.backward()
            if world > 1 and is_world:
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
            if torch.isfinite(grad_norm) and torch.isfinite(loss):
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                if rank == 0:
                    print(f'{{"step": {step}, "skipped_nonfinite_grad": true}}', flush=True)
            if step % 10 == 0 and rank == 0:
                record = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "grad_norm": float(grad_norm),
                    "horizon": horizon,
                    "learning_rate": learning_rate,
                    "elapsed_s": time.perf_counter() - start_time,
                }
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + "\n")
                log.flush()
            if step % args.validate_every == 0 or step == args.steps:
                if world > 1:
                    dist.barrier()
                if rank == 0:
                    result = validate(model, validation, device, args.validation_horizon, args.validation_count)
                    result.update(
                        step=step,
                        elapsed_s=time.perf_counter() - start_time,
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    )
                    print(json.dumps(result), flush=True)
                    log.write(json.dumps(result) + "\n")
                    log.flush()
                    saved = {
                        "architecture": arch_tag,
                        "model": model.state_dict(),
                        "model_config": model.config,
                        "step": step,
                        "optimizer": optimizer.state_dict(),
                        "validation": result,
                        "config": {
                            "schema": "object_particle_nerd_v1" if is_world else OBJECT_SCHEMA,
                            "architecture": arch_tag,
                            "model": model.config,
                            "dt": train.records[0][1]["dt_s"],
                            "features": {"support": 0.07},
                            "active_k": None if is_world else OBJECT_PIPELINE["active_k"],
                            "displacement_gate": not is_world,
                            "pipeline": None if is_world else OBJECT_PIPELINE,
                            "contact_settings": None if is_world else CONTACT_SETTINGS,
                        },
                    }
                    torch.save(saved, args.output / "last.pt")
                    if result["validation_loss"] < best:
                        best = result["validation_loss"]
                        torch.save(saved, args.output / "best.pt")
                if world > 1:
                    dist.barrier()
    finally:
        if log:
            log.close()
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
