# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differentiable recurrent training on unchanged episode splits; default horizon 0.5 s."""

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
import warp as wp
from torch.nn.parallel import DistributedDataParallel as DDP

from particle_nerd.transolver.features import features, material_features
from particle_nerd.transolver.inference import box_ground_state, manifest_colliders, manifest_sdf
from particle_nerd.transolver.model import ParticleTransolver


class PositionEpisodes:
    def __init__(self, root, split, history=3):
        self.root = Path(root)
        self.history = history
        self.records = []
        for path in sorted(self.root.rglob("*.campaign.json")):
            record = json.loads(path.read_text())
            if record["split"] == split:
                record["path"] = path.parent / record["file"]
                self.records.append(record)
        if not self.records:
            raise ValueError(f"No {split} episodes under {root}")
        ids = [r["index"] for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate episode IDs; keep one accepted sample per episode")

    def sample(self, rng, device="cuda:0", episode=None, frame=None):
        record = self.records[int(rng.integers(len(self.records))) if episode is None else episode]
        with h5py.File(record["path"], "r") as f:
            m = json.loads(f.attrs["manifest"])
            steps = f["frame_steps"][:]
            first = int(np.searchsorted(steps, m["settled_step"]))
            if frame is None:
                lower = first if rng.random() < 0.8 else 0
                frame = int(rng.integers(lower, len(steps) - 1))

            def tensor(a):
                return torch.as_tensor(np.asarray(a), device=device, dtype=torch.float32)

            frames = [max(0, frame - k) for k in reversed(range(self.history))]
            positions = np.stack([f["particle_position"][i] for i in frames], axis=1)
            poses = np.stack([f["body_pose"][steps[i]] for i in frames])
            return dict(
                x=tensor(f["particle_position"][frame]),
                position_history=tensor(positions),
                pose_history=tensor(poses),
                next_x=tensor(f["particle_position"][frame + 1]),
                next_v=tensor(f["particle_velocity"][frame + 1]),
                pose=tensor(f["body_pose"][steps[frame]]),
                body_velocity=tensor(f["body_velocity"][steps[frame]]),
                material=tensor(material_features(m, float(f["particle_mass"][0]), float(f["particle_radius"][0]))),
                mass=m["body_mass_kg"],
                index=record["index"],
                frame=frame,
                manifest=m,
            )


class Sequences(PositionEpisodes):
    def sequence(self, rng, steps, device, episode=None):
        sample = self.sample(rng, device, episode=episode)
        record = next(r for r in self.records if r["index"] == sample["index"])
        with h5py.File(record["path"], "r") as f:
            frames = f["frame_steps"][:]
            if len(frames) <= steps:
                raise ValueError("Rollout length exceeds available trajectory")
            start = min(sample["frame"], len(frames) - steps - 1)
            if start != sample["frame"]:
                index = self.records.index(record)
                sample = self.sample(rng, device, episode=index, frame=start)
            interval = np.diff(frames) * sample["manifest"]["dt"]
            if not np.allclose(interval, 0.02):
                raise ValueError("This experiment requires the existing uniform 0.02 s recording interval")
            sample["targets"] = torch.as_tensor(f["particle_position"][start + 1 : start + steps + 1], device=device)
            selected = frames[start : start + steps]
            sample["poses"] = torch.as_tensor(f["body_pose"][selected], device=device)
            sample["velocities"] = torch.as_tensor(f["body_velocity"][selected], device=device)
        return sample


def position_loss(predicted, targets, initial, dt, mode="weighted"):
    """Position supervision; balanced mode gives each nonempty motion group equal weight.

    A particle-time sample is moving when its reference displacement from the
    sequence start exceeds 0.5 mm. This preserves the original baseline labels.
    No predicted motion or region selector determines the supervision mask.
    """
    moving = (targets - initial[None]).norm(dim=-1) > 0.0005
    error = ((predicted - targets) / dt).square().sum(-1)
    counts = torch.stack((moving.sum(), (~moving).sum()))
    sums = torch.stack(((error * moving).sum(), (error * ~moving).sum()))
    means = sums / counts.clamp_min(1)
    if mode == "balanced":
        loss = means.sum() / (counts > 0).sum().clamp_min(1)
    elif mode == "weighted":
        weight = 1.0 + 4.0 * moving.float()
        loss = (error * weight).sum() / weight.sum()
    else:
        raise ValueError(f"Unknown loss mode: {mode}")
    return loss, means, counts


def rollout(model, sample, config):
    history = sample["position_history"]
    kinds, sizes, friction = manifest_colliders(sample["manifest"], history.device)
    predictions = []
    for pose, velocity in zip(sample["poses"], sample["velocities"]):
        poses, velocities = box_ground_state(pose, velocity)
        inputs = features(
            history,
            poses,
            velocities,
            kinds,
            sizes,
            friction,
            sample["material"],
            config["dt"],
            **config["features"],
            sdf=manifest_sdf(sample["manifest"], history.device),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            displacement = model(*inputs)
        x = history[:, -1] + displacement
        predictions.append(x)
        # Keep the graph: subsequent losses reach earlier predicted states through Warp features.
        history = torch.cat((history[:, 1:], x[:, None]), dim=1)
    return torch.stack(predictions)


@torch.no_grad()
def validate(model, data, config, samples, steps, device):
    model.eval()
    rng = np.random.default_rng(90517)
    total_error = moving_error = stationary_error = frozen_error = 0.0
    count = moving_count = stationary_count = 0
    for i in range(min(samples, len(data.records))):
        sample = data.sequence(rng, steps, device, episode=i)
        predicted = rollout(model, sample, config)
        error = (predicted - sample["targets"]).square().sum(-1)
        frozen = (sample["position_history"][:, -1][None] - sample["targets"]).square().sum(-1)
        moving = frozen > 0.0005**2
        total_error += float(error.sum())
        frozen_error += float(frozen.sum())
        moving_error += float(error[moving].sum())
        stationary_error += float(error[~moving].sum())
        count += error.numel()
        moving_count += int(moving.sum())
        stationary_count += int((~moving).sum())
    return dict(
        position_rmse_m=float(np.sqrt(total_error / count)),
        moving_position_rmse_m=float(np.sqrt(moving_error / max(1, moving_count))),
        stationary_position_rmse_m=float(np.sqrt(stationary_error / max(1, stationary_count))),
        moving_count=moving_count,
        stationary_count=stationary_count,
        balanced_position_rmse_m=float(
            np.sqrt(
                (moving_error / max(1, moving_count) + stationary_error / max(1, stationary_count))
                / max(1, int(moving_count > 0) + int(stationary_count > 0))
            )
        ),
        frozen_position_rmse_m=float(np.sqrt(frozen_error / count)),
        rollout_steps=steps,
        rollout_horizon_s=steps * config["dt"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--initialize", type=Path, help="Initialize weights from a compatible displacement checkpoint")
    horizon = p.add_mutually_exclusive_group()
    horizon.add_argument("--rollout-seconds", type=float, help="Recurrent training horizon in seconds (default: 0.5)")
    horizon.add_argument("--rollout-steps", type=int, help="Explicit legacy step-count override; each step is 0.02 s")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--loss-mode", choices=("weighted", "balanced"), default="weighted")
    p.add_argument("--evaluate-every", type=int, default=500)
    p.add_argument("--validation-samples", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--slices", type=int, default=64)
    p.add_argument("--seed", type=int, default=90516)
    args = p.parse_args()
    dt = 0.02
    if args.rollout_steps is None:
        seconds = 0.5 if args.rollout_seconds is None else args.rollout_seconds
        if not np.isfinite(seconds) or seconds <= 0:
            p.error("--rollout-seconds must be positive and finite")
        args.rollout_steps = round(seconds / dt)
        if not np.isclose(args.rollout_steps * dt, seconds, atol=1e-9, rtol=0):
            p.error("--rollout-seconds must be an integer multiple of the 0.02 s prediction interval")
    args.rollout_seconds = args.rollout_steps * dt
    if min(args.steps, args.rollout_steps, args.evaluate_every, args.validation_samples) < 1:
        p.error("Training/rollout/validation counts must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("--learning-rate must be positive and finite")
    rank, local, world = [
        int(os.environ.get(k, 1 if k == "WORLD_SIZE" else 0)) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    ]
    wp.init()
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + rank * 100003)
    train, validation = Sequences(args.data, "train"), Sequences(args.data, "validation")
    if {r["index"] for r in train.records} & {r["index"] for r in validation.records}:
        raise ValueError("Training and validation episode IDs overlap")
    model = ParticleTransolver(hidden=args.hidden, layers=args.layers, heads=args.heads, slices=args.slices).to(device)
    with h5py.File(train.records[0]["path"], "r") as source:
        manifest = json.loads(source.attrs["manifest"])
    config = dict(
        dt=dt,
        model=model.config,
        features=dict(support=0.07),
        body_size_m=manifest["body_size_m"],
        geometry=manifest.get("geometry", "box"),
        body_contact_friction=manifest.get("body_contact_friction", 0.6),
    )
    if args.initialize:
        initial = torch.load(args.initialize, map_location=device, weights_only=False)
        if initial["architecture"] != "particle_transolver_v1" or initial["config"]["model"] != model.config:
            raise ValueError("Initialization architecture differs from the requested model")
        model.load_state_dict(initial["model"])
    # Static graph supports repeated shared-network forwards in a differentiable rollout.
    forward = DDP(model, device_ids=[local], static_graph=True) if world > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    args.output.mkdir(parents=True, exist_ok=True)
    run = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        config=config,
        train_ids=[r["index"] for r in train.records],
        validation_ids=[r["index"] for r in validation.records],
        world_size=world,
        output="displacement only; velocity = displacement / dt",
        selection="all particles",
        recurrence=dict(
            prediction_dt_s=dt,
            training_steps=args.rollout_steps,
            training_horizon_s=args.rollout_seconds,
            particle_inputs="Recorded history once, then predicted particle positions at every subsequent step",
            rigid_inputs="Recorded current rigid-body pose and velocity at every step",
            gradients="Full backpropagation through every recurrent step, including Warp particle/collider features",
            supervision="Position loss across every predicted step; no particle-state resets within the rollout",
        ),
        loss_definition=dict(
            mode=args.loss_mode,
            moving_threshold_m=0.0005,
            reference="displacement from sequence start at each target step",
            balanced="equal mean squared position error for each nonempty motion group",
        ),
        reference_revision="mnabian/physicsnemo@fa1d68a67cc77baa176db8b4c1c5ef2ecdce00f7",
        caveat=(
            "Single dynamic rigid tool and fixed ground; arbitrary collider count is not yet validated on robot data"
        ),
    )
    if rank == 0:
        (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    if rank == 0:
        print(json.dumps(dict(training_recurrence=run["recurrence"])), flush=True)
    started, best = time.time(), float("inf")
    for step in range(1, args.steps + 1):
        model.train()
        sample = train.sequence(rng, args.rollout_steps, device)
        optimizer.zero_grad(set_to_none=True)
        predicted = rollout(forward, sample, config)
        loss, group_means, group_counts = position_loss(
            predicted, sample["targets"], sample["position_history"][:, -1], config["dt"], args.loss_mode
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / args.steps)))
        optimizer.step()
        if rank == 0 and (step == 1 or step % 100 == 0):
            print(
                json.dumps(
                    dict(
                        step=step,
                        loss=float(loss),
                        elapsed_s=time.time() - started,
                        moving_position_rmse_m=float(group_means[0].sqrt()) * config["dt"],
                        stationary_position_rmse_m=float(group_means[1].sqrt()) * config["dt"],
                        moving_count=int(group_counts[0]),
                        stationary_count=int(group_counts[1]),
                        peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                    )
                ),
                flush=True,
            )
        if step % args.evaluate_every == 0 or step == args.steps:
            if world > 1:
                dist.barrier()
            if rank == 0:
                metrics = validate(model, validation, config, args.validation_samples, args.rollout_steps, device)
                row = dict(
                    step=step,
                    elapsed_s=time.time() - started,
                    peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                    **metrics,
                )
                score = (
                    metrics["balanced_position_rmse_m"]
                    if args.loss_mode == "balanced"
                    else metrics["position_rmse_m"] + metrics["moving_position_rmse_m"]
                )
                row["selection_score_m"] = score
                saved = dict(
                    architecture="particle_transolver_v1",
                    config=config,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=step,
                    run=run,
                    metrics=metrics,
                )
                torch.save(saved, args.output / "last.pt")
                if score < best:
                    best = score
                    torch.save(saved, args.output / "best.pt")
                with (args.output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                (args.output / "progress.json").write_text(json.dumps(row, indent=2) + "\n")
                print(json.dumps(row), flush=True)
            if world > 1:
                dist.barrier()
    if rank == 0:
        (args.output / "COMPLETE.json").write_text(
            json.dumps(dict(elapsed_s=time.time() - started, **config), indent=2) + "\n"
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
