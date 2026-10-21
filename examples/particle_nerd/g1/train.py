# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train Transolver on G1 material motion with actual foot and boundary geometry."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import warp as wp
from torch.nn.parallel import DistributedDataParallel

from particle_nerd.g1.data import Episodes, collider_state
from particle_nerd.transolver.features import features
from particle_nerd.transolver.model import ParticleTransolver


def rollout(model, sample, dt, support=0.07):
    history, geometry = sample["history"], sample["geometry"]
    predicted = []
    for pose, velocity in zip(sample["poses"], sample["velocities"], strict=True):
        poses, velocities = collider_state(pose, velocity, geometry)
        inputs = features(
            history,
            poses,
            velocities,
            geometry["kinds"],
            geometry["sizes"],
            geometry["friction"],
            sample["material"],
            dt,
            support=support,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            displacement = model(*inputs)
        x = history[:, -1] + displacement
        predicted.append(x)
        history = torch.cat((history[:, 1:], x[:, None]), 1)
    return torch.stack(predicted)


def errors(prediction, sample, dt):
    squared = (prediction - sample["targets"]).square().sum(-1)
    frozen = (sample["history"][:, -1][None] - sample["targets"]).square().sum(-1)
    moving = frozen > (0.025 * dt) ** 2
    counts = torch.stack((moving.sum(), (~moving).sum()))
    sums = torch.stack(((squared * moving).sum(), (squared * ~moving).sum()))
    means = sums / counts.clamp_min(1)
    loss = (means / (dt * dt)).sum() / (counts > 0).sum().clamp_min(1)
    return loss, sums, counts, frozen.sum()


@torch.no_grad()
def validate(model, data, steps, samples, device):
    model.eval()
    rng = np.random.default_rng(42027)
    sums = np.zeros(2)
    counts = np.zeros(2)
    frozen = 0.0
    for i in range(min(samples, len(data.records))):
        s = data.sequence(rng, steps, device, episode=i)
        pred = rollout(model, s, data.dt)
        _, a, b, c = errors(pred, s, data.dt)
        sums += a.cpu().numpy()
        counts += b.cpu().numpy()
        frozen += float(c)
    return dict(
        position_rmse_m=float(np.sqrt(sums.sum() / max(1, counts.sum()))),
        moving_rmse_m=float(np.sqrt(sums[0] / max(1, counts[0]))),
        stationary_rmse_m=float(np.sqrt(sums[1] / max(1, counts[1]))),
        balanced_rmse_m=float(np.sqrt((sums / np.maximum(counts, 1)).sum() / max(1, (counts > 0).sum()))),
        frozen_rmse_m=float(np.sqrt(frozen / max(1, counts.sum()))),
        particle_time_counts=counts.tolist(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--rollout-steps", type=int, default=2)
    p.add_argument("--stride", type=int, default=1, help="1=200 Hz, 2=100 Hz, 4=50 Hz")
    p.add_argument("--evaluate-every", type=int, default=250)
    p.add_argument("--validation-samples", type=int, default=24)
    p.add_argument("--initialize", type=Path)
    p.add_argument("--seed", type=int, default=20261020)
    args = p.parse_args()
    if min(args.steps, args.rollout_steps, args.stride, args.evaluate_every, args.validation_samples) < 1:
        p.error("Counts must be positive")
    torch.set_num_threads(2)
    rank, local, world = [
        int(os.environ.get(k, 1 if k == "WORLD_SIZE" else 0)) for k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
    ]
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    wp.init()
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + rank * 100003)
    train = Episodes(args.data, "train", stride=args.stride)
    valid = Episodes(args.data, "validation", stride=args.stride)
    assert not ({r["index"] for r in train.records} & {r["index"] for r in valid.records})
    model = ParticleTransolver().to(device)
    if args.initialize:
        saved = torch.load(args.initialize, map_location=device, weights_only=False)
        if saved["architecture"] != "particle_transolver_v1" or saved["config"]["model"] != model.config:
            raise ValueError("Initialization checkpoint architecture mismatch")
        model.load_state_dict(saved["model"])
    forward = DistributedDataParallel(model, device_ids=[local], static_graph=True) if world > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(
        dt=train.dt,
        model=model.config,
        features=dict(support=0.07),
        geometry="g1_collision_boxes",
        schema="g1_particle_nerd_v1",
        stride=args.stride,
    )
    if rank == 0:
        (args.output / "run.json").write_text(
            json.dumps(
                dict(
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    config=config,
                    train_ids=[r["index"] for r in train.records],
                    validation_ids=[r["index"] for r in valid.records],
                    world_size=world,
                    selection="all particles",
                    supervision="position displacement; foot/body motion is reference during training",
                ),
                indent=2,
            )
            + "\n"
        )
    started = time.time()
    best = float("inf")
    for step in range(1, args.steps + 1):
        model.train()
        sample = train.sequence(rng, args.rollout_steps, device)
        optimizer.zero_grad(set_to_none=True)
        pred = rollout(forward, sample, train.dt)
        loss, sums, counts, _ = errors(pred, sample, train.dt)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        for group in optimizer.param_groups:
            group["lr"] = 3e-4 * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / args.steps)))
        optimizer.step()
        if rank == 0 and (step % 25 == 0 or step == 1):
            print(
                "TRAIN",
                json.dumps(
                    dict(
                        step=step,
                        loss=float(loss.detach()),
                        elapsed_s=time.time() - started,
                        peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                    )
                ),
                flush=True,
            )
        if step % args.evaluate_every == 0 or step == args.steps:
            if world > 1:
                dist.barrier()
            if rank == 0:
                metrics = validate(model, valid, args.rollout_steps, args.validation_samples, device)
                if not all(np.isfinite(metrics[k]) for k in metrics if k != "particle_time_counts"):
                    raise FloatingPointError(f"Nonfinite validation at {step}: {metrics}")
                record = dict(step=step, elapsed_s=time.time() - started, **metrics)
                print("VALIDATION", json.dumps(record), flush=True)
                with (args.output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                payload = dict(
                    architecture="particle_transolver_v1",
                    config=config,
                    model=model.state_dict(),
                    step=step,
                    validation=metrics,
                    optimizer=optimizer.state_dict(),
                )
                torch.save(payload, args.output / "last.partial.pt")
                os.replace(args.output / "last.partial.pt", args.output / "last.pt")
                if metrics["balanced_rmse_m"] < best:
                    best = metrics["balanced_rmse_m"]
                    torch.save(payload, args.output / "best.partial.pt")
                    os.replace(args.output / "best.partial.pt", args.output / "best.pt")
            if world > 1:
                dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
