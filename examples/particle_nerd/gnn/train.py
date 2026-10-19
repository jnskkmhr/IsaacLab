# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the 50 Hz all-particle GNN, with episode-held-out validation and checkpoints."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from particle_nerd.gnn.data import Episodes
from particle_nerd.gnn.features import Features
from particle_nerd.gnn.model import ParticleGNN, integrate


@torch.no_grad()
def evaluate(model, episodes, features, samples=24):
    model.eval()
    rng = np.random.default_rng(914)
    errors = []
    samples = max(samples, len(episodes.records))
    for index in range(samples):
        # Cover every eligible episode at least once, with fixed sampled times
        # across checkpoints. Extra samples cycle through the same episode list.
        s = episodes.sample(rng, device=features.device, episode=index % len(episodes.records))
        inputs = features.prepare(
            s["x"],
            s["history"],
            s["pose"],
            s["body_velocity"],
            s["material"],
            s["mass"],
            compact=True,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            correction = model(*inputs)
        px, pv = integrate(s["x"], s["history"], correction, model.config["dt"])
        near = inputs[3][:, 1, 0] > 0.0
        moving = torch.linalg.vector_norm(s["next_v"], dim=-1) > 0.05
        xerror = (px - s["next_x"]).square().sum(-1)
        verror = (pv - s["next_v"]).square().sum(-1)
        baseline = (s["x"] + 0.02 * s["history"][:, -3:] - s["next_x"]).square().sum(-1)
        errors.append(
            [
                xerror.mean().item(),
                verror.mean().item(),
                baseline.mean().item(),
                xerror[near].mean().item() if near.any() else np.nan,
                xerror[moving].mean().item() if moving.any() else np.nan,
                baseline[near].mean().item() if near.any() else np.nan,
                baseline[moving].mean().item() if moving.any() else np.nan,
            ]
        )
    array = np.asarray(errors)
    mean = np.array([column[np.isfinite(column)].mean() if np.isfinite(column).any() else np.nan for column in array.T])
    model.train()
    return dict(
        position_rmse_m=float(np.sqrt(mean[0])),
        velocity_rmse_m_s=float(np.sqrt(mean[1])),
        constant_velocity_position_rmse_m=float(np.sqrt(mean[2])),
        near_box_position_rmse_m=float(np.sqrt(mean[3])) if np.isfinite(mean[3]) else None,
        moving_particle_position_rmse_m=float(np.sqrt(mean[4])) if np.isfinite(mean[4]) else None,
        constant_velocity_near_box_position_rmse_m=float(np.sqrt(mean[5])) if np.isfinite(mean[5]) else None,
        constant_velocity_moving_particle_position_rmse_m=float(np.sqrt(mean[6])) if np.isfinite(mean[6]) else None,
        transitions=samples,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--evaluate-every", type=int, default=500)
    p.add_argument("--validation-samples", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--velocity-noise", type=float, default=0.0005)
    p.add_argument("--seed", type=int, default=7103)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    if world > 1:
        dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed + rank * 100003)
    train = Episodes(args.data, "train", args.history)
    validation = Episodes(args.data, "validation", args.history)
    assert not ({r["index"] for r in train.records} & {r["index"] for r in validation.records})
    model = ParticleGNN(hidden=args.hidden, passes=args.passes, history=args.history).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    start_step = 0
    best = float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = saved["step"]
        best = saved["best"]
    forward = DistributedDataParallel(model, device_ids=[local]) if world > 1 else model
    features = Features(16384, history=args.history, device=device)
    args.output.mkdir(parents=True, exist_ok=True)
    run = dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        config=model.config,
        train_episodes=len(train.records),
        validation_episodes=len(validation.records),
        validation_sampling=(
            "Every eligible validation episode at least once, using fixed seeded frame selection across checkpoints"
        ),
        world_size=world,
        model_input=(
            "current particle position, three velocity states, current box "
            "pose/velocity, material parameters, ground geometry"
        ),
        model_output="next particle position and endpoint velocity at 50 Hz",
        split="episode hash; test episodes are never used for training or checkpoint selection",
        particle_count=16384,
        grid_spacing_m=0.04,
    )
    if rank == 0:
        (args.output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    start = time.time()
    losses = []
    for step in range(start_step + 1, args.steps + 1):
        s = train.sample(rng, device)
        # Velocity-history noise; correct towards the clean
        # reference next state, including the noisy baseline displacement.
        history = s["history"] + torch.randn_like(s["history"]) * args.velocity_noise
        inputs = features.prepare(
            s["x"],
            history,
            s["pose"],
            s["body_velocity"],
            s["material"],
            s["mass"],
            compact=True,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            correction = forward(*inputs)
        px, pv = integrate(s["x"], history, correction, 0.02)
        near = inputs[3][:, 1, 0].detach()
        moving = (torch.linalg.vector_norm(s["next_v"], dim=-1) > 0.05).float()
        weight = 1.0 + 3.0 * near + moving
        position = ((px - s["next_x"]) / 0.02).square().sum(-1)
        velocity = (pv - s["next_v"]).square().sum(-1)
        # Every particle contributes; emphasize particles affected by the box.
        loss = ((position + 0.5 * velocity) * weight).sum() / weight.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = args.learning_rate * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        losses.append(float(loss.detach()))
        if rank == 0 and step % 25 == 0:
            print(
                json.dumps(
                    dict(
                        event="training",
                        step=step,
                        loss=float(np.mean(losses[-25:])),
                        elapsed_s=time.time() - start,
                    )
                ),
                flush=True,
            )
        if step % args.evaluate_every == 0 or step == args.steps:
            if world > 1:
                dist.barrier()
            if rank == 0:
                metrics = evaluate(model, validation, features, args.validation_samples)
                row = dict(
                    step=step,
                    train_loss=float(np.mean(losses[-args.evaluate_every :])),
                    elapsed_s=time.time() - start,
                    **metrics,
                )
                # Equal weight to global, near-box and moving-particle errors
                # prevents quiet particles from dominating checkpoint selection.
                selection = [
                    metrics[k]
                    for k in (
                        "position_rmse_m",
                        "near_box_position_rmse_m",
                        "moving_particle_position_rmse_m",
                    )
                    if metrics[k] is not None
                ]
                score = float(np.mean(selection))
                row["checkpoint_selection_score_m"] = score
                with (args.output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                improved = score < best
                best = min(best, score)
                checkpoint = dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=step,
                    best=best,
                    config=model.config,
                    run=run,
                )
                torch.save(checkpoint, args.output / "last.partial.pt")
                os.replace(args.output / "last.partial.pt", args.output / "last.pt")
                if improved:
                    torch.save(checkpoint, args.output / "best.pt")
                (args.output / "progress.json").write_text(json.dumps(row, indent=2) + "\n")
                print(json.dumps(dict(event="validation", **row)), flush=True)
            if world > 1:
                dist.barrier()
    if rank == 0:
        (args.output / "COMPLETE.json").write_text(
            json.dumps(
                dict(
                    steps=args.steps,
                    best_validation_selection_score_m=best,
                    elapsed_s=time.time() - start,
                ),
                indent=2,
            )
            + "\n"
        )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
