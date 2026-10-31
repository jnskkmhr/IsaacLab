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

from particle_nerd.g1.data import Episodes, collider_state, compact_step, infer_collider_slots, resolve_active_k, rotate
from particle_nerd.transolver.model import ParticleTransolver


def rollout(model, sample, dt, support=0.07, checkpoint_model=True, active_k=None, anchor_m=512):
    history, geometry = sample["history"], sample["geometry"]
    predicted = []
    for pose, velocity in zip(sample["poses"], sample["velocities"], strict=True):
        poses, velocities = collider_state(pose, velocity, geometry)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            displacement = compact_step(
                model,
                history,
                poses,
                velocities,
                geometry,
                sample["material"],
                dt,
                support=support,
                active_k=active_k,
                anchor_m=anchor_m,
                checkpoint_model=checkpoint_model,
            )
        x = history[:, -1] + displacement
        predicted.append(x)
        history = torch.cat((history[:, 1:], x[:, None]), 1)
    return torch.stack(predicted)


def balanced_position_errors(prediction, sample, dt):
    squared = (prediction - sample["targets"]).square().sum(-1)
    frozen = (sample["history"][:, -1][None] - sample["targets"]).square().sum(-1)
    moving = frozen > (0.025 * dt) ** 2
    counts = torch.stack((moving.sum(), (~moving).sum()))
    sums = torch.stack(((squared * moving).sum(), (squared * ~moving).sum()))
    means = sums / counts.clamp_min(1)
    loss = (means / (dt * dt)).sum() / (counts > 0).sum().clamp_min(1)
    return loss, sums, counts, frozen.sum()


def contact_squared_errors(prediction, sample, contact_band=0.02):
    """Reference-defined particles within the configured foot-surface distance.

    Geometry masks are detached reference quantities. No reference data is used
    at inference, and the particle architecture and active selection are unchanged.
    """

    geometry = sample["geometry"]
    slots, _ = infer_collider_slots(geometry["sizes"], geometry.get("foot_slots"))
    total = prediction.new_zeros(())
    count = prediction.new_zeros(())
    for t, (pose, velocity) in enumerate(zip(sample["poses"], sample["velocities"], strict=True)):
        with torch.no_grad():
            cp, _ = collider_state(pose, velocity, geometry)
            q = cp[slots, 3:].clone()
            q[:, :3] *= -1
            offset = sample["targets"][t, :, None] - cp[None, slots, :3]
            local = rotate(q[None].expand(len(offset), -1, -1), offset)
            distance = local.abs() - geometry["sizes"][None, slots]
            sdf = distance.clamp_min(0).norm(dim=-1) + distance.amax(dim=-1).clamp_max(0)
            mask = sdf.abs().amin(dim=-1) <= contact_band
        total = total + ((prediction[t] - sample["targets"][t]).square().sum(-1) * mask).sum()
        count = count + mask.sum()
    return total, count


def errors(prediction, sample, dt, contact_weight=0.0, contact_band=0.02):
    loss, sums, counts, frozen = balanced_position_errors(prediction, sample, dt)
    if contact_weight > 0.0:
        total, count = contact_squared_errors(prediction, sample, contact_band)
        loss = (1.0 - contact_weight) * loss + contact_weight * total / count.clamp_min(1) / dt**2
    return loss, sums, counts, frozen


@torch.no_grad()
def validate(model, data, steps, samples, device, active_k=None, anchor_m=512, contact_weight=0.0, contact_band=0.02):
    model.eval()
    rng = np.random.default_rng(42027)
    sums = np.zeros(2)
    counts = np.zeros(2)
    frozen = 0.0
    contact_total = 0.0
    contact_count = 0.0
    eligible = [r for r in data.records if r["frames"] > data.first_frame + steps * data.stride]
    if not eligible:
        raise ValueError("No recorded trajectories can supply the requested validation horizon")
    evaluated = min(samples, len(eligible))
    for i in range(evaluated):
        s = data.sequence(rng, steps, device, episode=i)
        pred = rollout(model, s, data.dt, active_k=active_k, anchor_m=anchor_m)
        _, a, b, c = errors(pred, s, data.dt)
        sums += a.cpu().numpy()
        counts += b.cpu().numpy()
        frozen += float(c)
        contact_s, contact_n = contact_squared_errors(pred, s, contact_band)
        contact_total += float(contact_s)
        contact_count += float(contact_n)
    balanced = (sums / np.maximum(counts, 1)).sum() / max(1, (counts > 0).sum())
    contact_mse = contact_total / max(1, contact_count)
    return dict(
        contact_region_rmse_m=float(np.sqrt(contact_mse)),
        selection_score_rmse_m=float(np.sqrt((1.0 - contact_weight) * balanced + contact_weight * contact_mse)),
        position_rmse_m=float(np.sqrt(sums.sum() / max(1, counts.sum()))),
        moving_rmse_m=float(np.sqrt(sums[0] / max(1, counts[0]))),
        stationary_rmse_m=float(np.sqrt(sums[1] / max(1, counts[1]))),
        balanced_rmse_m=float(np.sqrt((sums / np.maximum(counts, 1)).sum() / max(1, (counts > 0).sum()))),
        frozen_rmse_m=float(np.sqrt(frozen / max(1, counts.sum()))),
        particle_time_counts=counts.tolist(),
        eligible_trajectory_count=len(eligible),
        evaluated_trajectory_count=evaluated,
        prediction_steps=steps,
        horizon_s=steps * data.dt,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10000)
    horizon = p.add_mutually_exclusive_group()
    horizon.add_argument("--rollout-seconds", type=float, help="Recurrent horizon in seconds (default: 0.5)")
    horizon.add_argument("--rollout-steps", type=int, help="Explicit legacy step-count override")
    p.add_argument("--checkpoint-model", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--stride", type=int, default=1, help="1=200 Hz, 2=100 Hz, 4=50 Hz")
    p.add_argument("--evaluate-every", type=int, default=250)
    p.add_argument("--validation-samples", type=int, default=24)
    p.add_argument("--initialize", type=Path)
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument(
        "--active-k",
        type=str,
        default=None,
        help="Active particle count ('auto', 'none', or integer K) with confining cup boundary",
    )
    p.add_argument("--anchor-m", type=int, default=512, help="Number of strided background anchors M")
    p.add_argument(
        "--settled-frame",
        type=int,
        default=150,
        help="Skip initial MPM gravity settling frames before this physics frame (default: 150 = 0.75 s)",
    )
    p.add_argument(
        "--contact-loss-weight",
        type=float,
        default=0.5,
        help="Fraction of loss assigned to near-foot position error (0 preserves the original loss)",
    )
    p.add_argument(
        "--contact-band",
        type=float,
        default=0.02,
        help="Reference-particle distance from the foot box surface in metres; G1 collision margin is 0.02 m",
    )
    p.add_argument(
        "--pushforward-steps",
        type=int,
        default=0,
        help="Max no-grad unroll steps before the 100-step differentiable BPTT window to eliminate long-rollout drift",
    )
    args = p.parse_args()
    if not np.isfinite(args.contact_loss_weight) or not 0.0 <= args.contact_loss_weight <= 1.0:
        p.error("--contact-loss-weight must be finite and in [0, 1]")
    if not np.isfinite(args.contact_band) or args.contact_band <= 0.0:
        p.error("--contact-band must be positive and finite")
    if args.stride < 1:
        p.error("--stride must be positive")
    dt = 0.005 * args.stride
    if args.rollout_steps is None:
        seconds = 0.5 if args.rollout_seconds is None else args.rollout_seconds
        if not np.isfinite(seconds) or seconds <= 0:
            p.error("--rollout-seconds must be positive and finite")
        args.rollout_steps = round(seconds / dt)
        if not np.isclose(args.rollout_steps * dt, seconds, atol=1e-9, rtol=0):
            p.error("Training horizon must be an integer multiple of the prediction timestep")
    args.rollout_seconds = args.rollout_steps * dt
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("--learning-rate must be positive and finite")
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
    train = Episodes(args.data, "train", stride=args.stride, settled_frame=args.settled_frame)
    valid = Episodes(args.data, "validation", stride=args.stride, settled_frame=args.settled_frame)
    assert not ({r["index"] for r in train.records} & {r["index"] for r in valid.records})
    model = ParticleTransolver().to(device)
    if args.initialize:
        saved = torch.load(args.initialize, map_location=device, weights_only=False)
        if saved["architecture"] != "particle_transolver_v1" or saved["config"]["model"] != model.config:
            raise ValueError("Initialization checkpoint architecture mismatch")
        model.load_state_dict(saved["model"])
    forward = DistributedDataParallel(model, device_ids=[local], static_graph=True) if world > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    args.output.mkdir(parents=True, exist_ok=True)
    probe_seq = train.sequence(np.random.default_rng(args.seed), 1, device, episode=0)
    resolved_k = resolve_active_k(args.active_k, len(probe_seq["history"]), probe_seq["material"])
    del probe_seq
    config = dict(
        contact_loss=dict(
            weight=args.contact_loss_weight,
            reference_band_m=args.contact_band,
            selection="weighted balanced and contact-region MSE",
        ),
        dt=train.dt,
        model=model.config,
        features=dict(support=0.07),
        geometry="g1_collision_boxes",
        schema="g1_particle_nerd_v1",
        stride=args.stride,
        training_horizon_s=args.rollout_seconds,
        training_rollout_steps=args.rollout_steps,
        activation_checkpointing=args.checkpoint_model,
        active_k=resolved_k,
        anchor_m=args.anchor_m if resolved_k is not None else 0,
        confining_boundary=bool(resolved_k is not None),
        settled_frame=args.settled_frame,
        settled_time_s=args.settled_frame * 0.005,
        recurrence=(
            "Full backpropagation; predicted particle history; recorded current robot state; no resets or detachment"
        ),
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
                    selection=(
                        f"active_k={resolved_k} + anchor_m={args.anchor_m} with confining cup boundary"
                        if resolved_k is not None
                        else "all particles"
                    ),
                    supervision=(
                        "Balanced moving/stationary position loss with optional near-contact position loss; "
                        "reference foot motion"
                    ),
                ),
                indent=2,
            )
            + "\n"
        )
    if rank == 0:
        print(
            "TRAINING_HORIZON",
            json.dumps(
                dict(
                    dt_s=train.dt,
                    steps=args.rollout_steps,
                    horizon_s=args.rollout_seconds,
                    activation_checkpointing=args.checkpoint_model,
                    active_k=resolved_k,
                    anchor_m=args.anchor_m if resolved_k is not None else 0,
                )
            ),
            flush=True,
        )
    started = time.time()
    best = float("inf")
    for step in range(1, args.steps + 1):
        model.train()
        pf_max = min(int(args.pushforward_steps), int(args.pushforward_steps * step / max(1, args.steps // 2)))
        pf = int(rng.integers(0, pf_max + 1)) if pf_max > 0 else 0
        sample = train.sequence(rng, args.rollout_steps + pf, device)
        if pf > 0:
            with torch.no_grad():
                pf_sample = dict(
                    history=sample["history"],
                    poses=sample["poses"][:pf],
                    velocities=sample["velocities"][:pf],
                    geometry=sample["geometry"],
                    material=sample["material"],
                )
                pf_pred = rollout(
                    model,
                    pf_sample,
                    train.dt,
                    checkpoint_model=False,
                    active_k=resolved_k,
                    anchor_m=args.anchor_m,
                )
                h_len = sample["history"].shape[1]
                cat_hist = torch.cat((sample["history"], pf_pred.permute(1, 0, 2)), dim=1)
                sample = dict(
                    history=cat_hist[:, -h_len:].detach().contiguous(),
                    targets=sample["targets"][pf:],
                    poses=sample["poses"][pf:],
                    velocities=sample["velocities"][pf:],
                    geometry=sample["geometry"],
                    material=sample["material"],
                    trial_id=sample["trial_id"],
                    frame=sample["frame"] + pf,
                )
        optimizer.zero_grad(set_to_none=True)
        pred = rollout(
            forward,
            sample,
            train.dt,
            checkpoint_model=args.checkpoint_model,
            active_k=resolved_k,
            anchor_m=args.anchor_m,
        )
        loss, sums, counts, _ = errors(pred, sample, train.dt, args.contact_loss_weight, args.contact_band)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / args.steps)))
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
                metrics = validate(
                    model,
                    valid,
                    args.rollout_steps,
                    args.validation_samples,
                    device,
                    active_k=resolved_k,
                    anchor_m=args.anchor_m,
                    contact_weight=args.contact_loss_weight,
                    contact_band=args.contact_band,
                )
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
                if metrics["selection_score_rmse_m"] < best:
                    best = metrics["selection_score_rmse_m"]
                    torch.save(payload, args.output / "best.partial.pt")
                    os.replace(args.output / "best.partial.pt", args.output / "best.pt")
            if world > 1:
                dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
