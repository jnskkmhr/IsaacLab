# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the Particle NeRD network on the dataset that ``record.py`` wrote.

Every step rolls the network out over one window of a training episode, starting from the reference state and
continuing on its own predictions, and compares the result with MPM (``losses.world_particle_loss``). The window
grows in six equal stages of 8, 16, 32, 64, 100 and 100 steps, capped at ``--horizon``. The learning rate follows
a cosine from ``--lr`` to a tenth of it.

Resting episodes (an object resting on the bed, and the bare bed) are used in two ways. With ``--rest-fraction``
a share of the steps rolls one of them out over ``--rest-steps`` steps, so that the network sees the slow drift
it would build up at rest. With ``--rest-windows`` they are also drawn as ordinary windows.

Every ``--validate-every`` steps the network is rolled out over ``--validation-horizon`` steps on fixed windows of
the validation episodes, and saved as ``last.pt``, and as ``best.pt`` when the validation loss improved. These
files, ``config.json`` and ``metrics.jsonl`` go to ``logs/neural/<run name>/`` unless ``--output`` names a folder.

Run from the repository root, on one GPU:

    uv run python scripts/neural/train.py --dataset /data/particle_nerd --run-name my_run --rest-windows

On four GPUs of one node (each GPU draws its own windows and the gradients are averaged):

    uv run torchrun --standalone --nproc_per_node=4 scripts/neural/train.py \\
        --dataset /data/particle_nerd --run-name my_run --rest-windows
"""

import argparse
import hashlib
import inspect
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import warp as wp
from dataset import DEFAULT_CATEGORIES, TERRAINS, EpisodeWindows
from losses import balanced_loss, rest_unroll, world_particle_loss
from rollout import rollout

from isaaclab_tasks.contrib.particle_nerd.network import nerd_network
from isaaclab_tasks.contrib.particle_nerd.network.nerd_network import ParticleNeRDNetwork

HORIZON_STAGES = (8, 16, 32, 64, 100, 100)
"""Window length [steps] of the six equal stages of a training run."""
LOG_INTERVAL = 10
"""Training steps between two logged losses."""


@torch.no_grad()
def validate(model, dataset, device, horizon, count):
    """Roll the model out on fixed validation windows and average the errors per episode.

    About ``count`` episodes are chosen, spread over object shape and actuation mode. Each one contributes two
    windows of ``horizon`` steps: one from the first frame, and one that starts where the interaction starts.
    """
    model.eval()
    rng = np.random.default_rng(9042)
    metrics = []
    groups = {}
    for index, (record, _, _) in enumerate(dataset.records):
        groups.setdefault((record["shape"], record["mode"]), []).append(index)
    selected = []
    per_group = max(1, count // len(groups))
    for indices in groups.values():
        selected.extend(indices[i] for i in np.linspace(0, len(indices) - 1, min(per_group, len(indices))).astype(int))
    for index in selected:
        record = dataset.records[index][0]
        per_episode = []
        for start in (0, record["contact_start"]):
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


def source_hashes() -> dict:
    """SHA-256 of the files that define the network and of the scripts in this folder."""
    package = Path(inspect.getfile(nerd_network)).parents[1]
    network = [*sorted((package / "network").glob("*.py")), *sorted((package / "modules").glob("*.py"))]
    network.append(package / "nerd_kernels.py")
    scripts = sorted(Path(__file__).parent.glob("*.py"))
    return {
        "network_source_sha256": {
            str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest() for p in network
        },
        "script_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
    }


def main(argv: list[str] | None = None) -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Folder of the dataset that record.py wrote.")
    parser.add_argument("--run-name", default=None, help="Results go to logs/neural/<run name>/; a time by default.")
    parser.add_argument("--output", type=Path, default=None, help="Folder of the results, instead of the default.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint to continue, with its step count.")
    parser.add_argument("--finetune-from", type=Path, default=None, help="Checkpoint to start from, at step 1.")
    parser.add_argument("--steps", type=int, default=12000, help="Training steps.")
    parser.add_argument("--horizon", type=int, default=100, help="Longest training window; 100 steps are 0.5 s.")
    parser.add_argument("--validate-every", type=int, default=500, help="Steps between two validations.")
    parser.add_argument("--validation-count", type=int, default=24, help="Validation episodes, about.")
    parser.add_argument("--validation-horizon", type=int, default=200, help="Validation rollout; 200 steps are 1 s.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate at the start.")
    parser.add_argument("--far-support-scale", type=float, default=5.0, help="Reach of the long-range collider input.")
    parser.add_argument("--far-gap-weight", type=float, default=0.3, help="Weight of the long-range gap, in [0, 1].")
    parser.add_argument(
        "--far-influence-weight", type=float, default=0.15, help="Weight of the long-range influence, in [0, 1]."
    )
    parser.add_argument(
        "--rest-fraction", type=float, default=0.0, help="Share of the steps that roll a resting bed out for long."
    )
    parser.add_argument("--rest-steps", type=int, default=1000, help="Steps of one long resting rollout.")
    parser.add_argument("--rest-segment", type=int, default=100, help="Steps of one gradient segment of it.")
    parser.add_argument(
        "--rest-windows", action="store_true", help="Also draw the resting episodes as ordinary training windows."
    )
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES), help="Categories to train on.")
    parser.add_argument("--terrains", nargs="+", default=None, help="Terrains to train on; all by default.")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the weights and of the drawn windows.")
    args = parser.parse_args(argv)

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.output is None:
        run_name = datetime.now().strftime("particle_nerd_%Y%m%d_%H%M%S") if args.run_name is None else args.run_name
        args.output = Path("logs/neural") / run_name
    if not args.dataset.is_dir():
        parser.error(f"{args.dataset} is not a folder. Record the dataset first with scripts/neural/record.py")
    unknown = sorted(set(args.categories) - set(DEFAULT_CATEGORIES)) + sorted(set(args.terrains or ()) - set(TERRAINS))
    if unknown:
        parser.error(f"unknown category or terrain: {', '.join(unknown)}")
    if min(args.steps, args.horizon, args.validate_every, args.validation_count, args.validation_horizon) < 1:
        parser.error("--steps, --horizon, --validate-every, --validation-count and --validation-horizon start at 1")
    if not 0.0 <= args.rest_fraction <= 1.0:
        parser.error("--rest-fraction is a share of the steps, between 0 and 1")
    if args.rest_segment < 1 or args.rest_steps < args.rest_segment:
        parser.error("--rest-steps must be at least --rest-segment, and --rest-segment at least 1")
    for name in ("resume_from", "finetune_from"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            parser.error(f"--{name.replace('_', '-')} names {path}, which is not a file")
    if (args.output / "last.pt").exists():
        parser.error(f"{args.output} already holds a checkpoint. Name a new run with --run-name or --output")
    if not torch.cuda.is_available():
        parser.error("training needs a CUDA GPU: the network builds its inputs with Warp kernels on the GPU")

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    wp.init()
    wp.set_device(device)
    torch.set_num_threads(4)
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device(device), timeout=timedelta(minutes=30))
    try:
        train = EpisodeWindows(args.dataset, "train", categories=tuple(args.categories), terrains=args.terrains)
        validation = EpisodeWindows(
            args.dataset, "validation", categories=tuple(args.categories), terrains=args.terrains
        )
    except ValueError as error:
        parser.error(f"{error}. Record them with scripts/neural/record.py")
    model = ParticleNeRDNetwork(
        far_support_scale=args.far_support_scale,
        far_gap_weight=args.far_gap_weight,
        far_influence_weight=args.far_influence_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_step = 1
    best = float("inf")
    if args.finetune_from is not None:
        # Drops the weights of removed modules and fails on any other mismatch
        model.load_checkpoint_state(torch.load(args.finetune_from, map_location=device, weights_only=False)["model"])
        if rank == 0:
            print(f"Initialized from {args.finetune_from}", flush=True)
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        missing, _ = model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt and not missing:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0)) + 1
        best = float("inf") if missing else float(ckpt.get("validation", {}).get("validation_loss", float("inf")))
        if rank == 0:
            print(f"Resumed from {args.resume_from} at step {start_step - 1} (new keys={len(missing)})", flush=True)
    if world > 1:
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
    rng = np.random.default_rng(args.seed + rank + 1000 * (start_step - 1))

    def checkpoint(step: int, result: dict) -> dict:
        return {
            "architecture": nerd_network.ARCHITECTURE,
            "model": model.state_dict(),
            "model_config": model.config,
            "step": step,
            "optimizer": optimizer.state_dict(),
            "validation": result,
            "config": {
                "schema": "object_particle_nerd_v1",
                "architecture": nerd_network.ARCHITECTURE,
                "model": model.config,
                "dt": train.records[0][1]["dt_s"],
                "features": {"support": 0.07},
            },
        }

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "config.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "architecture": nerd_network.ARCHITECTURE,
                    "model": model.config,
                    "world_size": world,
                    "train_trajectories": len(train.records),
                    "validation_trajectories": len(validation.records),
                    **source_hashes(),
                    "horizon_curriculum": f"Six equal stages of {HORIZON_STAGES} steps, capped at --horizon",
                    "learning_rate_schedule": "Cosine from the initial learning rate to 10% of it",
                    "validation_windows": "Two per episode: from the first frame, and from the start of the contact",
                    "loss": "0.6 position + 0.25 velocity + 0.15 rest drift; moving and resting particles weigh equal",
                },
                indent=2,
                default=str,
            )
        )
        if args.resume_from is not None:
            # The run can be resumed again even if it stops before its first validation
            resumed = torch.load(args.resume_from, map_location="cpu", weights_only=False)
            resumed["model"] = {k: v.cpu() for k, v in model.state_dict().items()}
            resumed["model_config"] = model.config
            torch.save(resumed, args.output / "last.pt")
            torch.save(resumed, args.output / "best.pt")
    start_time = time.perf_counter()
    log = (args.output / "metrics.jsonl").open("a") if rank == 0 else None
    try:
        # Resting episodes are only seen in the long rollouts unless --rest-windows also draws them as windows.
        window_indices = [i for i, (r, _, _) in enumerate(train.records) if args.rest_windows or not r["rest_unroll"]]
        rest_indices = [i for i, (r, _, _) in enumerate(train.records) if r["rest_unroll"]]
        if rank == 0:
            resting = len(set(window_indices) & set(rest_indices))
            print(
                f"Window episodes: {len(window_indices)} ({resting} resting), long-rollout pool: {len(rest_indices)}",
                flush=True,
            )
        for step in range(start_step, args.steps + 1):
            stage = min(5, 6 * (step - 1) // args.steps)
            horizon = min(args.horizon, HORIZON_STAGES[stage])
            learning_rate = args.lr * (0.1 + 0.9 * (1 + np.cos(np.pi * (step - 1) / args.steps)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            if rest_indices and rng.random() < args.rest_fraction:
                loss = rest_unroll(model, train, rest_indices, rng, device, args.rest_steps, args.rest_segment)
            else:
                index = int(rng.choice(window_indices)) if window_indices else None
                sample = train.sequence(rng, horizon, device, index=index)
                pred, pred_vel = rollout(model, sample, return_vel=True)
                loss = world_particle_loss(pred, pred_vel, sample["targets"], sample["history"][:, -1], sample["dt"])
                loss.backward()
            if world > 1:
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
            if step % LOG_INTERVAL == 0 and rank == 0:
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
                    # Rank zero validates while the other ranks wait.
                    result = validate(model, validation, device, args.validation_horizon, args.validation_count)
                    result.update(
                        step=step,
                        elapsed_s=time.perf_counter() - start_time,
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    )
                    print(json.dumps(result), flush=True)
                    log.write(json.dumps(result) + "\n")
                    log.flush()
                    saved = checkpoint(step, result)
                    torch.save(saved, args.output / "last.pt")
                    if result["validation_loss"] < best:
                        best = result["validation_loss"]
                        torch.save(saved, args.output / "best.pt")
                if world > 1:
                    dist.barrier()
        if rank == 0:
            print(f"The checkpoints and metrics are in {args.output.resolve()}.", flush=True)
    finally:
        if log:
            log.close()
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
