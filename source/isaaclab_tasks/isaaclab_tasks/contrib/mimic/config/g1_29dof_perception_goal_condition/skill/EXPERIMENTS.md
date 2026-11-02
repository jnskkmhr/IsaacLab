# Experiment record and operations

Status snapshots below were checked on 2026-11-02 UTC. Re-query services before
claiming a run is still active. W&B links use the user's Forge frontend; the
configured W&B API was used to inspect metadata. No credentials belong here.

## Current run: task-episode style reward

- OSMO: [g1-mimic-stairs-task-style50-1](https://us-west-2-aws.osmo.nvidia.com/workflows/g1-mimic-stairs-task-style50-1)
- W&B: [3o1kz32r](https://forge.coreweave.com/wandb/jnskkmr/g1_mimic_perception_goal_condition/runs/3o1kz32r)
- W&B name: `2026-11-02_03-39-21_stairs_task_style50`.
- At 03:45 UTC, OSMO was RUNNING and W&B reported iteration 5125. The saved W&B
  configuration explicitly contains `TaskMotionStyle` at weight 2.0 with the
  documented four phase rates. This verifies PPO has started, not learning quality.
- Pool: `isaac-lab-l40s-03`, platform `ovx-l40s`.
- Four OSMO training workers, one GPU and 1,024 environments per worker: 4,096 total.
  Requested worker resources: 8 CPUs, 64 GiB memory, 88 GiB storage.
- Start checkpoint: `r7x8av8e/model_5000.pt`, rather than the canceled standing run's
  latest checkpoint. This keeps the original starting checkpoint for comparison.
- Requested 25,000 additional iterations; final imitation probability 0.5.
- Current four standing-tail reference clips and success-based reset curriculum.
- Code snapshot: `snapshots/20261102T033657Z-ayPrRj7asS/source/` under the output root.
- Image pinned to `nvcr.io/nvidia/isaac-lab:3.0.0-rc1@sha256:87c58ac2a48714048bf2fdf540a8ed62a569dd405ba1ee0689d8a2d21b16d44c`.
- Submission log: `logs/osmo/submit-20261102T033657Z-Q7s2.log` (local, not committed).

Output root:
`swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/g1-perception/runs`.
Outputs use `<root>/<workflow-id>/rank-<rank>/logs`; rank-zero logs are periodically
checkpointed to storage. Source snapshots are immutable. Editing local rewards
cannot change an already running job.

The user explicitly requested stopping the prior standing-tail run and starting
this replacement. Automatic approval initially blocked cancellation because the
OSMO skill required separate confirmation. The user then explicitly confirmed,
and cancellation succeeded before the replacement was submitted. Do not report
that the old run remains active or that the new job is only a local smoke check.

## Previous runs and decisions

| Run/workflow | What is established |
| --- | --- |
| `89ljv3g6`, checkpoint 22500 | Earlier 50% imitation checkpoint referenced by the mixed fine-tune example; user liked its inference appearance |
| 20% imitation comparison | Compared using deterministic videos; user preferred 50% and later requested canceling 20%. Exact run ID is not retained in this handoff; do not guess it |
| `r7x8av8e` | Earlier broad-randomization experiment stalled near zero task success; checkpoint 5000 is the common starting point for later curriculum/standing/style experiments |
| `g1-mimic-stairs-reset-curriculum50-3` / `l6ddoj98` | Completed through iteration 29999 with the original 174-frame pyramid clips; sampled final 2k iterations had about 83.4% task success and reset difficulty about 0.995; a single latest sample was 73.6% |
| `g1-mimic-stairs-standing-tail50-1` | Failed startup: rank 1 absent and only 3/4 ranks rendezvoused before a 901 s timeout. No PPO/W&B training established |
| `g1-mimic-stairs-standing-tail50-2` / `n5dsar3b` | Retry using the extended standing reference, 50% imitation, success-based resets. No task style reward. Explicitly canceled for replacement; OSMO verified FAILED_CANCELED |
| `g1-mimic-stairs-task-style50-1` / `3o1kz32r` | Current replacement; new local style reward, remaining standing-run settings preserved |

[n5dsar3b](https://forge.coreweave.com/wandb/jnskkmr/g1_mimic_perception_goal_condition/runs/n5dsar3b)
was launched by this assistant earlier, not by an unknown collaborator. Its W&B
host matched OSMO trainer-0 UUID `d789292af7f04f54b91892aed13966f6`.
Its name was `2026-11-02_01-19-19_stairs_standing_tail_reset_curriculum50`.
The confusion came from saying no *new style training* had been submitted while
this earlier standing-tail run was still running.

Earlier reset-curriculum retries included a CUDA failure followed by a successful
completed run. That demonstrates the error was not reproduced in the successful
retry; it does not prove the cluster was the root cause. Diagnose actual logs
before repeating failed jobs.

Historical decisions:

1. Task-only training at the end lost visible motion style. Restore 50% imitation;
   20% was an exploratory comparison, not the selected default.
2. Fine-tuning on generated stairs appeared more aggressive. Match inference
   initial conditions before attributing this to terrain geometry or critic input.
3. Remove direct `goal_progress` incentive; keep position/heading/settling rewards.
4. Expand start pose/velocity and goal variation. A fixed broad expansion was
   difficult; switch to success-based per-environment reset difficulty.
5. Append a standing finish to the stair reference to supervise post-arrival pose.
6. Controlled evaluation still found style degradation under full task starts/goals.
   Add explicit local style reward during task episodes; do not assume shared
   imitation automatically preserves style everywhere.

## Controlled style evaluation of l6ddoj98/model_29999.pt

These measurements predate the standing-tail/style-reward training changes.
They used the original pyramid reference, 64 environments per case, deterministic
mean actions, and controlled resets. Observation noise and startup physical
randomization were disabled. Goal/reset randomization was enabled only where
specified. Reference-specific terminations were removed for the comparison so
all cases shared a 10 s horizon and physical failure conditions.

Motion IDs cycled through all four cardinal approaches. The same actor observation
and state produced the same action when only imitation/task episode mode was
flipped. Long physical rollouts were nearly equal, not claimed bitwise identical:
maximum joint/root differences were about 0.164 rad / 0.03 m over 10 s.

| Condition | Reached goal | Falls within evaluation | Ascent time (s) | Leg velocity RMS (rad/s) | Time-aligned leg pose error (rad) |
| --- | --- | --- | --- | --- | --- |
| Imitation reference reset | 64/64 | 0 | 2.96 | 1.865 | 0.199 |
| Task mode, same reference reset and goal | 64/64 | 0 | 2.96 | 1.864 | 0.199 |
| Standing reset near reference, same goal | 64/64 | 64, after arrival | 2.77 | 2.17 | 0.274 |
| Standing reset 0.9 m farther back, same goal | 64/64 | 0 | 1.20 | 3.38 | 0.297 |
| Standing start and goal shifted forward 0.75 m | 64/64 | 0 | 0.96 | 4.52 | 0.319 |
| Full task start/goal randomization | 63/64 | 1 | 0.92-1.06 | 4.03-4.84 | 0.303-0.324 |

Ascent time measures root Z rising from initial reference root Z +0.1 m to final
reference root Z -0.1 m. It is not total episode or start-to-goal duration.
Leg pose error uses dynamic time warping of the 12 leg joint angles during ascent;
it is a local style proxy, not a formal human naturalness score. The randomized
row includes the observed variation reported by the diagnostic analysis.

Conclusion: switching the mode flag alone did not cause aggressive behavior.
Changing the start state/distance and goal changed the actor's behavior despite
identical physical stair geometry. All 63 completed fully randomized examples
climbed substantially faster than the reference; no style-preserving subgroup
was found in that sample. Goal-reaching success can coexist with poor style or
post-arrival collapse. Critic-only reference input does not explain an inference
action difference at an otherwise identical actor observation.

### Local evidence pointers

These were produced locally and are not required for running the task:

- `model29999_initialization_style_comparison.mp4`: four-panel comparison in the
  repository root; reference initialization, nominal task, and closest/farthest
  randomized examples (environment indices 55/58). This is a kinematic replay
  of recorded physical rollouts, not a second independently simulated policy run.
- `/tmp/style_eval/model_29999.pt`: evaluated checkpoint.
- `/tmp/style_eval/*.npz`: recorded trajectories; `style_metrics.json` and
  `summary.json`: measured results.
- `/tmp/evaluate_style_modes.py`, `/tmp/analyze_style_modes.py`,
  `/tmp/export_style_replays.py`: temporary evaluation/analysis/video scripts.
- `finetune_model31500_*stairs*.mp4` and corresponding JSON: earlier matched
  original/generated-stair comparisons. Do not treat unmatched videos as an ablation.
- `mimic_imitation50_model22500_4envs*.mp4`,
  `mimic_imitation20_model22500_4envs*.mp4`: original 50/20 comparison, including
  closer camera versions and combined videos.
- `kimodo_pyramid_stairs_reference.mp4` and
  `kimodo_pyramid_stairs_standing_reference.mp4`: kinematic dataset previews.
- `/tmp/extend_stairs_standing.py`: temporary standing-tail generation script;
  the durable outputs and source hashes are in the dataset/provenance files.
- `/tmp/check_task_style.py`: actual-motion reward checks.
- `/tmp/smoke_task_style.py`, `/tmp/smoke_task_style.log`: three-iteration PPO smoke.

Temporary paths may disappear. Preserve measurements here, and recreate diagnostics
from the checked-in implementation/data when needed. Videos and local checkpoints
were deliberately left out of the handoff commit.

## Reproduce the current submission

This submits a new workflow; do not execute it just to inspect an existing run.
The wrapper uploads the current checkout including uncommitted/untracked task
files selected by Git and deployment exclusions. It reuses assets by content hash.

```bash
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40s-03 \
  --set num_nodes=4 num_gpu=1 \
  --set-string \
    workflow_name=g1-mimic-stairs-task-style50 \
    platform=ovx-l40s \
    image=nvcr.io/nvidia/isaac-lab:3.0.0-rc1@sha256:87c58ac2a48714048bf2fdf540a8ed62a569dd405ba1ee0689d8a2d21b16d44c \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/g1-perception/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Mimic-G1-29dof-Perception-Goal-Condition --num_envs 1024 --logger wandb --viz newton_gl --video --checkpoint 'wandb:jnskkmr/g1_mimic_perception_goal_condition/r7x8av8e?checkpoint=5000' --max_iterations 25000 --run_name stairs_task_style50 env.rewards.task_motion_style.weight=2.0"
```

Read-only status checks:

```bash
osmo workflow query g1-mimic-stairs-task-style50-1 --format-type json
osmo workflow logs g1-mimic-stairs-task-style50-1 --task trainer-0 -n 100
```

For future comparisons, record the actual motion paths, reward config, reset
curriculum, starting checkpoint/optimizer policy, and source snapshot from each
run. The current local defaults alone do not identify a historical run's settings.
