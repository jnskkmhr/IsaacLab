# G1 perception training on OSMO

Run from the repository root. The helper uploads the current checkout, including
local edits and new task files, then launches `isaaclab train_multigpu` on each
OSMO training task. The locked environment includes `compositional_rsl_rl`.

## Train

```bash
# AME: four training tasks, one GPU each (1,024 environments total).
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40s-03 \
  --set num_nodes=4 num_gpu=1 \
  --set-string \
    platform=ovx-l40s \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/g1-perception/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Rough-G1-29dof-Perception-AME --num_envs 256 --logger wandb --viz newton_gl --video"

# CNN: same resources, height-only CNN encoder.
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40s-03 \
  --set num_nodes=8 num_gpu=1 \
  --set-string \
    platform=ovx-l40s \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/g1-perception/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Rough-G1-29dof-Perception-CNN --num_envs 512 --logger wandb --viz newton_gl --video --wandb_run 0o6giq1o"

# reach environment
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40-06 \
  --set num_nodes=10 num_gpu=1 \
  --set-string \
    platform=ovx-l40 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/g1-perception/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Rough-G1-29dof-Perception-CNN --num_envs 512 --logger wandb --viz newton_gl --video"

bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40-07 \
  --set num_nodes=16 num_gpu=1 \
  --set-string \
    platform=ovx-l40 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/g1-perception/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Rough-G1-29dof-Perception-Reach-CNN --num_envs 256 --logger wandb --viz newton_gl --video"

bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40s-03 \
  --set num_nodes=16 num_gpu=1 \
  --set-string \
    platform=ovx-l40s \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/g1-perception/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Rough-G1-29dof-Perception-Reach-CNN --num_envs 256 --logger wandb --viz newton_gl --video"


bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40s-03 \
  --set num_nodes=16 num_gpu=1 \
  --set-string \
    platform=ovx-l40s \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/g1-perception/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Mimic-G1-29dof-Perception --num_envs 256 --logger wandb --viz newton_gl --video"



bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40s-07 \
  --set num_nodes=4 num_gpu=1 \
  --set-string \
    platform=ovx-l40s \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/g1-perception/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Mimic-G1-29dof-Perception-Goal-Condition --num_envs 1024 --logger wandb --viz newton_gl --video"
```

Change the pool, platform, storage URI, and W&B credential to match your account.
`num_nodes` is the number of OSMO training tasks; `num_gpu` is the GPU count per
task. Total ranks are `num_nodes * num_gpu`, and `--num_envs` is **per GPU**.
For one task with four GPUs, use `--set num_nodes=1 num_gpu=4 num_cpu=32 memory=128`
on a pool/platform that supports that allocation. OSMO controls physical placement.

For goal reaching, replace `--task` with either:

- `IsaacContrib-Velocity-Rough-G1-29dof-Perception-Reach-AME`
- `IsaacContrib-Velocity-Rough-G1-29dof-Perception-Reach-CNN`

Append `--dry-run` to the submission command to render without uploading or
starting training. For a short training check, use two GPUs, `--num_envs 4`, and
`--max_iterations 2` in `args`. Add `--wandb_run RUN_ID` to resume a matching
checkpoint. Remove `--video` if recording is unnecessary.

## Source and outputs

The helper excludes logs, virtual environments, and credentials from the source
snapshot. Robot assets are uploaded separately and reused by content hash. If
Warp is pinned to Git, the helper also uploads the locally built Warp wheel.
Each trainer installs the snapshot's locked dependencies with the `rsl-rl`,
`wandb`, and `video` extras. No separate compositional-RSL-RL installation is needed.
The default image is `nvcr.io/nvidia/isaac-lab:3.0.0-rc1`; custom images must provide
Git and either uv or curl.

Logs and artifacts are stored under `<output_url>/<workflow-id>/rank-<node-rank>/`.
Rank zero's logs are checkpointed every ten minutes. Submission logs stay in
`logs/osmo/submit-*.log`. Submit again to include later code edits; running jobs
retain their submitted snapshot.
