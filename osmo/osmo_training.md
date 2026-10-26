# Moving-patch MPM training on OSMO

Run from the repository root. The generic submission helper
[`osmo/run_multinode.sh`](run_multinode.sh) syncs the current
checkout before running [`osmo/osmo_multi_gpu.yaml`](osmo_multi_gpu.yaml).
The workflow accepts normal IsaacLab training options through `args`, like
`docker/cluster/osmo_multi_gpu_workflow.yaml`.

The default image matches that standard workflow: `nvcr.io/nvidia/isaac-lab:3.0.0-rc1`.
It includes Git and curl. The launcher installs uv into `/tmp/uv` if needed, so
setup works as the image's non-root user without installing system packages.
Custom images must also provide Git and either uv or curl.

```bash
# finetune pretrained policy with MPM
# export osmo_node="isaac-lab-l40-06"
# export platform="ovx-l40"
export osmo_node="isaac-lab-l40s-03"
export platform="ovx-l40s"
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool "$osmo_node" \
  --set num_nodes=40 num_gpu=1 \
  --set-string \
    platform="$platform" \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/mpm-moving-patch/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-MixedTerrain --num_envs 64 --wandb_run tyiby5k6 --logger wandb --viz newton_gl --video --wandb_project g1_29dof_soft_flat_ppo"

# train from scratch 
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40-07 \
  --set num_nodes=40 num_gpu=1 \
  --set-string \
    platform=ovx-l40 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/mpm-moving-patch/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-MixedTerrain --num_envs 64 --logger wandb --viz newton_gl --video"

bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40-06 \
  --set num_nodes=40 num_gpu=1 \
  --set-string \
    platform=ovx-l40 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/mpm-moving-patch/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-MixedTerrain --num_envs 64 --logger wandb --viz newton_gl --video"

bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40s-03 \
  --set num_nodes=40 num_gpu=1 \
  --set-string \
    platform=ovx-l40s \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/mpm-moving-patch/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-MixedTerrain --num_envs 64 --logger wandb --viz newton_gl --video"


# resume training 
export osmo_node="isaac-lab-l40s-03"
export platform="ovx-l40s"
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool "$osmo_node" \
  --set num_nodes=40 num_gpu=1 \
  --set-string \
    platform="$platform" \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/mpm-moving-patch/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-MixedTerrain --num_envs 64 --logger wandb --viz newton_gl --video --wandb_run gzi8enaw"
```

```bash
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40-07 \
  --set num_nodes=8 num_gpu=1 \
  --set-string \
    platform=ovx-l40 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/g1-soft/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Flat-G1-29dof-Soft --num_envs 512 --logger wandb --viz newton_gl --video"
```


```bash 
export osmo_node="isaac-lab-l40-07"
export platform="ovx-l40"
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool "$osmo_node" \
  --set num_nodes=40 num_gpu=1 \
  --set-string \
    platform="$platform" \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/mpm-moving-patch/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-MixedTerrain \
--num_envs 64 \
--wandb_run tyiby5k6 \
--wandb_project g1_29dof_soft_flat_ppo \
--logger wandb \
--log_project_name g1_29dof_mpm_mixed_terrain \
--run_name soft_contact_finetune_bias5 \
--viz newton_gl --video \
agent.load_checkpoint=model_19999.pt \
agent.algorithm.learning_rate=0.0001 \
agent.env_interface.restore_terrain=false \
env.rewards.ang_vel_xy_l2.weight=-0.05 \
env.rewards.base_height.params.target_height=0.75 \
env.rewards.feet_air_time.weight=2.0 \
env.rewards.feet_trailing_distance.weight=0.0 \
env.curriculum.gravity_roll_bias.params.modify_params.start_step=480000 \
env.curriculum.gravity_roll_bias.params.modify_params.end_step=600000 \
env.curriculum.gravity_pitch_bias.params.modify_params.start_step=480000 \
env.curriculum.gravity_pitch_bias.params.modify_params.end_step=600000"
```

Append `--dry-run` to render the workflow without submitting or syncing files.
Change the task, checkpoint, logger, video, or other training settings in `args`.
The example allocates four training tasks with one GPU each and 64 environments
per GPU. Physical placement is controlled by the OSMO pool.

## Source syncing

The official OSMO template runs the code packaged in its Docker image; it does
not copy local edits. This workflow adds submission-time syncing:

1. The helper snapshots the checkout, including local edits and new files.
   It applies `.dockerignore` and `.gitignore` exclusion filters and excludes root
   `data/` and `.env*` files. Logs, virtual environments, and credentials stay local.
2. The helper separates `source/isaaclab_assets/data` and hashes its contents. Assets
   are uploaded directly with `osmo data upload` to `<output_url>/assets/<hash>/`
   only when that version is missing. A completion marker prevents reuse of partial
   uploads. Changing assets creates a new version; code edits reuse the existing one.
3. The helper uploads the code snapshot directly with `osmo data upload` to a
   unique `<output_url>/snapshots/<submission-id>/source/` directory. It submits
   the workflow only after both uploads succeed; no staging task is needed.
4. OSMO provides every trainer with the same code snapshot and cached assets as
   separate inputs. Assets are restored at their original repository path.
5. Each training task installs the locked uv environment and invokes
   `uv run isaaclab train_multigpu` with the supplied `args`.

The submission command is unchanged. The first submission uploads the assets;
subsequent submissions skip that upload until asset contents change. Trainers still
download assets from storage for each run. With the current checkout, the code
upload is about 47 MB instead of 533 MB including assets. `--dry-run` computes the
asset URL and renders the workflow without uploading or submitting anything.

There is no tar archive or fixed source URL to update. Submit again to include later
edits; a running job retains its submitted snapshot. The helper uses ordinary rsync
only for preparing the local snapshot; remote transfers use `osmo data upload`.
This avoids the OSMO 6.3.1 rsync client bug that can report success without uploading
files. Submission output is saved under `logs/osmo/submit-*.log`.

## Checkpoints, videos, and outputs

Checkpoint loading uses the normal training entrypoint's `--wandb_run` or
`--checkpoint` support. No custom downloader or distributed Python launcher is
needed. RSL-RL handles W&B metrics, model checkpoints, and recorded-video uploads
on global rank zero; the normal `--video` behavior applies on every rank.
TorchRun provides rendezvous and waits for the other training agents on exit.

Each task writes training logs into its OSMO output directory. Artifacts are
stored under `<output_url>/<workflow-id>/rank-<node-rank>/`; rank zero's logs are
also checkpointed every ten minutes. Source is retained under
`<output_url>/snapshots/<submission-id>/source/`, with its URL in the submitted workflow inputs.

The container, platform, resource counts, uv extras, RL library, credential
mapping, and output storage are workflow parameters. No task name, W&B run, or
user-specific storage path is embedded in the launch helper or workflow.
