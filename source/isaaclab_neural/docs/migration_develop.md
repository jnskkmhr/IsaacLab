# NeRD on the develop branch

The `source/isaaclab_neural` package was migrated from the sibling
IsaacLab-NeRD checkout based on release-3.0-beta2. Its dynamics models,
HDF5 data tools, contact encoders, registered NeRD environments, and RSL-RL
entry points are included. Existing checkpoint formats and NeRD task IDs
are preserved.

## Setup

From the repository root:

```bash
uv sync
uv run --extra test python -m pytest source/isaaclab_neural/test
```

The workspace installs the package automatically and uses this repository's
Newton version. Do not install the old package's `all` extra, which pinned
Newton 1.2.1. For W&B logging, use `uv run --with wandb ... --enable-wandb`
with dynamics training, or `--logger wandb` with RSL-RL.

## Command changes

- Use `uv run python -m isaaclab_neural.<module>` from the repository root.
- Use `--visualizer none` for runs without a viewer; `--headless` was removed.
- Built-in task names no longer have `-v0`, for example `Isaac-Cartpole`.
  Package-owned NeRD and Dataset-Gen task names still have `-v0`.
- The physical solver preset is `newton_mjwarp`, replacing `newton`.
  Dataset generation selects it automatically. NeRD tasks already select
  their own physics config; omit a physics preset for neural simulation.
- Launcher helpers now come from `isaaclab.app.sim_launcher`.
- Direct writes to Newton joint state must call `invalidate_fk()` before
  `forward()`. Both NeRD state-writing adapters now do so.

## Generate a small dataset

```bash
uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Cartpole --num-envs 2 \
  --num-transitions 32 --trajectory-length 16 \
  --dataset-dir /tmp/nerd-data --dataset-name dataset_train.hdf5 \
  --device cuda:0 --visualizer none
```

For Anymal with native contact tokens, select
`Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0` or
`Isaac-Velocity-Rough-Anymal-C-Dataset-Gen-v0` and add
`--contact-mode newton_native --contact-representation contact_tokens
--num-contacts-per-env 64 --max-contact-tokens 64`.

## Train and use a dynamics model

Set the dataset paths and training budget in a copy of a configuration from
`isaaclab_neural/train/cfg/`, then run:

```bash
uv run python -m isaaclab_neural.train.train \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --cfg /path/to/training.yaml --num-envs 64 \
  --logdir ./logs/nerd --device cuda:0 --visualizer none
```

Use a contact representation and history length compatible with the dataset.
For rough terrain, use the rough NeRD task and a rough training configuration.

Check a trained model's full and partial resets:

```bash
uv run python -m isaaclab_neural.eval.reset_smoke \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --checkpoint /path/to/dynamics/nn/final_model.pt \
  --num-envs 2 --repeat-steps 3 --device cuda:0 --visualizer none
```

Train an RL policy on that model:

```bash
uv run python -m isaaclab_neural.rl.rsl_rl.train \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --neural-model-path /path/to/dynamics/nn/final_model.pt \
  --num_envs 64 --max_iterations 5 --logger tensorboard \
  --device cuda:0 --visualizer none
```

## Validation and scope

Migration validation covered the package tests, CPU state-reset regression
cases, GPU Cartpole and flat/rough Anymal dataset generation, tiny dynamics
training runs, rollout evaluation, full/partial neural history resets,
one PPO iteration, PPO playback, and wheel contents. These are compatibility
checks, not model-accuracy or policy-convergence results.

Datasets, pretrained checkpoints, and the sibling repository's `osmo_scripts/`
workflows were not copied. References to those paths and historical performance
results in the older workflow guides describe the source project, not newly
validated results on develop. Supply compatible checkpoints and datasets to
run full experiments. Interactive viewers, distributed training, and long runs
were not validated in this migration.
