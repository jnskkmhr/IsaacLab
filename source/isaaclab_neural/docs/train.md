# IsaacLab-NeRD Local Training

This guide covers local data collection and training for:

- Cartpole with fixed-ground contacts
- Anymal-C with fixed-ground contacts
- Anymal-C with Newton native contacts, using either **flat slots** or **contact tokens**

All commands assume the current directory is the `IsaacLab-NeRD` repository root.

After training the dynamics model, see [RL policy learning](rl.md) for RSL-RL velocity tracking on NeRD.

## Setup

```bash
uv sync
```

Data is written to `./data/datasets` by default, and training outputs go to `./data/trained_models`.

For local debugging, reduce `--num-transitions`, `--num-envs`, and `--trajectory-length`. Use the full sizes below when reproducing experiments.

## Recommended Preset Runner

Use `osmo_scripts/run_experiment.py` and `osmo_scripts/presets/*.yaml` to keep
local and OSMO training configurations consistent. Presets select the task,
dataset specification, contact mode, training configuration, and run size.
Training hyperparameters still come exclusively from the YAML files under
`source/isaaclab_neural/isaaclab_neural/train/cfg/`.

Run the Cartpole preset locally:

```bash
uv run python osmo_scripts/run_experiment.py \
  --preset cartpole_fixed_ground \
  --workflow-base-name cartpole-fixed-ground \
  --dataset-subdir cartpole-fixed-ground \
  --dataset-cache-mode off \
  --output-root ./data/trained_models/osmo-local-cartpole \
  --dataset-dir ./data/datasets
```

Run the Anymal-C fixed-ground preset locally:

```bash
uv run python osmo_scripts/run_experiment.py \
  --preset anymal_fixed_ground \
  --workflow-base-name anymal-c-fixed-ground \
  --dataset-subdir anymal-c-fixed-ground \
  --dataset-cache-mode off \
  --output-root ./data/trained_models/osmo-local-anymal-c \
  --dataset-dir ./data/datasets
```

Run the Anymal-C Newton-native preset locally:

```bash
uv run python osmo_scripts/run_experiment.py \
  --preset anymal_newton_native \
  --workflow-base-name anymal-c-newton-native \
  --dataset-subdir anymal-c-newton-native \
  --dataset-cache-mode off \
  --output-root ./data/trained_models/osmo-local-anymal-c-native \
  --dataset-dir ./data/datasets
```

To reuse existing local HDF5 data and skip generation, run the
`isaaclab_neural.train.train` commands below directly.

## OSMO Training

The OSMO training entry point is `osmo_scripts/start.sh`, with configurations
in `osmo_scripts/presets/*.yaml`. By default, it:

1. Packages the current code and uploads it to `IsaacLab-NeRD-Code`.
2. Looks for cached data in `IsaacLab-NeRD-Datasets/<dataset_subdir>/<env_name>/`.
3. Generates missing data in OSMO and uploads it to `IsaacLab-NeRD-Datasets`.
4. Uploads logs, TensorBoard files, and checkpoints to
   `IsaacLab-NeRD-Output/<workflow_name>/`.

Example:

```bash
export NGC_API_KEY=<your-nvapi-key>
export NVDATASET_TENANTID=<your-tenant-id>

./osmo_scripts/start.sh \
  --preset anymal_fixed_ground \
  --pool <osmo-pool>
```

See `osmo_scripts/README.md` for more information about OSMO and NV-Datasets.
To log to Weights & Biases, use `osmo_scripts/start.sh --enable-wandb` and set
`WANDB_API_KEY`. See the W&B section of `osmo_scripts/README.md` for details.

## Memory and Storage Notes

Use `--write-chunk-transitions` to write generated data to HDF5 in chunks.
Each chunk's complete schema, shapes, and dtypes are validated before writing.
If an append fails, resized datasets are rolled back to avoid leaving a
partially written HDF5 file.

In `SequenceModelTrainer`'s ``eager`` mode, initialization loads up to
`algorithm.dataset.max_capacity` into CPU memory. With DDP, trajectories are
first limited by the global ``max_capacity``, then divided into non-overlapping
shards across ranks. Together, the ranks hold approximately one copy of the
training data rather than one copy per rank. Validation data is still loaded
in full on rank 0 only.

``lazy`` mode keeps only metadata and an HDF5 handle in memory, reading data
by batch. CUDA training enables pinned memory, non-blocking copies, and
persistent workers by default. Use these settings to tune data loading:

```yaml
algorithm:
  dataset:
    load_mode: lazy
    num_data_workers: 4
    pin_memory: auto
    non_blocking: auto
    persistent_workers: auto
    prefetch_factor: 2
    # Optional global cap divided across DDP ranks.
    max_total_workers: 32
```

``eager`` generally offers higher throughput, while ``lazy`` uses the least
memory. On OSMO, start with ``eager`` rank sharding. Switch to ``lazy`` if
validation or contact-token data still exceeds node memory. Reduce
``max_capacity`` or ``num_data_workers`` for smoke tests.

## Cartpole Fixed Ground

Configuration file:

```text
source/isaaclab_neural/isaaclab_neural/train/cfg/Cartpole/transformer.yaml
```

Key settings:

```yaml
contact_mode: fixed_ground
states_frame: world
```

### Generate Train Dataset

```bash
uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Cartpole \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_train.hdf5 \
  --env-name Cartpole \
  --robot-name Cartpole \
  --sample-mode joint_f \
  --initial-states-source sample \
  --contact-mode fixed_ground \
  --states-frame world \
  --num-envs 256 \
  --num-transitions 1000000 \
  --trajectory-length 100 \
  --seed 0 \
  --visualizer none \
  --force-overwrite
```

### Generate Validation Datasets

```bash
uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Cartpole \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_valid.hdf5 \
  --env-name Cartpole \
  --robot-name Cartpole \
  --sample-mode joint_f \
  --initial-states-source sample \
  --contact-mode fixed_ground \
  --states-frame world \
  --num-envs 256 \
  --num-transitions 100000 \
  --trajectory-length 100 \
  --seed 1 \
  --visualizer none \
  --force-overwrite

uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Cartpole \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_passive_valid.hdf5 \
  --env-name Cartpole \
  --robot-name Cartpole \
  --sample-mode joint_f \
  --initial-states-source sample \
  --contact-mode fixed_ground \
  --states-frame world \
  --zero-actions \
  --num-envs 256 \
  --num-transitions 100000 \
  --trajectory-length 100 \
  --seed 2 \
  --visualizer none \
  --force-overwrite
```

### Train

```bash
uv run python -m isaaclab_neural.train.train \
  --task Isaac-Cartpole-NeRD-v0 \
  --cfg ./source/isaaclab_neural/isaaclab_neural/train/cfg/Cartpole/transformer.yaml \
  --logdir ./data/trained_models/Cartpole \
  --num-envs 256 \
  --seed 0 \
  --visualizer none \
  --update-dataset-statistics \
  --skip-check-log-override
```

## Anymal-C Fixed Ground

Configuration file:

```text
source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer.yaml
```

Key settings:

```yaml
contact_mode: fixed_ground
states_frame: body
anchor_frame_step: every
```

### Generate Train Dataset

```bash
uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_train.hdf5 \
  --env-name Anymal-C \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode fixed_ground \
  --randomize-pd-gains \
  --kp-min 30.0 \
  --kp-max 200.0 \
  --kd-min 0.0 \
  --kd-max 4.0 \
  --num-envs 1024 \
  --num-transitions 20000000 \
  --write-chunk-transitions 10000000 \
  --trajectory-length 400 \
  --seed 0 \
  --visualizer none \
  --force-overwrite
```

### Generate Validation Datasets

```bash
uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_valid.hdf5 \
  --env-name Anymal-C \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode fixed_ground \
  --randomize-pd-gains \
  --kp-min 30.0 \
  --kp-max 200.0 \
  --kd-min 0.0 \
  --kd-max 4.0 \
  --num-envs 1024 \
  --num-transitions 1000000 \
  --write-chunk-transitions 10000000 \
  --trajectory-length 400 \
  --seed 10 \
  --visualizer none \
  --force-overwrite

uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_zero_action_valid.hdf5 \
  --env-name Anymal-C \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode fixed_ground \
  --zero-actions \
  --num-envs 1024 \
  --num-transitions 1000000 \
  --write-chunk-transitions 10000000 \
  --trajectory-length 400 \
  --seed 20 \
  --visualizer none \
  --force-overwrite

uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_lstm_actuator_zero_action_valid.hdf5 \
  --env-name Anymal-C \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode fixed_ground \
  --zero-actions \
  --num-envs 1024 \
  --num-transitions 1000000 \
  --write-chunk-transitions 10000000 \
  --trajectory-length 400 \
  --seed 30 \
  --visualizer none \
  --force-overwrite
```

### Train

```bash
uv run python -m isaaclab_neural.train.train \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --cfg ./source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer.yaml \
  --logdir ./data/trained_models/Anymal-C \
  --num-envs 1024 \
  --seed 0 \
  --visualizer none \
  --update-dataset-statistics \
  --skip-check-log-override \
  presets=newton_mjwarp
```

## Anymal-C Newton Native

Newton native mode records dynamic contacts from the collision pipeline.
This branch supports two **contact representations**. They are incompatible:
do not mix their datasets or checkpoints.

| | Flat slots (default) | Contact tokens |
|---|---|---|
| ``contact_representation`` | ``flat`` (may be omitted) | ``contact_tokens`` |
| Capacity field | ``num_contacts_per_env`` | ``max_contact_tokens`` |
| Packing | Flat packing, such as ``penetration_priority`` | Pair-atomic body round-robin inside ``ContactSetEncoder`` |
| Main HDF5 fields | ``contact_points_*`` / ``normals`` / ``depths`` / ``masks`` | ``contact_tokens`` ``[..., K, 17]`` + overflow |
| Training config | ``transformer_native.yaml`` | ``transformer_native_contact_tokens.yaml`` |
| Recommended data directory | ``./data/datasets/Anymal-C-Native/`` | ``./data/datasets/Anymal-C-Native-ContactTokens/`` |

The **flat** workflow is described below; see the next section for **contact tokens**.

### Flat contact representation (default)

Configuration file:

```text
source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_native.yaml
```

Key settings:

```yaml
contact_mode: newton_native
# contact_representation defaults to flat and may be omitted.
num_contacts_per_env: 64
contact_packing_policy: penetration_priority
states_frame: body
anchor_frame_step: every
```

Native flat datasets pack contacts into 64 fixed slots. The default packing
policy, `penetration_priority`, prioritizes contacts with the smallest signed
surface separation and uses geometry to break ties deterministically.

Native flat contact input conventions:

- Contact sides are canonicalized with the primary robot articulation first.
- `contact_normals` point from the robot side toward the other side.
- `contact_depths` store signed surface separation rather than the old raw normal distance.
- `contact_masks` indicate whether a slot contains a native contact.
- Inactive slots are zeroed again during preprocessing and at the model input
  boundary so normalization or noise cannot turn padding into nonzero features.
- Contact RMS statistics use only valid contacts with `contact_masks=True`.
  Slots with too few samples fall back to pooled field statistics.

Native flat inputs include `contact_points_0` and `contact_points_1`. Both are
stored in the world frame during collection and converted to the body frame
during training preprocessing.

### Generate Train Dataset

```bash
uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_train.hdf5 \
  --env-name Anymal-C-Native \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --num-contacts-per-env 64 \
  --contact-packing-policy penetration_priority \
  --randomize-pd-gains \
  --kp-min 30.0 \
  --kp-max 200.0 \
  --kd-min 0.0 \
  --kd-max 4.0 \
  --num-envs 1024 \
  --num-transitions 20000000 \
  --write-chunk-transitions 5000000 \
  --trajectory-length 400 \
  --seed 0 \
  --visualizer none \
  --force-overwrite
```

### Generate Validation Datasets

```bash
uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_valid.hdf5 \
  --env-name Anymal-C-Native \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --num-contacts-per-env 64 \
  --contact-packing-policy penetration_priority \
  --randomize-pd-gains \
  --kp-min 30.0 \
  --kp-max 200.0 \
  --kd-min 0.0 \
  --kd-max 4.0 \
  --num-envs 1024 \
  --num-transitions 1000000 \
  --write-chunk-transitions 5000000 \
  --trajectory-length 400 \
  --seed 10 \
  --visualizer none \
  --force-overwrite

uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_zero_action_valid.hdf5 \
  --env-name Anymal-C-Native \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --num-contacts-per-env 64 \
  --contact-packing-policy penetration_priority \
  --zero-actions \
  --num-envs 1024 \
  --num-transitions 1000000 \
  --write-chunk-transitions 5000000 \
  --trajectory-length 400 \
  --seed 20 \
  --visualizer none \
  --force-overwrite

uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_lstm_actuator_zero_action_valid.hdf5 \
  --env-name Anymal-C-Native \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --num-contacts-per-env 64 \
  --contact-packing-policy penetration_priority \
  --zero-actions \
  --num-envs 1024 \
  --num-transitions 1000000 \
  --write-chunk-transitions 5000000 \
  --trajectory-length 400 \
  --seed 30 \
  --visualizer none \
  --force-overwrite
```

### Train

```bash
uv run python -m isaaclab_neural.train.train \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --cfg ./source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_native.yaml \
  --logdir ./data/trained_models/Anymal-C-Native \
  --num-envs 1024 \
  --seed 0 \
  --visualizer none \
  --update-dataset-statistics \
  --skip-check-log-override \
  presets=newton_mjwarp
```

## TensorBoard

```bash
python3 -m tensorboard.main \
  --logdir ./data/trained_models \
  --host 0.0.0.0 \
  --port 6006
```

## Local Smoke Testing

For a quick local test, reduce data generation scale first:

```bash
--num-envs 32
--num-transitions 2000
--trajectory-length 20
--write-chunk-transitions 2000
```

For native experiments, always regenerate datasets after changing contact
schema, ``contact_representation``, packing policy, ``num_contacts_per_env`` /
``max_contact_tokens``, or contact point frame semantics.

## Contact Token Sets

PhysicsNeMo-style **contact tokens** store directed contacts as

```text
contact_tokens: [trajectories, steps, max_contact_tokens, 17]
contact_token_overflow: [trajectories, steps]   # dropped contacts beyond K
```

instead of flat 64-slot fields. Use this path for set-encoder experiments; keep
flat native for the default Anymal-C native baseline and for the validated
flat RL recipe in [rl.md](rl.md).

Token channels (``CONTACT_TOKEN_DIM = 17``):

```text
0  valid
1  body_slot
2  other_body_slot
3  other_is_dynamic
4-6   point xyz
7-9   normal xyz
10-12 lever_arm xyz
13    gap (signed separation)
14-16 relative_velocity xyz
```

### Config

Flat smoke / Anymal-C native tokens:

```text
source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_native_contact_tokens.yaml
```

Rough A/B:

```text
source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_rough_native_contact_tokens.yaml
```

Key settings:

```yaml
env:
  # Solver/runtime env label in the training YAML (may differ from HDF5 folder name).
  env_name: Anymal-C-Native
  neural_solver_cfg:
    contact_mode: newton_native
    contact_representation: contact_tokens
    max_contact_tokens: 64
    # Flat slot width is unused for capacity; kept for pipeline compatibility.
    num_contacts_per_env: 64
    # Metadata only; packing is implemented in ContactSetEncoder.
    contact_packing_policy: body_round_robin_pair_atomic
inputs:
  low_dim: [states_embedding, joint_f, gravity_dir]
  contact_set:
    dim: 17
    encoder_layers: 2
    encoder_heads: 4
    num_latent_queries: 8
    hidden_size: 384
```

Notes:

- Capacity is ``max_contact_tokens`` (not ``num_contacts_per_env``).
- Eager/lazy loaders preserve ``contact_tokens`` shape ``[..., K, 17]``.
- Token HDF5 must include ``root_body_q`` and ``gravity_dir`` for body-frame training.
- Flat and token checkpoints / datasets are **not** interchangeable.
- Generate path is ``{dataset_dir}/{env_name}/{dataset_name}``. To match the
  token YAML paths, use ``--env-name Anymal-C-Native-ContactTokens`` with
  ``--dataset-dir ./data/datasets``.

### Generate Train Dataset

```bash
uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_train.hdf5 \
  --env-name Anymal-C-Native-ContactTokens \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --contact-representation contact_tokens \
  --max-contact-tokens 64 \
  --num-contacts-per-env 64 \
  --randomize-pd-gains \
  --kp-min 30.0 \
  --kp-max 200.0 \
  --kd-min 0.0 \
  --kd-max 4.0 \
  --num-envs 1024 \
  --num-transitions 20000000 \
  --write-chunk-transitions 5000000 \
  --trajectory-length 400 \
  --seed 0 \
  --visualizer none \
  --force-overwrite
```

### Generate Validation Datasets

```bash
uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_valid.hdf5 \
  --env-name Anymal-C-Native-ContactTokens \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --contact-representation contact_tokens \
  --max-contact-tokens 64 \
  --num-contacts-per-env 64 \
  --randomize-pd-gains \
  --kp-min 30.0 \
  --kp-max 200.0 \
  --kd-min 0.0 \
  --kd-max 4.0 \
  --num-envs 1024 \
  --num-transitions 1000000 \
  --write-chunk-transitions 5000000 \
  --trajectory-length 400 \
  --seed 10 \
  --visualizer none \
  --force-overwrite

uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Flat-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_zero_action_valid.hdf5 \
  --env-name Anymal-C-Native-ContactTokens \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --contact-representation contact_tokens \
  --max-contact-tokens 64 \
  --num-contacts-per-env 64 \
  --zero-actions \
  --num-envs 1024 \
  --num-transitions 1000000 \
  --write-chunk-transitions 5000000 \
  --trajectory-length 400 \
  --seed 20 \
  --visualizer none \
  --force-overwrite
```

### Train

```bash
uv run python -m isaaclab_neural.train.train \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --cfg ./source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_native_contact_tokens.yaml \
  --logdir ./data/trained_models/Anymal-C-Native-ContactTokens \
  --num-envs 1024 \
  --seed 0 \
  --visualizer none \
  --update-dataset-statistics \
  --skip-check-log-override \
  presets=newton_mjwarp
```

### Rough contact tokens

Full-scale notes live in [train.md](train.md). For a local 1M-transition smoke
(data gen + short train), see [smoke_train.md](smoke_train.md).

### Diagnostics

```bash
uv run python -m isaaclab_neural.eval.contact_distribution_stats --dataset PATH
uv run python -m isaaclab_neural.eval.contact_regime_eval --dataset PATH --overflow-gate
uv run python -m isaaclab_neural.eval.contact_reconstruction_diagnostic \
  --task Isaac-Velocity-Flat-Anymal-C-NeRD-v0 \
  --checkpoint PATH \
  --dataset PATH \
  --num-envs 16
```

Pass ``--overflow-gate`` to fail when capacity truncates tokens.
Regenerate HDF5 whenever ``max_contact_tokens`` or the 17-channel schema changes.
