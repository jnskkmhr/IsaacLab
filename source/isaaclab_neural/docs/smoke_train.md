# Rough Native Contact-Token Smoke Train

This small local workflow generates a rough-terrain dataset using
**Newton native contacts + contact tokens**, then runs a short training smoke
test. See [train.md](train.md) for the full-scale workflow.

The current directory is assumed to be the `IsaacLab-NeRD` repository root.

## Goal

| item | smoke scale |
|---|---:|
| train transitions | ``1000000`` |
| valid / zero-action transitions | ``100000`` each |
| ``num_envs`` (data gen) | ``256`` |
| ``trajectory_length`` | ``100`` |
| contact | ``newton_native`` + ``contact_tokens`` (K=64) |
| train epochs | ``5`` (smoke cfg) |

Output directories:

```text
./data/datasets/Anymal-C-Rough-Native-ContactTokens/
./data/trained_models/Anymal-C-Rough-Native-ContactTokens-Smoke/
```

## Setup

```bash
uv sync
```

Key flags for rough terrain with contact tokens, shared by all generation commands:

```text
--task Isaac-Velocity-Rough-Anymal-C-Dataset-Gen-v0
--contact-mode newton_native
--contact-representation contact_tokens
--max-contact-tokens 64
--num-contacts-per-env 64
--env-name Anymal-C-Rough-Native-ContactTokens
```

The HDF5 file path is ``{dataset_dir}/{env_name}/{dataset_name}``.

## 1. Generate train dataset (1M)

```bash
uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Rough-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_train.hdf5 \
  --env-name Anymal-C-Rough-Native-ContactTokens \
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
  --num-envs 256 \
  --num-transitions 1000000 \
  --write-chunk-transitions 1000000 \
  --trajectory-length 100 \
  --seed 0 \
  --visualizer none \
  --force-overwrite \
  presets=newton_mjwarp
```

## 2. Generate validation datasets (100k each)

### Randomized-PD valid

```bash
uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Rough-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_valid.hdf5 \
  --env-name Anymal-C-Rough-Native-ContactTokens \
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
  --num-envs 256 \
  --num-transitions 100000 \
  --write-chunk-transitions 100000 \
  --trajectory-length 100 \
  --seed 10 \
  --visualizer none \
  --force-overwrite \
  presets=newton_mjwarp
```

### Zero-action valid (same Dataset-Gen task)

```bash
uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Rough-Anymal-C-Dataset-Gen-v0 \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_zero_action_valid.hdf5 \
  --env-name Anymal-C-Rough-Native-ContactTokens \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --contact-representation contact_tokens \
  --max-contact-tokens 64 \
  --num-contacts-per-env 64 \
  --zero-actions \
  --num-envs 256 \
  --num-transitions 100000 \
  --write-chunk-transitions 100000 \
  --trajectory-length 100 \
  --seed 20 \
  --visualizer none \
  --force-overwrite \
  presets=newton_mjwarp
```

The smoke training configuration requires only the three HDF5 files above.
Run optional step 3 to match the full rough-terrain configuration, which also
includes validation datasets generated with LSTM actuators.

## 3. Optional LSTM actuator valids

Use the stock rough-terrain task with its LSTM actuator and the same contact-token settings:

```bash
POLICY=./pretrained/control_policy/anymal_c_rough_terrain/model_1499.pt

uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Rough-Anymal-C \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_lstm_actuator_zero_action_valid.hdf5 \
  --env-name Anymal-C-Rough-Native-ContactTokens \
  --robot-name Anymal-C \
  --sample-mode action \
  --initial-states-source env \
  --contact-mode newton_native \
  --contact-representation contact_tokens \
  --max-contact-tokens 64 \
  --num-contacts-per-env 64 \
  --zero-actions \
  --num-envs 256 \
  --num-transitions 100000 \
  --write-chunk-transitions 100000 \
  --trajectory-length 100 \
  --seed 30 \
  --visualizer none \
  --force-overwrite \
  presets=newton_mjwarp

uv run python -m isaaclab_neural.generate.generate_dataset \
  --task Isaac-Velocity-Rough-Anymal-C \
  --dataset-dir ./data/datasets \
  --dataset-name dataset_lstm_actuator_policy_valid.hdf5 \
  --env-name Anymal-C-Rough-Native-ContactTokens \
  --robot-name Anymal-C \
  --sample-mode policy \
  --policy-checkpoint "$POLICY" \
  --policy-agent rsl_rl_cfg_entry_point \
  --initial-states-source env \
  --contact-mode newton_native \
  --contact-representation contact_tokens \
  --max-contact-tokens 64 \
  --num-contacts-per-env 64 \
  --num-envs 256 \
  --num-transitions 100000 \
  --write-chunk-transitions 100000 \
  --trajectory-length 100 \
  --seed 40 \
  --visualizer none \
  --force-overwrite \
  presets=newton_mjwarp
```

The full training configuration requires these LSTM validation files:

```text
source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_rough_native_contact_tokens.yaml
```

## 4. Smoke train

```bash
uv run python -m isaaclab_neural.train.train \
  --task Isaac-Velocity-Rough-Anymal-C-NeRD-v0 \
  --cfg ./source/isaaclab_neural/isaaclab_neural/train/cfg/Anymal/transformer_rough_native_contact_tokens_smoke.yaml \
  --logdir ./data/trained_models/Anymal-C-Rough-Native-ContactTokens-Smoke \
  --num-envs 64 \
  --seed 0 \
  --visualizer none \
  --update-dataset-statistics \
  --skip-check-log-override \
  --save-interval 1 \
  --eval-interval 1 \
  presets=newton_mjwarp
```

Key smoke configuration settings:

- ``contact_representation: contact_tokens`` / ``max_contact_tokens: 64``
- ``num_epochs: 5``, ``num_iters_per_epoch: 200``, ``batch_size: 128``
- ``max_capacity: 1000000``
- Validation uses only ``exp_trajectory`` and ``zero_action_trajectory``.

## 5. Quick eval (optional)

```bash
uv run python -m isaaclab_neural.eval.eval \
  --task Isaac-Velocity-Rough-Anymal-C-NeRD-v0 \
  --checkpoint ./data/trained_models/Anymal-C-Rough-Native-ContactTokens-Smoke/<timestamp>/nn/final_model.pt \
  --num-envs 16 \
  --num-steps 32 \
  --seed 0 \
  --contact-mode newton_native \
  --num-contacts-per-env 64 \
  --visualizer none \
  presets=newton_mjwarp
```

``--checkpoint`` restores the embedded ``neural_solver_cfg`` (including
``contact_representation: contact_tokens``).

Token data diagnostics:

```bash
DS=./data/datasets/Anymal-C-Rough-Native-ContactTokens/dataset_train.hdf5
uv run python -m isaaclab_neural.eval.contact_distribution_stats --dataset "$DS"
uv run python -m isaaclab_neural.eval.contact_regime_eval --dataset "$DS" --overflow-gate
```

## Even smaller debug

For an initial end-to-end smoke test, reduce the workload further:

```text
--num-envs 32
--num-transitions 2000
--trajectory-length 20
--write-chunk-transitions 2000
```

Also reduce ``algorithm.num_epochs`` and ``num_iters_per_epoch`` in the smoke
configuration, or use:

```bash
--cfg-overrides algorithm.num_epochs 1 algorithm.num_iters_per_epoch 20
```

## Notes

- Flat slots and contact tokens are **incompatible**. Do not mix flat data from
  ``Anymal-C-Rough-Native`` with the token data directory used by this smoke test.
- Rough terrain requires terrain context; the smoke configuration enables
  ``require_terrain_context``.
- To reproduce the full experiment, use the OSMO preset
  ``osmo_scripts/presets/anymal_rough_newton_native_contact_tokens.yaml``
  (20M training / 1M validation transitions) and the full
  ``transformer_rough_native_contact_tokens.yaml`` configuration.
