# OSMO: data collection, training, and evaluation

Run commands from the owning IsaacLab worktree, currently `~/isaac/IsaacLab-particle-nerd`. This guide follows the recent particle-GNN campaign structure using the consolidated modules. The commands submit real GPU jobs when executed; documenting them does not submit a job.

## Workflow map

```text
Local source + installed backend
  -> stage.py -> upload source bundle
                    |
                    v
collection.yaml -> shard directories containing HDF5 episodes + campaign splits
                    |
                    v
training.yaml   -> model/best.pt + training metrics + automatic test evaluation
                    |
                    v
 evaluation.yaml -> evaluate an existing checkpoint without retraining
```

| Stage | Workflow | Executed module | Default allocation / work |
| --- | --- | --- | --- |
| Collection | [collection.yaml](../osmo/collection.yaml) | `particle_nerd.data_collection.campaign` | 32 independent tasks, one GPU each; 1,000 episodes total |
| Training | [training.yaml](../osmo/training.yaml) | `particle_nerd.gnn.verify`, then `torchrun --module particle_nerd.gnn.train`, then `particle_nerd.gnn.evaluate` | Four GPUs on one node; 10,000 optimizer steps; automatic 12-episode evaluation |
| Independent evaluation | [evaluation.yaml](../osmo/evaluation.yaml) | `particle_nerd.gnn.evaluate` | One GPU; up to 12 test episodes, up to 250 steps each |

All templates currently select platform `ovx-l40`. Collection shards each request 8 CPUs, 64 GiB memory, and 88 GiB storage. Training requests 16 CPUs, 128 GiB memory, and 250 GiB storage. Independent evaluation requests 8 CPUs, 64 GiB memory, and 250 GiB storage. Choose a compatible accessible pool; do not assume the previous pool has capacity today.

**Evaluation scope:** particles are predicted recurrently while the box follows its recorded reference trajectory. This is not the dynamic-box force-estimation benchmark. See [implementation.md](implementation.md) for the distinct fully coupled runner and force-only evaluation.

## 1. Configure the CLI and storage

```bash
osmo version
osmo profile list
osmo pool list

RUN=examples/particle_nerd/run.sh
OSMO_DIR=examples/particle_nerd/osmo
POOL=isaac-lab-l40-07
STORAGE_ROOT=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/particle-gnn
OUTPUT_URL="$STORAGE_ROOT/runs"
```

The pool and storage prefix above match the recent experiment configuration; teammates must use a pool and output prefix they can access. If authentication is missing, run `osmo login` with the team's service configuration. Keep credentials out of YAML and source files.

## 2. Package and upload the source

```bash
SNAPSHOT_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
BUNDLE_DIR="/tmp/particle-nerd-source-$SNAPSHOT_ID"
SOURCE_URL="$STORAGE_ROOT/snapshots/$SNAPSHOT_ID/"

"$RUN" -m particle_nerd.osmo.stage --output "$BUNDLE_DIR"
osmo data upload "$SOURCE_URL" \
  "$BUNDLE_DIR/bundle.tar.gz" "$BUNDLE_DIR/bundle.sha256"
osmo data list "$SOURCE_URL" --no-pager
```

`stage` only writes local files. It copies this worktree's IsaacLab source, launcher configuration, and `examples/particle_nerd`; it also bundles the installed Newton/Warp/schema packages and pins other dependencies. `snapshot.json` records file hashes and backend provenance. The output directory must not already exist. The following `osmo data upload` command uploads the archive and checksum.

Each task checks the archive checksum, extracts it, and runs [bootstrap.sh](../osmo/bootstrap.sh) to create its Python environment. Repackage after source changes, particularly an architecture change. Use a new snapshot URL rather than overwriting a previous experiment's source. The dataset can be reused with a new source snapshot when its format and physical assumptions remain compatible.

## 3. Collect reference trajectories

```bash
osmo workflow submit "$OSMO_DIR/collection.yaml" --pool "$POOL" \
  --set num_shards=32 num_episodes=1000 \
  --set-string "source_url=$SOURCE_URL" "output_url=$OUTPUT_URL" \
  --format-type json
```

Record the workflow ID returned by the command:

```bash
COLLECTION_ID=REPLACE_WITH_RETURNED_COLLECTION_ID
DATASET_URL="$OUTPUT_URL/$COLLECTION_ID/"
osmo workflow query "$COLLECTION_ID"
osmo workflow logs "$COLLECTION_ID" --task collect-0 -n 100
```

Shard k records episode indices k, k+32, and so on. These are separate one-environment processes; they do not constitute one IsaacLab environment with 32 parallel worlds. See [implementation.md](implementation.md) for exact terrain, particle, mass, and material settings.

Expected final storage structure:

```text
runs/<collection-id>/
  shard-0/snapshot.json
  shard-0/runtime.log
  shard-0/episodes/episode-*.hdf5
  shard-0/episodes/episode-*.campaign.json
  shard-0/episodes/COMPLETE.json
  shard-1/...
```

Wait for every shard to complete and verify episode files and split sidecars before training. The five-minute collection checkpoint uploads progress/campaign JSON only; HDF5 trajectories upload as final task output. Progress JSON alone is not a usable dataset. Mount the entire collection workflow prefix as `dataset_url`; the loader recursively discovers all shards. Do not combine duplicate episode IDs from multiple campaigns.

A small runtime pilot can use `--set num_shards=2 num_episodes=10`. Small pilots may lack sufficient train/validation/test coverage; do not treat them as training datasets without checking the split records. The recent campaign's recorded dataset prefix was `swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/particle-gnn/runs/particle-gnn-collect-1/`. Check its availability and contents before reuse; this guide does not assert current remote state.

## 4. Train the particle model

Reuse `DATASET_URL` from the completed collection. If the model code changed, repeat packaging/upload and use the new `SOURCE_URL`.

```bash
osmo workflow submit "$OSMO_DIR/training.yaml" --pool "$POOL" \
  --set num_steps=10000 \
  --set-string "source_url=$SOURCE_URL" "dataset_url=$DATASET_URL" \
    "output_url=$OUTPUT_URL" \
  --format-type json

TRAINING_ID=REPLACE_WITH_RETURNED_TRAINING_ID
MODEL_URL="$OUTPUT_URL/$TRAINING_ID/model/"
osmo workflow query "$TRAINING_ID"
osmo workflow logs "$TRAINING_ID" --task train -n 100
```

Training uses episode-held-out validation to select `best.pt`. The task then automatically evaluates that checkpoint on the test split: up to 12 episodes with up to 250 recurrent particle steps. Current evaluation excludes episode 2 because it was inspected during development. The four-GPU request and `torchrun --nproc_per_node=4` are both fixed in this template; change both deliberately if adapting GPU count.

Expected outputs under `runs/<training-id>/`:

| Path | Contents |
| --- | --- |
| `snapshot.json`, `runtime.log` | Source/backend provenance and task log |
| `verification.json` | Feature/integration verification result |
| `model/run.json` | Training arguments and model configuration |
| `model/best.pt`, `model/last.pt` | Selected and latest checkpoints |
| `model/metrics.jsonl`, `model/progress.json`, `model/COMPLETE.json` | Training/validation progress and training completion |
| `evaluation/evaluation.json` | One-step and recurrent particle metrics |
| `evaluation/rollout-*.hdf5` | Predicted particle trajectories |

`model/COMPLETE.json` means training finished, not that the subsequent evaluation succeeded. Check workflow completion and evaluation outputs. Model checkpoint files are uploaded every five minutes; this does not automatically resume the optimizer after a failed task. Do not rerun a failed training job blindly or overwrite its output.

## 5. Evaluate an existing checkpoint independently

The new evaluation template uses a model directory whose root contains `best.pt`. It does not train or modify that checkpoint.

```bash
osmo workflow submit "$OSMO_DIR/evaluation.yaml" --pool "$POOL" \
  --set num_episodes=12 rollout_steps=250 \
  --set-string "source_url=$SOURCE_URL" "dataset_url=$DATASET_URL" \
    "model_url=$MODEL_URL" "output_url=$OUTPUT_URL" split=test \
  --format-type json

EVALUATION_ID=REPLACE_WITH_RETURNED_EVALUATION_ID
osmo workflow query "$EVALUATION_ID"
osmo workflow logs "$EVALUATION_ID" --task evaluate -n 100
osmo data download "$OUTPUT_URL/$EVALUATION_ID/" \
  "/tmp/particle-nerd-evaluation-$EVALUATION_ID"
```

Use `split=validation` for development comparisons; reserve test evaluation for the final comparison. The source snapshot must support the checkpoint's architecture and output semantics. Evaluation writes `snapshot.json`, `checkpoint.sha256`, `runtime.log`, and `evaluation/` outputs. Metrics include one-step errors, recurrent position error, ground violations, graph overflow/failure information, and particle-step GPU/wall timing. Inspect failures and valid frame counts even when the process exits successfully.

For a local rerun, the equivalent module is:

```bash
"$RUN" -m particle_nerd.gnn.evaluate --data /path/to/downloaded-dataset \
  --checkpoint /path/to/best.pt --output /path/to/new-evaluation \
  --episodes 12 --steps 250 --split test
```

## Validation and experiment records

Add `--dry-run` to a submission command to render the workflow without launching it. Check resolved inputs, outputs, GPU requests, and module arguments. A successful render does not establish GPU capacity, remote data availability, or runtime success. Independent evaluation was added with this guide; treat its first actual remote execution as an integration check.

For each real run, retain the workflow ID, source snapshot URL, dataset URL, model URL, rendered specification, and final metrics. Check status with `osmo workflow query`; use `osmo workflow events` for scheduling/start failures and `osmo workflow logs` for task failures. When reporting a run, include the available workflow/Grafana/dashboard links returned by OSMO. Never infer remote completion from local preparation files.

Optional activity-selector training has its own [region_training.yaml](../osmo/region_training.yaml). It is separate from the three-stage base-model workflow above. Model input/output changes may also require selector changes; see [model_changes.md](model_changes.md).
