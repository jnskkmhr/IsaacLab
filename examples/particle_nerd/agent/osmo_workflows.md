# OSMO: data collection, training, and evaluation

Run commands from the owning IsaacLab worktree (`~/isaac/IsaacLab-particle-nerd`).

## Workflow templates

| Stage | Workflow | Executed module |
| --- | --- | --- |
| Box collection | `examples/particle_nerd/osmo/collection.yaml` | `particle_nerd.data_collection.campaign` |
| Box Transolver training | `examples/particle_nerd/osmo/transolver_training.yaml` | `particle_nerd.transolver.train` + `particle_nerd.transolver.evaluate` |
| Shovel collection | `examples/particle_nerd/osmo/shovel_collection.yaml` | `particle_nerd.data_collection.campaign` (`--geometry shovel`) |
| Shovel Transolver training | `examples/particle_nerd/osmo/shovel_training.yaml` | `particle_nerd.transolver.train` + `particle_nerd.transolver.evaluate` |
| G1 collection | `examples/particle_nerd/g1/osmo/collect_500.yaml` | `particle_nerd.g1.collect` |
| G1 full-bed training | `examples/particle_nerd/g1/osmo/train_500.yaml` | `particle_nerd.g1.train` |
| G1 active-compaction training | `examples/particle_nerd/g1/osmo/train_compact_500.yaml` | `particle_nerd.g1.train` (`--active-k auto --settled-frame 150`) |

## Package and upload source bundle

```bash
RUN=examples/particle_nerd/run.sh
SNAPSHOT_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
BUNDLE_DIR="/tmp/particle-nerd-source-$SNAPSHOT_ID"
"$RUN" -m particle_nerd.osmo.stage --output "$BUNDLE_DIR"
```
