# Particle NeRD: agent handoff

Implementation snapshot: 2026-10-19. This directory is maintained project documentation for a teammate using Codex, Claude, or another coding agent. Source code and the selected checkpoint configuration take precedence if they have changed since this snapshot.

## Start here

Give your agent this prompt from the IsaacLab repository root:

```text
Read the repository AGENTS.md and examples/particle_nerd/agent/README.md.
Then read the linked implementation and architecture-change notes relevant to my task.
Inspect the current git diff before editing; preserve unrelated work.
Use examples/particle_nerd as the canonical source.
Explain the interfaces affected by the requested model change, implement it,
and validate training/inference consistency and held-out rollout behavior.
Distinguish execution checks from measured model accuracy.
Task: <describe the architecture or experiment change here>
```

These files are not automatically loaded by every agent. Explicitly point the agent to this README. Repository and directory-specific instructions still apply.

| Read | Purpose |
| --- | --- |
| [Current implementation](implementation.md) | Data, model inputs/outputs, contact mechanics, and coupling order |
| [OSMO workflows](osmo_workflows.md) | Source upload, collection, training, independent evaluation, and output locations |
| [Changing the model](model_changes.md) | Replacement contract, affected callers, checkpoint compatibility, validation |
| [Example README](../README.md) | Supported launch commands and source layout |
| [Migration record](../MIGRATION.md) | Consolidation history and previously completed execution checks |

## Research objective and current status

Replace the expensive MPM particle update with a learned particle-motion model. Recover the material-on-rigid-body contact wrench with an analytical iterative impulse solver, then advance rigid bodies with Newton MJWarp. There is no learned force head.

| Path | Particle motion | Rigid-body motion | What it establishes |
| --- | --- | --- | --- |
| `data_collection.collect` | MPM reference | Dynamic MJWarp with MPM feedback | Reference trajectories and contact wrenches |
| `gnn.play`, `gnn.evaluate` | Recurrent GNN | Recorded reference box trajectory | Particle prediction under reference box motion |
| `force_estimation.evaluate` | Recorded reference particles | Recorded states used for reconstruction | Force reconstruction with reference particle input |
| `runner.play` | Recurrent GNN | Dynamic MJWarp with estimated wrench | Fully coupled learned simulation; no MPM solver |
| Separate G1 force benchmark | Live MPM | Dynamic articulated MJWarp with reference or estimated feedback | Force estimator in a robot system; not a learned-particle demonstration |

The unified coupled runner is new and experimental. Its short execution checks do not establish long-horizon accuracy. Earlier reports include different model generations and must not be treated as results for the current runner.

**Pending model change:** the user requested displacement-only prediction, deriving particle velocity from displacement divided by timestep. Current code still has six outputs and predicts displacement-rate and endpoint-velocity corrections separately. This request has not yet been implemented or retrained. See [model changes](model_changes.md).

## Constraints to preserve

- Learned inference must not run an MPM solve, including a low-iteration MPM correction step. MPM remains the data/reference solver.
- Simulation-step geometry, neighborhoods, contact solving, and state updates use GPU Warp; the neural network uses Torch. Keep CPU transfers outside the physics step. Offline analysis and recording can use CPU code.
- Use material particles and rigid collider geometry as physical roles. Do not introduce semantic classes such as shovel versus foot. Current geometry support is narrower than this intended generalization.
- Do not change terrain dimensions, particle spacing, material parameters, or scene appearance incidentally while changing a model. Record deliberate dataset changes.
- Say “solver iterations.” Keep MPM iterations separate from contact-estimator iterations.
- Report force exerted by which object on which body, coordinate frame, torque origin, and sample interval. Use explicit component names, including “Collider encoder.”
- Preserve dataset splits, original checkpoints, and existing results. New architecture experiments need separate output directories.

## Local source and artifact locations

The active worktree at this snapshot is `/home/jkamohara/isaac/IsaacLab-particle-nerd`, branch `junnosuke/particle_nerd`. The original `~/isaac/IsaacLab` checkout is not the example's working copy. Inspect `git status` before editing: consolidation changes may still be uncommitted.

Canonical modules live in the parent of this directory. External `experiments/particle_gnn` files include compatibility links and reporting scripts; edit the canonical modules. Do not restore retired flat modules from the archive.

Existing local example assets, not bundled with the source:

```text
/home/jkamohara/isaac/particle_nerd/particle_nerd_plan/results/
  particle_gnn_full_training/model/best.pt
  particle_gnn_region_full/selected.pt
  particle_gnn_trained_demo.hdf5
```

These are useful for execution checks, not automatic evidence of held-out accuracy or the latest training run. Check checkpoint `config`/`run`, dataset manifests, and campaign split records. The demo was inspected during development; use genuinely held-out episodes for final evaluation. Full training data are supplied through `--data`; do not infer dataset conditions from an artifact filename.

The separate G1 implementation is under `/home/jkamohara/isaac/IsaacLab-g1-force/examples/particle_nerd_g1`. Website source/results are under `/home/jkamohara/isaac/particle_nerd/particle_nerd_plan`. Methodology and experiment reports are published at [the project website](https://jnskkmhr.github.io/particle-nerd-plan/). Source and run metadata, rather than slide page numbers, define the implementation.
