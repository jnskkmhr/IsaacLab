# Particle NeRD: agent handoff

This directory documents the maintained `ParticleTransolver` + analytical/RFT contact-force architecture across `examples/particle_nerd` and `examples/particle_nerd_batch`.

## Start here

| Read | Purpose |
| --- | --- |
| [Transolver and batched simulation](transolver_and_batch.md) | Current architecture, active particle compaction, source owners, launch commands, and timings |
| [Batch runner README](../../particle_nerd_batch/README.md) | Interactive GL/RTX launch, full-step CUDA graph capture, profiling, and partial reset |
| [OSMO workflows](osmo_workflows.md) | Source staging, data collection, and distributed recurrent training on OSMO |
| [Shovel experiment](shovel.md) | Concave SDF geometry, recurrent training, and dynamic coupling |
| [Changing the model](model_changes.md) | Replacement contract, affected callers, checkpoint compatibility, and validation |
| [Example README](../README.md) | Supported launch commands and source layout |
| [G1 README](../g1/README.md) | 200 Hz G1 humanoid sand-bed collection, training, active compaction, and closed-loop play |

## Architecture overview

| Path | Particle motion | Material force on rigid bodies | Scope |
| --- | --- | --- | --- |
| `transolver/`, `runner/` | `ParticleTransolver` (`particle_transolver_v1`) | `ContactEstimator` (`jacobi`, `body_velocity`, or `rft_body_velocity`) | Single dynamic box/shovel; recorded-body particle evaluation in `transolver/evaluate.py` |
| `particle_nerd_batch/` | Batched `ParticleTransolver` (full `16,384` or `--active-particles` + quadrature anchors) | Batched `ContactEstimator` (`analytical` or `rft`) | One free intruder per environment; full-step CUDA graph; scales to 256 worlds on one GPU |
| `g1/` | G1 `ParticleTransolver` (full `88,360` or `--active-k auto` `5,312` + `512` anchors) | `G1ProxyEstimator` (`jacobi`, `body_velocity`, or `rft_body_velocity`) | Native actuated G1 dynamics at 200 Hz (single or multi-environment) |

## Constraints to preserve

- Learned inference must not run an MPM solve. MPM remains the data-collection and reference solver.
- Simulation-step geometry, neighborhoods, contact solving, and state updates use GPU Warp; the neural network uses Torch. Keep CPU transfers outside the physics step.
- Use material particles and rigid collider geometry as physical roles without semantic object-ID classes.
- Preserve dataset splits, original checkpoints, and existing results.
- Preserve the 0.5 s recurrent training horizon: 25 steps at 50 Hz for box/shovel, or 100 steps at 200 Hz for G1.
