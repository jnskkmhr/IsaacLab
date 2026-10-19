# Replacing the particle model

Use [implementation.md](implementation.md) for the current feature and output definitions. Read [repository instructions](../../../AGENTS.md) and the contribution guide before editing. Keep new experiments separate from existing checkpoints and results.

## Smallest replacement boundary

For a backbone change with unchanged features and output meaning, preserve:

```text
model.forward(nodes, edges, boundary, influence, src, dst, degree, valid)
    -> float32 correction tensor [N, 6] on the input GPU

ParticlePredictor.step(current_body_pose, current_body_velocity)
    -> updated particle position [N, 3], velocity [N, 3]
```

The coupled runner also reads predictor `x`, `v`, `dt`, and `device`. Initialization and selection use `history` and `model.config`. Preserve these interfaces or explicitly update their callers. `boundary` information must continue to reach particle embeddings; masking, edge orientation, and normalization must remain consistent.

| Change | Source owners/callers to inspect |
| --- | --- |
| Backbone, hidden width, message passing | [model.py](../gnn/model.py), [train.py](../gnn/train.py), [inference.py](../gnn/inference.py), [evaluate.py](../gnn/evaluate.py) |
| Input representation, geometry, material history | [features.py](../gnn/features.py), [data.py](../gnn/data.py), collector metadata, checkpoint configuration |
| Output semantics/integration | `model.integrate`, Warp `inference.advance`, training loss, evaluation, [region_selection.py](../gnn/region_selection.py) |
| State/history initialization | [dataset.py](../data_collection/dataset.py), evaluator initialization, predictor constructor |
| Coupled feedback | [simulation.py](../runner/simulation.py), [estimator.py](../force_estimation/estimator.py) |
| Verification of changed behavior | [verify.py](../gnn/verify.py), [verify_selection.py](../gnn/verify_selection.py), [GPU interface tests](../tests/test_gpu_interfaces.py) |

`train.py`, `inference.py`, and `evaluate.py` construct `ParticleGNN` directly. There is no general architecture registry. If introducing another model class, update these construction sites together. Add an explicit architecture/output version to checkpoint metadata when compatibility changes, and fail clearly on unsupported checkpoints. Do not silently load six-output weights into a three-output model or use permissive state loading to hide missing layers.

Current checkpoints contain `model`, `config`, `optimizer`, `step`, `best`, and `run`. Loading uses Python serialization: load trusted project checkpoints. Resuming training currently constructs the model from CLI arguments before loading weights; match the saved configuration or improve this behavior deliberately.

## Requested next experiment: displacement-only output

This is requested, not implemented at this snapshot:

```text
displacement = network(current particle state/history, current collider state)
next_position = position + displacement
next_velocity = displacement / dt
```

An equivalent residual parameterization can use `displacement = dt * current_velocity + learned_displacement_correction`, provided that meaning is explicit in training, inference, and checkpoint metadata. The velocity must then come from the resulting displacement, not a second predicted vector.

Update the decoder, Torch training integration, Warp inference update, loss, evaluation, and selector integration together. Preserve the old checkpoint path as an explicit legacy mode or reject it clearly. Do not merely discard three decoder outputs from the existing checkpoint and call the result a trained displacement-only model.

Finite-difference velocity is the average velocity over a recording interval; recorded MPM velocity is an endpoint state. They are not identical during acceleration/contact. A smaller inference timestep alone does not fix this unless data and model temporal assumptions also change. Before attributing force errors to the new network, evaluate force reconstruction using reference displacement-derived velocity at the intended interval. Compare that with reconstruction using reference endpoint velocity. Keep frame alignment explicit and retrain or regenerate velocity histories consistently with the chosen definition.

## Known generalization limits

- The graph network can structurally process different N, but training and one-step evaluation allocate `Features(16384)`. Geometry is fixed-size and there are exactly two collider slots. Variable-particle-count or multi-object support requires changes outside the neural backbone.
- Radius, neighbor density, mass/radius scaling, and material state affect behavior. A larger sand bed or changed sampling density is not automatically within the training distribution.
- Current initialization requires uniform particle mass and radius. Material scalars are broadcast across particles.
- Stress/deformation history is absent. Persistent material state may not be observable from three velocity frames and current geometry. Treat this as an unresolved model assumption, not a proven explanation for every rollout failure.
- Previously observed cohesion-like drift, particle inflation during lifting, and ground violations require recurrent evaluation. One-step RMSE can be dominated by stationary particles.
- The current selector still computes the full GNN. Evaluate the backbone without a selector first; report selector-assisted results separately.

## Commands and validation sequence

Run from the owning IsaacLab repository root; use [run.sh](../run.sh) to resolve the worktree imports. The launcher uses the existing interpreter plus a uv overlay. Set `PARTICLE_NERD_PYTHON` if needed; do not modify a shared environment incidentally.

```bash
RUN=examples/particle_nerd/run.sh
"$RUN" -c 'import particle_nerd; print(particle_nerd.__file__)'
"$RUN" -m particle_nerd.gnn.train --help
"$RUN" -m particle_nerd.gnn.train --data /path/to/dataset --output /path/to/new-model --steps 10000
"$RUN" -m particle_nerd.gnn.evaluate --data /path/to/dataset --checkpoint /path/to/new-model/best.pt --output /path/to/new-evaluation
"$RUN" -m particle_nerd.runner.play --checkpoint /path/to/new-model/best.pt --episode /path/to/episode.hdf5 --headless --steps 5 --output /tmp/particle-nerd-coupled-check.json
```

Paths above are placeholders. See [the example README](../README.md) for collection, independent force evaluation, optional selection, and interactive demos. Read the HDF5 manifest and checkpoint configuration before running comparisons. Existing evaluation excludes episode 2 from final test metrics because it was inspected during development.

1. Check feature dimensions, output units, timestep, device, and checkpoint loading. Compare Torch training integration with Warp inference integration for identical inputs.
2. Run relevant existing mechanics/interface checks: `"$RUN" -m pytest examples/particle_nerd/tests -q` and `"$RUN" -m particle_nerd.gnn.verify --output /tmp/gnn-check.json`. If selection changed, also run `particle_nerd.gnn.verify_selection`. For test edits, follow the repository's test-audit instructions.
3. Complete a short training/validation/checkpoint-reload check. Label it an execution check, not a trained-model result.
4. Measure held-out one-step and recurrent particle errors, including near-box/moving-particle errors, ground violations, and neighborhood overflow. Preserve episode-level splits; never select checkpoints on final test performance.
5. Evaluate force reconstruction with reference particle inputs, then learned inputs under the same scene and interval. Report force/torque RMSE, impulse over contact events, and sign/frame conventions.
6. Evaluate the fully coupled dynamic box for translation, rotation, stability, penetration, and force time series. Recorded-box playback is not this test. Include lateral/rotational motion and mid-trajectory disturbances when assessing robustness.
7. Profile warmed GPU stages separately: graph/features, network, gradient fitting, contact solve, rigid stepping. Also report synchronized end-to-end wall time per simulation step, scene size, active/allocated edges, iteration counts, and capture/compile settings. Exclude rendering from solver timing and report viewing cost separately.

Do not launch a full new campaign for a documentation or interface-only task. Scale validation to the requested change and label skipped or unavailable checks. Preserve previous results and record source revision plus local diff, checkpoint/configuration, data manifests, split IDs, random seeds, hardware, and commands for any new accuracy claim. Update these handoff notes when behavior changes.
