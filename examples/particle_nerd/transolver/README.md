# Particle Transolver

An experimental displacement-only Transolver variant for the existing particle
prediction dataset. The GNN and its checkpoints remain separate.

## Why this model is relevant

Transolver maps many material particles to a small set of learned latent groups
("slices"), applies attention between those groups, and maps the result back to
each particle. A slice is a soft grouping of learned features, not a spatial
region selector. All particles are updated.

For N particles and S slices, attention avoids an N-by-N particle matrix.
Particle-to-slice work grows with N times S, while slice attention grows with
S squared. This removes particle-neighbor construction from this variant.
It does not enforce contact, material constitutive laws, or conservation.

The [crash paper](https://arxiv.org/pdf/2510.15201) compares Transolver and
MeshGraphNet, including autoregressive rollout training. Its vehicle-impact
benchmark is not evidence of generalization to independently moving robot feet
and granular particles. The current
[crash example](https://github.com/mnabian/physicsnemo/tree/main/examples/structural_mechanics/crash)
also includes GeoTransolver. This implementation uses **standard Transolver
slice attention**, not GeoTransolver or Transolver++.

## Inputs and outputs

For each particle:

```text
particle embedding = Particle encoder(particle features)
                   + sum over colliders [distance weight * Collider encoder(pair features)]

particle embeddings -> Transolver blocks -> displacement [dx, dy, dz]
next position = current position + displacement
next velocity = displacement / timestep
```

| Input | Values | Units / meaning |
| --- | ---: | --- |
| Current particle position | 3 | World position relative to the fixed configured origin; divided by 1 m |
| Particle motion history | 6 | Two finite-difference velocities from three position frames; divided by 1 m/s |
| Material and sampling | 12 | Existing ten normalized material parameters, particle mass, particle radius |
| Timestep | 1 | dt / 0.02 s |
| Gravity | 3 | World gravity / 9.81 m/s² |
| Each particle–collider pair | 8 | Signed particle-surface gap / 0.07 m; world outward normal (3); collider surface velocity minus particle velocity (3), divided by 1 m/s; friction coefficient |

The signed gap is the collider signed distance minus the particle radius.
The weight is `max(0, 1 - max(gap, 0) / support)`, with default support 0.07 m.
Each pair passes through the same Collider encoder. Its weighted output is
summed into that particle's token **before slice attention**. Multiple nearby
colliders contribute simultaneously. Distant colliders have zero direct weight;
attention can still transmit their effect through other particles.

There are no learned IDs or semantic classes for ground, foot, or shovel.
Collider IDs belong in the simulation adapter to associate geometry with rigid
bodies. Reordering collider rows does not change the prediction. Static ground
has zero surface velocity; robot foot and object surface velocities include
rotation. This is prescribed *input conditioning* to the particle network;
it does not mean rigid bodies must be prescribed in the coupled simulation.

Current geometry supports any number of planes, oriented boxes, and spheres.
A box can approximate a robot foot. Robot mesh SDF queries and a G1 scene adapter
are **not implemented** here. Mesh/SDF geometry can use the same eight pair
features once its distance, normal, and surface velocity queries are supplied.
No robot policy, actuator, or force estimator is changed by this model addition.

## Model and training

Default: four blocks, width 128, four attention heads, 64 slices, GELU MLPs,
pre-normalization and residual connections. A shared collider MLP conditions
particle tokens; a three-value linear decoder outputs displacement in metres.
The decoder is initialized to zero and scaled by 0.02 m. The paper uses a larger
six-layer, width-256, eight-head, 128-slice configuration; CLI options support it.
The reference source revision and attribution are in [NOTICE.md](NOTICE.md).

The default is `--rollout-seconds 0.5`: 25 recurrent predictions at 50 Hz.
Initialize with reference position history, predict the next position, append
that prediction to the history, and repeat. Loss is evaluated at every predicted
step; there is no teacher reset within the sequence. Gradients pass through
predicted history and the Warp feature kernels. Collider states at each step
come from the recorded reference trajectory; no future collider state is passed
to an individual prediction.

Loss: particle position squared error / dt², with weight 5 for particle/time
pairs whose reference displacement from the sequence start exceeds 0.5 mm and
weight 1 otherwise. There is no velocity target. Validation selects checkpoints
using all-particle and moving-particle errors. A frozen-particle baseline is
reported so a model that misses material motion cannot be mistaken for success.
The original episode splits are preserved for a full experiment.

The crash paper's autoregressive scheme predicts acceleration and integrates it.
Our displacement output, collider conditioning, dataset, weighting, and
rollout length are deliberate adaptations, not a numerical reproduction of its
crash results. Full-step CUDA graph capture is not enabled.

## Commands

Run from the IsaacLab worktree root:

```bash
RUN=examples/particle_nerd/run.sh
"$RUN" -m particle_nerd.transolver.train \
  --data /path/to/dataset --output /path/to/transolver-model \
  --steps 10000 --rollout-seconds 0.5 --loss-mode balanced
"$RUN" -m particle_nerd.transolver.evaluate \
  --data /path/to/dataset --checkpoint /path/to/transolver-model/best.pt \
  --output /path/to/transolver-evaluation --mode all --steps 250
"$RUN" -m particle_nerd.transolver.profile \
  --checkpoint /path/to/transolver-model/best.pt --episode /path/to/episode.hdf5 \
  --output /path/to/transolver-timing.json
"$RUN" -m pytest examples/particle_nerd/tests/test_transolver.py -q
```

The shared displacement evaluator recognizes the checkpoint architecture and
uses the same 12 matched trajectories as the GNN study. It replays reference
box motion; it is not a closed-loop force-estimator evaluation.

[OSMO template](../osmo/transolver_training.yaml): stage/upload a new source
bundle using the [OSMO guide](../agent/osmo_workflows.md), then submit with the
existing dataset URL and a new output prefix. Parameters: `training_steps`,
`rollout_seconds`, `source_url`, `dataset_url`, `output_url`. No new PhysicsNeMo
package is needed. The template completed as `particle-transolver-1` on four L40S GPUs in `isaac-lab-l40s-03`.

## Half-second recurrent training and continuation

```bash
"$RUN" -m particle_nerd.transolver.train \
  --data /path/to/box-or-shovel-dataset --output /path/to/new-model \
  --initialize /path/to/matching-previous-model/best.pt \
  --steps 10000 --rollout-seconds 0.5 --loss-mode balanced --learning-rate 0.0001
```

The 0.5 s horizon contains 25 predictions at the dataset's 0.02 s interval.
All 25 predictions contribute to the loss. Gradients pass through all recurrent
steps; predicted particle history is never detached or reset during the sequence.
Recorded rigid-body pose and velocity condition each step. `run.json` records the
prediction interval, number of recurrent steps, horizon in seconds, and recurrence
rules. An incompatible fractional step count is rejected. `--rollout-steps` remains
an explicit legacy override and cannot be combined with `--rollout-seconds`.

Evaluate both previous and new checkpoints on the same unchanged validation/test
splits with `--all-episodes --steps 250 --mode all`. The evaluator records particle
position RMSE after 0.5 s and at the end of each rollout (up to 5 s), plus stationary
particle drift and the fixed-particle baseline. Checkpoint selection uses validation
errors only. These evaluations use recorded rigid-body motion, not estimated-force
feedback.

## Inference interface

`Predictor(saved, history, material, kinds, sizes, friction)` accepts:

- Particle history `[N, H, 3]`, normalized material vector `[13]`.
- Collider kinds `[B]` int32: plane=0, box=1, sphere=2. These select geometry
  queries; their numeric values are not network inputs.
- Sizes `[B,3]`: box half extents; sphere radius in column 0; ignored for planes.
- Friction `[B]`.

`step(poses, velocities)` accepts `[B,7]` world xyz + xyzw quaternion and `[B,6]`
world linear/angular velocities. Linear velocity must be measured at each
collider pose origin, not an unrelated robot link center of mass. A plane uses
its local z=0 surface and local +z normal. Poses must contain unit quaternions.
All tensors are contiguous on the same CUDA device, with float32 physical values.

The predictor exposes `x`, `v`, `dt`, and `device`. A convenience single-box
adapter accepts `[7]` / `[6]` and adds the fixed ground. The general constructor
supports variable N and B; the current dataset reader supplies only one box and
ground. Legacy coupled-play initialization does not yet construct this predictor.

## Verification and limitations

Focused tests cover simultaneous collider features, collider/particle permutation
symmetry, exclusion of distant colliders, geometry gradients against finite
differences, and GPU-only particle state updates. A six-update real-data training
check exercises four-step backpropagation and checkpoint reload. Those execution checks use
the existing temporary development dataset, separately from the full-cloud accuracy experiment.
Full training and matched particle evaluation completed; see the results below. Robot-contact accuracy is not established.

Internal material stress and plastic deformation history are still absent from
the inputs. A transformer does not remove that observability limitation. Learned
slices may also smooth narrow contact regions. Before using predicted particle
motion to estimate rigid-body forces, measure moving-particle rollout error,
penetration, and force sensitivity to displacement-derived velocity. The latter
is an interval-average velocity, not necessarily the MPM endpoint velocity.

## Completed cloud experiment

[Report and measurements](https://jnskkmhr.github.io/particle-nerd-plan/transolver.html).
OSMO run `particle-transolver-1`: four L40S GPUs, 10,000 optimizer updates,
four-step differentiable training, 776 training / 109 validation episodes.
Training took 810.7 s, including validation and checkpoint writes. The selected
checkpoint is update 10,000. Its four-step validation position RMSE is 0.885 mm.

All twelve matched five-second test rollouts completed without nonfinite values,
but **long-rollout particle drift is substantial**:

| Method | Mean final position RMSE | Reference displaced-particle RMSE |
| --- | ---: | ---: |
| Transolver, all particles | 5.07 cm | 7.83 cm |
| GNN v2 with CNN selection | 2.32 cm | 5.53 cm |
| Frozen particles | 2.28 cm | 5.39 cm |

Of 72,764 reference-stationary particle instances, 69,433 (95.4%) drift more than
1 cm at some point in the Transolver rollout. Reference-stationary means staying
within 1 mm of the reference initial position throughout five seconds. Displaced
particles move more than 1 cm between reference initial and final positions.
Metrics are trajectory means over the same twelve test IDs.

This is an implementation experiment, not an isolated attention ablation:
architecture, collider representation, particle selection, and training recurrence
differ from GNN v2. These position-error measurements do not establish per-step displacement or
force-estimator accuracy; those require separate evaluation. Do not interpret
low four-step validation error as long-rollout position accuracy.

Initial trained-checkpoint L40S timing: features 0.224 ms, network 2.113 ms,
complete particle step 2.668 ms (median synchronized wall time; 100 repetitions,
16,384 particles, two colliders, BF16 Torch / FP32 Warp, no compile/capture).
This excludes force estimation, rigid-body integration, and rendering. The dedicated matched benchmark `particle-matched-profile-1` measured 2.690 ms
for Transolver versus 11.084 ms for GNN v2 with learned selection on the same
L40S GPU (4.12× faster). This compares these implementations, including their
different selection/geometry processing, not an isolated architecture ablation.

Cloud outputs:
`swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/particle-gnn/runs/particle-transolver-1/`.
Checkpoint `model/best.pt`; configuration/splits `model/run.json`; twelve rollout
metrics `evaluation/evaluation.json`; component timings `profile.json`.

## Local implementation-check measurements

The 31 example tests passed. Six optimizer updates with four-step rollout loss,
checkpoint reload, and a 12-step recurrent execution check passed. These historical execution checks are separate from the full cloud experiment above.
The repository's scoped Python/style checks pass; its global changelog hook
still reports pre-existing fragment issues outside this example.

Warmed RTX 5090 timing, 16,384 particles, one box plus ground, 50 measurements,
BF16 neural operations and FP32 Warp geometry, without graph capture or compile:

| Operation | Median GPU time | Median synchronized wall time |
| --- | ---: | ---: |
| Particle/collider features | 0.133 ms | 0.141 ms |
| Transolver network | 1.448 ms | 1.460 ms |
| Complete particle step | 1.718 ms | 1.730 ms |

The complete particle step includes feature construction, network execution,
and position/velocity/history updates. It excludes force estimation, rigid-body
integration, and rendering. Timings use the short execution-check checkpoint;
they establish compute cost, not model accuracy or a controlled GNN comparison.

## Long-horizon experiment

The follow-up experiment uses `--rollout-steps 25 --loss-mode balanced`: 0.5 s
at the unchanged 0.02 s recording interval. It starts from scratch with the same
architecture, seed, dataset split, and 10,000 optimizer updates. All 25 predicted
states remain in the differentiation graph; collider states are recorded inputs.
This changes both the training horizon and loss, not the model architecture.

The baseline `weighted` loss remains the default for reproducibility. Both modes
use the original reference labels: a particle-time sample is moving when its
reference displacement from the sequence's initial position exceeds 0.5 mm.
`balanced` computes the mean squared position error separately for moving and
nearly stationary samples and averages the nonempty groups with equal weight.
The position error is divided by the fixed timestep before squaring, as in the
baseline. A group with no samples is omitted, not replaced by an artificial zero
error. Nearly stationary samples retain their reference targets; no particle is
frozen and no region selector is introduced.

Validation reports both groups separately. Balanced training selects checkpoints
by the square root of the average of pooled group mean squared position errors.
These 25-step validation numbers must not be compared directly to the old
four-step validation numbers. Final evaluation uses the same twelve five-second
trajectories; those previously examined trajectories are not a fresh independent
generalization test. A future controlled ablation would separate the horizon
change from the loss change.

Workflow `particle-transolver-long-1` runs a separate two-update, 25-step smoke
check before starting full training. Periodic model checkpoints are uploaded
every five minutes. The live report is
[transolver_long_horizon.html](https://jnskkmhr.github.io/particle-nerd-plan/transolver_long_horizon.html).
No accuracy improvement is asserted until the matched evaluation completes.


## Unseen box-path evaluation

The trajectory benchmark tests the two frozen checkpoints on six new
five-second reference simulations: a 0.76 m diameter circle, an S curve, and a
figure eight (the latter two span 0.76 × 0.52 m), each in both traversal directions.
A quintic time parameter gives zero target velocity and acceleration at the ends.
Dragging acts from 0.25 to 4.75 s. The box settles at the bed center. A 0.75 s
force-driven lead-in moves the target to the path start, followed by 3.75 s of
path traversal. The box remains dynamic in MPM–MJWarp: a
bounded spring–damper force acts at its center, and no target pose is assigned.
The actual box path can differ from the requested path; both are reported.

```bash
./examples/particle_nerd/run.sh -m particle_nerd.transolver.trajectory_benchmark \
  --output /path/to/trajectory-results \
  --checkpoint /path/to/25-step-model/best.pt \
  --baseline-checkpoint /path/to/4-step-model/best.pt
```

This collects references, runs both recurrent particle predictors for 250 steps,
and saves per-step displacement errors as well as accumulated position errors.
Both predictors receive recorded actual box states. This is a particle-motion
test, not a closed-loop force-estimator test. No retraining or region selection
is performed. Scene geometry and material parameters are unchanged; box mass is
7.5 kg, initial velocities are zero, and all six tests use seed 620000. Initial
box position is the bed center. The centered dragging force differs from
the off-center dragging anchors used in the training data.

Outputs include `reference/`, `long/`, `baseline/`, `collection.json`, and
`COMPLETE.json` (checkpoint hashes). `motion_metrics.json` defines a moving
sample by reference displacement greater than 0.1 mm in one 20 ms step.
Reference `drag_target_position` records the requested center position at each
physics step; compare this with `body_pose`, not with predicted particle motion.
Videos and path plots are published in the
[new-trajectory report](https://jnskkmhr.github.io/particle-nerd-plan/transolver_long_horizon.html#trajectories).

The preliminary OSMO run `particle-transolver-paths-2` placed boxes at path
starts before settling. Edge-start boxes slid outward; its circle/S actual paths
were incomplete. Those records are retained as a setup diagnostic. The corrected
six-case run uses a center start and smooth lead-in on the local RTX 5090, with
identical checkpoint SHA256 hashes. Do not merge the preliminary and corrected
trajectory metrics or call the preliminary paths successful circle/S tests.


## Airborne release evaluation

Use `data_collection.collect --case airborne --launch-velocity VX VY VZ` for a
five-second rollout beginning in the air. The box center starts at (0, 0, 0.75) m
above the rigid ground z=0. During two seconds of material preparation, an
upward external force balances box gravity. At release, that force is removed
and the specified initial linear velocity is assigned once; angular velocity is
zero. There are no subsequent external pushes or dragging forces. The same
open-ground scene, material, particle spacing, mass, and solver settings apply.

The five requested tests use velocity vectors (0,0,0), (1,0,0), (-1,0,0),
(0,1,0), and (0,-1,0) m/s, mass 7.5 kg, and seed 620000. The manifest records
`release_step`, `release_body_pose`, and `release_body_velocity`. For compatibility
with the existing evaluator, `settled_step` also points to this release boundary;
it does **not** mean the box settled. The initial three particle-position frames
come from the final 40 ms before release. Recurrent evaluation predicts the next
250 particle steps while reading actual recorded box states at each step.
The intentional initial velocity assignment is excluded from post-release
force-balance diagnostics. Videos use time since release, not time since settling.

Example:

```bash
./examples/particle_nerd/run.sh -m particle_nerd.data_collection.collect \
  --case airborne --launch-velocity 1 0 0 --duration 5 --body-mass 7.5 \
  --seed 620000 --name episode-11001 --output /path/to/airborne/reference
```

[Airborne results and videos](https://jnskkmhr.github.io/particle-nerd-plan/transolver_long_horizon.html#airborne)
use the frozen 25-step Transolver checkpoint, without training or region selection.


## Full dynamic box comparison

`closed_loop.py` connects the frozen Transolver to the analytical contact estimator
and MJWarp. Only the initial three particle frames and current box state come from
MPM. Thereafter the predicted particles and independently dynamic box feed back
into each other. No MPM solver is constructed on the learned side.

```bash
RUN=examples/particle_nerd/run.sh
"$RUN" -m particle_nerd.transolver.closed_loop \
  --reference /path/to/episode-10000.hdf5 \
  --checkpoint /path/to/best.pt \
  --path wide-circle --effective-mass-scale 300 --output /path/to/closed-loop-circle
# Use episode-10004.hdf5 and --path figure-eight for the figure-eight test.
```

Use reference episodes from the corrected center-start trajectory benchmark.
Both simulations apply the same bounded spring-damper drag controller at the box
center, evaluated against their own box position and velocity every 5 ms. The
reference uses 5 ms MPM updates; the learned simulation predicts particles and
reconstructs material forces every 20 ms, holding the latter for four MJWarp
substeps. The estimator uses the initial external force for its 20 ms free-motion
estimate. Targets are checked against the reference's external controller targets;
future reference body/particle states are read only for offline metrics.

Outputs: `rollout.hdf5`, `comparison.npz`, and `metrics.json`. The HDF5
`sampled_body_pose` and `body_velocity` contain each 20 ms state;
`body_pose` uses reference absolute indices only for rendering (only entries in
`frame_steps` are valid). Material and external forces are recorded separately.
The `render_rollout.py --dynamic-body` experiment renderer must be used, otherwise
the older particle-only renderer mode intentionally copies reference box motion.

The published closed-loop comparison now uses `--effective-mass-scale 300`.
Both five-second tests completed with converged contact solves. Box-position RMS
differences from MPM were 10.7 cm (circle) and 8.4 cm (figure eight); material-force
vector RMS differences were 81.7 N and 73.8 N. These compare different evolving
trajectories, not identical physical states. The old published ×10 videos and
measurements were replaced. The CLI default remains 10; use the explicit flag
above to reproduce the requested ×300 experiment.


## Controlled force-estimator input comparison

```bash
RUN=examples/particle_nerd/run.sh
"$RUN" -m particle_nerd.transolver.force_ablation \
  --reference /path/to/reference/episode-10000.hdf5 \
  --prediction /path/to/long/rollout-10000.hdf5 \
  --case wide-circle --output /path/to/force-ablation/wide-circle
```

Repeat with episode/rollout 10004 and `--case figure-eight`, or 11000 and
`--case vertical-drop`. Predictions must be complete five-second particle-only
rollouts conditioned on the exact reference box trajectory. The runner verifies
matching body poses, frame indices, and initial particle positions.

A uses MPM x(t) and v(t+dt); B uses MPM x(t) and displacement/dt; C uses recurrent
Transolver x(t) and displacement/dt. Each uses the same recorded body pose,
velocity, and external force at t. The estimator is unchanged. Primary targets
are raw MPM material reactions averaged over four 5 ms steps; torque is shifted
to the interval-start box COM. Applied coupler feedback is retained as a secondary
force reference. Outputs are `forces.npz` and `summary.json` with force/torque
errors, support means, impact impulse, and convergence statistics.

Initial results: pre-drag support with reference particle inputs is about 37.6 N
versus 73.6 N from MPM. Velocity differencing barely changes this deficit. This
identifies a baseline estimator/integration problem in this setup; it does not
justify attributing the closed-loop sinking solely to Transolver particle errors.
