# Shared object dynamics experiment

## Collecting the 1,000-trajectory training set

The training set has 1,000 native-MPM trajectories of 6 s at 200 Hz, all with the same sand bed and material:

| Part | Trajectories | Command module | Contents |
| --- | ---: | --- | --- |
| Box, sphere, cylinder with prescribed motion | 360 (120 per shape) | `learning.collect --mode prescribed` | Horizontal push, vertical intrusion, circling and holding; the pose is imposed |
| Box, sphere, cylinder as free bodies | 240 (80 per shape) | `learning.collect --mode dynamic` | Drops, slides, throws and tumbling releases; the body reacts to the sand |
| Free-falling sand blocks and columns | 400 | `learning.collect_freefall` | Unconfined particle clouds released onto flat ground; no rigid object |

Robot cases (Franka, Go2, G1) are **not** part of this set. The split is fixed by trajectory ID: 800 train, 100
validation, 100 test. `learning.verify` rejects missing, duplicate or mis-phased trajectories.

One command records the whole set (objects, then free-fall, then the check). From the repository root:

```bash
bash examples/particle_nerd_benchmark/learning/collect_dataset.sh --output /path/to/dataset --num_gpu 4
```

The same collection on OSMO, written directly to a private bucket:

```bash
bash examples/particle_nerd_benchmark/learning/osmo/collect_all.sh \
  --output swift://PRIVATE_BUCKET/particle-nerd-object-benchmark --pool POOL
```

The individual steps behind these commands follow. Each part runs in its own process on its own GPU:

```bash
export PYTHONPATH=$PWD/examples
OUT=/path/to/dataset            # about 135 GB; the object trajectories are about 220 MB each

# 600 object trajectories: six independent jobs, one GPU each
for shape in box sphere cylinder; do
  for mode in prescribed dynamic; do
    uv run --no-sync python -m particle_nerd_benchmark.learning.collect \
      --shape "$shape" --mode "$mode" --output "$OUT/$shape-$mode" --device cuda:0
  done
done

# 400 free-fall trajectories: sharded over the GPUs of one node
uv run --no-sync torchrun --standalone --nproc_per_node=4 \
  --module particle_nerd_benchmark.learning.collect_freefall --output "$OUT/freefall"

# Check the complete set and write its audit
uv run --no-sync python -m particle_nerd_benchmark.learning.verify --data "$OUT" --output "$OUT/data_audit.json"
```

Add `--limit N` to `learning.collect` for a quick local check of the first
`N` trajectories. Collection resumes: a complete trajectory with a matching manifest is skipped.

On OSMO the six object jobs run in parallel and the free-fall set uses four 4-GPU shards. `collect_all.sh` submits both;
to submit them separately:

```bash
OUT=swift://PRIVATE_BUCKET/particle-nerd-object-benchmark
bash examples/particle_nerd_benchmark/learning/osmo/run_collection.sh $OUT --pool POOL
bash examples/particle_nerd_benchmark/learning/osmo/run_collection.sh \
  --workflow examples/particle_nerd_benchmark/learning/osmo/collect_freefall.yaml $OUT --pool POOL
```

Training expects all seven folders (six object folders and the free-fall folder) under one data root. The training,
evaluation and fine-tuning workflows read the collected set from `data_url`; pass the bucket prefix that holds it with
`--set-string data_url=swift://YOUR_BUCKET/particle-nerd-object-benchmark`.

## Current Particle NeRD retraining

Use `osmo/train_nerd.yaml` for fresh Particle Transolver training. The existing
`osmo/train.yaml` is retained for the separate WorldParticle work.

The corrected NeRD path calls the same compact update as G1: automatic Top-K,
512 background samples, selection every four steps, Argon's displacement gate,
free-surface ballistics and tray projection. Primitive distances use the actual
box, sphere or cylinder geometry. The gate runs during training and evaluation.
Dynamic recordings provide beginning-of-step body states for neural inputs;
rewound proxy states remain the inputs to contact reconstruction.

New checkpoints record schema `object_particle_nerd_topk_v2` and the complete
pipeline. The old step-9,000 full-particle checkpoint does not meet this contract.
Its existing published results are historical and must not be presented as
results of this retraining. Evaluation rejects that old checkpoint rather than
silently switching its inference method.

The dataset remains 600 flat-sand object trajectories: 480 training, 60 validation
and 60 test. The new run starts from random network weights, with 12,000 optimizer
steps and a recurrent horizon that grows to 100 physics steps (0.5 s at 200 Hz).
The held-out evaluation uses the same corrected update. Its metrics and videos
must be regenerated from the new checkpoint before updating the website.


This experiment trains one Particle Transolver on Box, Sphere, and Cylinder interactions with the finite flat sand bed used by the benchmark gallery. Dataset collection uses native implicit MPM. Model evaluation replays recorded interacting-body inputs and advances the model's own particle predictions, without replacing them with reference positions after initialization.

## Data protocol

| Quantity | Setting |
|---|---|
| Complete trajectories | 600; 200 per shape |
| Prescribed motion | 360; 120 per shape |
| Dynamic motion | 240; 80 per shape |
| Train / validation / test | 480 / 60 / 60 complete trajectories |
| Recording interval | 0.005 s, native 200 Hz |
| Trajectory duration | 6 s, including initial settling |
| Sand bed | 2.0 × 1.2 × 0.30 m; nominal fill level at z = 0 (actual particle centers are recorded) |
| MPM voxel / particle spacing | 0.04 / 0.032 m |
| Particles per trajectory | 20,646, with persistent particle IDs |
| Particle initialization | Uniform coordinate jitter up to 0.0064 m, truncated at the initial bed bounds; independently seeded per episode |
| Object dimensions | Box: 0.24 × 0.18 × 0.12 m; Sphere: radius 0.10 m; Cylinder: radius 0.10 m, height 0.12 m, local z axis |
| Object mass | 2 kg for dynamic bodies; prescribed bodies have zero native solver mass/inertia |
| Sand density / material friction | 1,600 kg/m³ / 0.48 |
| Collider friction | 1.0, as imported by the existing benchmark; stored per shape |
| Sand Young's modulus / yield pressure | 10¹⁵ Pa / 10¹⁵ Pa |

The other material parameters, tray geometry, coupling, and solver settings remain those of `core/physics.py` and `core/terrain.py`. Each manifest stores the resolved configuration and source hashes. The initial particle state is deliberately included: the model is not restricted to recordings that begin after settling.

Prescribed cases contain horizontal pushes, vertical intrusion, circular motion, and a stationary embedded object. Their pose and velocity are imposed; sand forces cannot deflect the object. Dynamic cases contain drops, slides, throws, and tumbling releases. Gravity is compensated for the first second while sand settles. After release, the body evolves freely under gravity and contact forces, with no further pose overwrites. Motion parameters and particle jitter differ between episodes. Randomization does not change material properties, geometry dimensions, particle spacing, or timestep.

Trajectory IDs determine the split before simulation. Entire trajectories belong to exactly one split. No frames from a held-out trajectory enter training.

## Stored quantities

Each training HDF5 file contains particle positions, rigid-body states, and static metadata. It contains no recorded particle velocities, forces, deformation tensors, or policy actions. Particle velocities used by the model are finite differences of positions.

`particle_position[t]` and `body_state[t]` are states after t completed physics steps. Body state contains position, quaternion in xyzw order, and world-frame linear/angular velocity. `collider_state[t]` records the rigid proxy state at entry to the MPM step from t to t+1, after the coupler removes lagged material feedback. This separately recorded rigid-body state avoids substituting an incorrectly timed collider pose or velocity. Linear velocity is measured at the body center of mass, coincident with the shape origin for these objects.

The five static tray boxes are stored in the manifest. Native shape parameters and friction, particle mass/radius, and body mass/inertia are static datasets. Coordinates use world axes with the origin at the initial tray center. Native kinematic-body mass is zero; the configured nominal mass is also recorded to avoid confusing these cases with freely moving 2 kg bodies.

The dataset and model checkpoints are private. Public reports may contain only the permitted website, videos, plots, and their provenance, not trajectory or checkpoint downloads.

## Collection

The editable benchmark package must be installed, or set `PYTHONPATH=$PWD/examples` when invoking the modules from the repository root.

```bash
uv run --no-sync python -m particle_nerd_benchmark.learning.collect \
  --shape box --mode prescribed --output /path/to/private/box-prescribed
```

Repeat for `sphere` and `cylinder`, and for `--mode dynamic`. The six-task OSMO launcher snapshots source and preserves the lockfile:

```bash
bash examples/particle_nerd_benchmark/learning/osmo/run_collection.sh \
  swift://PRIVATE_BUCKET/particle-nerd-object-benchmark \
  --pool isaac-lab-l40s-03
```

A separate evaluation-reference option records native MPM material forces. These reference files are not part of the state-only training set:

```bash
uv run --no-sync python -m particle_nerd_benchmark.learning.collect \
  --shape box --mode prescribed --reference horizontal-flat --output /path/to/reference/horizontal-flat
uv run --no-sync python -m particle_nerd_benchmark.learning.collect \
  --shape box --mode dynamic --reference static-support --output /path/to/reference/static-support
```

The first reproduces the nominal six-second horizontal push. The second records 20 seconds of free-body support after release, exposing delayed material or support drift. Force samples are the MPM material-on-object impulse divided by 0.005 s; direct contacts with the rigid tray are excluded.

## Training and evaluation

`learning/train.py` uses the shared Particle Transolver architecture from `contrib/legacy/particle_nerd/network/model.py`. The current method selects Top-K active particles and 512 background samples, then applies the shared G1 blending, displacement stabilization and tray projection. There is no force filter. The feature encoder includes the interacting object and all five actual tray boxes. Cylinder distance features use an analytic capped cylinder along local z; plane, box, sphere, and grid-SDF encodings retain their existing definitions.

`learning/network.py` loads those same production source files in a private package namespace. This avoids the eager Gym task registration in `isaaclab_tasks.__init__` and dependencies belonging to unrelated RL environments. The model and feature implementations are neither copied nor modified by this loader. Distributed training allows 30 minutes for synchronization because nonzero ranks wait while rank zero evaluates the validation windows.

Training windows are recurrent: predicted positions replace the oldest history frame, and gradients flow through subsequent predicted states. The loss gives equal weight to nonempty moving and stationary reference groups; moving means cumulative reference displacement greater than 0.5 mm within the window. Checkpoint selection uses only fixed validation windows, stratified across shape and motion mode. A frozen-particle baseline is recorded. Test trajectories are reserved for final evaluation. The planned run uses 12,000 optimizer steps on four GPUs, one independently sampled trajectory window per GPU. The recurrent horizon increases through 4, 8, 16, 32, 64, and 100 native physics steps (a maximum of 0.5 s). The learning rate decreases from 0.0002 to 0.00002 with a cosine schedule. Checkpoint selection uses two 200-step (1 s) recurrent windows per selected validation trajectory: initial settling and a window covering the object interaction. Horizontal pushes and circular motion start at 3 s, vertical motion at 2 s, holding at 5 s, and dynamic releases at 1 s. The saved configuration contains the actual training duration, horizons, split counts, dataset-manifest hashes, and model/feature source hashes.

`learning/metrics.py` follows the benchmark repository's `metrics/README.md`: RMS Euclidean particle error without division by three; normalized physical-mass deposition; reference-mass-weighted velocity error and coverage; area-sampled symmetric point-to-mesh surface distances; and raw signed Fx/Fy/Fz errors. Surface reconstruction uses the same normalized density kernel and grid for both methods. Evaluation records all thresholds, grid settings, seeds, failures, and checkpoint/data hashes. Videos must render the same saved predicted/reference states as their paired plots.

After all six collection tasks complete, submit training against their common private output directory:

```bash
bash examples/particle_nerd_benchmark/learning/osmo/run_collection.sh \
  --workflow examples/particle_nerd_benchmark/learning/osmo/train_nerd.yaml \
  swift://PRIVATE_BUCKET/particle-nerd-object-benchmark \
  --pool isaac-lab-l40s-03 \
  --set-string data_url=swift://PRIVATE_BUCKET/particle-nerd-object-benchmark/COLLECTION_WORKFLOW/
```

Before optimization starts, the job verifies all 600 episode IDs against the fixed protocol, checks complete state arrays and manifest hashes, and hashes every trajectory file. The expected split is 480 training, 60 validation, and 60 test trajectories. A partial collection cannot silently become the training dataset. After optimization, the job evaluates the selected checkpoint on all 60 test trajectories, split across the four GPUs. Each rank writes separate episode directories and a summary; no test result influences checkpoint selection. Surface reconstruction uses scikit-image 0.27.0rc1, and the evaluator records its actual dependency versions.

Evaluate the selected checkpoint on test trajectories or on a separate evaluation-reference directory:

```bash
uv run --no-sync --with scikit-image==0.27.0rc1 python -m particle_nerd_benchmark.learning.evaluate \
  --data /path/to/private/data --checkpoint /path/to/model/best.pt \
  --output /path/to/evaluation
uv run --no-sync python -m particle_nerd_benchmark.learning.plot \
  /path/to/evaluation/EPISODE_ID
uv run --no-sync python -m particle_nerd_benchmark.learning.video \
  /path/to/evaluation/EPISODE_ID
```

The evaluator also records the mass of particle centers inside the interacting object and mass beneath it. The lower region is the horizontal rectangle enclosing the body's world x/y projection, from 0 to 0.10 m below its lowest world z, clipped above the rigid tray floor. It does not rotate away from gravity when a sphere rolls. Local mass-difference maps sum the shared grid along z or y and report kilograms per projected grid column. Surface error uses a 0.02 m grid, with a 0.03 m reconstruction for sensitivity analysis. Position and force errors are sampled at 200 Hz, field metrics at 10 Hz, surface metrics at 5 Hz, and mass maps at 1 Hz. Videos render saved states at 50 fps; they do not run another simulation.

The current object experiment shares the deployed G1 Top-K update and stabilization, adapted to primitive collider geometry. Test errors are accumulated recurrent errors, not one-step prediction errors. Reference-particle force reconstruction is reported separately from force reconstruction using recurrent predictions; neither comparison lets the body respond to the reconstructed force. The recurrent evaluation replays the recorded body motion to isolate particle prediction. The separate `learning/closed_loop.py` evaluation lets the object respond to NeRD material force and torque, as described below.

`learning/summarize.py EVALUATION_DIR --expected 60 --protocol-test` combines the held-out results only after every expected test episode is present. It rejects mixed checkpoints or mixed network/evaluation implementations. Time means are computed within each trajectory before equally weighted aggregation, with sample standard deviations across trajectories. Complete rollouts and available-duration measurements including failed rollouts are reported separately. Missing regions and reconstruction failures remain unavailable rather than becoming zero error. Force references additionally include one-second world-frame Fx/Fy/Fz means and frame-level standard deviations; these describe temporal variability, not uncertainty across independent trajectories.

Plot generation writes `plot_provenance.json`, binding each figure to its evaluation metadata, measured arrays, and predicted states. Video generation records the same prediction/checkpoint/reference hashes and exact frame indices in `video_provenance.json`. The benchmark repository's `plots/build_object_report.py` checks these bindings before exporting the shared-model report and four detailed case pages. Its `plots/OBJECT_EVALUATION.md` documents staging and publication. Raw state arrays and checkpoints remain private.


## Fully dynamic object and particle evaluation

`learning/closed_loop.py` initializes from one dynamic drop-and-support MPM
reference and then evolves its own object and particles. Only the initial states
and fixed scene parameters are used. Future reference poses, velocities, forces,
and particles are never supplied to the simulated NeRD system.

```bash
PYTHONPATH=examples uv run --no-sync python -m particle_nerd_benchmark.learning.closed_loop \
  --reference /path/to/private/box-static-support/box-static-support.h5 \
  --checkpoint /path/to/private/model/best.pt \
  --output /path/to/new/box-dynamic
```

Use the corresponding sphere/cylinder references for those geometries. Each
reference directory must contain its original `manifest.json`. The object-model
checkpoint must use schema `object_particle_nerd_topk_v2`. The previously published
step-9,000 comparison used the old raw model and must be regenerated with the new
checkpoint. Implementation hashes are saved for every run.

The object receives an upward force balancing gravity until 1 s, then is released.
Two MJWarp substeps of 0.0025 s advance each 0.005 s coupling interval. The
previous interval's estimated material force and center-of-mass torque are
applied to the dynamic object. Its beginning-of-step pose and velocity condition
compact neural inference. The force estimator separately uses the rewound free
velocity. Physical masses, inertia, collision geometry and friction come from
the reference. The current G1 contact solver uses normal/tangent/horizontal/upward
mass factors 100/40/8/2, normal exponent 3, stabilization 0.01 and 200 contact
iterations. No force filtering or native MPM solve is used.


`prediction.h5` records every particle/body state, raw material wrench, applied
lagged wrench, external wrench, and contact convergence statistics.
`--disable-feedback` is a diagnostic that omits the material wrench from the
rigid integrator; it must not be presented as the coupled result.

The benchmark website's `plots/render_object_motion_video.py` renders these
saved states beside the reference at 50 fps. `plots/plot_object_closed_loop.py`
exports the corresponding object-position and Fx/Fy/Fz curves. Their hashes bind
the figures and video to the same recordings. Full trajectories and checkpoints
remain private.
