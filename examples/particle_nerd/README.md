# Particle NeRD

This is the maintained source directory for the current particle-motion GNN,
activity selector, analytical contact-force estimator, and IsaacLab data collector.
Training and inference use Torch and Warp. Only data collection imports IsaacLab's
manager-based environment. The learned runner uses Newton MJWarp for the rigid box;
it does not construct an MPM solver.

For teammates and coding agents, start with the [agent handoff](agent/README.md)
for implementation context, architecture replacement instructions, and validation boundaries.

## Source layout

| Directory | Responsibility | Main entry points |
| --- | --- | --- |
| `data_collection/` | Dynamic MPM–MJWarp recordings, scene configuration, manager terms, episode reader | `collect.py`, `campaign.py`, `dataset.py` |
| `gnn/` | Particle GNN, GPU features, training, optional activity selection, particle inference and evaluation | `model.py`, `train.py`, `inference.py`, `region_selection.py`, `play.py` |
| `force_estimation/` | GPU velocity-gradient fitting and analytical contact impulse solve | `estimator.py`, `kernels.py`, `evaluate.py` |
| `runner/` | Connect the current GNN to force estimation and dynamic rigid-body integration | `simulation.py`, `rigid.py`, `play.py` |
| `osmo/` | Package this worktree and run collection/training in OSMO | `stage.py`, `collection.yaml`, `training.yaml`, `region_training.yaml` |
| `tests/` | Contact mechanics, geometry, rigid motion and GPU integration checks | `test_gpu_interfaces.py` and retained reference tests |

`force_estimation/body_contact_reconstruction.py` and its geometry/optimization
helpers are the offline numerical reference. They may use NumPy, SciPy and Numba.
The full runner uses `ContactEstimator`, whose simulation-step operations use Warp
on the GPU. `NewtonRigidStepper.step()` is also an offline reference interface;
`CoupledSimulation.step()` uses captured MJWarp calls without its CPU snapshots.

## Interfaces and step order

- `ParticlePredictor.step(body_pose, body_velocity)` takes current box pose and
  velocity as CUDA tensors. It updates particle position, velocity and velocity
  history. `SelectedPredictor` has the same interface and freezes inactive particles.
- `ContactEstimator.estimate(position, next_velocity, body_pose, body_velocity,
  external_wrench)` takes Warp GPU arrays: current particle positions `[N]`,
  predicted next particle velocities `[N]`, and the current box state `[1]`.
  It returns a `[1]` material-on-box wrench: world force in N followed by world
  torque in N m about the box center of mass. Known external forces are supplied
  separately; gravity is included internally in the free-motion calculation.
- `CoupledSimulation.step(external_wrench=None)` predicts particles, estimates the
  material wrench, then advances the dynamic box with that wrench and any known
  external wrench. Four 5 ms rigid steps cover one 20 ms GNN step. The force is held
  constant during those four steps. Predicted particle velocities already include
  contact, so the runner does not apply a second contact impulse to particles.

This is explicit coupling: it does not iterate between the GNN and the rigid
solver. The current-model connection is new and experimental; short smoke tests
are not evidence of accurate long-horizon coupled simulation. Only MJWarp is graph
captured. The combined Torch/Warp step has not been graph captured. Initialization,
file I/O, diagnostics and viewing may access the CPU; simulation-step state remains
on the GPU.

The estimator retains the existing approximations: effective particle mass scale
10, velocity-fit radius 0.065 m, stabilization 0.01, tolerance
1e-9 and a maximum of 2000 contact solver iterations with early termination. The
current-format runner/evaluator use the collector's box contact friction of 0.6
(recorded as `body_contact_friction` in new files; older recordings use 0.6). These
are contact-estimator settings, separate from the collector's 25 MPM solver
iterations. They have not been calibrated for the current 50 Hz coupled runner.
Ground contact on the box is handled by MJWarp, but it is not included in the
estimator's free-motion calculation; simultaneous box–ground and box–sand contact
is therefore a limitation of this interface.

## Run locally

### Set up the command paths (run once)

```bash
cd ~/isaac/IsaacLab-particle-nerd
RUN=examples/particle_nerd/run.sh
RESULTS="$HOME/isaac/particle_nerd/particle_nerd_plan/results"
```

The launcher puts this worktree's source and examples on `PYTHONPATH`. It uses this
worktree's `.venv`, or the parent checkout's `.venv` for a Git worktree. Set
`PARTICLE_NERD_PYTHON` to select another tested IsaacLab interpreter. It runs through
`uv` with the existing NumPy/USD/HDF5 overlay; it does not modify the shared venv.
Use module entry points, not direct paths to files inside subdirectories.

### Collect one MPM simulation episode

```bash
HEADLESS=1 "$RUN" -m particle_nerd.data_collection.collect \
  --case continuous-drag --body-mass 7.5 --duration 5 \
  --randomize-motion --visualizer none --output /path/to/previews
```

### Collect a dataset of 1,000 MPM episodes

```bash
HEADLESS=1 "$RUN" -m particle_nerd.data_collection.campaign \
  --episodes 1000 --output /path/to/dataset
```

The current collection setup is open ground at z=0, an initial 1 x 1 x 0.5 m sand
volume, 0.04 m MPM grid spacing and a 0.12 x 0.12 x 0.08 m box. There is no enclosing
tray. Campaigns sample box mass from 5–10 kg and use rest, sustained push and dragging
cases. The solver runs at 200 Hz; particle states are recorded at 50 Hz. Each accepted
trajectory contains five seconds after settling. The scene has 16,384 particles,
with initial position jitter. The current campaign starts one environment per
process; OSMO distributes campaign shards across workers.

### Train the particle-motion GNN

```bash
"$RUN" -m particle_nerd.gnn.train --data /path/to/dataset \
  --output /path/to/model --steps 10000
```

The dataset retains its existing HDF5 format and `.campaign.json` train/validation/
test assignments. Checkpoint formats are unchanged. Model parameters and optimizer
behavior are unchanged by the move. The current GNN encodes one fixed-size box and
a ground plane. Variable box geometry is not supported by this checkpoint. The
activity selector suppresses updates; it currently does not reduce GNN graph work.

### Inspect activity-selector training options

```bash
"$RUN" -m particle_nerd.gnn.train_selector --help
```

### View GNN + region selector with recorded box motion (website setup)

The box follows the MPM recording; mouse dragging does not change its trajectory.

```bash
"$RUN" -m particle_nerd.gnn.play \
  --checkpoint "$RESULTS/particle_gnn_full_training/model/best.pt" \
  --selector "$RESULTS/particle_gnn_region_full/selected.pt" \
  --episode "$RESULTS/particle_gnn_region_full/reference/episode-00117-attempt-0.hdf5"
```

### Run interactive GNN + force estimator with a dynamic box (experimental)

Drag the box with the mouse. No MPM solver runs; the episode supplies initialization only.
This coupled setup can exhibit excessive sinking and is not the website evaluation.

```bash
"$RUN" -m particle_nerd.runner.play \
  --checkpoint "$RESULTS/particle_gnn_full_training/model/best.pt" \
  --selector "$RESULTS/particle_gnn_region_full/selected.pt" \
  --episode "$RESULTS/particle_gnn_region_full/reference/episode-00117-attempt-0.hdf5"
```

The full runner reads only the settled initial state and particle velocity history.
The box then moves dynamically. The viewer's mouse interaction applies an external
wrench to the box. Omit `--selector` to use the motion GNN alone. Add
`--headless --steps 5 --output /tmp/coupled.json` for a short execution check.

### Evaluate the force estimator using recorded MPM particle motion

```bash
"$RUN" -m particle_nerd.force_estimation.evaluate \
  --episode /path/to/episode.hdf5 --output /path/to/forces.json
```

This reports force and torque RMSE against the MPM wrench averaged over each 20 ms
particle recording interval. It is not an evaluation of learned particle motion.

## OSMO

Follow the [OSMO workflow instructions](agent/osmo_workflows.md) for source
packaging/upload, distributed data collection, training, evaluation, and results.

| Operation | Workflow | Output |
| --- | --- | --- |
| Data collection | `osmo/collection.yaml` | Sharded reference episodes and train/validation/test assignments |
| Training + automatic evaluation | `osmo/training.yaml` | Model checkpoints, training metrics, and particle rollout evaluation |
| Evaluate an existing checkpoint | `osmo/evaluation.yaml` | Particle rollout metrics and trajectories, without retraining |

`"$RUN" -m particle_nerd.osmo.stage --output /path/to/new-source-bundle` only
creates the local source archive and checksum. Upload that bundle before submitting
any workflow. Regenerate it after source changes. Evaluation uses recorded box
motion to assess particle prediction; it does not measure fully coupled box motion.

## Verification and migration

```bash
"$RUN" -m pytest examples/particle_nerd/tests -q
"$RUN" -m particle_nerd.gnn.verify --output /tmp/gnn-check.json
"$RUN" -m particle_nerd.gnn.verify_selection --output /tmp/selector-check.json
```

The former flat example and its earlier model/checkpoint generation were archived
before cleanup. They are not imported by the current runner. The external
`experiments/particle_gnn` core files are compatibility links to this directory;
website reporting scripts, results, datasets and checkpoints remain outside the
source tree. `gpu_kernels.py` is a small compatibility import for the separate G1
proxy experiment. See [MIGRATION.md](MIGRATION.md) for the original source locations
and verification record.
