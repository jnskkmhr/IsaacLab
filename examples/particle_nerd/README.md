# Particle NeRD: MPM reference data and force reconstruction

This example implements the data collection and force reconstruction experiments
for milestones 1 and 2. It uses IsaacLab's `junnosuke/mpm_rl_task` branch and the
Newton version installed in that checkout's `.venv`.

## Code organization

| File | Responsibility |
| --- | --- |
| `scene_cfg.py` | Define the sand material, initial particle distribution, and rigid box geometry. |
| `env_cfg.py` | Configure parallel environments, manager terms, and MJWarp–MPM two-way coupling. |
| `terms/events.py` | Reset particle positions, velocities, body poses, and MPM solver history. |
| `terms/recordings.py` | Record particle states, body states, actions, and reference contact impulses. |
| `newton_access.py` | Map Newton indices to particle IDs and body `object_id`; aggregate contact impulses. |
| `collect.py` | Generate independent trajectories and export episodes through IsaacLab's recorder manager. |
| `dataset.py` | Check episode alignment, coordinates, identifiers, finite values, and momentum balance. |
| `reconstruction.py` | Estimate body reactions from particle motion and an explicit internal-force assumption. |
| `evaluate_forces.py` | Compare reconstructed forces and torques against recorded MPM measurements. |

No IsaacLab framework changes are required. The experimental particle trainer and
analytical-force rollout evaluator are described below; they are not yet a validated simulator.

## Milestone 1: collect reference motion

The scene contains 500 material particles in a 32 cm by 32 cm by 16 cm sand bed.
Six rigid boxes represent the ground beneath the sand, four retaining walls,
and a moving box. The initial sand surface is at environment-relative `z = 0`.
Every body has geometry, pose, velocity, friction, and an `object_id`.
There are no ground/tool semantic labels in the data representation.

The 2 kg box is dynamic. At reset, sample its center within x/y = ±2.5 cm
and z = 13–20 cm above the initial sand surface, roll/pitch = ±0.35 rad, and
yaw = ±pi. Sample initial linear velocity x/y = ±0.1 m/s, z = -0.2–0 m/s,
and angular velocity = ±0.5 rad/s per axis. No actions are applied after reset.

MJWarp integrates the rigid bodies. Newton implicit MPM integrates the sand.
A lagged proxy coupler passes rigid geometry and motion to MPM and returns sand
reactions to MJWarp. The ground and four retaining walls remain kinematic;
the box responds to gravity, material contact, and any direct rigid contacts.

MPM uses APIC transfer, a sparse grid, a 4 cm voxel size, 25 iterations maximum,
and tolerance 1e-4. Physics and recording timesteps are 5 ms. Both subsolvers
use one substep and the coupler uses one pass. Forward collider velocity follows
the existing G1 coupled configuration. CUDA graph capture is disabled.

**Earlier results used a prescribed box trajectory.** Those files remain
schema version 1; they do not validate the response of a freely moving box.
The corrected dynamic collector introduced schema version 2. Version 3 additionally
records box/sphere/cylinder geometry and static/dynamic physical roles.

From the IsaacLab repository root, using its existing environment:

```bash
uv run --no-project --python .venv/bin/python python examples/particle_nerd/collect.py \
  --output /tmp/particle_nerd/train.hdf5 \
  --num_envs 4 --episodes_per_env 2 --steps 400 --seed 42 --split train

uv run --no-project --python .venv/bin/python python examples/particle_nerd/collect.py \
  --output /tmp/particle_nerd/validation.hdf5 \
  --num_envs 2 --episodes_per_env 2 --steps 400 --seed 137 --split validation
```

The default launch has no viewer. Existing output files are rejected.
Choose separate seeds and files for training, validation, and test trajectories;
never split adjacent frames of the same episode between these sets. Collection
buffers one episode per environment before export, so memory use grows with
particle count, environment count, and episode length.

### Dataset contract

Each HDF5 episode lives under `data/demo_<id>`:

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `states/particle_position` | `(T+1, N, 3)` | Material particle positions [m]. |
| `states/particle_velocity` | `(T+1, N, 3)` | Material particle velocities [m/s]. |
| `states/body_pose` | `(T+1, B, 7)` | Body center and quaternion in `xyzw` order. |
| `states/body_velocity` | `(T+1, B, 6)` | Body linear velocity followed by angular velocity. |
| `actions` | `(T, 0)` | No actuator or trajectory commands. |
| `reference/body_impulse` | `(T, B, 6)` | Material-on-body linear impulse [N s] and torque impulse [N m s]. |
| `reference/applied_body_impulse` | `(T, B, 6)` | Previous MPM feedback applied during this rigid-body transition. |
| `reference/next_body_impulse` | `(T, B, 6)` | New MPM feedback available for the next rigid-body transition. |
| `reference/rigid_contact_impulse` | `(T, B, 6)` | Direct rigid contact impulses during this transition. |
| `constants/*` | `(1, ...)` | Particle masses, radii, IDs; body sizes, friction, IDs, masses, inertia, COM offsets; environment origin and timestep. |

Positions are relative to each environment's origin. Vector components use
world-aligned axes. Torque is measured about each box's center, which is also its
center of mass. `reference/body_impulse[t]` belongs to the transition from
`states[t]` to `states[t+1]`. Divide the impulse by `dt` to obtain the average force
or torque over that step. Reference impulses are evaluation targets, not inputs
to reconstruction or future force-network labels.

For the dynamic box, validate `mass * (v[t+1] - v[t])` against the sum of
`mass * gravity * dt`, `applied_body_impulse[t]`, and `rigid_contact_impulse[t]`.
Do not compare the box acceleration to the newly computed MPM reaction without
accounting for the lag. Future body acceleration is an evaluation reference,
not an input to forward force reconstruction. The direct rigid-contact recorder
currently requires one rigid substep and zero local COM offsets, as configured.

The file attribute `particle_nerd_manifest` records the split, seed, material
parameters, solver settings, IsaacLab revision and dirty-file list, and installed
Newton/Warp/PyTorch versions and available installation provenance. A missing
completion manifest indicates an interrupted collection.

## Milestone 2: original particle-force baseline

For each recorded transition, the evaluator:

1. Computes material particle acceleration from successive velocities.
2. Connects nearby material particles with equal-and-opposite central forces.
3. Finds candidate contacts with rigid boxes from their geometry.
4. Solves particle force balance with compressive internal forces and contact
   forces inside a four-sided approximation of the Coulomb friction cone.
5. Sums the opposite contact forces and their moments on each body.
6. Compares those body reactions with the MPM impulses divided by the timestep.

The force balance is written in plain text as:

```text
sum(internal particle forces) + sum(body-on-particle contact forces)
    = particle_mass * ((velocity_next - velocity_current) / dt - gravity)
```

A quadratic penalty selects smaller force coefficients when several solutions
satisfy the same motion. This is an **experimental material-force assumption**.
It is not Newton's granular material law, the cable method's rod law, or a full
PBD solver. The contact distance is a candidate-contact shell; it is not an exact
MPM contact reconstruction. The experiment does not enforce a sliding-friction
law, evolve material stress history, or model APIC particle angular momentum.

Particle motion does not uniquely determine the distribution of reaction forces
among several bodies. A small force-balance residual is therefore insufficient:
per-body force error and runtime must also be measured. The reference impulses
are read only after reconstruction has produced its estimate.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --no-project --python .venv/bin/python python examples/particle_nerd/evaluate_forces.py \
  /tmp/particle_nerd/validation.hdf5 \
  --output /tmp/particle_nerd/forces.json --start_step 99 --stride 100

uv run --no-project --python .venv/bin/python python -m pytest \
  examples/particle_nerd/tests/test_reconstruction.py -q
```

The JSON report contains per-body force and torque RMSE, force-balance residuals,
solver convergence, reconstruction time, and the measured and estimated reactions.
The inverse least-squares iteration limit is separate from Newton's 25 MPM
iterations. This CPU evaluator is intended for mechanics experiments, not online
coupling.

## Dynamic-body contact reconstruction

`body_contact_reconstruction.py` estimates a dynamic box reaction from current
box state and predicted next material velocities. It does not run MPM and does
not read future box acceleration. The original force-only evaluations substitute recorded
MPM particle motion. The learned-motion experiments below instead supply GNN
predictions and feed the estimated reactions back into rigid-body integration.

The calculation is:

1. Advance the current box velocity under gravity and the gyroscopic term.
2. Find particle contacts with the oriented box. Optionally estimate particle
   velocity at the box surface using a local affine velocity fit.
3. Solve unilateral contact conditions for the box velocity. A finite effective
   material mass allows relative normal motion. A small penetration correction
   requests a separating velocity when a particle lies inside the box.
4. Add Coulomb friction through projected contact iterations, when enabled.
5. Convert the box momentum correction into force and torque. Return the
   opposite impulse on each contacting particle for momentum checks.

The effective contact mass and penetration coefficient are calibrated scalar
approximations. They are not individual grain masses or the MPM material law.
The frictional solver minimizes tangential kinetic correction inside a circular
Coulomb disk. Its sliding impulse opposes the final relative contact velocity,
including for off-center contact with anisotropic body inertia.

An optional history variant applies compliance to changes from the preceding
contact impulse. It starts from zero estimated impulse, advances history at
every step, and discards separated contacts. It never initializes from recorded
forces. `history_weight=0` retains the memoryless baseline; experiments with
nonzero history are reported separately.

The implementation handles one dynamic box, sphere, or cylinder. Static ground and wall
reactions are not required to advance those fixed bodies, but their forces are
included when checking the complete material momentum equations. This is a
physical distinction between dynamic and fixed geometry, not an object category
such as shovel or foot. Material properties affect reference particle motion;
this contact approximation does not independently reconstruct granular stress.

### Files and evaluation protocol

| File | Task |
| --- | --- |
| `body_contact_reconstruction.py` | Compute body reactions from material velocity and rigid-body contact conditions. |
| `contact_iterations.py` | Solve normal contact and Coulomb friction with effective material compliance. |
| `evaluate_body_contact.py` | Fit scalar contact parameters on four complete episodes, validate on four other episodes, then evaluate a separate file. |
| `score_frozen_contacts.py` | Evaluate previously selected parameters without fitting; report episode-bootstrap intervals and direct rigid-contact subsets. |
| `replay_body_motion.py` | Integrate box motion with estimated forces while retaining recorded particle trajectories. |
| `check_particle_balance.py` | Check whether predicted box loads can coexist with particle force balance and static-body reactions. |
| `check_contact_sensitivity.py` | Measure force changes under particle-position/velocity perturbations and contact-iteration limits. |
| `network_contact_reconstruction.py` | Experimental joint velocity projection through normal particle-particle and particle-body contacts; also measures changes to particle motion. |
| `pressure_reconstruction.py` | Experimental inverse isotropic-pressure closure; preliminary accuracy was inadequate. |

`check_particle_balance.py` uses the original central-force approximation for
internal forces. A compatible solution does not establish unique forces or
validate that material law. Alternative box loads can also satisfy particle
force balance by changing static-body reactions. Additional contact equations
are therefore essential.

### Numerical environment

Use a separate Python 3.12 environment for offline analysis. The original
simulation environment has a NumPy version incompatible with the installed
Numba; do not modify that environment to run the contact iterations.

```bash
uv venv /tmp/particle_nerd_force/.venv --python 3.12
uv pip install --python /tmp/particle_nerd_force/.venv/bin/python \
  -r examples/particle_nerd/requirements-force.txt
```

Set `FORCE_PYTHON=/tmp/particle_nerd_force/.venv/bin/python` in the commands below.
Collection continues to use the existing IsaacLab simulation environment.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --no-project --python "$FORCE_PYTHON" python examples/particle_nerd/evaluate_body_contact.py \
  /path/to/calibration.hdf5 /path/to/development_evaluation.hdf5 \
  --frictional --affine --dense-fitting --geometry start \
  --offsets 0 --mass-scales 3,10,30,100 --stabilization-grid 0,.002,.01,.05 \
  --output /path/to/contact_fit.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --no-project --python "$FORCE_PYTHON" python examples/particle_nerd/score_frozen_contacts.py \
  --datasets /path/to/new_test.hdf5 --experiments /path/to/contact_fit.json \
  --output /path/to/frozen_test.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --no-project --python "$FORCE_PYTHON" python examples/particle_nerd/replay_body_motion.py \
  /path/to/new_test.hdf5 /path/to/contact_fit.json --output /path/to/replay.json

uv run --no-project --python "$FORCE_PYTHON" python -m pytest \
  --confcutdir=examples/particle_nerd/tests examples/particle_nerd/tests -q
```

The replay uses the collector's one-step force delay. It compares against a
reference-force replay to check integration. It rejects reference episodes with
direct rigid contact and flags predicted box-ground or box-wall intersections.
Recorded material cannot respond to box errors, so this is **not a fully coupled
NeRD rollout**. Future body acceleration is never a force-estimator input.

### Current dynamic measurements

With a 2 kg box, zero contact offset, effective contact mass equal to ten
particle masses, penetration coefficient 0.01, affine surface velocity, and
Coulomb friction 0.6:

- 32 new episodes, 12,800 transitions: force-vector RMSE **9.41 N**; force
  component MSE **29.50 N²**; torque-vector RMSE **0.344 N m**.
- Episode-bootstrap 95% interval for force-vector RMSE: **8.91–9.87 N**.
- Rigid-box replay: mean episode position RMSE **0.72 cm**, orientation RMSE
  **4.35 degrees**. No rigid-ground or rigid-wall intersections occurred.
- Reference-force replay: mean position RMSE **1.25 micrometers**.
- Parameters were selected before collecting these 32 episodes. A subsequently
  corrected friction-direction bug changed force RMSE by less than 0.001 N;
  reports preserve the original and corrected results separately.

These measurements concern one material, one box geometry, and recorded particle
motion. Force-vector error remains about 35% relative to the reference force
norm. Heavier boxes and perturbed particle velocities expose additional errors.
No general material model or real-time coupled performance has been established.

The contact-iteration limit is distinct from the MPM iteration limit. On 320
validation samples, 50 contact sweeps matched the converged result; 25 sweeps
changed forces by only 0.0000165 N RMSE but did not always meet the strict
1e-9 N s impulse-change tolerance. Reference MPM runs retain 25 iterations by
default, with an explicit `--mpm-iterations 50` sensitivity option. Collection
also accepts `--body-mass` for physical mass generalization experiments.

### Contact history and timed evaluation

Transporting the preceding estimated contact impulse and recalibrating four
scalar contact parameters reduced force-vector RMSE from 9.41 N to **6.66 N**
on the same 32 development episodes. Component MSE fell from 29.50 to
**14.80 N²**; mean box-position RMSE fell from 0.72 to **0.38 cm**.
However, upward-force error at the reference impact time increased from
40.55 to **62.22 N RMSE**. Lower average error does not establish better
impact-force reconstruction. These episodes have now been reused for method
comparison and are no longer an untouched final test set.

`overnight_study.py` compares mean-force and impact-force objectives using
64 new fitting episodes, 32 validation episodes, and 64 independent test
episodes. All recordings use randomized dynamic bodies and two-way MJWarp–MPM
coupling. Force estimation uses particle motion and analytical contact iterations;
it never invokes MPM or a force network. Histories advance at every timestep,
including when only a subset of output frames is scored.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --no-project --python "$FORCE_PYTHON" python examples/particle_nerd/overnight_study.py \
  --data-dir /path/to/study_recordings \
  --initial-experiment /path/to/history_parameters.json \
  --baseline-experiment /path/to/memoryless_parameters.json \
  --deadline 2026-10-15T14:00:00+00:00 \
  --publish-site /path/to/particle_nerd_plan
```

The directory must contain `calibration.hdf5`, `validation.hdf5`, and
`test.hdf5`; an existing `study.json` is never overwritten. The runner checks
reference momentum balance, records source and dataset hashes, and selects
candidates using validation force error and box-motion checks. Test forces
are scored only after selection. Replay holds particle trajectories fixed,
so it remains a diagnostic rather than a coupled NeRD simulation.

Fitting stops 45 minutes before the specified deadline to leave time for
independent evaluation. Completed trials and failures are saved locally.
Optional publication updates only aggregate progress, the designated HTML
results panel, and its PDF in an existing slide repository. Raw recordings
and detailed per-frame outputs remain local. The runner does not publish
IsaacLab code. Collection also accepts `--material-friction` and
`--material-density` for material sensitivity experiments.

## Learned particle motion (experimental)

The learning study uses a 0.02 m MPM grid and 4,000 particles in the small bed.
Reference rigid bodies move dynamically under MJWarp–MPM coupling. Initial
conditions are randomized; optional linear/angular launch velocities apply only
at reset. Body geometry is selected with `--body-shape` and `--body-size`.

| File | Task |
| --- | --- |
| `collect_learning_data.py` | Collect separate training, validation, and test parameter combinations; optionally expand training with independently varied body mass. |
| `geometry.py` | Query signed distance, outward normal, and closest surface point for analytic shapes or sampled distance fields. |
| `learning_data.py` | Load complete episodes while preserving split boundaries and particle identity. |
| `particle_model.py` | Encode particle/rigid geometry features, exchange graph messages, and predict every particle's next position. |
| `train_particles.py` | Train on next positions, optionally backpropagating through several predicted steps. |
| `evaluate_particles.py` | Measure conditional particle rollouts and coupled particle/body rollouts. |
| `newton_rigid.py` | Advance rigid bodies and rigid contacts with Newton MJWarp using estimated material forces. |
| `evaluate_sdf.py` | Isolate sampled-distance-field error using recorded states and the analytical force estimator. |

### Inputs, outputs, and timestep order

The particle network receives current particle positions, recent particle
velocities, particle mass/radius, material density/friction/Young modulus/Poisson
ratio, and rigid geometry/state. Each rigid body contributes signed distance,
outward normal, relative surface velocity, inverse mass, and friction. Its
object ID associates these quantities; there is no learned object-category
embedding. Static bodies have zero inverse mass. All material particles enter
the graph; only graph edges and rigid influences have local distance cutoffs.

The decoder predicts a correction to a gravity-only step. In plain notation:

```text
velocity_next = velocity_current + dt * gravity + velocity_scale * decoder_output
position_next = position_current + dt * velocity_next
```

This parameterization predicts next position while making the velocity passed
to force reconstruction explicit. Recorded MPM position differences and stored
velocities agree to approximately 0.0000005 m/s RMS in a checked sphere episode.
The force estimator has no neural output head and receives no force labels.

The collector uses lagged coupling, reproduced by the deployment diagnostic:

1. Advance the rigid body with the previous estimated contact wrench and gravity.
2. Predict particle positions using current particles and the advanced rigid geometry.
3. Reconstruct the current material-on-body wrench using current particle positions,
   predicted next particle velocities, and the rigid state before step 1.
4. Store that wrench for the next rigid transition.

Training supplies the recorded advanced rigid state for step 2. Deployment
computes it from its own previous wrench, so future reference body states are
not required. Conditional evaluation keeps recorded rigid motion to isolate
particle-model error. Coupled evaluation supplies only initial states and
advances both particles and the dynamic body using predictions.

The contact solver's auxiliary particle impulse is not applied again to the
network's predicted particle velocity. The network was trained on motion that
already includes contact. This composition does not enforce the complete
particle momentum equations or uniquely recover material internal forces.
Rigid momentum balance follows from applying the estimated wrench; agreement
with reference motion and reference forces is a separate accuracy test.

### Training and evaluation

```bash
# Use the simulation environment for collection and the analysis environment
# (requirements-force.txt plus torch) for learning and reconstruction.
uv run python examples/particle_nerd/collect_learning_data.py \
  --output-dir /tmp/particle_learning --simulation-python /path/to/simulation/python

uv run python examples/particle_nerd/train_particles.py /tmp/particle_learning \
  --output /tmp/particle_model --steps 30000 --unroll 4

uv run python examples/particle_nerd/evaluate_particles.py /tmp/particle_learning \
  /tmp/particle_model/best.pt --output /tmp/particle_validation --split validation
```

One-step training starts from recorded states. With `--unroll 4`, the next three
inputs use the model's predicted particle positions and velocities. Gradients
pass through those predictions and analytic distance queries; neighbor indices
are rebuilt discretely. A six-frame history and deeper message passing are
configurable. The optional `--early-transition-probability 0.75` draws 75% of rollout starts
from the first third of an episode and the remainder uniformly. This increases
exposure to initial impacts in the drop/launch dataset; every particle remains
in the loss, and no measured force determines the sampling weights. It is a
dataset-specific training ablation, disabled by default.
Force labels do not enter training. Material parameters are inputs,
but stress, plastic strain, and MPM grid state are absent.

`--projection static` in evaluation projects particles outside static rigid
geometry after each learned update. `--projection all` also projects against
the moving body. Both are explicit ablations, disabled by default. Projection
against the moving body is one-way and may change force inference substantially;
it is not a jointly solved particle/body contact constraint.

Report one-step error, 300-step particle error, rigid-body position/orientation
error, contact-force error, and the number of completed rollouts. Do not rank
an early-terminated rollout by its smaller partial-horizon RMSE. The free-body
integrator stops if the moving body intersects the rigid tray; an actual Newton
rigid-contact adapter is available with `--rigid-solver newton`. This adapter
constructs only rigid geometry in Newton, applies the previous material wrench
once, and uses MJWarp for rigid motion and direct tray contacts. It uses the
collector's Euler integrator. A reference-force replay on one 300-step box episode
reproduced body-center positions to 0.00000053 m RMS and orientation to
0.000068 degrees RMS. Independent tests check free-flight momentum and a sphere
settling on rigid ground. Direct rigid-contact forces are recorded separately
from material reactions. Runtime
measured during simultaneous GPU experiments is not a performance benchmark.

For large graphs, `--edge-chunk-size 65536` uses the same trained network while
limiting temporary edge-MLP storage. Every edge and every particle is updated in
every layer; node updates begin after all edges have consumed the previous node
states. This is not a region selector. A parity test covers cross-chunk messages
and an isolated particle. Three reference-state queries on the 296,875-particle
G1-sized bed used up to 3.1 GB of allocated GPU tensor memory with the 64-channel,
three-layer model. This checks memory feasibility only, not rollout stability or
speedup over MPM.

The initial study uses 48 training, 6 validation, and 12 reserved test episodes.
The expanded collection contains 156 training episodes in total.
The expanded study adds training combinations; the split manifests in each run
record exactly which episodes were used. All three shape families occur in
training, so these tests do not establish generalization to unseen mesh shapes.

The architecture follows the encoder/message-passing/decoder pattern in the
[terrain manipulation paper](https://arxiv.org/abs/2503.23270) and its
[public implementation](https://github.com/Chaoqi-LIU/lgbnd). The
[PhysicsNeMo Lagrangian example](https://github.com/NVIDIA/physicsnemo/tree/main/examples/cfd/lagrangian_mgn)
provides additional reference for velocity history, feature normalization, and
training noise. This example is an independent PyTorch implementation and does
not use the paper's region selector or its terrain/tool/wall category encoding.

## Historical prescribed-motion validation

The collection experiment completed four two-second episodes in two parallel
worlds, including resets: 1,600 transitions, 500 particles per world. All four
trajectories contact the sand. Checks passed for action/state alignment, particle
retention in the bed, finite values, episode resets, and settling after release.
The recorded material/body linear momentum balance had relative residual
`8.36e-6`, normalized by the larger of material weight and total body reaction.

The initial mechanics tests covered: free fall, weight transmission through
a two-particle column, sliding friction and torque, and rotated-box contact geometry.
These establish implementation conventions, not force accuracy on MPM trajectories.
The expanded numerical suite contains 22 mechanics tests and two study-selection tests; all 24 pass.

Reconstruction results are recorded separately after comparison with MPM.

The example-specific formatting and lint checks pass. The repository-wide
changelog hook reports existing fragment errors and missing fragments in other
packages; those files are outside this example's changes.


## Historical DEM-inspired contact-force experiment

The measurements in this section use the earlier prescribed-box dataset.

`dem_reconstruction.py` adds a separate nonlearned estimator; the original
`reconstruction.py` retains the central-force baseline. `evaluate_dem.py` compares both on the
same recorded frames. No MPM solve, particle integration, or force network is
used by the estimator.

The normal force uses a linear spring and approach-only damping, following the
structure of [Warp's DEM example](https://github.com/NVIDIA/warp/blob/main/warp/examples/core/example_dem.py).
Unlike that cohesive example, this experiment gives separated contacts a zero
force prior. Two friction models are compared: velocity-dependent damping and
a tangential spring with contact history. Body contact velocity includes linear
velocity and rotation about the body's center. Histories are advanced at every
recorded frame, projected into the current tangent plane, corrected on sliding,
and discarded after separation. Create a new `ContactHistory` at episode reset.

The inverse objective is:

```text
0.5 * squared particle force-balance error
  + 0.5 * law_weight * squared physical contact-force error relative to DEM
```

Four nonnegative rays represent each particle-particle and particle-body friction
cone. Unlike the original baseline, internal pairs can transmit tangential force.
The penalty acts on the physical 3D force, not ray coefficients. A zero-prior
ablation uses identical geometry, rays, and penalty weight to isolate the effect
of the constitutive target. The bound-constrained CPU solver reports its projected
gradient residual, iterations, and convergence; low force loss alone is not enough.

### Explicit approximations

- Effective particle-pair rest distance is the sum of recorded radii times a
  configurable scale. These material points are not individual physical grains.
- The default body contact offset is zero: recorded MPM particle centers rest
  approximately on rigid surfaces. The stored radius is not used as wall overlap.
- The old 6.5 cm neighbor radius and 4 cm body candidate shell are retained to
  isolate the constitutive experiment. A separated candidate has a zero DEM
  prior, but **balance fitting may still assign it nonzero force**. This is a soft
  contact-law constraint, not exact DEM contact activation.
- Pair tangential forces omit physical grain rotation, rolling resistance, and
  APIC angular momentum. Only translational particle balance is fitted. Body
  torque is summed about each box center; angular consistency is not established.
- Stiffness is an effective force/length parameter, not the MPM Young modulus.
  All contacts currently share one stiffness and damping ratio. Friction uses
  material friction for particle pairs and body friction for rigid contacts.
- Current contact state gives instantaneous force estimates; targets are MPM
  impulses divided by the 5 ms interval. Position noise tests isolate geometry
  effects while leaving recorded velocities and accelerations unchanged.

### Reproduce the measurement

Use the existing uv-managed environment explicitly when this isolated worktree
has no `.venv`. Set `PYTHON_BIN` to that environment's Python executable:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --no-project --python "$PYTHON_BIN" python examples/particle_nerd/evaluate_dem.py \
  /path/to/validation.hdf5 --output /path/to/dem_experiment.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --no-project --python "$PYTHON_BIN" python examples/particle_nerd/evaluate_dem.py \
  /path/to/validation.hdf5 --output /path/to/dem_original_baseline.json --baseline-only
```

The protocol selects stiffness (100, 1,000, 10,000 N/m) and prior weight (0.001,
0.1) on `demo_0` and `demo_1`, then evaluates on `demo_2` and `demo_3`.
The primary loss is squared force error averaged across frames, six bodies, and
three force components, in N². Box force vector RMSE is also reported separately.
The fixed evaluation schedule densely covers the impact interval, plus the
reference peak frame for each episode as an explicit peak diagnostic. Direct
contact-law estimates are additionally scored over all 800 evaluation transitions.
The previously inspected validation file is reused; this is a preliminary
comparison, not a pristine final test set or a NeRD rollout evaluation.

Parameters are selected by force MSE among converged calibration trials with a
mean particle-balance residual at most 5%. If none meet that gate, the report says
so and retains the best converged candidate for diagnosis. MPM reactions are
used for this scalar parameter calibration and scoring, never as inference inputs.
The estimator is nonlearned but the selected effective parameters are calibrated.
Position sensitivity uses persistent Gaussian offsets with standard deviations
0.1 mm and 1 mm per coordinate, with fixed seeds and separately reset histories.


### Measured result: first DEM-prior experiment

On 52 matched frames from `demo_2` and `demo_3`:

| Method | Force MSE [N²] | Box force-vector RMSE [N] | Mean balance residual | Converged frames |
| --- | ---: | ---: | ---: | ---: |
| Original baseline | 5290.03 | 214.85 | 0.039% | 0/52 |
| Zero prior, new basis | 6876.50 | 225.21 | 14.78% | 52/52 |
| DEM velocity friction | 6553.98 | 223.92 | 13.86% | 52/52 |
| DEM contact history | 6523.56 | 224.01 | 13.92% | 52/52 |

The history model increased force MSE by 23.3% versus the original baseline.
At the exact reference force peaks (515.1 and 518.6 N upward on the moving box),
it predicted -2.51 and -2.51 N. The original baseline predicted 10.88 and 13.01 N.
Neither reconstructs the impact load accurately.

Both DEM variants selected stiffness 100 N/m and prior weight 0.1. **No calibration
candidate passed both convergence and the 5% particle-balance criterion.**
These selected parameters are diagnostic fallbacks, not validated material parameters.
The original solves reached their iteration limit; new solves met a normalized
projected-gradient criterion of 1e-4. Their convergence flags use different criteria.
The zero-prior ablation is a better controlled comparison of the added physical
prior; that prior helps within the new basis but does not outperform the original.

Median CPU assembly-plus-fit times were 0.521 s (velocity),
0.611 s (history), and 6.695 s (original).
These are indicative timings from concurrent jobs, not a controlled speed benchmark
or evidence of an end-to-end simulation speedup. Histories evolve from the prescribed
motion and constitutive friction cap, not from the adjusted inverse-solve forces.

For persistent 0.1 mm and 1 mm position offsets, force-change RMSE versus clean
estimates was 0.0778 and 0.3876 N per component on 12 matched evaluation frames.
This isolates geometric sensitivity and does not establish accurate loads.

Results and plots are retained in the sibling `particle_nerd/particle_nerd_plan/results/`
and `assets/` directories: `dem_experiment.json`, `dem_original_baseline.json`,
`dem_summary.json`, `dem-force-history.svg`, and `dem-force-loss.svg`.
Eight mechanics tests pass. Scoped formatting/lint checks pass; the repository-wide
changelog gate still fails on pre-existing unrelated fragments. No framework source
was changed, and no MPM inference or GPU implementation was introduced.

### G1-sized level testbed and grid resolution

Use the G1 tilting testbed's 2 × 2 × 0.3 m sand volume, with a level fixed tray:

```bash
uv run --no-project --python .venv/bin/python python examples/particle_nerd/collect.py \
  --output /tmp/particle_nerd/g1_grid002.hdf5 --scene g1-testbed --voxel-size 0.02 \
  --num_envs 1 --episodes_per_env 1 --steps 400 --seed 20261024 --split validation
```

This records 296,875 material points at approximately 0.016 m spacing. The
0.12 × 0.12 × 0.08 m, 2 kg box remains freely dynamic. The tray uses 0.1 m
floor/wall thickness; its walls reach the initial sand surface. The preset
copies the testbed geometry, without its actuated roll/pitch joints or G1 robot.
The manifest records actual grid spacing, lattice size, particle radius, and
bed dimensions; collider sizes are recorded from the configured scene.

For large recorded scenes, optional `velocity_gradient_device: "cuda:0"` in the
contact parameter dictionary computes the same affine velocity fit with Warp.
Install `warp-lang` in the force-analysis environment to use it. Every material
point and its neighbors inside the fit radius remain included. This changes
computation only; it introduces no material-region selection or force model.
CPU evaluation uses bounded temporary storage. GPU/CPU numerical parity is
checked independently of force calibration.

Plot vertical **sand-on-box force** in newtons, positive upward. For a plot
aligned with box motion, use `reference/applied_body_impulse / dt` and delay
replay-produced forces by one step, matching the collector's lagged coupling.
The 2 kg box's weight is 19.62 N. These are material sample spheres, not DEM grains.

For controlled horizontal or spinning launches, override only the initial box
velocity. Initial pose randomization remains controlled by `--seed`:

```bash
# Horizontal launch with an initial spin about the world x axis.
uv run --no-project --python .venv/bin/python python examples/particle_nerd/collect.py \
  --output /tmp/particle_nerd/launch_roll.hdf5 --scene g1-testbed --voxel-size 0.02 \
  --num_envs 1 --episodes_per_env 1 --steps 400 --seed 20261025 --split validation \
  --initial-linear-velocity 1.5 0 -0.1 --initial-angular-velocity 8 0 0
```

Linear velocity is in m/s; angular velocity is in rad/s about world x, y, and z.
Use angular velocity `0 0 0` for translation only, `0 8 0` for a world-y spin,
and `0 0 8` for a world-z spin. These are angular-velocity vector components,
not Euler-angle derivatives. No velocity or trajectory is prescribed after reset.
