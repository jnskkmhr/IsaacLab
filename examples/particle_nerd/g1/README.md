# G1 particle NeRD

The G1 examples use the current native G1 MPM task in `~/isaac/IsaacLab`.
The model is Transolver. No GNN, region selector, or MPM correction network is used.

## Runtime

Use the native IsaacLab checkout's runtime and only overlay these example modules:

```bash
cd ~/isaac/IsaacLab
export PYTHONPATH="$HOME/isaac/particle_nerd/worktrees/IsaacLab-scalability/examples${PYTHONPATH:+:$PYTHONPATH}"
```

Do not add this worktree's `source/` directories to `PYTHONPATH`: the reference
collector uses the newer native G1 task and Newton/MJWarp versions.

## Reference simulation

```bash
uv run --no-sync python -m particle_nerd.g1.play \
  --output /tmp/g1-reference --duration 5 --viz newton_rtx
```

Omit `--checkpoint` to use native coupled MPM–MJWarp. The pretrained robot policy
runs at 50 Hz; physics runs at 200 Hz. `--headless` disables the live viewer.
The output contains 200 Hz force and robot traces and 25 Hz replay snapshots.

## Learned simulation

The selected full-data checkpoint is optimizer step 9,750 from the 500-trajectory
experiment. This is an experimental checkpoint: short predictions are accurate,
but local long rollouts fail. See the [measured results and videos](https://jnskkmhr.github.io/particle-nerd-plan/g1-training.html).

```bash
CHECKPOINT="$HOME/isaac/particle_nerd/experiments/g1-nerd-osmo-20261020/full-training-final/model/best.pt"
uv run --no-sync python -m particle_nerd.g1.play \
  --checkpoint "$CHECKPOINT" --output /tmp/g1-learned \
  --duration 5 --effective-mass-scale 100 --iterations 200 \
  --contact-relaxation 0.02 --viz newton_rtx
```

This replaces every material step with Transolver and an analytical foot-force
estimator. The native robot solver handles actuators, joints and rigid dynamics.
No MPM solve runs during learned inference. The existing native task still allocates
its MPM state during setup for particle initialization and reset compatibility.
The first history is extrapolated from the initial particle position and velocity;
no future reference particle frames are supplied.

The command above uses material-mass multiplier 100, 200 solver iterations, and
Jacobi relaxation 0.02. The smaller relaxation avoids the divergence observed on
a captured G1 contact problem with the original 0.1 setting. It does not fix
particle drift or establish accurate long-rollout physics. Parser defaults remain
10, 50, and 0.1; pass these options explicitly to reproduce the newer evaluation.

The estimator uses independent native virtual-foot responses, not the full
articulated contact response matrix. `--effective-mass-scale` multiplies the
auxiliary material mass; it does not change robot mass. This evaluation runner
uses one environment and eager dispatch; it is not the optimized batch benchmark.

For matched command scenarios, add the same `--schedule commands.json --trial 12`
and `--seed 120000` to both reference and learned commands. Native termination
ends the recorded run. The robot policy observes each run's evolving robot state.

## Ten-second coupled evaluation

```bash
cd ~/isaac/IsaacLab
export PYTHONPATH="$HOME/isaac/IsaacLab-particle-nerd/examples${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT="$HOME/isaac/particle_nerd/experiments/g1-nerd-osmo-20261020/full-training-final/model/best.pt"
SCHEDULE="$HOME/isaac/particle_nerd/particle_nerd_plan/assets/g1-command-plan-500/commands.json"
uv run --no-sync python -m particle_nerd.g1.play \
  --checkpoint "$CHECKPOINT" --schedule "$SCHEDULE" --trial 2 --seed 120000 \
  --output /tmp/g1-learned-10s --duration 10 \
  --effective-mass-scale 100 --iterations 200 --contact-relaxation 0.02 \
  --viz newton_rtx
```

Use `--trial 106` for curved motion or `--trial 306` for lateral motion. Omit
`--checkpoint` and use a different output directory for the matched MPM reference.
The requested horizon is 10 seconds; native termination or a nonfinite state ends
the recording earlier. Failed runs are not padded or reset into a second episode.
The command after the scheduled switch remains constant through the extended
horizon. Training trajectories were at most five seconds.

`metadata.json` records the actual duration, requested duration, solver settings,
and native termination terms when available. Learned `trace.npz` also records
per-foot `contact_stats`: contact count, iterations used, and maximum unrelaxed
impulse correction (N s). These diagnostics copy to the CPU for recording; they
are not part of a simulation throughput benchmark.

## Record reference data

```bash
uv run --no-sync python -m particle_nerd.g1.collect \
  --schedule /path/to/commands.json --output /path/to/data \
  --num-envs 2 --limit 50 --duration 5 --headless
```

`--shard R --shards N` partitions scheduled trials. Each HDF5 file records one
trajectory at 200 Hz; native resets are excluded. Static data contain particle
masses, collision-box geometry, and material parameters. `applied_generalized_force`
is Newton `Control.joint_f`, which excludes the native actuator output. It must
not be interpreted as total actuator torque.

## Train Transolver

```bash
uv run --no-sync python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m particle_nerd.g1.train --data /path/to/data --output /path/to/model \
  --steps 2000 --rollout-steps 2 --validation-samples 50
```

The trainer streams HDF5 windows; it does not load the complete dataset into RAM.
All particles are processed. The validation split selects `best.pt`; the test
split is not used for training or checkpoint selection. Two-step training and
validation cover 10 ms, so long rollouts must be evaluated separately.

## Evaluate particle motion with recorded robot geometry

```bash
uv run --no-sync python -m particle_nerd.g1.evaluate \
  --checkpoint /path/to/g1-best.pt --episode /path/to/validation-episode.hdf5 \
  --output /tmp/g1-particle-evaluation --duration 5
```

Only the first three particle frames initialize the rollout. Later particle
predictions feed the next step. Robot geometry follows the recording throughout.
This measures particle prediction; it is not closed-loop robot simulation.

## Render paired dynamic simulations

```bash
uv run --no-sync python -m particle_nerd.g1.render \
  --reference /tmp/g1-reference --prediction /tmp/g1-learned \
  --output /tmp/g1-comparison.mp4
```

Newton RTX renders the saved states without advancing physics. Both views use
brown particles, a 10 mm display radius, and identical cameras. The chart shows
vertical force exerted by sand on each robot foot. Omit `--prediction` for a
reference-only video. `--preview` writes one JPEG instead of the video.

## Plot robot motion, foot forces, and particle error

```bash
uv run --no-sync python -m particle_nerd.g1.report \
  --reference /tmp/g1-reference --prediction /tmp/g1-learned \
  --particle-evaluation /tmp/g1-particle-evaluation --output /tmp/g1-report
```

The report checks matching initial particle and robot poses. It plots the pelvis
trajectory and height, vertical sand forces on both feet, and particle position
error against the frozen-particle baseline. Force differences between diverging
robot trajectories are not a force-estimation error measured on the same motion.

## Reproduce the cloud experiment

The [collection workflow](osmo/collect_500.yaml) and
[training workflow](osmo/train_500.yaml) pin the exact source snapshots, dataset,
and pilot initialization used for this experiment. They use OSMO direct storage;
the full dataset remains in the cloud. These are reproduction configurations,
not automatically updated snapshots of your local source edits.

```bash
osmo workflow submit examples/particle_nerd/g1/osmo/collect_500.yaml --pool isaac-lab-l40s-03
```

Collection writes to a new workflow-ID prefix. To train on that new collection,
update the dataset input URL in `train_500.yaml` after collection completes.
Leaving it unchanged reproduces training on the original completed collection.

```bash
osmo workflow submit examples/particle_nerd/g1/osmo/train_500.yaml --pool isaac-lab-l40s-03
```

The mandatory [dataset audit](osmo/audit_dataset.py) checks every recorded numeric
value, trajectory IDs, splits, rates, and geometry before training. It also exports
measured pelvis paths at 10 Hz. Checkpoint selection uses validation trajectories;
short-horizon test metrics are computed only after selection.
