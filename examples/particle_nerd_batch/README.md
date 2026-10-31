# Batched Particle NeRD simulation

Transolver-only inference, analytical contact-force estimation and Newton MJWarp
rigid dynamics. This package does not import `particle_nerd`, GNN or GNN_V2.
It contains no dataset collector or trainer. It reads an existing checkpoint and
the settled initial history from an episode; it does not read future motion.

## Simulation step

```text
Current particle history + current rigid poses/velocities
    → batched particle/collider features (Warp)
    → one batched Transolver forward (Torch, BF16)
    → particle positions and velocities at the next timestep (Warp)
    → velocity-gradient fitting and contact-force estimation (Warp)
    → four batched MJWarp rigid substeps with the estimated forces
```

The complete step is captured in a CUDA graph on a shared Torch/Warp stream.
There is no Python loop over environments in the simulation step. Particle
attention stays within each environment. Contact equations use one cooperative
GPU block per environment, with separate convergence checks. The particle
neighbor index uses environment-specific cell keys and instance-owned GPU
buffers; it does not capture a native hash-grid host descriptor.

Current scope: equal particle counts, one dynamic intruder and fixed ground per
world. Worlds share rigid geometry, mass/inertia, contact settings and checkpoint
weights; histories, material feature vectors, initial rigid states and external
loads can differ. Supported intruder feature geometries are box, sphere and the
included shovel SDF. The rigid contact estimator is for free bodies, not robot
articulations. Geometry support does not establish checkpoint generalization.

## Run locally

### Interactive batch simulation — Newton OpenGL

```bash
cd ~/isaac/IsaacLab-particle-nerd
uv run isaaclab -p examples/particle_nerd_batch/play_shovel.py \
  --num-envs 4 --env-spacing 6
```

This uses the same launch pattern and default shovel checkpoint/initial episode
as `examples/particle_nerd/play_shovel.py`. Override them with `--checkpoint` and
`--episode` when needed. Right-click and drag an intruder; Space pauses/resumes.

### Interactive batch simulation — Newton RTX (OVRTX)

```bash
uv run --extra ovrtx isaaclab -p examples/particle_nerd_batch/play_shovel.py \
  --viz newton_rtx --num-envs 4 --env-spacing 6 --compile
```

The `ovrtx` extra installs the renderer. Full-step CUDA graph capture is always
enabled here; `--compile` additionally optimizes Transolver. Compilation remains
optional because numerical differences affected earlier long rollouts. Omit
`--compile` to use the default uncompiled model.

### Set the effective material mass multiplier

```bash
uv run --extra ovrtx isaaclab -p examples/particle_nerd_batch/play_shovel.py \
  --viz newton_rtx --num-envs 4 --env-spacing 6 --compile \
  --effective-mass-scale 180
```

The default is 180 for the shovel and 10 for other supported intruder shapes.
The multiplier scales the auxiliary material mass used by the analytical contact
solver: `effective_mass = multiplier * particle_mass`. It does not change the
intruder's mass or the material parameters supplied to Transolver. The same
argument is available in the headless `particle_nerd_batch.play` command.

### Set up paths

```bash
cd ~/isaac/IsaacLab-particle-nerd
RUN=examples/particle_nerd_batch/run.sh
CHECKPOINT="$HOME/isaac/particle_nerd/particle_nerd_plan/results/shovel/training/model/best.pt"
EPISODE="$HOME/isaac/particle_nerd/experiments/particle_gnn/shovel/reference/wide-circle.hdf5"
```

The launcher selects this checkout's UV-managed interpreter and source packages.
Set `PARTICLE_NERD_PYTHON` explicitly when running from another worktree.

### Interactive simulation with explicit artifact paths

```bash
"$RUN" -m particle_nerd_batch.view \
  --checkpoint "$CHECKPOINT" --episode "$EPISODE" --num-envs 4
```

Right-click and drag a rigid intruder to apply force. Space pauses/resumes.
The display offsets the worlds into a grid with 4 m between environment origins.
Use `--env-spacing 6` for wider spacing. Simulation coordinates remain local to
each world. Particles are brown, with a displayed radius of 0.012 m.
Use `--viz newton_rtx` for Newton RTX, or `--steps 100` to limit the run.
The previous `--renderer rtx` spelling remains accepted for existing commands.

### Run and time the complete batched simulation

```bash
"$RUN" -m particle_nerd_batch.play \
  --checkpoint "$CHECKPOINT" --episode "$EPISODE" \
  --num-envs 32 --warmup 20 --steps 60 --output /tmp/particle-batch-32.json
```

This command runs headlessly. Wall time includes the complete simulation step
and waits for GPU completion. It excludes initialization, compilation, capture,
diagnostics and rendering. Without external inputs the intruder moves under
gravity and material/ground reactions. `--no-capture` runs the same batched
operations without the outer graph for debugging.

### Test optional compiled Transolver

```bash
"$RUN" -m particle_nerd_batch.play \
  --checkpoint "$CHECKPOINT" --episode "$EPISODE" \
  --num-envs 32 --compile --warmup 20 --steps 60 \
  --output /tmp/particle-batch-compiled-32.json
```

Both commands use complete-step graph capture. `--compile` additionally fuses
neural operations with TorchInductor. It retains checkpoint weights but changes
numerical rounding; previous five-second shovel rollouts diverged. It is off by
default and is not an established accuracy-preserving optimization.

### Measure neural scaling separately

```bash
"$RUN" -m particle_nerd_batch.benchmark_nn \
  --checkpoint "$CHECKPOINT" --episode "$EPISODE" \
  --envs 1 8 32 --particles 512 2048 16384 \
  --output /tmp/transolver-particle-scaling.json
```

This uses fixed subsets of recorded features and CUDA graph replay. It measures
neural computation, not physical accuracy at reduced particle counts. Compare
equal total particle counts as well as fixed particles per environment.

### Run integration tests

```bash
"$RUN" -m pytest examples/particle_nerd_batch/tests/test_simulation.py -q
```

## Python runner interface

```python
import torch
from particle_nerd_batch.simulation import Simulation

sim = Simulation.from_files(checkpoint, episode, environments=32)
loads = torch.zeros((32, 6), device="cuda:0")
loads[:, 0] = 20.0  # External world-x force [N] on each intruder.
sim.step(loads)

x = sim.positions  # CUDA [E,N,3], particle positions [m]
v = sim.velocities  # CUDA [E,N,3], particle velocities [m/s]
pose = sim.body_pose  # CUDA [E,7], position [m] and xyzw quaternion
velocity = sim.body_velocity  # CUDA [E,6], linear [m/s], angular [rad/s]
wrench = sim.contact_wrench  # CUDA [E,6], material force [N], COM torque [N m]

mask = torch.zeros(32, dtype=torch.bool, device="cuda:0")
mask[3] = True
sim.reset(mask)  # Reset only world 3 to its initial particle/rigid state.
```

Outputs are live GPU views, not copies; clone them if retaining past states.
Treat state views as read-only. `step()` orders its stream with the caller's
Torch stream. Passing no load means zero external force/torque. Gravity and
material contact are applied internally. The external load is held across the
rigid substeps. `diagnostics()` explicitly synchronizes and returns host data;
leave it out of the performance-critical step.

For different initial states, call `io.load(...)`, modify the returned initial
arrays, then construct `Simulation(saved, manifest, **initial)`. A partial reset
also clears the selected worlds' MJWarp warm starts and applied-force buffers.

## Numerical settings

- Particle timestep comes from the checkpoint: 0.02 s for the supplied shovel.
- Rigid timestep defaults to 0.005 s: four substeps per particle update.
- Effective material mass multiplier: 180 for shovel, 10 for other geometries.
- Contact relaxation: 0.075 for shovel, 0.1 otherwise; stabilization: 0.01.
- Contact iteration limit: 2,000; convergence tolerance: 1e-9. The body-velocity
  initializer usually reduces the subsequent contact iteration count sharply.
- Particle velocity-gradient support radius: 0.065 m.
- The learned particle update already includes interaction. The estimator uses
  an auxiliary material response and does not apply another impulse to the
  predicted particle states.

## Code layout

| File | Responsibility |
|---|---|
| `simulation.py` | State ownership, complete-step capture, batched step and reset |
| `features.py`, `state.py` | Particle/collider features and particle integration |
| `model.py` | Checkpoint-compatible batched Transolver |
| `force/` | GPU neighbor index, gradients, contact equations and convergence |
| `rigid.py` | One Newton/MJWarp model containing all worlds |
| `io.py`, `sdf.py`, `assets/` | Initial-state loading and collider data |
| `play.py`, `view.py`, `benchmark_nn.py` | Headless timing, interactive play and NN scaling |

The code is an inference refactor. It does not fix the existing checkpoint's
material-motion accuracy or establish generalization to new interactions.
