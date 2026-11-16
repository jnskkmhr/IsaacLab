# Native MPM benchmark environments

Manager-based Isaac Lab environments for the benchmark gallery. Every video, force plot,
and dataset in a run directory comes from one simulation. No Particle NeRD solver is
instantiated by this suite.

## Cases

| Motion | Terrain variants | Controller |
| --- | --- | --- |
| Horizontal box push | Flat, ascend, descend, wave | Prescribed collider pose and velocity |
| Vertical box intrusion | Flat | Prescribed collider pose and velocity |
| Box circling | Flat | Prescribed collider pose and velocity |
| Sphere rolling | Flat, ascend, descend, wave | Prescribed translation and spin |
| Shallow box projectile | Flat | Dynamic rigid body after release |
| Franka pushing sand | Flat | End-effector differential IK and PD joint drives |
| Go2 crawl | Flat | Free base; leg IK, crawl gait, attitude feedback, and PD drives |
| G1 walking | Flat | Saved MPM-trained RL policy |

The quadruped uses a **0.12 m sand layer above a rigid floor**. Its feet may also
contact that floor; this is not a deep-sand bearing-capacity test. There is no
prescribed base motion or external support wrench. The G1 bed is 0.30 m deep.

## Run through the Isaac Lab CLI

Install the example's task-discovery entry point once, from the IsaacLab-particle-nerd
repository root:

```bash
uv pip install --no-deps -e examples/particle_nerd_benchmark
```

Then use the normal `isaaclab play --task ...` command:

```bash
uv run isaaclab play --task IsaacBench-MPM-Box-Push-Flat-v0 \
  --num_envs 1 --viz newton_gl --real-time

uv run isaaclab play --task IsaacBench-MPM-Franka-Push-Flat-v0 \
  --viz newton_gl --real-time

uv run isaaclab play --task IsaacBench-MPM-Go2-Crawl-Flat-v0 \
  --viz newton_gl --real-time
```

Each task selects its prescribed-motion, IK, or saved-policy controller automatically;
**omit `--rl_library`** for these reference examples. They are playback benchmarks,
not RL training tasks. Gym registrations live in each `config/cube`, `config/sphere`,
and `config/robot` directory. Configurations use `ManagerBasedRLEnv` for the standard
Gym interface, with empty reward and termination managers. The physical settings and
controllers are shared with dataset collection.

| Task ID | Motion |
| --- | --- |
| `IsaacBench-MPM-Box-Push-Flat-v0` | Horizontal push, flat bed |
| `IsaacBench-MPM-Box-Push-Ascend-v0` | Horizontal push, ascending slope |
| `IsaacBench-MPM-Box-Push-Descend-v0` | Horizontal push, descending slope |
| `IsaacBench-MPM-Box-Push-Wave-v0` | Horizontal push, wave bed |
| `IsaacBench-MPM-Box-Intrude-Flat-v0` | Downward intrusion |
| `IsaacBench-MPM-Box-Circle-Flat-v0` | Circular motion |
| `IsaacBench-MPM-Box-Projectile-Flat-v0` | Dynamic projectile after release |
| `IsaacBench-MPM-Sphere-Roll-Flat-v0` | Prescribed rolling, flat bed |
| `IsaacBench-MPM-Sphere-Roll-Ascend-v0` | Prescribed rolling, ascending slope |
| `IsaacBench-MPM-Sphere-Roll-Descend-v0` | Prescribed rolling, descending slope |
| `IsaacBench-MPM-Sphere-Roll-Wave-v0` | Prescribed rolling, wave bed |
| `IsaacBench-MPM-Franka-Push-Flat-v0` | Manipulator differential IK |
| `IsaacBench-MPM-Go2-Crawl-Flat-v0` | Free-base quadruped leg IK |
| `IsaacBench-MPM-G1-Walk-Flat-v0` | G1 saved RL policy |

Playback lasts six simulated seconds by default; use `--duration` to change it.
Omitting `--viz` runs headless. These reference cases support one environment and
reject `--num_envs` values other than 1. `--real-time` limits playback to simulation
speed when the solver can keep up. Closing the viewer stops playback.

G1 requires the MPM-trained policy checkpoint and its adjacent `training_config.json`:

```bash
uv run isaaclab play --task IsaacBench-MPM-G1-Walk-Flat-v0 \
  --checkpoint /path/to/model_19999.pt --viz newton_gl --real-time
```

The reference gallery uses W&B run `jqur5jhm`, iteration 19,999. The local files are
in `benchmark_environment/checkpoints/jqur5jhm/` of the benchmark repository.
Its actor takes 10-frame term histories plus a seven-value velocity/heading command.
The runner checks the saved actor dimensions and loads weights strictly. Control is
50 Hz; particle/body states and MPM forces are recorded at 200 Hz.

### Record video and aligned data

Ordinary playback writes no dataset. Add `--video` for a 50 fps MP4; add `--output`
to stream the 200 Hz HDF5 trajectory and save provenance alongside it:

```bash
uv run isaaclab play --task IsaacBench-MPM-Box-Push-Wave-v0 \
  --duration 6 --video --output /tmp/box-wave

uv run python -m particle_nerd_benchmark.core.evaluate /tmp/box-wave
```

With `--video` alone, video and metadata go under `logs/particle_nerd_benchmark/`.
An existing output directory is rejected to preserve prior runs. Positions use world
axes in metres, +z upward, with x=y=0 at the tray centre.

The collection entrypoint and complete 14-case suite remain available for automation:

```bash
uv run python -m particle_nerd_benchmark.core.run \
  --task IsaacBench-MPM-Box-Push-Wave-v0 --duration 6 --output /tmp/box-wave-new

uv run python -m particle_nerd_benchmark.core.run_suite \
  --checkpoint /path/to/model_19999.pt --output /tmp/mpm-benchmark-suite
```

## Simulation conditions

- Physics: 0.005 s per step; video: 50 fps, 1280 × 720.
- Native implicit MPM: sparse grid, 0.04 m voxels, APIC transfer, Q1 velocity basis,
  PIC27 collider basis, 25 iterations, tolerance 1e-4. CUDA graph capture is disabled
  for these recording runs. Collection wall time is not a solver-speed measurement.
- Initial particle lattice spacing: 0.032 m; particle radius: 0.016 m. Each initial
  coordinate receives uniform jitter of up to 0.0064 m (20% of spacing), truncated
  at the tray boundaries and initial sand surface. Seed 42 makes the particle
  positions reproducible; particle count, mass, and initial velocity are preserved.
- Material: density 1600 kg/m³, Young's modulus 1e15 Pa, Poisson ratio 0.3,
  material friction 0.48, yield pressure 1e15 Pa; tensile yield ratio, yield stress,
  hardening, dilatancy, and viscosity are zero.
- Slopes: initial particle surface z=±0.12x. Wave: z=0.06 sin(2πx/1.2).
  These define initial particle placement; the free surface subsequently evolves.
- A rigid floor and four retaining walls surround each bed. Separate coincident
  static colliders are assigned to the rigid and MPM solvers. Franka's walls end at
  z=0.04 m, 4 cm above its initial flat sand surface, to clear the arm's motion;
  other cases retain walls ending at z=0.20 m.
- Robot foot collider margin: 0.02 m. Robot masses are the asset masses.
  Proxy mass scale is 1.0. Each run records its resolved physical configuration.
- Object cases allow 1 s of material preparation while the object is above the bed.
  Robot controllers and the material start evolving together at reset.

## Dataset and alignment

Each run contains `trajectory.h5`, `metadata.json`, `simulation.mp4` (unless disabled),
and a snapshot of the executed Python package in `source/`. Evaluation adds
`validation.json`, `forces.png`, `forces.svg`, and `poster.jpg`.

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `state_time` | T+1 | Initial state followed by post-step timestamps, seconds |
| `particle_position`, `particle_velocity` | (T+1,P,3) | World positions and velocities, m and m/s |
| `body_pose` | (T+1,B,7) | World position and quaternion in xyzw order |
| `body_velocity` | (T+1,B,6) | Linear and angular velocity, m/s and rad/s |
| `material_force`, `material_torque` | (T,B,3) | MPM impulse divided by physics dt, N and N m |
| `force_time` | T | End of the physics interval producing each force sample |
| `applied_action` | (T,A) | Manager action used during each physics interval |
| `joint_position`, `joint_velocity` | (T+1,J) | Robot joints, rad and rad/s (robot cases) |
| `rigid_tray_contact_count` | T+1 | Franka–tray contacts with nonpositive gap, including contact margins; must remain zero |

Torques are about each body's centre of mass. Forces act **from the sand on the
body**, in world axes. Direct rigid floor/wall forces are excluded. Static material
properties, particle masses/radii, body masses/COMs, names, and shape geometry arrays
are stored alongside the trajectories. This is a prediction/evaluation dataset,
not a complete native MPM restart checkpoint with constitutive history.

`metadata.json` maps every video frame to an exact HDF5 state index; frame zero uses
state 4 (0.02 s). SHA-256 hashes bind the dataset and video to the run. Plots read this
HDF5 file without smoothing or discarded samples. A failed simulation/validation is
not eligible for gallery publication.

## Layout

```text
core/                   # Collection, validation, and plotting
  mdp/                  # Shared manager action, observation, and reset terms
config/
  cube/env_cfg/         # Standalone box scene, actions, observations, events, physics
  sphere/env_cfg/       # Standalone sphere configuration
  robot/env_cfg/        # Franka, Go2, and G1 configurations
```

The website and its publisher live in the separate benchmark repository at
`benchmark_environment/`.
There is no website copy in this simulation package. From the benchmark repository:

```bash
uv run --no-project python benchmark_environment/publish.py \
  --runs /tmp/mpm-benchmark-suite
```

The publisher only accepts validated runs and refuses to replace a different
published run. See that repository's README for GitHub Pages and dataset downloads.
Runtime validation establishes that these examples execute and contain the stated
motions; it does not establish agreement with physical experiments or NeRD accuracy.
