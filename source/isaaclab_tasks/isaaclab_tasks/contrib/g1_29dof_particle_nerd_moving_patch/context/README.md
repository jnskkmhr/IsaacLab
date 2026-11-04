# Moving-patch Particle NeRD: implementation and performance context

Recorded 2026-11-04. This note preserves the investigation and verified measurements
for continuing work on this task. Training statuses below are historical observations,
not a live status check. Commands are run from the repository root.

## Main findings and comparison rules

The matched G1 moving-patch benchmark with sand in every environment measured
Particle NeRD at **1.51–1.64× faster than implicit MPM** where both completed on an
RTX 5090. The largest successful tested counts were 32 for MPM and 128 for NeRD;
these are tested points, not exact maximum capacities.

Three experiments must remain separate:

1. **Matched all-sand NeRD versus MPM:** the latest direct solver comparison below.
2. **Mixed-terrain NeRD:** half of all G1 environments have sand; the rest have rigid
   terrain. These are the earlier 16/128/256-environment measurements and the current
   training jobs.
3. **Original G1 example versus flat moving patch:** different particle budgets and
   domains; verified only at 1 and 4 environments.

**Never substitute the box example for G1.** The green curves in the old
`video/multi_terrain/plots/env_scaling_benchmark.png` and `.pdf` were generated with
`examples/particle_nerd_batch/simulation.py`: one box collider per environment.
The approximately 29.95 ms result at 256 environments is a box result. Its MPM
curve was imported from a separate archived workload whose matching provenance was
not established. That figure is invalid as evidence of G1 speed or scalability;
this context note does not replace or repair the old image itself.

The old 50 Hz particle-update explanation and 16,384-particle count also describe
that box experiment. The actual G1 example used 200 Hz physics. No verified
256-environment result for the original G1 example has been established here.

Preserve the validated checkpoint, geometry, effective material scales, timestep,
and particle-selection settings when comparing implementations. Any diagnostic
change must be recorded explicitly. Do not change these silently to improve speed,
make a run fit, or alter terrain softness.

## Current implementation

The task owns its environment, scene, observations, rewards, PPO configuration,
and train/play registrations. Reusable neural dynamics, contact reconstruction,
and particle recycling live in the sibling `particle_nerd` package.

- MJWarp advances G1 rigid dynamics; the neural solver advances the sand particles.
  Two ankle-roll bodies supply the foot colliders for lagged two-way proxy coupling.
- The moving patch has fixed particle storage. Particles inside the overlapping
  portion retain their deformation; departing slots are recycled at the entering
  edge, with fresh particle state and three identical history frames.
- There is **no full particle-state archive**. Footprints are retained only while
  inside the moving patch and are not restored after returning to a departed area.
- There are **zero native MPM construction, settling, warm-up, or simulation steps**
  in the NeRD task. `MPMObjectCfg`, material schemas, and reused terrain utilities
  provide storage/geometry; their names do not imply native MPM stepping.
- Shared fixes from the fixed-terrain G1 integration include proxy-inertia refresh,
  beginning-of-step collider inputs, and triangle-mesh collision classification.
  Contact reconstruction uses coupler-assigned virtual inertia; the robot's
  physical link masses were not changed by the proxy-inertia correction.
- The moving patch uses a terrain height chart, pinned particle boundary, and
  recycling. The original fixed testbed has retaining walls. Matching contact
  parameters does not establish identical trajectories or neural accuracy.

Task-level settings verified for this investigation:

| Setting | Value |
|---|---|
| Physics / policy interval | 0.005 s / 0.02 s (200 Hz / 50 Hz) |
| Rigid substeps per coupling exchange | 2 |
| Simulated patch interior | 1.5 × 1.5 m |
| Pinned boundary width on each side | 0.2 m |
| Sand depth / sampling voxel size | 0.3 m / 0.04 m |
| Stored particles per sand environment | 36,000, including boundary particles |
| Active selection | `active_k="auto"`; resolves to 2,176 for this patch |
| Background features | 512 per sand environment |
| Selection refresh | Every 4 physics steps; neural prediction still runs each physics step |
| Normal / tangential effective material scales | 100 / 40 |
| Horizontal / upward scales | 8 / 2 |
| Contact iteration limit / relaxation | 200 / 0.02 |
| Sand foot margin | 0.02 m |
| Network / contact batching | `batch_inference=True`, `batch_contacts=True` |
| Checkpoint | `particle_nerd/checkpoint/best.pt`, resolved directly by task config |

Use the task's `env_cfg/physics_cfg.py` to determine effective settings. For example,
the reusable solver config has a normal-scale default of 50, but this task explicitly
overrides it to 100.

### CUDA graphs and dependency pin

The NeRD task sets `use_cuda_graph=False` and `capture_solver=True`. The first flag
turns off the generic whole-step capture path; the custom manager still captures
and replays coupled rigid/neural solver work using PyTorch-owned CUDA graphs and
Warp interoperability. Four solver graphs were captured and reused in successful
benchmarks. Particle recycling and global collision work remain outside those
custom solver graphs. `use_cuda_graph=False` alone does not mean inference is eager.

The checkout pins Newton to
`e1feec68109c49aae8d5cfb59e8a933d7fa19ad9` in `pyproject.toml`, associated with the
mesh-contact progress/synchronization fix. Preserve and record this dependency when
reproducing the stall investigation.

### Mixed-terrain task

`IsaacContrib-Velocity-G1-29dof-ParticleNeRD-MovingPatch-MixedTerrain`
and its `-Play` variant require an even environment count. Half the environments
contain neural sand; half have rigid ground. Only sand environments allocate neural
particle histories and inference buffers. Group assignments remain fixed across
resets. Rigid environments use zero foot margin.

Consequently, **512 mixed environments per GPU means 256 sand environments**, not
512 sand environments. Dividing mixed-task wall time by all environments includes
rigid-only environments in the denominator.

## Latest matched all-sand benchmark

Hardware: RTX 5090, 32 GB GPU memory; workstation RAM approximately 62 GiB.
Both methods execute the registered G1 moving-patch tasks, using policy
`jnskkmr/g1_29dof_mpm_mixed_terrain/5f1xyc21` (`model_39998.pt`) and seed 42.
Rendering, recording, and real-time pacing are disabled.

| G1 environments | MPM ms/step | MPM ms/step/env | NeRD ms/step | NeRD ms/step/env |
|---:|---:|---:|---:|---:|
| 1 | 5.175 | 5.175 | 3.147 | 3.147 |
| 4 | 8.027 | 2.007 | 5.234 | 1.309 |
| 16 | 23.033 | 1.440 | 15.237 | 0.952 |
| 32 | 44.807 | 1.400 | 28.460 | 0.889 |
| 64 | GPU OOM | — | 57.674 | 0.901 |
| 128 | Not attempted | — | 115.877 | 0.905 |
| 256 | Not attempted | — | GPU OOM | — |
| 512 | Not attempted | — | Host-memory stop | — |

`ms/step` is wall time to advance the entire batch by one 5 ms physics step.
`ms/step/env` is that batch time divided by the environment count; it is an
amortized throughput measure. It is not the latency of advancing one environment
independently. For example, 128 NeRD environments advance together in 115.877 ms,
not 0.905 ms.

### Matching and timing boundaries

The MPM benchmark temporarily adopts NeRD's 1.5 × 1.5 m patch interior and 0.3 m
sand depth, producing 36,000 particles per environment in both methods. MPM's
production interior is 1.2 × 1.2 m and its play configuration uses 0.35 m depth.
These benchmark overrides did not edit production configurations.

Saved robot, height-scanner, terrain-generator, and particle-patch configurations
match exactly for every successfully paired count. Both use the difficult rough
terrain generator, the same forward command and reset settings, and disabled
pushes, mass/material/gain randomization, and command/gravity-bias curricula.
Full overrides are recorded in the benchmark script and resolved configuration files.

MPM retains its native implicit solver, sparse unpadded grid, APIC transfer,
P0 strain basis, 25-iteration limit, 1e-4 tolerance, and existing sparse-grid
capacities. NeRD retains its trained checkpoint and the selection/contact settings
above. Guards reject native MPM construction or stepping in NeRD runs.

Each process runs alone on the GPU. The protocol is:

1. 400 physics steps of warm-up to exclude compilation and initial capture.
2. Three rollout windows of 200 physics steps each.
3. 200 separately timed `sim.step` calls, with CUDA synchronization before and after
   each call. Every measured call advances one physics step.

Warm-up here means executing the selected solver before collecting timings;
it does not introduce native MPM warm-up into NeRD.

The headline physics interval includes collision detection, rigid dynamics,
material simulation/prediction, contact-force reconstruction/coupling, recycling,
and boundary restoration. It excludes policy inference, external action submission,
observations, rewards, resets, and rendering. Separate rollout timings include
policy and environment processing. Their earlier measurement windows differ from
the headline latency window, so their means need not be greater in every run.

### Observed capacity limits

- **MPM 64:** GPU allocation failed during initialization of a coupled contact
  buffer. MPM 128, 256, and 512 were not attempted.
- **NeRD 256:** CUDA OOM during solver graph capture in terrain-coordinate
  processing. PyTorch reported about 30.06 GiB process GPU usage, including
  11.82 GiB in private pools.
- **NeRD 512:** explicitly attempted, then stopped with SIGTERM during initialization
  because the process reached approximately 44 GiB resident host memory and system
  swap usage reached 26 GiB. There is no timing result and no verified CUDA OOM at
  this count. The saved outcome is `stopped_host_memory_pressure`.

These limits apply to the tested configuration and hardware. The L40/L40S training
jobs use 48 GB GPUs and mixed terrain. Their successful 512-environment allocation
does not establish that 512 all-sand environments fit on the RTX 5090.

Evidence and reproduction: [full report](../../../../../../video/benchmarks/g1_moving_patch_matched_20261104/README.md) · [CSV](../../../../../../video/benchmarks/g1_moving_patch_matched_20261104/results.csv) · [PDF figure](../../../../../../video/benchmarks/g1_moving_patch_matched_20261104/g1_moving_patch_scaling.pdf) · [benchmark harness](../../../../../../video/benchmarks/g1_moving_patch_matched_20261104/benchmark.py) · [sweep script](../../../../../../video/benchmarks/g1_moving_patch_matched_20261104/run_sweep.py) · [validation](../../../../../../video/benchmarks/g1_moving_patch_matched_20261104/validation.json) · [checkpoint hashes](../../../../../../video/benchmarks/g1_moving_patch_matched_20261104/checkpoint_hashes.json).

## Earlier mixed-terrain results and speedups

These RTX 5090 measurements use the mixed task with half sand and half rigid
terrain, at 200 Hz physics with policy `5f1xyc21` and no rendering. They must not be
joined to the all-sand table as one scaling curve.

| Total G1 environments | Sand environments | NeRD ms/physics step | ms/step/total environment |
|---:|---:|---:|---:|
| 16 | 8 | 8.812 | 0.551 |
| 128 | 64 | 60.521 | 0.473 |
| 256 | 128 | 121.092 | 0.473 |

Network batching combines compatible sand-world inputs into one Transolver call.
The sequential fallback remains available. Contact batching then combines the
independent per-foot initializers and iterative solves into one launch each,
using separate per-foot scratch arrays. Per-world/foot contact preparation still
contains separate launches.

In the 128-total-environment mixed profile, network batching reduced inference
from 30.565 to 8.115 ms while contact reconstruction remained about 37 ms.
Adding contact batching then reduced its profiled cost to 14.242 ms. These
are instrumented component intervals for that experiment.

With network batching already enabled, contact batching reduced whole physics-step
latency from 11.558 → 8.812 ms at 16 total environments, 85.026 → 60.521 ms at 128,
and 170.718 → 121.092 ms at 256. Material scales, geometry, particle budgets, and
iteration settings were retained. The saved validation includes identical-input
force comparisons, graph reuse, reset/recycling checks, zero-native-MPM checks,
and five passing integration tests. This does not prove long-horizon trajectory
identity after batched floating-point operations.

Evidence: [mixed-task batching report](../../../../../../video/benchmarks/particle_nerd_contact_speedup_20261103/README.md).

### Where time still goes

A separate instrumented profile at **256 mixed environments (128 sand)** measured:

| Component | GPU interval, entire batch (ms) |
|---|---:|
| Global collision detection | 33.45 |
| Contact reconstruction, including preparation | 28.68 |
| Terrain-coordinate and particle/history processing | 21.09 |
| Network inference | 17.36 |
| Selection and network features | 8.02 |
| Displacement reconstruction | 2.39 |
| Rigid integration | 1.50 |

That profile's uninstrumented wall total was 111.95 ms; coarse instrumented wall
time was 115.39 ms. Its earlier measurement window differs from the 121.09 ms
mixed-task benchmark. Component timings are diagnostic GPU intervals, not a
replacement for synchronized wall-clock latency.

In a more heavily instrumented pass, gradient fitting took 18.10 ms, contact
preparation 6.78 ms, and neighbor-grid construction 3.20 ms. The already batched
contact initializer and iterative solve took about 0.458 and 0.045 ms. Thus the
remaining contact cost is predominantly preparation. Do not add those more heavily
instrumented subtimings to the coarse contact total.

Mesh–mesh signed-distance collision queries against the shared terrain mesh were
another major cost. Candidate mesh-edge visits are not accepted contacts. Changing
robot meshes, removing self-collision, or reducing terrain geometry would change
the physical comparison and requires explicit documentation.

Evidence: [component profile](../../../../../../video/benchmarks/particle_nerd_comparative_profile_20261103/README.md). Use only its current-G1 measurements for this discussion; its box-reference comparison is not a G1 baseline.

## Actual G1 example versus moving patch

The original G1 reference is `examples/particle_nerd/g1`, exercised through the
`video/multi_terrain_5f1xyc21_zero_warmup_50fps` recording harness with rendering
removed. Both reference and moving patch use **200 Hz physics** and **50 Hz policy**.

| Setting | Original G1 recording setup | Flat moving patch |
|---|---:|---:|
| Stored particles per sand environment | 890,625 | 36,000 |
| Active K | 7,104 | 2,176 |
| Background features | 512 | 512 |
| Sampling voxel size | 0.02 m | 0.04 m |
| Domain | Fixed 6 × 2 m bed | 1.5 × 1.5 m interior plus boundary |

| Environments | Original G1 example ms/physics step | Flat moving patch ms/physics step |
|---:|---:|---:|
| 1 | 4.074 | 2.713 |
| 4 | 13.330 | 4.289 |

The reference begins outside the bed at x = −4.5 m. The accepted timing window is
3–4 simulated seconds, after contact with sand; initial approach-only diagnostics
were excluded. Nonzero sand contacts were verified. The reference retains its
original upstream `/home/jkamohara/isaac/IsaacLab/source` imports and constructs a
native material object before replacing its step; native MPM **steps** are guarded
against. The moving-patch task rejects native MPM construction as well as stepping.

These results show no slowdown of the moving patch at the tested counts. Different
particle budgets and scene domains prevent an implementation-only speedup claim.
Both actual G1 implementations prepare contacts separately per foot; claims that
the original example globally batches that preparation came from the box example
and must not be applied to G1.

Evidence: [actual G1 example comparison](../../../../../../video/benchmarks/g1_example_vs_moving_patch_20261103/README.md).

## Training runs and playback

Two separate eight-GPU distributed training jobs were submitted, each with 512
mixed environments per GPU: **4,096 G1 environments total, including 2,048 sand**.
The second submission did not cancel the first.

| OSMO workflow | Pool |
|---|---|
| `g1-nerd-mixed-8gpu-512-1` | `isaac-lab-l40-07` |
| `g1-nerd-mixed-l40s-8gpu-512-1` | `isaac-lab-l40s-03` |

Both use the same uploaded source snapshot, `20261104T083556Z-uSiF2weM2b`, including
the current neural checkpoint and batching implementation. The launcher uses
`osmo/osmo_multi_gpu.yaml`, eight workers, and one GPU per worker. Each worker runs
`isaaclab train_multigpu`; 512 is the per-worker environment count. The checkpoint
source project is `g1_29dof_mpm_mixed_terrain`; new training logs use
`g1_29dof_particle_nerd_mixed_terrain`. `--wandb_run 5f1xyc21` loads the starting
checkpoint and does not overwrite that source run. Recording uses 200-frame clips.

As last observed on 2026-11-04, both jobs were running. The L40S run had reached
iteration 40,008/59,998. Its initialized W&B run is
[9gr8sg4w](https://wandb.ai/jnskkmr/g1_29dof_particle_nerd_mixed_terrain/runs/9gr8sg4w).
Use that ID rather than the preliminary CLI announcement `5ayidm1r`.

Post-initial-video L40S iterations took 35.25 and 35.66 s, versus an earlier L40
window of 36.49–36.70 s. L40S iterations during the first video took about 53 s;
its compile-heavy first iteration took about 365 s. These are training observations
from different windows, not a controlled GPU benchmark. Do not compare those
iteration durations directly to the physics-step timings above.

- [L40 workflow](https://us-west-2-aws.osmo.nvidia.com/workflows/g1-nerd-mixed-8gpu-512-1)
- [L40S workflow](https://us-west-2-aws.osmo.nvidia.com/workflows/g1-nerd-mixed-l40s-8gpu-512-1)
- [Saved submission commands, source hashes, logs, and launch reports](../../../../../../logs/osmo/particle-nerd-512-launch)

Playback with the requested policy:

```bash
uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Velocity-G1-29dof-ParticleNeRD-MovingPatch-MixedTerrain-Play \
  --num_envs 128 --wandb_run 5f1xyc21 \
  --wandb_project g1_29dof_mpm_mixed_terrain --viz newton_rtx \
  'env.video_recorders=[]'
```

The `IsaacContrib-Velocity-Sand-G1-29dof-ParticleNeRD-MovingPatch-Play` registration selects the all-sand
task; it is a different benchmark layout from the mixed task above. Its current
terrain selection should be read from the config rather than inferred from old
screenshots or task documentation.

## Continuing the investigation

1. Use the matched all-sand benchmark as the NeRD-versus-MPM speed reference. Record
   hardware, terrain layout, particle count, K, timestep, policy, graph mode, and
   measurement window with every new result.
2. Investigate NeRD graph-buffer and initialization memory before claiming support
   for larger all-sand batches. The 256/512 failures identify different stages.
   Graph-pool reuse or buffer reuse are candidates to test, not validated fixes.
3. Profile and batch remaining contact preparation and particle/history processing.
   Optimize terrain collision queries while retaining the same collision geometry
   when measuring an implementation speedup.
4. Preserve numerical checks for reconstructed forces, reset independence, recycling,
   graph reuse, and zero native MPM calls. Verify contacts are actually loaded when
   comparing contact-reconstruction costs.
5. Separate performance conclusions from neural prediction accuracy and locomotion
   stability. Fixed-bed parity does not establish moving-patch generalization.

Implementation entry points: [task physics](../env_cfg/physics_cfg.py) · [neural solver](../../particle_nerd/solver.py) · [graph manager](../../particle_nerd/manager.py) · [contact batching](../../particle_nerd/contact/batch.py) · [task overview](../README.md).

The raw experiment and launch directories are local artifacts and may not be
present in a fresh clone. Preserve their manifests, logs, configs, and raw samples
when transferring the investigation; do not treat a plotted curve alone as evidence.

## 2026-11-04 Follow-Up Optimization Results

Addressing the three bottlenecks identified above reduced the 256-environment mixed-terrain physics step from `115.17 ms` to `21.21 ms` (**5.43× speedup**) and the matched all-sand moving-patch benchmark to **5.60×–7.14× faster than Implicit MPM** at 16–32 environments while eliminating the 256-environment CUDA OOM:

1. **Moving-Patch Convex-Hull Collision Preservation (`manager.py`)**:
   - Kept `GeoType.CONVEX_MESH` on robot collision links when `fixed_bed is None` so terrain–robot broadphase pairs use BVH midphase culling instead of global mesh–mesh edge reduction (`collision`: `33.85 ms` $\rightarrow$ `0.72 ms`, **47.0× faster**).
2. **Grouped-HashGrid Batched Contact Preparation (`contact/batch.py`, `contact/proxy.py`)**:
   - Replaced the per-foot Python loop (`128` `HashGrid.build` + `256` gradient/contact/scan/compaction launches) with 7 batched GPU launches using `wp.HashGrid` group partitioning (`contact_total`: `28.68 ms` $\rightarrow$ `0.92 ms`, **31.2× faster**).
3. **Zero-Clone Multi-World Batched Solver & Sparse Terrain Charting (`solver.py`, `kernels.py`, `network/features.py`, `network/inference.py`, `network/model.py`)**:
   - Eliminated per-world `[36000, 3, 3]` history clones and Python loops in `ParticleNeRDSolver.step`, restricted 3-frame terrain-chart ray queries and post-step height queries to the $K + M = 2,688$ selected particles per world, and factored the linear collider projection and `bfloat16` residual stream in `ParticleTransolver` (`world_inputs + selection_features + displacement + world_outputs`: `31.50 ms` $\rightarrow$ `1.68 ms`; `network`: `17.35 ms` $\rightarrow$ `13.99 ms`; peak reserved GPU memory reduced by `2.3×`, enabling `256` all-sand environments / `9.22M` particles in `41.30 ms/step`).

| All-Sand Envs | Total Particles | Implicit MPM (ms/step) | NeRD Before (ms/step) | NeRD After (ms/step) | NeRD After (ms/step/env) | Speedup vs MPM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36,000 | 5.175 | 3.147 | **2.237** | **2.237** | **2.31×** |
| 4 | 144,000 | 8.027 | 5.234 | **2.827** | **0.707** | **2.84×** |
| 16 | 576,000 | 23.033 | 15.237 | **4.114** | **0.257** | **5.60×** |
| 32 | 1,152,000 | 44.807 | 28.460 | **6.276** | **0.196** | **7.14×** |
| 64 | 2,304,000 | GPU OOM | 57.674 | **10.385** | **0.162** | **~8.6×** (est.) |
| 128 | 4,608,000 | Not attempted | 115.877 | **20.338** | **0.159** | **~8.8×** (est.) |
| 256 | 9,216,000 | Not attempted | GPU OOM | **41.304** | **0.161** | **~8.6×** (est.) |
| 512 | 18,432,000 | Not attempted | Host-memory stop | **75.450** | **0.147** | **~9.5×** (est.) |
