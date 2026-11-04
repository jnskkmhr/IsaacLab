# Fixed-terrain validation — 2026-11-03

The integrated solver was compared with the actual G1 example adapter and with
policy `5f1xyc21` on flat, ascending, descending and wave terrain. The reference
parameters remain those of `video/multi_terrain_5f1xyc21_zero_warmup_50fps`.

## Corrections and evidence

Three migration bugs affected dynamics despite matching the visible configuration:

- The example prepared robot collision meshes as triangle meshes and zeroed
  kinematic retaining-body inertia; the integrated task had omitted that step.
- Neural collider features used the virtual proxy's velocity after rigid
  integration rather than the rigid body's beginning-of-step velocity.
- The force estimator cached 0.608 kg foot-link masses before the coupler assigned
  articulated virtual proxy masses of approximately 1.964 kg. It now refreshes
  cached masses on body-inertia notifications and invalidates solver graphs.

Collider ordering, scalar material-feature precision and proxy rewind arithmetic
also match the example. The material scales remain 100 normal and 40 tangential;
no material, checkpoint or policy retuning was used to compensate for the bugs.

## Common-input replay

The original `LearnedMaterial` adapter and the packaged solver independently
predicted particles and reconstructed foot wrenches from identical histories and
free proxy states. The maximum measured differences were zero for both particle
positions and wrench components on every checked step. Selected particle IDs also
matched on every step: 800 physics steps each for flat, ascending and wave, and
740 steps for descending before its lateral exit. Comparisons synchronized GPU
work and enforced tolerances of 1e-6 m and 1e-3 N or N·m.

## Policy rollouts

| Terrain | Fresh G1 example | Integrated solver | Normal RSL-RL + RTX playback |
| --- | --- | --- | --- |
| Flat | Crossed, 8.02 s | Crossed, 8.44 s | Crossed, 8.42 s |
| Ascending | Crossed, 9.34 s | Crossed, 9.14 s | Crossed, 9.12 s |
| Descending | Lateral exit, 3.66 s | Lateral exit, 3.62 s | Lateral exit, 4.70 s |
| Wave | Crossed, 8.36 s | Crossed, 8.44 s | Crossed, 8.28 s |

Independent closed-loop paths vary, including reruns of the original example.
These checks establish common-input solver parity and matching rollout outcomes,
not bit-identical trajectories or a statistical policy success rate. Descending
was absent from the archived three-terrain campaign; its fresh example also exits
sideways with this policy.

All four normal-runner checks blocked native MPM construction and stepping. They
used the existing downloaded policy checkpoint, whose bytes match `5f1xyc21`,
and recorded RTX videos at 50 fps. Camera/headless options and the bounded
recording loop were the only playback overrides beyond task, policy and environment
count. W&B downloading itself was not retested.

## Regression checks

The fixed-task suite passed three tests, covering the archived initial particle
state and two-world flat/tilting integration. The existing moving-patch regression
also passed. Isolated negative controls disabled each of the three fixes and
failed the intended assertions: cached virtual mass, initial neural collider
velocity, and robot collision-mesh representation. Existing API deprecation
warnings remain.

The flat, ascending and wave initial particle positions still match their archived
SHA-256 hashes exactly, with 890,625 particles, mass 0.006467368453741074 kg and
radius 0.007964757271111012 m each. Automatic Top-K resolves to 7,104 particles plus
512 background anchors, refreshed every four physics steps.

See the [full report, traces and videos](../../../../../video/particle_nerd_solver_parity_20261103/REPORT.md)
for source hashes, measured errors, plots, recordings, reproduction scripts and
test logs. The archived campaign was preserved unchanged.
