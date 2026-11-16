# Agent handoff: fixed-bed G1 foot forces and feedback filtering

Snapshot: 2026-11-06, America/Los_Angeles. Repository branch during experiments:
`junnosuke/particle_nerd`, HEAD `f9b9704fe` plus local benchmark files.
This document records completed work and unresolved questions; it is not a new task authorization.

## Problem and current conclusion

The user observed stepping at zero commanded velocity in policies trained with moving-patch
Particle NeRD. They requested a comparison of foot ground reactions against implicit MPM in
the **fixed-terrain G1 task**. In the initial six-second comparison, an MPM-trained policy
stands steadily with MPM but develops oscillating foot loads with NeRD near 4 s.
The latest request was to determine whether that oscillation stops and test force filtering.

**Completed: four 30-second zero-command rollouts with the same MPM-trained policy.**
The unfiltered NeRD oscillation subsides around 19 s, but this is misleading: the feet sink
onto the rigid ground beneath the sand. Sand support disappears, and rigid contact supports
the robot. Neither of the tested online filters fixes the problem. Leave filtering disabled
by default; no production solver/task defaults were changed.

Start with the [benchmark report](../benchmark/README.md), then the plots and raw data linked
below. Do not infer a successful standing recovery from small force variance late in the run.

## Scope and matched setup

- This is **G1**, not a box benchmark. Fixed flat sand bed; **no moving-patch recycling**.
- Base task: `IsaacContrib-Velocity-Sand-Flat-G1-29dof-ParticleNeRD-Play`.
  The recorder constructs the MPM counterpart using the same NeRD scene and task terms,
  substituting native implicit MPM as the particle solver. See `make_config` in
  [record.py](../benchmark/record.py).
- One environment, seed 42, deterministic actor inference, zero velocity command.
  Randomized evaluation events/observation corruption are disabled by the evaluation setup.
- Robot initial root: (−2, 0, 0.76) m, directly over sand. The ordinary task starts on the
  rigid approach platform; starting there at zero command would not test sand contact.
- Sand: 6 × 2 × 0.3 m; initial surface world z = 0; rigid floor top z = −0.30 m.
- Both solvers: 890,625 particles; voxel size 0.02 m; 1.25 particles per cell dimension;
  density 1,600 kg/m³; friction 0.48; Young's modulus 1e15 Pa; Poisson ratio 0.3.
- NeRD: Top-K = 7,104, 512 background anchors, selection refresh every four physics steps;
  effective normal mass scale 100, tangential scale 40. Complete settings are in metadata.
- Physics/recording 200 Hz (dt = 0.005 s), policy 50 Hz. CUDA graph capture enabled.
  **Zero native MPM settling/warm-up steps for NeRD.** Initial graph-capture passes are a
  separate implementation detail; do not describe them as native MPM warm-up.
- Robot mass 33.341 kg, weight approximately 327.08 N.
- Matching initial particle/checkpoint hashes, materials, particle count, and timestep were
  checked. These are **independent closed-loop trajectories**, not common-motion replays;
  pointwise force differences are not a measurement of pure NeRD prediction error.

## Policies and earlier findings

Frozen snapshots (hashes and downloaded training configs are in
[benchmark/data/checkpoints](../benchmark/data/checkpoints)):

| Role | W&B project under `jnskkmr` | Run | Checkpoint |
| --- | --- | --- | --- |
| MPM-trained reference; used in all extended tests | `g1_29dof_mpm_mixed_terrain` | `jqur5jhm` | `model_19999.pt` |
| NeRD trained from scratch | `g1_29dof_particle_nerd_mixed_terrain` | `iylbaqt3` | `model_9500.pt` |
| NeRD fine-tuned | `g1_29dof_particle_nerd_mixed_terrain` | `y4t5ybra` | `model_29000.pt` |

The six-second matrix tested all three policies with both solvers at zero command, plus
`jqur5jhm` at 0.5 m/s with both solvers. Both NeRD-trained policies also stepped when evaluated
with MPM. Their learned gait therefore persists without NeRD feedback during evaluation.
This does not establish what caused that behavior during training.

The historical training reward configurations differed substantially: MPM standing-posture
weights were 0.5 for all joints; the older NeRD runs had much weaker penalties for several
leg joints. The user subsequently requested exact reward parity with the MPM moving-patch
reference. That local reward configuration was aligned separately. Do not mistake current
reward parity for parity of the archived training runs. See
[saved_reward_differences.json](../benchmark/data/saved_reward_differences.json).

## Completed online filtering experiment

Only the feedback-filter setting changes between the extended NeRD cases. The existing
coupler `proxy_relaxation` in fixed mode implements:

```text
feedback[k] = alpha * raw_wrench[k] + (1 - alpha) * feedback[k-1]
alpha = 1 - exp(-0.005 / tau_seconds)
```

- Recorder option: `--force_filter_tau_ms`, default 0 (disabled).
- Tested tau = 10 ms (alpha 0.393469) and 20 ms (alpha 0.221199).
  These are **exponential time constants**, not finite moving-average windows.
- Filters the full sand-to-foot wrench: Fx, Fy, Fz **and torque**. Direct rigid-ground
  contact is not filtered. Initial feedback history is zero.
- The filtered buffer feeds the **next lagged rigid step**; this is an online physics
  intervention, not smoothing the saved plot. Particle displacement prediction and its
  settings remain unchanged, but subsequent particle trajectories can change with robot motion.
- Relevant implementation: `particle_nerd/solver.py::coupling_harvest_proxy_wrenches`,
  `isaaclab_contrib/coupling/coupler_cfg.py::CouplerProxyMappingCfg`, and Newton's
  `solver_coupled_proxy.py::_blend_proxy_feedback`.
- All four cases completed 30 s without termination. After the initial capture passes,
  the recorded six-component feedback obeys the EMA recurrence within 0.001 in the
  respective force/torque units. The unfiltered case has zero recurrence residual.

| Case | Fz std over 4–10 s, left/right (N) | Ankle-roll z excursion over 4–10 s, left/right (cm) | Mean summed sand / rigid Fz over 20–30 s (N) |
| --- | ---: | ---: | ---: |
| Implicit MPM | 0.8 / 0.8 | 0.8 / 0.9 | 327.1 / 0.0 |
| NeRD, unfiltered | 86.6 / 92.7 | 5.6 / 9.2 | 0.0 / 327.1 |
| NeRD, EMA 10 ms | 117.0 / 120.5 | 11.7 / 17.3 | 21.7 / 305.3 |
| NeRD, EMA 20 ms | 108.7 / 105.6 | 13.2 / 12.7 | 327.1 / 0.0 |

Unfiltered NeRD chronology:

1. Oscillation begins around 4 s.
2. Direct rigid foot reaction first exceeds 5 N at 15.650 s.
3. Summed sand Fz last exceeds 1 N at 16.735 s.
4. Loads become nearly steady around 19 s, supported by rigid contact.
5. Root z decreases from approximately 0.591 m at 3 s to 0.479 m at 30 s.
   Final ankle-roll body z is −0.238/−0.241 m relative to the **initial** sand surface.
   These are body positions, not foot-sole clearance from the evolving sand surface.

The 10 ms EMA increases early oscillation and eventually also transfers support to the
rigid ground. The 20 ms EMA keeps oscillating through 30 s; during 10–20 s its ankle-roll
z excursions are approximately 50 cm. It does not become a steady standing solution merely
because its late support is predominantly sand. MPM remains steady with sand support.

These results cover one policy, one seed, one fixed flat bed and two filter time constants.
They do not establish the cause of the original instability or rule out every possible filter.
Filter-induced delay is a possible contributor to the worse response, not a proven diagnosis.

## Artifacts and measurement semantics

- [30-second Fx/Fy/Fz PNG](../benchmark/plots/policy_jqur5jhm_zero_command_filtering.png).
- [Foot/root motion and support-source PNG](../benchmark/plots/diagnostics/policy_jqur5jhm_zero_command_filtering_motion.png).
- [Windowed statistics and recurrence validation](../benchmark/plots/filter_metrics.json).
- [Extended traces](../benchmark/data/filter_30s): `mpm_tau0`, `nerd_tau0`, `nerd_tau10`, `nerd_tau20`.
  Each has `trace.npz` and `metadata.json`; the latter includes complete physics settings.
- [Recorder](../benchmark/record.py), [filter comparison plotting](../benchmark/plot_filter.py),
  [original per-policy plotting](../benchmark/plot.py), [checkpoint downloader](../benchmark/download.py).

Trace keys:

- `time`: simulation seconds, one sample after each 0.005 s physics step.
- `force`: signed world-frame force on each foot = `sand_force + rigid_force`, ordered
  left ankle-roll body then right ankle-roll body. Positive z is upward.
- `sand_force`: current coupling feedback; filtered in the EMA cases.
- `rigid_force`: direct rigid-contact reaction, separate from sand coupling.
- `raw_sand_wrench`: NeRD estimator output before filtering; six components, force then torque.
  For MPM this field contains its unfiltered coupling feedback.
- `feedback_wrench`: coupling buffer after filtering, used by the next lagged rigid step.
- `foot_position`, `foot_velocity`, `root_position`, `root_velocity`, `root_quaternion`:
  measured robot states. `action` is sampled at the 50 Hz policy rate.

The original six-second traces and PNGs are preserved. The extended raw run reproduces the
instability but is not bitwise identical: root-position differences through 6 s stay below
3.1 mm compared with the earlier trace. Do not splice the traces as a single continuous run.

Initial offline analysis used a four-sample causal average (20 ms window, 7.5 ms group delay).
It reduced successive total-Fz sample changes but barely changed per-foot Fz variance. That
is a different experiment from the online EMA tests above; avoid conflating the two.

## Reproduce

Run from the repository root. Regenerate figures without launching simulation:

```bash
benchmark=source/isaaclab_tasks/isaaclab_tasks/contrib/g1_29dof_particle_nerd/benchmark
uv run python "$benchmark/plot_filter.py" \
  --data "$benchmark/data/filter_30s" --output "$benchmark/plots"
```

Download the frozen policy snapshots using existing W&B authentication, then rerun:

```bash
uv run --with wandb python "$benchmark/download.py" --output /tmp/grf-checkpoints
checkpoint=/tmp/grf-checkpoints/jqur5jhm/model_19999.pt
for tau in 0 10 20; do
  uv run python "$benchmark/record.py" --backend nerd --checkpoint "$checkpoint" \
    --duration 30 --force_filter_tau_ms "$tau" --output "/tmp/grf-filter/nerd_tau${tau}"
done
uv run python "$benchmark/record.py" --backend mpm --checkpoint "$checkpoint" \
  --duration 30 --output /tmp/grf-filter/mpm_tau0
uv run python "$benchmark/plot_filter.py" --data /tmp/grf-filter --output /tmp/grf-filter/plots
```

Local execution used `/tmp/isaaclab-nerd-foot-forces`, with the existing uv environment reused
and interpreter/import locations verified. Temporary logs/checkpoints are under
`/tmp/nerd-force-filter-30s` and `/tmp/nerd-foot-force-runs/checkpoints`; these are disposable.
Use the archived repository traces/manifests as the durable evidence. Source formatting/lint
checks passed; the already-failing repository changelog gate was skipped. No new unit tests
were added; validation used the actual rollouts, trace invariants, EMA recurrence, and PNG inspection.

## Unresolved work and constraints for the next agent

1. **Prescribed-motion replay has not been implemented.** Replay the same measured foot/body
   motion into both sand solvers to isolate particle/force response from policy feedback.
   Preserve collider geometry, timing, material parameters, particle initialization, and Top-K.
2. Locate the origin of progressive sand-support loss: inspect particles near each foot,
   predicted displacements, selected particle IDs, contact reconstruction, and direct rigid
   contact as the feet descend. Do not assume it is force noise or recycling; this test has no recycling.
3. Separate persistent learned stepping in the historical NeRD-trained policies from the
   instability of the MPM-trained policy under NeRD. Reward confounds and solver response are
   distinct unresolved contributors.
4. Do not silently adjust mass scales, particle resolution, geometry, warm-up, or solver rate.
   Change one variable per ablation and state what changed. Do not enable either tested EMA
   in production based on these results. No filtered-force retraining was performed.
5. The user wants one policy/command per PNG, Fx/Fy/Fz rows and left/right foot columns,
   dark-blue MPM and green NeRD. Use numbered `Figure X:` captions identifying every lettered
   panel. Avoid blue/orange defaults and purple. Label the force frame and measurement reference.
6. No new training submission, cancellation, or solver-default change is authorized by this
   handoff-writing request. The repository contains unrelated moving-patch/training changes;
   preserve them. Benchmark/context artifacts have not been committed or pushed in this work.
