# G1 foot ground-reaction comparison — 2026-11-06

## Problem

Compare the forces exerted by sand on the G1 feet with implicit MPM and Particle NeRD,
and investigate stepping at a zero velocity command. This benchmark uses the **fixed flat
sand bed in `g1_29dof_particle_nerd`**, without moving-patch recycling.

The measurements show two effects:

1. Both NeRD-trained policies lift their feet repeatedly **even when evaluated with native MPM**.
   Their stepping behavior therefore does not require NeRD force feedback during playback.
2. The MPM-trained policy stands steadily with MPM but develops oscillating load transfer with
   NeRD after roughly 3–4 seconds. The closed-loop solver response also matters.

These observations do not establish whether force dynamics, rewards, or another training
difference originally caused the learned gait. They do rule out treating the problem solely
as high-frequency noise in the currently evaluated NeRD force signal.

## Possible reasons

### The rewards are similar, but the standing-posture penalty differs substantially

The NeRD moving-patch configuration used to train the measured policies had weaker standing
posture weights than the MPM reference. The downloaded training configurations confirm this
difference for the actual runs (the local reward configuration was subsequently aligned with MPM):

| Joint family | MPM `jqur5jhm` standing weight | NeRD `iylbaqt3` and `y4t5ybra` | Reduction |
| --- | ---: | ---: | ---: |
| Hip pitch, knee, ankle pitch, ankle roll | 0.5 | 0.02 | 25× |
| Hip yaw | 0.5 | 0.08 | 6.25× |
| Hip roll | 0.5 | 0.3 | 1.67× |
| Waist and arms | 0.5 | 0.5 | None |

These coefficients multiply absolute joint-angle deviations inside `variable_posture_l1`;
the outer `joint_deviation` reward weight is −1.0. The standing branch applies below a
combined commanded linear/angular speed of 0.5, including zero command.

The saved MPM run had no `feet_slide` term; both NeRD runs have it at −0.25. The current
MPM source now also has that term, so inspecting current source alone misses this training
difference. The foot-clearance ground-height offset was 0.25 m in the saved MPM configuration
and 0.30 m in NeRD, matching their different configured sand depths. This is a geometry
difference, not evidence by itself of a reward bug.

Several shared terms permit alternating steps at zero command:

- `feet_air_time_positive_biped` gives no reward at zero command.
- `reward_soft_landing` also disables its touchdown-force penalty at zero command.
- `no_fly` penalizes both feet being airborne; it does not require both feet to stay in contact.
- `feet_slide` penalizes horizontal foot speed during contact, not vertical lifting of a swing foot.

See [saved reward differences](data/saved_reward_differences.json) and the archived
`data/checkpoints/<run>/training_config.json` files. No reward or solver parameters were
changed as part of this benchmark.

### Force variation is mostly slower than a short averaging window

A four-sample causal average at 200 Hz spans 20 ms and has 7.5 ms group delay.
Applied offline, it reduces the RMS difference between consecutive total vertical-force
samples by about 20–31% for NeRD. However, the standard deviation of each foot's vertical
force decreases by **less than 1%**. Most of the visible variation is the slower gait/load-transfer
cycle, which survives this filter. Only about 1.1–2.3% of the NeRD total vertical-force
spectral power is above 20 Hz in these traces.

Filtering might still affect closed-loop stability, because it changes feedback phase and gain.
The offline plots cannot establish that benefit. The initial six-second comparison used no
filter. The subsequent 30-second closed-loop filtering experiment is reported below; no
filtered-force training was performed and production defaults remain unchanged.

## Verification

### Matched evaluation setup

| Setting | Value |
| --- | --- |
| Robot | G1 29 DoF, identical robot/foot collision configuration in both solvers |
| Scene | Fixed 6 × 2 × 0.3 m flat sand bed from the NeRD task |
| Initial robot root position | (−2.0, 0.0, 0.76) m, directly over sand |
| Initial sand surface | World z = 0 m |
| Particle sampling | 0.02 m voxel size; 1.25 particles per cell dimension |
| Particle count | 890,625 for both solvers |
| Particle density / friction | 1,600 kg/m³ / 0.48 |
| Young's modulus / Poisson ratio | 1e15 Pa / 0.3 |
| NeRD selection | Top-K = 7,104, 512 background anchors; refresh every 4 physics steps |
| NeRD material-response scales | Normal 100; tangential 40; other fixed-task settings unchanged |
| Physics / policy rate | 200 Hz / 50 Hz |
| Recording rate | 200 Hz, after each physics step |
| Explicit native settling/warm-up | None |
| Initial random seed | 42 |
| Duration | 6 s per rollout; statistics use 1–6 s |
| Commands | All policies: zero; additionally `jqur5jhm`: 0.5 m/s forward |

The default fixed task starts on the rigid approach platform. That would not test sand at
zero command, so both solvers instead start G1 directly above the sand. Material/reset
randomization and observation corruption are disabled. The actor uses deterministic inference,
the saved training actor architecture, and the saved observation-history length. Action joint
ordering, scale, offset mode, and order preservation are checked against the training config.

Both solvers use the NeRD scene, observation, action, command, event, reward, and termination
configs. The MPM comparison changes the particle solver to native implicit MPM at 0.02 m.
All paired rollouts have identical initial particle-position checksums, policy checksums,
material settings, and particle count. Robot mass is 33.341 kg (weight approximately 327.08 N).

Forces are the world-frame coupling feedback exerted on the left/right ankle-roll bodies,
plus direct rigid-ground contact forces. Positive z is upward. Raw signed components are
retained, including short downward reaction transients; these are the solver's coupling
reactions, not a separate physical load-cell measurement. Direct rigid-ground force was zero
in all six standing rollouts, so those plots contain sand reactions only. The force samples
are taken before policy-step reset handling, not downsampled from the 50 Hz task cache.

These are **independent closed-loop trajectories**, not identical-foot-motion replays.
Consequently, pointwise NeRD-minus-MPM force error would combine different contact motions
with solver differences and is not reported as prediction accuracy.

### Frozen policy snapshots

| Policy | Run | Checkpoint |
| --- | --- | --- |
| MPM trained from scratch | [jqur5jhm](https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_mpm_mixed_terrain/runs/jqur5jhm) | `model_19999.pt` |
| NeRD trained from scratch | [iylbaqt3](https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_particle_nerd_mixed_terrain/runs/iylbaqt3) | `model_9500.pt` |
| NeRD fine-tuned | [y4t5ybra](https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_particle_nerd_mixed_terrain/runs/y4t5ybra) | `model_29000.pt` |

The NeRD runs were still training when downloaded. Checkpoint SHA-256 hashes are retained
in the manifests; subsequent run progress does not change this comparison. Training histories
and iteration counts differ, so this is not a comparison at equal training budget.

### Results at zero command

All six rollouts completed the full six seconds without termination or reset. Values below
are left/right foot measurements over 1–6 s.

| Policy | Solver | Vertical force standard deviation (N) | Foot-link vertical excursion (cm) |
| --- | --- | --- | --- |
| MPM trained | MPM | 1.6 / 1.5 | 1.3 / 1.3 |
| MPM trained | NeRD | 61.6 / 67.3 | 4.4 / 3.2 |
| NeRD scratch | MPM | 167.0 / 160.8 | 15.0 / 14.5 |
| NeRD scratch | NeRD | 168.9 / 160.2 | 16.1 / 18.7 |
| NeRD fine-tuned | MPM | 179.5 / 169.5 | 16.8 / 17.5 |
| NeRD fine-tuned | NeRD | 173.0 / 172.3 | 18.4 / 21.0 |

Mean total vertical foot force is 326.0–327.3 N across the six cases, close to body weight.
The distinction is time-varying load distribution and foot motion, not a large mean support-force bias.
Vertical excursion is max-minus-min **ankle-roll body position relative to the initial sand
surface**, not foot-sole clearance above the evolving sand surface.

- [Figure 1: MPM-trained policy, zero command, PNG](plots/policy_jqur5jhm_zero_command.png)
- [Figure 2: NeRD scratch policy, zero command, PNG](plots/policy_iylbaqt3_zero_command.png)
- [Figure 3: NeRD fine-tuned policy, zero command, PNG](plots/policy_y4t5ybra_zero_command.png)
- [Figure 4: MPM-trained policy, 0.5 m/s command, PNG](plots/policy_jqur5jhm_v=0.5.png)
- [Figure 5: foot-link motion diagnostic, PNG](plots/diagnostics/foot_motion_zero_command.png)
- [Figure 6: offline filter and force spectra diagnostic, PNG](plots/diagnostics/force_filter_diagnostic.png)
- [Numerical metrics](plots/metrics.json)

Each policy/command PNG overlays both solvers, with Fx, Fy, Fz rows and left/right foot columns.
The components are signed and expressed in the world frame. Each PNG includes a numbered
caption identifying its lettered panels. Dark blue denotes implicit
MPM; green denotes Particle NeRD. `trace.npz` retains all three force components, separate
sand/rigid forces, foot positions/velocities, root positions/velocities, and policy actions.

### Extended zero-command test: does filtering resolve the oscillation?

**The unfiltered NeRD oscillation subsides around 19 seconds because the feet sink to the
rigid ground beneath the sand. It is not recovery of steady support by the sand.** Direct
rigid contact first exceeds 5 N at 15.650 s. The summed sand Fz last exceeds 1 N at 16.735 s;
from 20–30 s, rigid contact supplies approximately the full 327 N robot weight. The robot
root drops from approximately 0.591 m at 3 s to 0.479 m at 30 s. The left/right ankle-roll
body positions end at z = −0.238/−0.241 m relative to the initial sand surface (z = 0).
The rigid ground beneath the sand has its top at z = −0.30 m.

Neither tested feedback filter solves the problem. Both cause larger force and foot-motion
oscillations during 4–10 s. The 10 ms case eventually also transfers support to the rigid
ground. The 20 ms case continues oscillating through 30 s, including substantial excursions
during 10–20 s. Completion without a termination signal is not evidence of satisfactory standing.

| Solver / feedback | Foot Fz std, 4–10 s, L/R (N) | Ankle-roll z excursion, 4–10 s, L/R (cm) | Mean summed sand / rigid Fz, 20–30 s (N) | First rigid foot reaction >5 N (s) |
| --- | ---: | ---: | ---: | ---: |
| Implicit MPM | 0.8 / 0.8 | 0.8 / 0.9 | 327.1 / 0.0 | None |
| NeRD, unfiltered | 86.6 / 92.7 | 5.6 / 9.2 | 0.0 / 327.1 | 15.650 |
| NeRD, EMA 10 ms | 117.0 / 120.5 | 11.7 / 17.3 | 21.7 / 305.3 | 12.365 |
| NeRD, EMA 20 ms | 108.7 / 105.6 | 13.2 / 12.7 | 327.1 / 0.0 | 11.565 |

The same frozen `jqur5jhm/model_19999.pt` policy, seed 42, zero command, initial particles,
material, and 200 Hz physics / 50 Hz policy rates were used for all four 30-second rollouts.
All four completed 30 s. The extended unfiltered NeRD run reproduces the onset near 4 s;
GPU rollouts are not bitwise identical to the earlier recording (root-position differences
remain below 3.1 mm through 6 s). The archived six-second plots are preserved.

The experiment uses the existing coupler's fixed `proxy_relaxation`, applied online to the
complete sand-to-foot wrench (Fx, Fy, Fz and torque), before the next lagged rigid step:

```text
feedback[k] = alpha * raw_wrench[k] + (1 - alpha) * feedback[k-1]
alpha = 1 - exp(-physics_dt / tau)
```

For tau = 10/20 ms, alpha = 0.393469/0.221199. These are **exponential time constants**,
not finite moving-average window lengths. Initial feedback is zero. Direct rigid-ground
contact forces are not filtered. NeRD displacement predictions, material-response scales,
selection settings, and policy weights are unchanged. CUDA graph capture remains enabled.
After the initial graph-capture passes, recorded feedback agrees with the configured
recurrence for all six wrench components within 0.001 in their respective units.

- [Figure 7: Fx, Fy, Fz over 30 seconds](plots/policy_jqur5jhm_zero_command_filtering.png)
- [Figure 8: foot/root motion and sand-versus-rigid support](plots/diagnostics/policy_jqur5jhm_zero_command_filtering_motion.png)
- [Windowed metrics and filter validation](plots/filter_metrics.json)
- Recorded traces and complete configurations: `data/filter_30s/<case>/`.

**Recommendation:** leave filtering disabled by default. These two EMA settings worsen this
closed-loop standing test; smoother instantaneous feedback would not by itself demonstrate a
fix. Filter delay is a possible contributor, but this experiment does not isolate the cause
of the original instability or the progressive loss of sand support. A prescribed-foot-motion
comparison should next separate NeRD particle/force response from policy feedback. This result
is limited to one policy, seed, fixed flat bed, and the two tested filter time constants.

Reproduce after downloading the frozen checkpoint as described below:

```bash
benchmark=source/isaaclab_tasks/isaaclab_tasks/contrib/g1_29dof_particle_nerd/benchmark
checkpoint=/tmp/grf-checkpoints/jqur5jhm/model_19999.pt
for tau in 0 10 20; do
  uv run python "$benchmark/record.py" --backend nerd --checkpoint "$checkpoint" \
    --duration 30 --force_filter_tau_ms "$tau" --output "/tmp/grf-filter/nerd_tau${tau}"
done
uv run python "$benchmark/record.py" --backend mpm --checkpoint "$checkpoint" \
  --duration 30 --output /tmp/grf-filter/mpm_tau0
uv run python "$benchmark/plot_filter.py" --data /tmp/grf-filter --output /tmp/grf-filter/plots
# Regenerate the archived extended-test figures:
uv run python "$benchmark/plot_filter.py" --data "$benchmark/data/filter_30s" --output "$benchmark/plots"
```

### Reproduce

Run from the repository root. Download the exact archived policy snapshots with your existing
W&B authentication; policy weights are not duplicated in this benchmark directory.

```bash
benchmark=source/isaaclab_tasks/isaaclab_tasks/contrib/g1_29dof_particle_nerd/benchmark
uv run --with wandb python "$benchmark/download.py" --output /tmp/grf-checkpoints

for run in jqur5jhm iylbaqt3 y4t5ybra; do
  case "$run" in
    jqur5jhm) iteration=19999 ;;
    iylbaqt3) iteration=9500 ;;
    y4t5ybra) iteration=29000 ;;
  esac
  for backend in mpm nerd; do
    uv run python "$benchmark/record.py" \
      --backend "$backend" --checkpoint "/tmp/grf-checkpoints/$run/model_$iteration.pt" \
      --speed 0 --duration 6 --seed 42 --output "/tmp/grf-reproduction/${run}_${backend}_0"
  done
done

for backend in mpm nerd; do
  uv run python "$benchmark/record.py" \
    --backend "$backend" --checkpoint /tmp/grf-checkpoints/jqur5jhm/model_19999.pt \
    --speed 0.5 --duration 6 --seed 42 --output "/tmp/grf-reproduction/jqur5jhm_${backend}_0.5"
done

uv run python "$benchmark/plot.py" --data /tmp/grf-reproduction --output /tmp/grf-reproduction/plots
# Or regenerate the included PNGs from the included measurements:
uv run python "$benchmark/plot.py" --data "$benchmark/data" --output "$benchmark/plots"
```

Validation covered eight six-second rollouts (six standing, two walking), both physics solvers,
matching particle initialization and materials, finite raw forces, checkpoint/action compatibility,
and regeneration of the PNGs. Results use one seed and a short fixed-bed rollout. They do not
measure recycling effects or establish the cause of the original moving-patch training behavior.

## Test plan

1. As the first training ablation, restore MPM's standing-posture weights in NeRD while leaving
   force estimation unchanged. Compare zero-command foot motion at matched training iterations.
2. Investigate the MPM-trained policy's load-transfer oscillation under NeRD separately from
   the already learned stepping gait. A prescribed-foot-motion replay would isolate material
   response from feedback-driven trajectory divergence.
3. The completed 30-second EMA ablation does not justify enabling filtering. Before considering
   other filter settings, investigate the loss of sand support and distinguish true standing
   stability from support supplied by the rigid ground beneath the sand.

No training jobs, rewards, material parameters, or production force paths were modified.
