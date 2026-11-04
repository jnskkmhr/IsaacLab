# Fixed-terrain G1 with Particle NeRD

This task integrates the G1 Particle NeRD example as a configurable physics solver.
The flat, ascending and wave demo defaults are pinned to the recorded
`video/multi_terrain_5f1xyc21_zero_warmup_50fps` campaign. It uses a fixed particle
pool: no moving patch, particle recycling, saved terrain pages, native MPM solver,
or MPM warm-up. The configs are local to this task; terrain spawners and MDP
functions are reused from `g1_29dof_mpm`. Neural dynamics and foot reactions live
in `../particle_nerd`.

## Playback

From the `IsaacLab-particle-nerd` checkout:

```bash
uv run --with wandb --extra ovrtx isaaclab play \
  --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Sand-Flat-G1-29dof-ParticleNeRD-Play \
  --num_envs 1 \
  --wandb_run 5f1xyc21 \
  --wandb_project g1_29dof_mpm_mixed_terrain \
  --viz newton_rtx \
  'env.video_recorders=[]'
```

The particle dynamics checkpoint is bundled at `../particle_nerd/checkpoint/best.pt`
and resolved relative to `env_cfg/physics_cfg.py`;
`--wandb_run` selects the robot policy. Physics runs at 200 Hz and policy control at 50 Hz.

Replace the task ID to select a terrain. Each also has a registration without
`-Play`; material randomization is disabled in both because neural material
features are fixed when the solver is constructed.

| Terrain | Task ID |
| --- | --- |
| Flat | `IsaacContrib-Velocity-Sand-Flat-G1-29dof-ParticleNeRD-Play` |
| Ascending slope | `IsaacContrib-Velocity-Sand-Ascending-G1-29dof-ParticleNeRD-Play` |
| Descending slope | `IsaacContrib-Velocity-Sand-Descending-G1-29dof-ParticleNeRD-Play` |
| Two wave/dune ridges | `IsaacContrib-Velocity-Sand-Dune-G1-29dof-ParticleNeRD-Play` |
| Tilting tray, stance | `IsaacContrib-Balance-Sand-Tilting-G1-29dof-ParticleNeRD-Play` |
| Tilting tray, walking | `IsaacContrib-Velocity-Sand-Tilting-G1-29dof-ParticleNeRD-Play` |

The corresponding reference task replaces `ParticleNeRD` with `MPM` in the ID.
The current MPM task defaults differ from the recorded demo. Apply the same material,
sampling, bed dimensions, policy, commands and initial state to that reference before
comparing trajectories.

## What the fixed scene controls

`env_cfg/scene_cfg.py` and `env_cfg/physics_cfg.py` preserve the recorded G1 demo:

| Parameter | Recorded value |
| --- | --- |
| Material density / friction | 1600 kg/m³ / 0.48 (`mild_sand`) |
| Young's modulus / Poisson ratio | 1e15 Pa / 0.3 |
| Sampling voxel size / particles per cell dimension | 0.02 m / 1.25 |
| Flat bed dimensions | 6 × 2 × 0.3 m |
| Particle count, flat/ascending/wave | 890,625 |
| Normal / tangential effective-mass scale | 100 / 40 |
| Horizontal / upward scale; normal exponent | 8 / 2; 3 |
| Contact iterations / relaxation | 200 / 0.02 |
| Top-K selection / background anchors | automatic K = 7,104 / 512 |
| Selection refresh interval | 4 physics steps (50 Hz) |
| Particle inference / rigid coupling | every physics step (200 Hz) |
| Foot collision margin | 0.02 m |

The network reads collider motion at the beginning of each physics step. The
force estimator separately reads the virtual proxy state after rigid integration
and lagged-force removal. Its virtual masses come from Newton's articulated
inertia calculation (about 1.964 kg per foot for this G1 setup); they are distinct
from the 100/40 multipliers applied to particle mass in material response.

All material fields are explicit in the scene config. The flat/ascending/wave Play
variants use deterministic root placement and a 1 m/s forward command, matching
the recording. These are archived example settings, not newly tuned parameters.
The saved campaign covers these three terrains; it does not establish equivalent
rollouts for the tilting variants.

Flat terrain supplies the actual retaining-box poses, dimensions and friction to
the neural feature encoder. Slope and wave terrain use the G1 example's vertical
height chart and five intrinsic boundary descriptors. The tilting case expresses
particles and foot colliders in the powered tray's frame, including relative
collider velocities. Predicted positions are transformed back to world space.
Environment origins are removed independently for every world.

For beds larger than 3.1 m, the default follows `examples/particle_nerd/g1/adapter.py`:
it uses a canonical feature window around the feet. This changes network inputs;
it does not move or recycle the fixed particle pool. Disable it explicitly to test
features from the entire fixed bed:

```bash
'++env.sim.physics.solver_cfg.entries.1.solver_cfg.fixed_bed.sliding_window=false'
```

The network predicts particle motion, followed by the example's `project_tray_bounds`
postprocessing. There is no native particle collision solve. Foot reactions are
reconstructed by the shared neural solver. The powered tray is
actuated by MJWarp; this implementation does not reconstruct particle reaction
wrenches on the tray. Tilting also changes gravity expressed in the tray frame,
but this integration keeps the network's gravity features fixed instead of rotating
them into the tray frame. The four-terrain parity validation does not cover tilting.

`examples/particle_nerd_batch` uses a single-intruder schema and a different
contact-estimation path. Matching its appearance is not a numerical parity test
for this two-foot G1 checkpoint. The directly comparable neural path is
`examples/particle_nerd/g1`.

## Diagnostic sequence

1. Compare this fixed task with its MPM counterpart using the same initial state
   and material. Inspect particle trajectories and foot reaction forces separately
   from whether the RL policy succeeds.
2. Compare identical recorded particle histories and collider states through the
   example and packaged neural inference paths. This isolates feature and network
   migration from feedback-driven rollout divergence.
3. Compare fixed and moving-patch NeRD with identical material, particle spacing,
   checkpoint, contact settings and active-particle count. Recycling, boundary
   pinning and terrain input coordinates then remain explicit differences.
4. Test slope, wave and tilting separately. A failed policy rollout alone cannot
   distinguish a contact-model error from particle prediction error or policy OOD.

Solver controls are under `env.sim.physics.solver_cfg.entries.1.solver_cfg`,
including `active_k`, `selection_interval`, `anchor_particles`, `effective_mass_scale`,
`tangent_mass_scale`, `compile_model`, and `fixed_bed.sliding_window`.
See [validation results](VALIDATION.md) for the scope of the checks performed.
