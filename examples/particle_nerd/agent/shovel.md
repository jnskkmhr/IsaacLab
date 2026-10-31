# Shovel experiment

For the optimized batched runner, current Transolver interfaces, measured MPM comparison and the separate G1 NN impulse models, read [transolver_and_batch.md](transolver_and_batch.md). This file retains the single-shovel experiment history; settings below belong to their named experiments.

The shovel uses the geometry from `newton/examples/mpm_moving_patch_beta/impl/shovel.py`, copied with its Apache-2.0 header into `shovel/source_geometry.py`. `shovel/asset.py` exports a portable USD, recentered collision meshes, a 5 mm signed-distance grid, and metadata. Preserve the separate blade prisms: a single convex hull would fill the bowl. The cylindrical neck is approximated by a 48-sided mesh. Mass and the full inertia tensor are retained from the reference builder (20.5064708 kg); all generated geometry is translated so the rigid-body origin is its center of mass.

## Collection

```bash
PYTHONPATH=examples uv run --no-sync python -m particle_nerd.data_collection.collect \
  --geometry shovel --case continuous-drag --drag-path wide-circle \
  --seed 710000 --output /path/to/reference --name wide-circle
```

Collection uses IsaacLab manager terms and the existing MPM–MJWarp coupler. The shovel is dynamic, moved by external forces, and is never position-overwritten during dragging. The bed is initially 1 × 1 × 0.5 m, without walls, with a 0.04 m MPM grid and 16,384 jittered particles. MPM and MJWarp use a 5 ms interval; particle frames are saved every 20 ms. Body and force records are saved every 5 ms. The generated asset must exist before running collection; regenerate with `python -m particle_nerd.shovel.asset` only when intentionally changing geometry.

`data_collection.campaign --geometry shovel` records the fixed reference shovel mass and randomizes initial motion and drag/push commands. It retains deterministic whole-episode train/validation/test splits. `osmo/shovel_collection.yaml` configures 128 episodes across 16 workers. Partial checkpoint directories can contain campaign JSON without HDF5: `osmo/shovel_training.yaml` normalizes this output, validates all episode IDs, and rejects conflicting duplicates before training.

## Particle predictor

`transolver/inference.py::manifest_colliders` identifies the shovel with geometric kind 3 and the fixed ground with kind 0. These are geometry types, not semantic classes. Each particle queries the shovel SDF and the ground; pair features include signed gap, normal, relative surface velocity and friction. Weighted collider embeddings enter the particle embedding. No shovel surface nodes or region selector are used.

`shovel/sdf.py` implements differentiable Warp trilinear SDF interpolation. `transolver/features.py` propagates gradients to particle position history during recurrent training. The predictor outputs displacement only; its velocity is displacement / 20 ms.

`transolver.train --initialize checkpoint.pt --rollout-steps 25 --loss-mode balanced` fine-tunes compatible box weights with a 0.5-second recurrent loss. Initialization is not a shovel accuracy result. Checkpoints store the geometry and body extents in config. Whole-split evaluation uses `transolver.evaluate --all-episodes --mode all`; all particles remain recurrent during a five-second rollout, while recorded shovel states are supplied at each step.

## Dynamic coupling

`runner/simulation.py` and `runner/rigid.py` support the shovel's measured inertia and compound geometry. `force_estimation/estimator.py` uses the same shovel SDF as the particle model. The estimator's auxiliary material-velocity correction remains enabled: `fixed_material_velocity=False`. The current shovel benchmark explicitly selects effective mass multiplier 180; it is an approximation, not an identified material law.

`transolver.closed_loop` runs predicted particles → estimated material wrench → MJWarp rigid motion. It contains no MPM solver on the learned side. The bounded external drag target is shared with the reference, but each simulation computes its force from its own shovel state. Initial particles/body states are shared; subsequent reference states are used only for comparison. Ground contact remains in MJWarp. The supplied material velocities are not overwritten by estimator correction; correction is internal to contact reconstruction.

## Validation and reporting

- GPU tests: `test_transolver.py`, `test_gpu_interfaces.py`, `test_shovel.py`.
- New geometry test checks the empty bowl and a finite-difference gradient.
- Collector asserts imported mass, COM and full inertia tensor.
- Separate contact benchmark uses true MPM particles to distinguish force-estimator error from particle-prediction error.
- Website: `https://jnskkmhr.github.io/particle-nerd-plan/shovel.html`.
- Report/render scripts and local references: `/home/jkamohara/isaac/particle_nerd/experiments/particle_gnn/shovel/`.
- Do not label recorded-body-motion evaluation as fully dynamic learned coupling. Do not label a smoke test with old box weights as trained shovel prediction.

The final dynamic shovel benchmark uses `--contact-relaxation 0.025 --contact-iterations 10000`. This reduces the parallel Jacobi impulse increment; it does not change effective mass or average forces over time. With the original 0.1 relaxation, the figure-eight solve diverged at step 147 with 152 contacts despite finite particle predictions. At 0.025, all 250 steps of both paths converge (maximum 2,110 iterations). Preserve this numerical setting when reproducing the published videos. Convergence does not imply accurate coupled motion: the full five-second position RMS errors are approximately 7.5 cm (circle) and 22.5 cm (figure eight).

## Completed run (2026-10-20)

- Collection: OSMO `particle-shovel-collect-1`, 128 episodes; 99 train / 16 validation / 13 test.
- Training: OSMO `particle-shovel-transolver-1`, 1,000 updates on four L40S GPUs; selected step 1,000.
- Local selected checkpoint: `/home/jkamohara/isaac/particle_nerd/particle_nerd_plan/results/shovel/training/model/best.pt`. SHA-256: `c749f82b2139e08ae4229e913ca5cfdfba27b46263699aa2f7a353fb7d35b959`. Checkpoints and HDF5 files are local/cloud artifacts, not committed to the website.
- Final local dynamic rollouts: website repository `results/shovel/closed-loop/{wide-circle,figure-eight}`. The original 0.1-relaxation diagnostic runs are preserved under `results/shovel/closed-loop-relaxation-0.1/`.
- Published measurements: `results/shovel/summary.json` and `results/shovel/training-summary.json`; videos under `assets/shovel/{recorded,dynamic}-{wide-circle,figure-eight}.mp4`. Website commit `cf39409`.
- Five-second recorded-motion final particle RMSE: circle 8.57 cm, figure eight 7.57 cm. Full dynamic shovel COM position RMS: 7.46 / 22.51 cm; orientation RMS: 56.07 / 17.84 degrees. Neither numerical convergence nor short-horizon training error establishes accurate dynamic coupling.

## Interactive demo

`runner.play` now detects Transolver checkpoints and initializes position history at the settled frame (no future trajectory replay). See the interactive shovel command in `../README.md`. Current interactive defaults are ×180 effective mass, 0.075 contact relaxation, 2,000 maximum iterations, with the original 1e-9 impulse-residual tolerance. The published original videos retain their 0.025 setting. Right-drag applies the Newton viewer picking wrench at the selected point; Space pauses; the window stays open until closed unless `--steps` is supplied. The default viewer is Newton OpenGL; `--viz newton_rtx` selects Newton RTX with the same picking-force integration.

## Iteration optimization (2026-10-20)

The interactive runner now uses 0.075 Jacobi relaxation with a 2,000-iteration safety limit, stopping early at the original 1e-9 residual. Do not confuse the limit with actual work: medians are 458 (circle) and 613 (figure eight), maximum 698, with all 500 solves converged. Identical-state replay of all 500 original rollout states matches the old converged wrench to force RMS below 1.2e-10 N and torque RMS below 3.8e-9 N m. The five-second off-center picking test, including an upward drag target and release, also converges throughout (median 646, maximum 782, up to 181 contacts).

Capping the original 0.025 solver at 100–800 iterations is not an equivalent optimization: force truncation errors change accumulated shovel rotation. The current choice accelerates convergence instead of accepting that truncation. Synchronized physics-step medians before/after: 26.94 → 12.03 ms (circle), 60.17 → 22.36 ms (figure eight), excluding rendering, diagnostics, disk writes and the first ten steps. These are single development rollouts, not general runtime guarantees.

Benchmark code: `shovel/iteration_budget.py`; closed-loop timing: `transolver/closed_loop.py`. Raw experiment records and comparison script: `/home/jkamohara/isaac/particle_nerd/experiments/particle_gnn/shovel/iteration-budget/`.

## Rigid-contact capacity fix (2026-10-20)

The rigid adapter no longer forces `nconmax=128` and `njmax=128`; Newton sizes both buffers from the scene. The current shovel scene allocates 2,640 rigid-contact slots. This buffer holds direct shovel–ground contacts, independently of the material-contact count in the analytical estimator. The user's step-550 pose generated 195 rigid contacts; the previous adapter sent only 128 to MJWarp. The regression `test_shovel_ground_contacts_are_not_truncated` reproduces that failure before the fix and verifies all contacts reach MJWarp afterward, without constraint overflow.

`runner.play` now prints the failing step even between periodic log entries and names nonfinite particle position/velocity, estimated wrench, rigid pose/velocity, or solver-buffer overflow. Diagnostics report generated/accepted rigid contacts and capacity. Simulation stepping remains on the GPU; diagnostic readback remains outside the physics step.

Validation: seven focused GPU/shovel tests pass. Two 600-step (12-second) learned simulations—unforced settling and off-center mouse drag followed by release—remain finite, converge on every estimator step, and have no rigid-contact truncation or buffer overflow. These are not a replay of the user's exact mouse interaction. The user's estimator non-convergence at steps 275 and 300 preceded rigid-contact overflow and is a separate unresolved case; the capacity fix does not prove arbitrary dragging converges.

## Current pipeline profile (2026-10-20)

`runner/profile.py` profiles the actual coupled runner without changing physics. It runs the circle and figure-eight force-controller trajectories twice: synchronized wall timing without instrumentation, then CUDA-event stage timing. Ten warmup steps are excluded from each 250-step run. Nested stage totals overlap; do not add the whole-estimator row to its kernel rows. Diagnostic readback and rendering are outside timings. The resulting final poses matched exactly between passes.

Current RTX 5090 / 16,384-particle measurements, circle / figure-eight medians: neural network 1.522 / 1.532 ms; velocity-gradient fitting 0.161 / 0.159 ms; iterative impulse solve 7.644 / 17.854 ms; whole force estimator 7.921 / 18.130 ms; four rigid substeps 0.676 / 0.677 ms. Uninstrumented physics wall time: 10.731 / 20.572 ms. Iterative solving occupies 96.8% / 98.4% of mean estimator GPU time. These use constant relaxation 0.075, effective mass ×180, tolerance 1e-9, and the rigid-contact capacity fix. They do not measure RTX viewer FPS or establish force accuracy.

Run from this worktree with `PYTHONPATH=examples uv run --extra ovrtx python -m particle_nerd.runner.profile --checkpoint CHECKPOINT --reference-dir REFERENCE_DIR --output profile.json`. The example module is not an installed package, so this module invocation needs `PYTHONPATH=examples`; the normal shovel demo launcher still does not. References must contain `wide-circle.hdf5` and `figure-eight.hdf5`. Published measurements: `results/shovel/force-pipeline-profile.json` on the plan website; presentation: `compute-profile.html#shovel-current`.

## Contact impulse warm start (2026-10-20)

Implemented as an opt-in experiment, disabled by default. Use `ContactEstimator(warm_start=True)`,
`CoupledSimulation(contact_warm_start=True)`, or the demo / `transolver.closed_loop` flag
`--contact-warm-start`. It introduces no learned component or CPU state transfer.

Each estimator owns one rigid body and persistent material-particle indices. Cache converged
contact impulses in world coordinates, transform them into the current contact frame, clamp the
normal impulse nonnegative and project tangential impulse onto the current friction disk. Initialize
auxiliary body velocity from the same impulses before iterating; the existing material response
already includes each contact's accumulated impulse. Contact candidates still use unmodified free
motion. Clear cached impulses for separated contacts, nonfinite impulses and unconverged solves.
Call `estimator.reset()` on the simulation stream after an episode reset, teleport or particle
reordering; construction starts with an empty cache. Reset does not alter the physical state.

Validation uses the same 0.075 relaxation, effective mass ×180, 1e-9 residual tolerance and 2,000
iteration cap. All 500 recorded-state solves converge with both initializations. Force-vector RMS
differences (warm minus cold): circle 9.65e-7 N, figure-eight 1.51e-8 N. Median whole-estimator wall
time: 8.524 → 8.577 ms and 17.943 → 18.004 ms. Median iterations: 459 → 463 and 613 → 560.5.
These timings are synchronized, exclude compilation and include cache initialization/update.

Paired five-second closed-loop runs also converge throughout. Median physics-step wall time:
circle 12.035 → 12.412 ms, figure-eight 22.020 → 22.174 ms, excluding the first ten steps,
rendering and logging. Warm versus cold body-position RMS differences: 5.15 / 0.90 mm;
orientation RMS: 2.60 / 0.126 degrees. Circle maximum position/orientation differences:
13.44 mm / 6.24 degrees. These are initialization comparisons, not accuracy against MPM.
No speedup is established; do not enable this by default or claim trajectory equivalence.

Reproduction scripts and raw results:
`/home/jkamohara/isaac/particle_nerd/experiments/particle_gnn/shovel/warm-start/{benchmark.py,dynamics.py,replay.json,closed-loop-comparison.json}`.
Run scripts from this worktree using `PYTHONPATH=examples uv run --extra ovrtx python SCRIPT`.
The recorded-state benchmark uses every consecutive state (`--stride 1`). Six GPU interface tests
cover analytical impact, repeated warm starts, rotated contact frames, contact release/reset,
fixed-material support, and absence of host transfers in the coupled step. Repository-wide
changelog checks remain blocked by pre-existing unrelated fragment errors.

## Current shovel default: body-velocity solve (2026-10-20)

The ordinary `play_shovel.py` launcher now selects `--contact-solver body_velocity`.
Use `--contact-solver jacobi` for the previous solver. `--contact-warm-start` alone
selects the optional Jacobi warm-start experiment; combining it with explicit
`--contact-solver body_velocity` is rejected. Programmatic `ContactEstimator` and
`CoupledSimulation` defaults remain Jacobi for reproducibility; select
`contact_solver="body_velocity"` explicitly. `transolver.closed_loop` exposes the
same CLI option. `runner.profile` defaults to the new shovel solver.

Implementation: `force_estimation/body_velocity.py` eliminates independent finite
material contact mobilities to express each contact impulse as a function of the
body's six post-contact velocity components. A semismooth Newton solve with an
analytic Jacobian, pivoted 6 × 6 elimination and backtracking enforces body momentum
balance. It uses the same supplied particle velocity, contact geometry, effective
mass and Coulomb law. It does not learn forces or change predicted particles.
This is a numerical Newton iteration implemented in Warp, not an MPM call.

Initialization starts from free body velocity, has a 30-iteration limit and targets
1e-12 in the six-component velocity residual. If the final residual is nonfinite
or exceeds 1e-8, discard the initializer and cold-start the existing Jacobi solver.
Otherwise initialize both impulses and the consistent auxiliary body response.
Always finish with the original 1e-9 impulse-correction check (maximum 2,000 Jacobi
iterations). `diagnostics()` separates `body_velocity_iterations` and
`impulse_iterations`; `iterations` is their sum. It also reports initializer
residual and fallback. No GPU-to-CPU transfer occurs inside estimate/step.

Scope: one rigid body with finite effective particle masses. Fixed-material-velocity
mode automatically uses Jacobi. This is not an articulated G1 or multiple-body
implementation. Warp loop unrolling and backward generation are disabled in this
module to avoid excessive inference-kernel compilation. First compilation is
excluded from all speed measurements.

Final validation on RTX 5090 / 16,384 particles:

- wide-circle: whole-estimator median 7.871 → 0.565 ms (13.9×); full-step median 10.514 → 3.160 ms (3.3×). Same-state force RMS difference 1.39e-06 N; torque 5.01e-08 N m. Paired body-position/orientation RMS differences 6.96 mm / 1.66°. Position RMS error against MPM 7.46 → 7.39 cm.

- figure-eight: whole-estimator median 17.842 → 0.639 ms (27.9×); full-step median 21.628 → 3.250 ms (6.7×). Same-state force RMS difference 2.72e-10 N; torque 3.23e-08 N m. Paired body-position/orientation RMS differences 2.46 mm / 0.64°. Position RMS error against MPM 22.51 → 22.37 cm.

All 500 identical-state solves and 500 paired candidate trajectory steps converge.
The 250-step and 600-step off-center drag/release tests also converge throughout;
the longer test reaches 183 material contacts and has no rigid-contact overflow.
Nine focused GPU tests pass: analytical impact, zero friction, GPU-only stepping,
fixed-material fallback, and previous warm-start/reset contracts. The ordinary
headless shovel launcher was also exercised for 30 steps. Ruff lint/format passes;
the repository-wide changelog gate still fails on pre-existing unrelated fragments.

No initializer fallback occurred on the 500 recorded states. Timing includes the
initializer and final contact verification. GPU profile medians for circle/eight:
Transolver 1.523/1.529 ms; G fitting 0.163/0.161 ms; body-velocity solve 0.321/0.362 ms;
impulse verification 0.024/0.042 ms; whole estimator 0.614/0.681 ms. Synchronized
whole-estimator replay times and nested GPU profile times are different measurements.
The particle model is now the largest individual GPU stage. Existing physical
prediction errors versus MPM remain; converged numerical agreement does not fix them.

Reproduction directory: `/home/jkamohara/isaac/particle_nerd/experiments/particle_gnn/shovel/solver-speed/`.
Use `benchmark_production.py`, `dynamics_production.py`, `mouse_production.py`,
`mouse_long.py` with `PYTHONPATH=examples uv run --extra ovrtx python SCRIPT` from
this worktree. `production-replay.json`, `production-closed-loop-comparison.json`,
`production-mouse-12s.json` and `production-profile.json` contain the final results.
The prototype adapters and earlier friction/warm-start results are experimental
records, not runtime dependencies. The cheaper friction bracket gave 11–15% savings
on 100 states but was not incorporated into the final implementation; the six-velocity
solve provides the measured large improvement without that additional change.

Public report: `compute-profile.html#shovel-current`; method: `index.html#body-velocity-method`.


## Interactive refinement divergence fix (2026-10-20)

User log: body-velocity residual 1.71e-14 at step 410, followed by 2,000 Jacobi
iterations and a nonfinite wrench. Rigid contacts did not overflow. A deterministic
shovel contact state reproduced the same failure mode at the default 1e-9 tolerance.

The stored shovel inertia has an approximately 1.49e-8 kg m² difference between
I_xz and I_zx from float32 asset accumulation. This makes the contact response
slightly asymmetric, whereas the 2 × 2 friction solve assumes symmetry. A valid
body-velocity solution therefore produced a spurious 6.09e-9 N s impulse correction.
Repeated simultaneous Jacobi updates amplified that error into divergence.

`ContactEstimator` now constructs the physical symmetric tensor as
`I = (I_input + I_input.T) / 2` in float64, before either solver uses it. This keeps
the effective-mass response consistent with the friction solve. The effective
mass multiplier, material prediction, scene, and 1e-9 stopping tolerance are unchanged.
No force clipping or convergence-check bypass was added.

Validation after the fix:
- 10 GPU interface tests pass. The dense sticking-impact regression fails without
  the fix and passes with it; it checks momentum and COM torque against an
  independent analytic solution at a supported strict 1e-12 impulse tolerance.
- All 20 shovel pose/velocity cases converge, with up to 2,122 material contacts.
  The case that previously diverged now needs one verification iteration.
- All 1,200 steps of a 24 s delayed mouse drag/release run converge; maximum
  impulse residual 2.38e-12 N s and one verification iteration on every step.
- The normal headless shovel launcher completes 600 steps with finite states.

Reproduction inputs and logs are under the sibling particle_nerd repository's
`experiments/particle_gnn/shovel/solver-stability/`; temporary run logs are
`/tmp/nerd-*-stability*.log`, `/tmp/nerd-shovel-dense*.log`, and
`/tmp/nerd-inertia-regression-before.log`. The exact user mouse trajectory was not
recorded, so the regression reproduces the numerical failure mechanism, not that
exact step-410 state. These are numerical stability checks, not improved motion
accuracy against MPM. Earlier timing/trajectory tables above predate this fix.


### Interactive MPM reference launcher

`play_shovel_reference.py` uses `data_collection.collect.create_collection_env_cfg`.
Run from this worktree with
`uv run --extra ovrtx isaaclab -p examples/particle_nerd/play_shovel_reference.py --viz newton_rtx`.
The collection environment runs without its own viewer. Reference and learned play
share `runner/viewer.py` (studio, camera, particle color/radius, frame drawing) and
`runner/rigid.py` (rigid model geometry). The reference display model has no solver;
MPM–MJWarp physics remains inside the IsaacLab collection environment.
The native viewer picking callback is registered with `NewtonManager` and acts on
the actual simulation state; the shared display/physics body index is zero.
Actions are zero, avoiding double application of dragging forces.

Both scripts load the same episode positions, velocities, body pose and body
velocity. Reference MPM stress/deformation history starts reset: the recording does
not store those internal fields, so this is visible-state parity, not a full MPM
checkpoint restore. `--bed-size` uses `dataset.initialize_fresh_bed` in both scripts
for identical fresh points, ordering, jitter, masses and body pose. MPMPointsCfg
authors these explicit initial particle states into the collection scene.
`--steps` counts 20 ms frames in both demos; reference uses four 5 ms MPM steps per
frame. `--headless` disables rendering. Data collection retains its original grid
spawning, initial randomization, material parameters and recording behavior.


### CUDA graphs in interactive play

Reference play enables `cfg.sim.physics.use_cuda_graph` and sets MPM
`max_active_cell_count` to the particle count (PIC integration, zero padding).
The native MPM internal graph remains enabled. Newton captures the complete
coupled step after its first eager step and checks sparse-grid overflow on replay.
Register viewer picking before this first step so the graph includes picking.

Learned play captures the Transolver neural forward in Torch, the analytical
estimator (hash-grid rebuild, gradient fitting, contact construction, solve and
wrench) in Warp, and MJWarp substeps in Warp. Particle features and particle
state update remain eager. The predictor keeps graph input/output tensors alive;
the estimator keeps graph instances for each rigid-state buffer layout. Particle
and collider counts must stay fixed within a predictor instance. The captures
retain GPU input updates and do not advance particle history during warmup.
`runner/profile.py` measures captured wall time in its primary pass. Its separate
per-kernel diagnostic pass uses eager NN/estimator launches because Python kernel
instrumentation does not execute during graph replay; do not report that detailed
breakdown as captured timing.

Validation (RTX 5090, 2026-10-20):
- Default 16,384-particle reference: synchronized `env.step` median decreased
  from 10.69 ms to 3.54 ms per 5 ms physics step (120 steps, first 20 excluded).
  Both inner and outer graph flags were verified; states remained finite.
- Fresh 2 x 2 x 0.5 m reference: 65,536 particles, 4.16 ms median per 5 ms step
  (40 steps, first 10 excluded), both graphs active, finite states.
- Learned simulation: 100 steps with changing applied forces, 3.15 ms eager
  versus 2.99 ms captured per 20 ms simulation interval (first 10 excluded).
  Final particles and every recorded body pose/wrench were identical. This is a
  modest median improvement, not a demonstrated tail-latency improvement.
- Native RTX picking and release verified after capture in both launchers.
- Larger learned launcher passed five steps with graph capture; this does not
  validate long-rollout accuracy for the larger terrain.
- 16 Transolver/GPU-interface/shovel tests pass. Ruff checks pass; the full format
  command still reports pre-existing repository-wide changelog errors.
Temporary measurement scripts, logs and JSON summaries: `/tmp/nerd-cuda-graphs/`.
