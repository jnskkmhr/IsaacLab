Experimental G1 fixed-bed MPM with native particle sleeping
=========================================================

Location and source revisions
-----------------------------

This worktree is isolated at ``~/isaac/experimental/IsaacLab`` on branch
``jnskkmhr/experimental-mpm-sleeping``. It has its own ``.venv``.
The original ``~/isaac/IsaacLab`` working files and environment are unchanged.
The task was copied from that checkout's working tree, including local task
configuration changes, rather than reconstructed from its HEAD alone.

* IsaacLab: PR #8981, ``96e7eb99dd557df7b898510669da07cd0b6dbbf9``.
* Newton: PR #5217, ``41cf3b44389205be786bbfa855f63ebc06c68978``,
  editable source in ``../newton``.
* Warp: PR #2270, ``feeb1d9c522172c352979d4cecfd456d1b571788``,
  merged with PR #2267, ``306164a86481785fbfcc2295404d6579160ec1a8``.
  The local merge is ``3d0f4e5b``, editable source in ``../warp``.
  The overlapping FEM test was resolved by retaining both sets of tests.
* CUDA build toolkit: 13.4.2, in ``../cuda``. Warp uses libmathdx 0.4.2
  from its own dependency manifest. Its native binaries were rebuilt for these sources.
* Installed versions: Warp 1.20.0.dev0, Newton 1.9.0.dev0,
  MuJoCo and MuJoCo Warp 3.17.0, Torch 2.11.0+cu130,
  RSL-RL 5.5.1+supermimic.2, compositional-rsl-rl 0.2.0.dev5.

The local asset copy is under
``source/isaaclab_assets/data/unitree/g1_29dof_rev_1_0_box_foot_improved_collision``.
USD files are ignored by the repository's existing asset rules; retain this
asset directory when moving the experiment to another machine.

Run the verified checkpoint
---------------------------

A copy of the existing 5f1xyc21 checkpoint is in this worktree's ignored logs
folder. It does not depend on exporting files into the original checkout::

    cd /home/jkamohara/isaac/experimental/IsaacLab
    uv run --extra ovrtx isaaclab play \
      --rl_library rsl_rl \
      --task IsaacContrib-Velocity-Sand-Flat-G1-29dof-MPM-Play \
      --num_envs 1 \
      --checkpoint logs/experimental_mpm/5f1xyc21/model_39998.pt \
      --viz newton_rtx \
      "env.video_recorders=[]"

This newer IsaacLab CLI uses ``--checkpoint`` for W&B as well. To download a
checkpoint, add ``--with wandb`` to ``uv run`` and replace the checkpoint argument
with ``--checkpoint wandb:jnskkmr/g1_29dof_soft_flat_ppo/5f1xyc21``.
The old ``--wandb_run`` / ``--wandb_project`` playback flags do not apply here.

Append ``"env.mpm_sleeping_enabled=false"`` to disable sleeping for comparison.
That comparison still uses this experimental backend and grid topology reuse;
it is not a return to the old Newton version.

Without a checkpoint::

    uv run isaaclab zero_agent \
      --task IsaacContrib-Velocity-Sand-Flat-G1-29dof-MPM-Play \
      --num_envs 1 --max_steps 100 \
      "env.sim.visualizer_cfgs=[]" "env.video_recorders=[]"

Implementation and scope
------------------------

The task uses the original fixed bed, particle material, robot gains, 29 actions,
930 policy observations plus 7 command observations, rewards and coupled foot
force extraction. Backend, actuator-limit and visualizer APIs were adapted to
this newer IsaacLab base. The viewer currently uses a fixed world overview
camera; the source branch's custom asset-following camera was not ported.

``env_cfg/physics_cfg.py`` enables ``reuse_grid_topology`` and configures
``MPMSleepingCfg(stress_filter_time=0.1)``. ``G1MPMEnv.load_managers`` applies
that config to the actual sand solver after solver construction and before
capture. Episode resets use the new coupled MPM reset helper, including
particle activity/history reset. The source material randomizer and accessors
are included locally in this task's ``mdp`` package.

Activity decisions happen on GPU. All particle states remain allocated on GPU;
this is not the CPU particle cache or recycling moving patch. Quiet material
can sleep, moving material and approaching colliders wake regions, and frozen
support particles surround simulated regions. Sleeping is an approximation:
it does not prove static equilibrium or preserve every slow-creep response.

The full evaluation capacity is retained for now. A deliberately small compact
working set is NOT enabled: startup and episode resets wake the full bed and
must not overflow a capacity chosen only for settled contact. Sleeping/grid
reuse are active, but no massive speedup or reduced full-state memory usage
has been established by these smoke tests.

Validation
----------

Validation used an RTX 5090. Logs are in ``logs/experimental_mpm/validation``.

* Native Newton ``test_captured_sleep_support_and_dirty_wake_cuda_0`` passed.
* Full flat bed: 593,750 particles, 80 random-action environment steps passed.
* Checkpoint 5f1xyc21: 250 environment steps with headless RTX rendering;
  robot positions, observations, rewards and actions were checked for finite
  values. An explicit reset at step 150 succeeded. The physics graph was
  captured once; the reset did not produce a second capture.
* Training configuration: material randomization, reset and 10 zero-action
  environment steps passed.
* Ascending bed (890,625 particles) and tilting bed: three physics/environment
  steps each passed. All 12 train/play configuration classes validated.
* Ruff lint and formatting checks passed for the 41 ported Python files;
  the uv lockfile check passed.

Particle activity samples from policy playback:

.. list-table:: Particle counts; the explicit reset occurs after step 150
   :header-rows: 1

   * - Environment step
     - Sleeping
     - Frozen support
     - Awake
   * - 50
     - 220
     - 83,784
     - 509,746
   * - 100
     - 444,062
     - 90,585
     - 59,103
   * - 150
     - 475,185
     - 61,644
     - 56,921
   * - 200
     - 220
     - 83,784
     - 509,746
   * - 250
     - 452,392
     - 89,147
     - 52,211

These are functionality checks, not a force-equivalence or training benchmark.
No changes have been pushed to GitHub.

Activity visualization
----------------------

Particle activity colors are enabled by default in this experimental task for
Newton GL and RTX: awake = orange, stationary support = cyan, sleeping = gray.
The colors use the solver's original particle IDs, not particle speed as a proxy.
All particle positions, including sleeping particles, remain visible. Updates run
on GPU at render cadence and are displayed on the following frame. Disable the
color overlay with ``"env.mpm_activity_colors=false"`` for timing comparisons.
No Newton or Warp source changes are required by this visualization.

Performance check (2026-11-11)
------------------------------

One RTX 5090, one flat-bed environment (593,750 particles), checkpoint 5f1xyc21,
400 control steps per trial, rendering and video disabled. Each control step is
20 ms of simulated time (four 5 ms physics steps). Reported wall time averages
steps 101--400, excluding the initial 100 steps that include warm-up and capture.
GPU synchronization occurs at each 100-step measurement boundary. Automatic
resets remain enabled, so these are whole-playback timings, not steady settled-bed
solver microbenchmarks. Separate trajectories can diverge between backends.

* Original ``~/isaac/IsaacLab``: 66.64 ms/control step.
* Experimental, sleeping disabled: 66.74 ms/control step.
* Experimental, sleeping enabled: 55.34 ms/control step (17% lower wall time).

Terrain dimensions, particle spacing, particle count, material, and MPM timestep
match for the tested flat bed. The experimental task enables sleeping and topology
reuse, but ``particle_working_set_capacity`` is still ``None``: compact evaluation
is not configured. Every reset wakes the full bed. At step 200, 460,278 particles
slept, 60,521 were support, and 72,951 were awake; at step 300 a reset had woken all
593,750 particles. These effects limit savings and make startup/reset timing unlike
settled local interaction. No timing test here changes physical particle size.

Raw measurements and the reproduction script are in
``logs/experimental_mpm/activity_comparison`` (ignored, local results).

Correction to the earlier RTX comparison: the 800x600 headless runs did not
request a camera image on every step. Headless viewer stepping skips actual
presentation, so those timings are not a valid interactive RTX comparison.
The raw files are retained for provenance but must not be cited as rendered FPS.
See the live-window dune measurements in
``logs/experimental_mpm/dune_performance`` instead.

Visualization validation: 150 policy steps each in GL and RTX, with exact GPU
color-to-activity checks at steps 100 and 150. Both rendered screenshots were
inspected; all three colors were visible. Screenshots and validation logs are in
``logs/experimental_mpm/activity_comparison``. Ruff lint/format and whitespace
checks passed for the changed Python files.

Live-window dune follow-up
--------------------------

The dune task uses 890,625 particles. With the same 5f1xyc21 checkpoint,
steps 101--200 averaged 8.03 control steps/s in the original checkout,
9.94 with experimental sleeping and activity colors, and 11.46 with
experimental sleeping and colors disabled. In separately synchronized profiling,
physics consumed about 105 ms/control step originally versus 61--62 ms
experimentally. Rendering used about 11 ms originally, 34 ms with experimental
activity colors, and 19 ms without those colors.

The original x36a2qh8 run confirmed two RTX images per submission: its viewer
presents DLSS-generated intermediate frames and counts them in displayed FPS
(about 16--17 FPS despite about 8 control steps/s). The experimental scene-camera
renderer publishes only the latest image to its GL window. Display FPS therefore
does not directly compare simulation throughput between these viewer paths.
Detailed logs, settings, raw timings, and the profiling script are in
``logs/experimental_mpm/dune_performance``.

Cached initial bed state
------------------------

The experimental task enables ``env.mpm_cache_initial_state=true``. Before the
first episode it advances only the sand solver for ``env.mpm_settle_time=1.0``
seconds of simulated time on the actual generated support mesh. It does not
advance robot dynamics, episode counters, rewards, or rendering during this
preparation. Newton's ``SolverImplicitMPM.save_particle_state`` owns the GPU
snapshot; ``restore_particle_state`` restores selected worlds without physics
steps. IsaacLab owns preparation timing and coupling-state synchronization.

The snapshot includes particle positions, velocities, elastic strain,
deformation transforms, velocity gradients, stress, plastic volume ratios, and
sleep/quiet history. Both canonical state buffers and coupled entry buffers are
updated on reset. Existing step graphs remain usable. Native material refresh
now preserves activity-buffer addresses and quiet history. Material parameters
are not restored from the snapshot: reset-time material randomization remains
in effect, so the initial mechanical state is approximate for changed material.
Geometry changes require rebuilding the snapshot. Position jitter is rejected
while snapshot restoration is enabled because it invalidates the saved state.

This is bounded initial relaxation, not guaranteed equilibrium. Moving particles
remain awake. The dune fixture exposed a small number of particles escaping its
existing support geometry and plastic-volume overflow during a longer two-second
preparation. No collider geometry or constitutive equations were changed to hide
that issue. Non-finite snapshots are rejected. The default one-second preparation
passed the two-world dune check with 386,749 sleeping, 855,379 support, and 539,122
awake particles out of 1,781,250 particles.

Validation checked exact particle/constitutive restoration for one world,
unchanged positions in the other world, no MPM calls during reset, graph reuse,
and finite robot state/rewards for 41 playback steps including a selective reset.
This does not yet implement robot-following region selection or CPU particle paging.

Use the existing play command. Optional overrides::

    "env.mpm_settle_time=1.0"
    "env.mpm_cache_initial_state=false"  # Original raw-spawn reset for comparison.

The native API changes live in ``../newton`` and are used by this checkout's
existing editable dependency. They are local changes, not a published dependency.

Final validation also passed single-world dune playback and two-world playback
with friction/density randomization enabled at reset. All 10 native activity
regressions passed. Restoring the previous material-refresh behavior made the
sleep-retention regression fail as intended. Logs and reproductions are in the
ignored ``logs/experimental_mpm/settled_state/`` directory.
