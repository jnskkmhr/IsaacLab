## Moviepy install
Need this for video logging
```bash
pip install 'moviepy<2'
```

# Agile G1 locomotion configuration

Install with `uv sync --extra wandb`. uv fetches the pinned compositional-rsl-rl
Git revision and its matching RSL-RL fork automatically; separate library checkouts
and editable installs are unnecessary.

This task follows Agile's
`G1VelocityHistoryNewtonFlashSACEnvCfg` and FlashSAC training recipe:

- Fifteen controlled joints: both legs followed by waist yaw, roll, and pitch. Waist yaw
  is an intentional extension of Agile's fourteen-action setup. The remaining
  joints receive randomized position targets with Agile's trapezoidal velocity profiles.
- BeyondMimic's implicit actuator gains, armatures, and effort/velocity limits, with
  per-environment command delays of 0–4 physics steps (0–20 ms) on every actuator group,
  including the arms. This intentionally replaces Agile's delayed DC motors. The Agile
  initial pose, zero robot collision margin, and selective self-collision exclusions remain.
- The reference rough terrain, terrain progression, reset and dynamics randomization,
  command sampling, contact sensors, and termination conditions.
- The same observation histories, reward terms, and initial reward weights. The actor
  receives 270 values; the FlashSAC critic receives those plus 85 privileged values,
  nine terrain heights, and fifteen candidate actions before its independent residual encoder.
  The terrain heights are an intentional extension of Agile: a separate `height_scan`
  observation group with history length 1, also supplied to the PPO critic. The torso-mounted,
  yaw-aligned scanner samples a 0.2 × 0.2 m grid at 0.1 m spacing once per control step.
- An additional foot-clearance reward matches `g1_29dof_rigid`'s rough-terrain term:
  weight 5.0, target clearance 0.1 m above the standing ankle-link height (0.03539 m),
  measured relative to the torso scanner's mean terrain height and weighted by horizontal foot speed.
- Independent 128/256-wide residual actor/critic features with two blocks, a squashed
  Gaussian policy, and two categorical Q estimates with 101 bins on [-5, 5].
- Reward smoothing, contact penalties, and forward command bounds are environment
  `CurriculumTermCfg` terms in `env_cfg/curriculum_cfg.py`, evaluated on resets using
  `env.common_step_counter`. Contact and command ramps run from step 5,000 to 10,000;
  this intentionally replaces Agile's wall-time schedule. Replay stores reward components and reconstructs sampled
  n-step rewards using the current weights, including past transitions.
- Seed 1, 2,048 environments, 200 Hz physics, 50 Hz control, and 30-second episodes.
  Training runs for 100,000 iterations by default (`--max_iterations` overrides this).
  Checkpoints are saved every 1,250 iterations and at the end. Video recording is disabled by default.

```bash
uv run --extra wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid-Agile \
  --agent rsl_rl_flash_sac_cfg_entry_point --num_envs 2048
```

The existing `Flat` task ID is retained for compatibility, but now uses Agile's reference
terrain, just like the `Rough` ID. Append `-Play` for playback. Select `--logger tensorboard`
when W&B logging is unnecessary. The ordinary `rsl_rl_cfg_entry_point` still selects PPO;
the iteration settings and replay reward curricula above belong to the FlashSAC entry point.

## Validation

The resolved scene, actions, events, commands, terminations, curricula, and simulator
configuration were compared against the native Agile task before the waist-yaw and delayed implicit actuator changes,
excluding visualization and adapter metadata. A simulator smoke test checks zero collision margins, final observations,
symmetry augmentation, and replay reward reconstruction. Native and ported randomized
joint trajectories and the original curriculum schedules were compared directly before converting the curricula to collection steps.

A 600-second run with 2,048 environments verified improving returns and episode lengths
for the original fourteen-action setup. The fifteen-action setup with BeyondMimic delayed
implicit actuators passed the simulator smoke check and a 7,000-iteration learning run
with 2,048 environments and seed 1. Its mean episode duration reached 14.29 seconds
at 6,000 iterations and 17.36 seconds at 7,000, using 100-record smoothing of rolling
episode metrics. Mean return at 7,000 iterations was 166.63.
This is a learning sanity check, not a full convergence or hardware evaluation. Identical
configuration does not imply identical random-number consumption or learning curves
across the two training backends.


## OSMO

`osmo/flash_sac_g1.yaml` uses the workflow/task name `isaac_flash_sac_g1` and
`isaac-lab-l40s-03`. It defaults to one GPU. Pass task, agent, environment count,
logging, and visualization flags through `--set-string args="..."`.
Submit through `osmo/run_multinode.sh` to upload the checkout and assets; see
`osmo/osmo_flash_sac_training.md` for complete commands.

Set `num_gpu=4` and pass `--num_envs 1024` inside `args` for one synchronized policy
across four GPUs with 1,024 environments per GPU. The launcher selects `train_multigpu` for
multiple GPUs. Replay capacity, warmup transitions, and minibatch size are per
GPU; algorithm and curriculum steps count synchronized vector collections.
Contact and command ramps are environment curriculum terms using each rank's environment step counter.
