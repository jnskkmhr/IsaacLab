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
- Agile's delayed DC motors, default pose, gains, armature, effort and velocity limits,
  zero robot collision margin, and selective self-collision exclusions.
- The reference rough terrain, terrain progression, reset and dynamics randomization,
  command sampling, contact sensors, and termination conditions.
- The same observation histories, reward terms, and initial reward weights. The actor
  receives 270 values; the critic receives those plus 85 privileged values and fifteen
  candidate actions before its independent residual encoder.
- Independent 128/256-wide residual actor/critic features with two blocks, a squashed
  Gaussian policy, and two categorical Q estimates with 101 bins on [-5, 5].
- Reward smoothing ramps by collection step; contact penalties and forward command
  bounds ramp by wall time. Replay stores reward components and reconstructs sampled
  n-step rewards using the current weights, including past transitions.
- Seed 1, 2,048 environments, 200 Hz physics, 50 Hz control, and 30-second episodes.
  The default training budget is 3,500 seconds including setup. Checkpoints are saved
  every 5,000 learner updates and at the end. Video recording is disabled by default.

```bash
uv run --extra wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid-Agile \
  --agent rsl_rl_flash_sac_cfg_entry_point --num_envs 2048
```

The existing `Flat` task ID is retained for compatibility, but now uses Agile's reference
terrain, just like the `Rough` ID. Append `-Play` for playback. Select `--logger tensorboard`
when W&B logging is unnecessary. The ordinary `rsl_rl_cfg_entry_point` still selects PPO;
the wall-clock budget and replay reward curricula above belong to the FlashSAC entry point.

## Validation

The resolved scene, actions, events, commands, terminations, curricula, and simulator
configuration were compared against the native Agile task before the waist-yaw extension,
excluding visualization and adapter metadata. A simulator smoke test checks zero collision margins, final observations,
symmetry augmentation, and replay reward reconstruction. Native and ported randomized
joint trajectories and curriculum schedules are compared directly.

A 600-second run with 2,048 environments verified improving returns and episode lengths
for the original fourteen-action setup. The fifteen-action setup has a separate simulator
smoke check; its learning curve has not been evaluated.
This is a learning sanity check, not a full convergence or hardware evaluation. Identical
configuration does not imply identical random-number consumption or learning curves
across the two training backends.
