# Agile G1 locomotion configuration

Install from this IsaacLab checkout with `uv sync --locked --extra wandb`. uv fetches the pinned
compositional-rsl-rl Git revision and its matching RSL-RL fork automatically; no separate library
checkouts or editable installs are needed. Use `uv run --extra wandb isaaclab train ...` below
for the task's default W&B logger, or select `--logger tensorboard` without the W&B extra.

This task ports the rewards, observations, and FlashSAC hyperparameters from
`agile.rl_env.tasks.locomotion.g1.flashsac_velocity_env_cfg.G1VelocityHistoryNewtonFlashSACEnvCfg`.
It controls fourteen joints: both legs followed by waist roll and pitch.
The remaining joints are held at their default positions.

The actor receives 255 values: five frames of each policy term, flattened term by term.
The Q critic receives that same policy history followed by 84 current privileged values,
then the learner's fourteen candidate actions before its residual encoder.
The actor and critic use independent 128/256-wide residual features with two blocks;
the critic has two categorical Q estimates with 101 bins on [-5, 5].
The reward terms, initial weights, tracking widths, and action-smoothing weight ramps
match the Agile task. Both terrain variants use the same rewards.

The robot is `G1_29DOF_BeyondMimic_delay`: BeyondMimic's pose and gains with implicit
drives delayed by 0–4 physics steps, resampled per environment and actuator group on reset.
The existing Agile `G1_29DOF_BeyondMimic` preset is unchanged.

```bash
uv run --extra wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid-Agile \
  --agent rsl_rl_flash_sac_cfg_entry_point --num_envs 2048 \
  'env.video_recorders=[]'
```

Use `Rough` instead of `Flat` for the terrain-generator task. Append `-Play` for playback.
The ordinary `rsl_rl_cfg_entry_point` remains PPO; its inputs and basic hyperparameters
are aligned with Agile's G1 PPO configuration. Recurrent PPO remains an optional configuration.

## Scope of parity

This matches the requested reward/observation/learner configuration, not the entire hardware recipe.
The requested BeyondMimic delayed implicit robot differs from Agile's FlashSAC robot,
which uses delayed DC motors and randomized arm targets.
Terrain generation, terrain progression, randomization, and termination conditions retain the
rigid task's definitions. Three-dimensional command sampling includes Agile's 0.1 dead zone.

The reference trainer also owns wall-clock command/contact curricula, replay reward relabeling,
and a wall-clock stopping budget. Those are not provided by this task configuration.
The current runner stores rewards with the weights at collection time; it does not relabel older
samples after reward-weight ramps. Its training budget is iteration-based (100,000 by default),
and periodic checkpoints are every 1,250 collection iterations. Replay requests one million
transitions under the same 3 GiB budget; the actual capacity is bounded by storage requirements.
Agile's optional PPO entropy annealing is not implemented here.
