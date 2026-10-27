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
configuration were compared against the native Agile task before the waist-yaw and delayed implicit actuator changes,
excluding visualization and adapter metadata. A simulator smoke test checks zero collision margins, final observations,
symmetry augmentation, and replay reward reconstruction. Native and ported randomized
joint trajectories and curriculum schedules are compared directly.

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
`isaac-lab-l40s-03`. It defaults to one GPU, 4,096 environments, W&B logging,
and `--video --viz newton_gl`. No `env.video_recorders` override is passed.
Submit through `osmo/run_multinode.sh` to upload the checkout and assets.

Set `num_gpu=4 num_envs=1024` for one synchronized policy across four GPUs; `num_envs=1024`
means 1,024 environments per GPU. The launcher selects `train_multigpu` for
multiple GPUs. Replay capacity, warmup transitions, and minibatch size are per
GPU; algorithm and curriculum steps count synchronized vector collections.
Contact and command ramps use rank zero's elapsed training time on every rank.
