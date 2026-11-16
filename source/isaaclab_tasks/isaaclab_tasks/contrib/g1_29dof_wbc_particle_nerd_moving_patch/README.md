# G1 whole-body commands on Particle NeRD moving terrain

The policy controls 15 joints: 12 leg joints and the three waist joints. The 14 arm
joints follow independent uniform targets across their full asset joint limits every
2–4 seconds, using AGILE's bounded trapezoidal trajectories. Arm and hand colliders
are disabled; foot and torso colliders remain enabled. Arm targets and trajectory limits are configured in
`env_cfg/action_cfg.py`; the random-arm term contributes no policy action dimensions.

The task retains the lower-body task's terrain, physics, velocity commands and curricula. Joint position and velocity observations
still contain all 29 joints. Previous-action observations contain only the 15 policy
actions. Joint regularization excludes the randomized arms; the upper-body action-rate
term becomes a waist-only term with the same weight. Symmetry augmentation remains enabled.

The actor and critic also receive a scalar `base_height` command, sampled uniformly
from 0.50–0.80 m every 10 seconds. Height means pelvis/root world z above the
initial terrain surface. The raycast reference adds the 0.30 m particle depth in
sand environments only; rigid environments use the ray-hit height directly. It does
not track the current deformed particle surface. The previous
fixed-height squared penalty is replaced by `10 * exp(-height_error**2 / std**2)`.
The height tolerance `std` decreases from 0.20 m to 0.05 m over 15,000 PPO iterations
(24 environment steps per iteration). The command is mirrored unchanged and is supplied
through the shared actor/critic command group. Existing checkpoints without this command
have incompatible input dimensions; train a fresh policy.

The actor and critic also receive `torso_pitch`, sampled independently from 0–30°
every 10 seconds (stored in radians). Positive pitch leans the torso forward.
`torso_link` roll targets zero, while yaw remains unconstrained by this command.
The torso reward is `5 * exp(-(roll_error**2 + pitch_error**2) / std**2)`;
`std` tightens from 20° to 5° over 15,000 iterations. The neutral waist-pitch
posture penalty is disabled so it does not oppose the command. Pelvis upright
regularization and all other gait rewards remain unchanged. Pitch is unchanged
by left/right symmetry. Green and blue arrows show target and measured torso
forward directions, respectively, using the current torso yaw.

This directory replaces `g1_29dof_particle_nerd_moving_patch_lower`. The new WBC
Gym IDs require the additional pitch input: previous lower-body checkpoints cannot
be loaded unchanged. The actor input is now 799 values and actions remain 15.

Configurations are local to this task. The environment classes and Particle NeRD
solver are reused from the original moving-patch task. There are no Newton or Warp changes.

## Train

Run from the repository root:

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Sand-G1-29dof-WBC-ParticleNeRD-MovingPatch \
  --num_envs 64 --viz newton_gl --logger wandb
```

For the same fixed 50/50 neural-sand/rigid allocation as the original mixed task:

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-G1-29dof-WBC-ParticleNeRD-MovingPatch-MixedTerrain \
  --num_envs 64 --viz newton_gl --logger wandb
```

Mixed terrain requires an even environment count of at least two. The sand-only task
supports one environment. W&B projects are `g1_29dof_wbc_particle_nerd` and
`g1_29dof_wbc_particle_nerd_mixed_terrain`, respectively.

## Play

Use a checkpoint trained with this task's 15-action layout:

```bash
uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Sand-G1-29dof-WBC-ParticleNeRD-MovingPatch-Play \
  --num_envs 1 --viz newton_rtx --wandb_run xxx 'env.video_recorders=[]'
```

The mixed task also has a `-Play` variant; use at least two environments. Play keeps
arm randomization enabled. Add `env.actions.random_arms.randomize=false` to hold the
arms at their default pose. Existing 29-action locomotion checkpoints do not match
this task's action and previous-action observation dimensions.

## Validation

```bash
uv run --no-sync python -m pytest source/isaaclab_tasks/test/contrib/test_particle_nerd_lower.py -q
```

The GPU test exercises arm trajectories, partial resets, symmetry with full-body
observations and lower-body actions, and isolation of arm motion from joint penalties.
It uses a smaller active particle set and disables network compilation to keep the
test short; the task's default solver configuration remains unchanged.


## Flat terrain with backward torso force

`IsaacContrib-Velocity-G1-29dof-WBC-ParticleNeRD-MovingPatch-MixedFlatTerrain`
keeps the same 50/50 rigid/Particle NeRD allocation, height commands, arm
randomization, and PPO setup. Both initial terrain surfaces are flat; particles
can still deform. Terrain-level curriculum is disabled.

This variant replaces the velocity-kick event with a force at the `torso_link`
center of mass, directed along **world -X**, regardless of robot heading. Each
environment independently samples a magnitude uniformly from 0–50 N and holds it
constant for 3 seconds. Pulse starts are independently spaced 10–15 seconds apart
(not 10–15 seconds of waiting after a pulse). Reset cancels the active force and
resamples the timer. The event ticks every control step to clear the force on time;
`pulse_interval_range_s` configures actual disturbance intervals. This event owns
the robot's permanent external-wrench buffer.

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-G1-29dof-WBC-ParticleNeRD-MovingPatch-MixedFlatTerrain \
  --num_envs 512 --viz newton_gl --video --logger wandb
```

Use the `-Play` variant to inspect both terrain groups with force pulses enabled.
