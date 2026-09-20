## Moviepy install 
Need this for video logging
```bash 
pip install 'moviepy<2'
```
## Wandb login
```bash
uv run --with wandb wandb login
```

## Training
```bash
uv run --with wandb isaaclab train --rl_library rsl_rl --task IsaacContrib-Mimic-G1-29dof-v0 --num_envs 4096 --viz newton_gl --video
```

## Inference
```bash
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Mimic-G1-29dof-Play-v0 --num_envs 4 --viz newton_gl --wandb_run 4blqnxbi
uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl --task IsaacContrib-Mimic-G1-29dof-Play-v0 --num_envs 4 --viz newton_rtx --wandb_run 4blqnxbi
```

## Reference frame and reset randomization

The root reset event captures its sampled XY translation and yaw as a fixed
per-episode reference transform, pivoted about the sampled motion root position.
Global anchor pose targets, world-space body velocity targets, observations,
termination checks, and the assistive wrench use this transformed reference.
Reference linear and angular velocities, including the initial root velocities,
rotate with the sampled yaw. Reset height, roll/pitch, and velocity noise remain
tracking errors. The transform is replaced on reset and does not follow later
robot drift or pushes.

Relative body-pose rewards still align XY/yaw to the current torso each step.
Global anchor rewards therefore remain useful for enforcing jump displacement
and heading. Removing them relaxes trajectory tracking; a landing-position and
heading objective would be needed to enforce a specific landing goal instead.
Existing checkpoints retain their input dimensions, but resumed runs use these
new reference targets when reset XY/yaw randomization is enabled.


## Anchor velocity tracking

`motion_anchor_lin_vel` and `motion_anchor_ang_vel` track the configured anchor
(`torso_link`) against the fixed episode reference velocities. Each uses
`exp(-squared_error / std**2)` multiplied by the tracking phase weight. The default
weights are 1.0, with kernel widths of 1.0 m/s and 3.14 rad/s, respectively.
Per-body velocity reward weights default to zero; set `motion_body_lin_vel.weight`
and `motion_body_ang_vel.weight` to nonzero values to restore that tracking.

The command reports ungated velocity-error norms as `error_anchor_lin_vel` [m/s]
and `error_anchor_ang_vel` [rad/s], including during stance. Existing checkpoints
can still load, but resumed training uses the new reward terms.


## Assistance during stance

The assistive force and torque are multiplied by the command's tracking weight:
full configured assistance during motion, smoothly reduced assistance during phase
transitions, and no assistance during full stance. This applies to both fixed and
adaptive assistance, including gravity compensation. The survival-based adaptive
gain remains frozen for the episode; phase gating does not alter its statistics.
The wrench arrows display the phase-scaled force and torque.
