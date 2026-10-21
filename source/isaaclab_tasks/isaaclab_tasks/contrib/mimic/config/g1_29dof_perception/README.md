## Terrain perception

The torso-mounted `height_scanner` matches the reach task's 1.6 × 1.0 m,
0.1 m-resolution grid, aligned with robot yaw. It casts against `/World/ground`,
including motion-paired terrain, once per control step (60 Hz by default).
Set `env.scene.height_scanner.debug_vis=true` to display the ray hits.

Both MLP observation groups append 187 terrain-height values, computed as torso
height minus terrain height minus 0.5 m and clipped to [-2, 2] m. Symmetry
augmentation reflects the grid laterally. These extra inputs require training a
perception policy; checkpoints from the non-perception task have different input
dimensions. Logs use `g1_mimic_perception`.

## Moviepy install
Need this for video logging
```bash
uv pip install 'moviepy<2'
```
## Wandb login
```bash
uv run --with wandb wandb login
```

## Sanity check 
```bash
uv run isaaclab -p scripts/environments/zero_agent.py \
  --task IsaacContrib-Mimic-G1-29dof-Perception \
  --num_envs 100 --viz newton_gl \
  env.sim.dt=0.005
```

## Training

Set `motion_files` and `terrain_mode` in [env_cfg/commands_cfg.py](env_cfg/commands_cfg.py).
The commands below use those settings and assume 50 Hz motion references.

```bash
uv run --with wandb isaaclab train \
  --rl_library rsl_rl \
  --task IsaacContrib-Mimic-G1-29dof-Perception \
  --num_envs 1024 --logger wandb --viz none \
  env.sim.dt=0.005
```

## Inference

```bash
uv run --extra ovrtx isaaclab play \
  --rl_library rsl_rl \
  --task IsaacContrib-Mimic-G1-29dof-Perception-Play \
  --num_envs 1 --viz newton_rtx \
  --checkpoint /absolute/path/to/model.pt \
  env.sim.dt=0.005
```

To load from W&B, add `--with wandb` to `uv run` and replace `--checkpoint ...`
with `--wandb_run YOUR_PERCEPTION_RUN_ID`.

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
Resumed perception runs use these reference targets when reset XY/yaw randomization is enabled.


## Anchor velocity tracking

`motion_anchor_lin_vel` and `motion_anchor_ang_vel` track the configured anchor
(`torso_link`) against the fixed episode reference velocities. Each uses
`exp(-squared_error / std**2)` multiplied by the tracking phase weight. The default
weights are 1.0, with kernel widths of 1.0 m/s and 3.14 rad/s, respectively.
Per-body velocity reward weights default to zero; set `motion_body_lin_vel.weight`
and `motion_body_ang_vel.weight` to nonzero values to restore that tracking.

The command reports ungated velocity-error norms as `error_anchor_lin_vel` [m/s]
and `error_anchor_ang_vel` [rad/s], including during stance. Resumed perception training uses these reward terms.


## Assistance during stance

The assistive force and torque are multiplied by the command's tracking weight:
full configured assistance during motion, smoothly reduced assistance during phase
transitions, and no assistance during full stance. This applies to both fixed and
adaptive assistance, including gravity compensation. The survival-based adaptive
gain remains frozen for the episode; phase gating does not alter its statistics.
The wrench arrows display the phase-scaled force and torque.


## Symmetry augmentation

The default RSL-RL runner augments PPO batches with sagittal reflections using
`MirrorObservationTermCfg` and `MirrorJointPositionActionCfg`. Reference joint
positions, robot joints, and actions swap left/right channels and negate roll/yaw
axes. Linear vectors reflect their Y component; angular vectors reflect X/Z.
Motion phase stays unchanged. Critic body terms also swap corresponding bodies;
6D rotations reflect both frames with `S @ R @ S`, where `S = diag(1, -1, 1)`.
The transform uses recorded batch values and supports observation histories.

Mirror rules use the explicit joint/body orders in the configuration. If those
orders are overridden, update the corresponding `mirror_params` as well. Joint
observations and actions are relative to default positions: augmentation treats
randomized defaults as reflected along with the robot, rather than reusing an
individual environment's asymmetric offsets for the reflected sample.

Observation/action dimensions and the reference file are unchanged. Existing
checkpoints remain compatible; resumed training with the new runner configuration
uses augmentation. Set `algorithm.symmetry_cfg.use_data_augmentation=False` to
disable augmentation. Mirror loss defaults to disabled and can be enabled
separately with `use_mirror_loss=True` and a positive `mirror_loss_coeff`.

## Export reference joint positions

Use `DatasetManager.save_csv(motion, "joint_pos.csv", joint_positions_only=True,
joint_names=JOINT_NAMES)` after loading the motion through a dataset term. Columns
follow `JOINT_NAMES` from `env_cfg/commands_cfg.py`; values are absolute joint
positions in radians. See the dataset README for the complete example.

Motion import and replay: [dataset pipeline](../../data/motions/README.md).

## Reference ghost

With `--viz newton_gl` or `--viz newton_rtx`, playback displays a translucent blue G1 at the current
motion-reference pose. Its root follows the episode's reference translation and
yaw, and its joints follow the selected clip. The ghost has no simulated
articulation or contacts. The soft Mimic scene inherits the same overlay.

Training disables the overlay by default. Set `env.scene.ghost.enabled=true` to
include it in training videos, or `env.scene.ghost.enabled=false` to hide it in play.
Every environment gets a reference ghost; there is no instance limit.
Adjust `env.scene.ghost.spawn.color` and `env.scene.ghost.spawn.opacity` for its
appearance. Kit is not supported. Enable the ghost before the first render so RTX
registers its mesh when building the scene.

The rigid Mimic task uses compositional PPO with the existing `policy` and `critic`
observation groups. Run `uv sync` to install the locked `compositional-rsl-rl` main
revision. The optional terrain snapshot interface is disabled for this plane task.

## Motion-paired terrain

Set `motion_files` and `terrain_mode` in `env_cfg/commands_cfg.py`. The default `terrain_mode="auto"`
loads terrain from those NPZ files for both standard RL train and policy play.
Clips without terrain metadata retain the existing scene plane. Use `"motion"`
to require paired terrain, or `"scene"` to retain the configured scene terrain.

Both training and playback use the command term configuration. For 50 Hz references,
the commands above set `env.sim.dt=0.005` with decimation 4. For 60 Hz references,
omit that override to use the default timing.
See the [terrain loading guide](../../utils/README.md) for playback, NPZ fields,
and shared patch placement. Runtime utilities live in `mimic/utils/`; sampling
and reset events remain in `mimic/mdp/`.
