# G1 locomotion controller

Run from the IsaacLab checkout. These tasks use Newton/MJWarp and standard RSL-RL PPO.

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-Locomotion --num_envs 1024 --viz newton_gl --logger wandb

uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-Locomotion-Play --num_envs 10 --viz newton_rtx \
  --wandb_run YOUR_HEIGHT_CONDITIONED_RUN "env.video_recorders=[]"
```

Configuration files are standalone; reusable MDP functions are imported where appropriate.
The control rate is 50 Hz, with a 200 Hz simulation step and two solver substeps.
There is no high-level planner or downstream policy composition in this task.

## Control and observations

The policy outputs **15 actions**: left leg (6), right leg (6), and waist yaw/roll/pitch (3).
`env_cfg/action_cfg.py` specifies the exact order. Actions are joint-position offsets in radians:
`target = default_joint_position + 0.25 * action`.

The full 29-joint G1 remains physically simulated. A separate, zero-action-dimensional
term drives both arms with AGILE trapezoidal trajectories, resampling every 2–4 s.
The bounded left/right joint ranges are symmetric. Arm motion continues while walking
and standing; policy actions cannot overwrite those arm targets. Episode resets clear
only the affected environments' arm trajectories.

Actor observations (83 values, no history) are all 29 relative joint positions,
all 29 joint velocities scaled by 0.05, previous 15 actions, base angular velocity
scaled by 0.25, projected gravity, and `[vx, vy, wz]` commands plus a scalar pelvis-height command. Thus the actor observes
measured arm positions and velocities. The critic additionally receives base linear
velocity, giving 86 values. Rewards regularize the controlled lower-body joints;
random arm motion is not penalized as a policy action.

During training, the actor receives projected gravity with independent roll/pitch
orientation biases sampled on each environment reset and held for that episode.
The ranges ramp from zero to ±3° over 120,000 steps (5,000 PPO iterations), matching
the moving-patch task. These are orientation biases, not additive gravity-vector noise.
The critic and the `-Play` task retain clean projected gravity.

Velocity commands are expressed in the robot yaw frame: vx -1–1.5 m/s, vy ±0.5 m/s,
and yaw rate ±1 rad/s, with 20% standing commands. Training uses flat rigid ground.

Pelvis height is sampled uniformly from **0.60–0.80 m** above each environment's flat
ground, independently every 10 seconds and on reset. Configure it through
`commands.base_height.height_range`. Height tracking uses `exp(-error² / std²)` with
weight +10.0. Its `std` ramps linearly from 0.20 m to 0.05 m over steps 0–360,000
(15,000 PPO iterations), following the velocity-tracking curriculum duration.
Height is unchanged under left/right reflection. With `commands.base_height.debug_vis=True`,
blue and green spheres show measured pelvis height and commanded height, respectively.
They sit beside the robot at configurable world XY offsets; their world Z values are
unchanged, including each environment's ground elevation.
Older 82-input locomotion checkpoints require retraining for this 83-input policy.

Both velocity-tracking reward standard deviations ramp linearly from 1.0 to
`sqrt(0.1)` over steps 0–360,000 (15,000 PPO iterations of 24 steps), configured in
`env_cfg/curriculum_cfg.py`.

The contact sensor covers both feet. `contact_impulse` uses the rigid task's
`reward_soft_landing` (weight −0.005): summed force magnitudes at first contact,
active when commanded planar speed plus absolute yaw rate exceeds 0.05. Despite
the term name, this measures landing force, not a time-integrated impulse.
`stance_foot_angle_penalty` penalizes squared
foot pitch relative to world horizontal while a foot contacts the ground, including
landing and continued stance (weight −8.0, zero angle tolerance). Swing feet contribute
zero. These match the rigid locomotion task and require no height scanner.

Foot yaw uses the rigid task's two terms, each weighted −0.5: `feet_yaw_mean`
penalizes the feet's mean yaw relative to pelvis yaw, and `feet_yaw_diff` penalizes
the wrapped yaw difference between the two feet. Together they discourage both
feet turning away from the pelvis heading and the feet pointing apart. Both apply
throughout the gait, including swing.

## Symmetry

PPO data augmentation and a mirror loss (coefficient 0.1) are enabled. Reflection
swaps left/right joints, negates roll/yaw joint values, reflects linear vectors,
and handles angular velocity as an axial vector. All measured arm channels are
included in the reflection. The random-arm action term contributes no policy channels.

## Validation

Four-environment reset/step checks (260 steps), two PPO iterations with symmetry
augmentation and mirror loss, checkpoint save/reload, and TorchScript export passed.
An additional probe shortened the arm resampling interval to exercise target changes
and partial episode resets. Zero-action robots fall during this probe; it does not
establish learned walking performance. No converged policy is supplied.

## Compositional PPO option

The standard PPO configuration remains the default. Select the compositional implementation with
`--agent comp_rsl_rl_cfg_entry_point`; it preserves this task's observation/action layout, PPO settings,
and symmetry augmentation:

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-Locomotion \
  --agent comp_rsl_rl_cfg_entry_point --num_envs 1024 --viz newton_gl
```

Use the same `--agent` override for playback. To use its actor as an object-pushing teacher, select
`state_dict_key=network_state_dict` and `prefix=terms.actor.` in that teacher's `pretrained_cfg`.
See `../g1_29dof_object_push/README.md` for the combined task.

## Disabled native SAC option

The SAC config and registration are commented out while the project uses the previous
SuperMimic RSL-RL dependency. The settings below are retained for reference; restore the
SAC-compatible dependencies and enable the config and registration before using them.

With SAC enabled:

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-Locomotion \
  --agent rsl_rl_sac_cfg_entry_point --num_envs 1024 --viz newton_gl --logger wandb
```

Use the same `--agent rsl_rl_sac_cfg_entry_point` for playback of an SAC checkpoint.
PPO remains the default; its checkpoints are not SAC training checkpoints.

The actor uses the existing policy observations, and two independent Q critics use
the critic observations plus the sampled actions. Symmetry augmentation and the
0.1 mirror loss are enabled. Squashed actions lie in [-4, 4]; the environment's
0.25 action scale converts these to joint-position offsets of up to +/-1 radian.
Wrapper action clipping is disabled so replay stores the actions actually applied.

Each iteration collects 24 environment steps and runs 24 SAC updates, preserving
the existing environment-curriculum timing. Each GPU holds up to one million
replay transitions, collects 100,000 warm-up transitions, and samples batches of
1,024 transitions before symmetry augmentation. The runner enables final observations
for correct replay across episode resets. CUDA graph capture is disabled, allowing
the same config to run on multiple GPUs. These are initial settings, not a tuned
or validated walking policy.

Validation: an eight-environment Newton/MJWarp smoke run completed two iterations
and four SAC updates with symmetry and 0.2-second episodes to exercise timeout
resets. The resulting checkpoint reloaded successfully; TorchScript and ONNX
exports matched eager policy inference. This smoke used reduced replay and batch sizes.
