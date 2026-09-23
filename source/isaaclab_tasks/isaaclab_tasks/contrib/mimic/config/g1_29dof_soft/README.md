# G1 soft-contact motion tracking

This task inherits observations, joint actions (29 outputs, scale 0.2), rewards,
reference motion, terminations, physics settings, and timing from `g1_29dof`.
The zero-dimensional `physics_callback` adds the same foot-contact model used by
soft G1 locomotion: 3D Warp RFT with box foot colliders. Its identity mirror keeps
PPO symmetry augmentation compatible with the unchanged policy action space.

Soft material parameters are randomized on reset using the locomotion ranges:
friction 0.1–1.0, stiffness parameter 0.2–0.9, packing ratio 0.5–1.0, and bulk
density 1000–3000 kg/m³. The rigid task's reset and randomization events remain.

Training and playback use a fixed 0.12 m soft layer, with terrain-level progression disabled.
Training starts each episode at the beginning of the reference. Reference origins stay at the
soft surface (world Z=0); the rigid floor is at −0.12 m and has 0.05 m solid backing for
single-tile MuJoCo compatibility. Material randomization remains enabled.

For push-recovery fine-tuning, training samples independent world-frame x/y velocity increments
in [−0.5, 0.5] m/s every 1–3 seconds. Vertical and angular velocity are unchanged. The event
only applies to environments with fully active stance rewards (`standing_weight >= 1 - 1e-6`)
and at least one foot in contact, using the existing hybrid rigid/soft contact helper. This
excludes the blended stance/tracking transition and skips delayed landings that are still
airborne. Skipped events are not queued for touchdown, so not every landing receives a push.
Assistance is disabled; rewards and policy observations are unchanged by this fine-tuning setup.

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl --task IsaacContrib-Mimic-G1-29dof-Soft --num_envs 4096 --viz newton_gl --video
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Mimic-G1-29dof-Soft-Play --num_envs 1 --viz newton_gl --wandb_run 5uq0oyh5

# finetune 
uv run --with wandb isaaclab train --rl_library rsl_rl --task IsaacContrib-Mimic-G1-29dof-Soft-Finetune --num_envs 4096 --viz newton_gl --video --wandb_run irsvxnpg
```

Training logs use `logs/rsl_rl/g1_jump_soft`. Existing rigid checkpoints retain
compatible observation/action dimensions; contact dynamics change in this task.

Soft-contact action, material-randomization, and terrain settings are defined in
this task's `env_cfg/` files. Tune these local configurations independently of the
velocity task; the shared soft-contact and symmetry MDP implementations are reused.

Training from frame zero prevents adaptive sampling from concentrating on short
final-stance segments and advancing terrain difficulty without learning the whole
motion. `Train/mean_episode_length` measures policy steps (60 steps/s here), not
whole-motion success when random start frames are enabled. Check
`Episode_Termination/end_of_reference` alongside failure terms and terrain level.

Restart training to apply these settings. Existing checkpoints remain loadable,
but a checkpoint trained mostly on final stance still needs to learn the earlier
motion. Use a new run name to keep the learning curves separate. To reproduce the
previous random-start setup, override `env.commands.motion.start_from_beginning=false`.

The local `env_cfg/reward_cfg.py` adds `motion_anchor_tilt` at weight `-0.5`.
It uses `motion_global_anchor_tilt_error_l2` to penalize squared distance between
unit gravity vectors in the reference and actual torso frames. This term ignores
yaw and remains active throughout stance and motion tracking. It penalizes excess
backward or forward tilt while allowing the reference's intended takeoff lean;
it does not require the torso to stay upright during a jump. All inherited reward
weights, including action-rate, torque, and joint-acceleration penalties, remain
unchanged. The policy observation and action interfaces are unchanged.

Restart or resume training in a new run to learn from the additional reward;
changing a reward does not change an existing checkpoint's playback behavior.
Compare torso pitch error during preparation and full-motion completion at matched
terrain depth. Disable the new term with `env.rewards.motion_anchor_tilt.weight=0.0`
for a comparison with the previous reward configuration.

### Ankle-pitch torque perturbations

`events.perturb_ankle_pitch` triggers independent signed torque pulses on the left
and right ankle-pitch bodies during fully weighted, grounded stance. Its defaults
are `torque_range=(-5.0, 5.0)` [N m], `duration_range_s=(0.2, 0.5)` [s], and
`interval_range_s=(1.0, 3.0)` [s]. The torque acts about each body's local y axis,
which is the G1 ankle-pitch axis. This is an external body torque, not a joint
position change or an internal motor torque.

`actions.ankle_pitch_perturbation` applies a sine-squared pulse envelope at physics
step midpoints through the instantaneous wrench buffer. It follows
`physics_callback`, whose contact forces remain in the permanent buffer; the
simulator combines both. The action consumes zero policy outputs, preserving the
policy action dimension. The pulse stops for a foot when contact is lost and for
both feet when full stance ends or the environment resets. Active pulses are not
restarted by another interval event.

Set `events.perturb_ankle_pitch=None` to disable new pulses. Set `events.push_foot=None`
when evaluating torque perturbations alone. Keep the perturbation action after
`physics_callback`, and keep its body and contact-sensor lists in matching
left/right order. These defaults are initial training settings, not a calibrated
model of MPM contact moments.
