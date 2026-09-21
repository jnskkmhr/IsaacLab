# G1 soft-contact motion tracking

This task inherits observations, joint actions (29 outputs, scale 0.2), rewards,
reference motion, terminations, physics settings, and timing from `g1_29dof`.
The zero-dimensional `physics_callback` adds the same foot-contact model used by
soft G1 locomotion: 3D Warp RFT with box foot colliders. Its identity mirror keeps
PPO symmetry augmentation compatible with the unchanged policy action space.

Soft material parameters are randomized on reset using the locomotion ranges:
friction 0.1–1.0, stiffness parameter 0.2–0.9, packing ratio 0.5–1.0, and bulk
density 1000–3000 kg/m³. The rigid task's reset and randomization events remain.

The terrain curriculum increases soft-layer depth by lowering the rigid floor
from 0 toward −0.12 m. Training starts each episode at the beginning of the reference. Completing the
whole clip without a failure promotes an environment; failed or incomplete
episodes demote it. Reference origins stay at the soft
surface (world Z=0), so changing floor depth does not lower the target motion.
The rigid floor has a 0.05 m solid backing for single-tile MuJoCo compatibility.
The Play task uses a fixed 0.12 m layer without terrain progression.

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl --task IsaacContrib-Mimic-G1-29dof-Soft --num_envs 4096 --viz newton_gl --video
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Mimic-G1-29dof-Soft-Play --num_envs 4 --viz newton_gl --wandb_run 5uq0oyh5
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
