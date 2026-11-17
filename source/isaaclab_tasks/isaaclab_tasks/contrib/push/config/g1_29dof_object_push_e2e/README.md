# G1 box pushing on rigid flat ground, single policy

`IsaacContrib-Push-Box-Flat-G1-29dof` trains one whole-body PPO policy, without teachers, to push a
box while tracking a forward velocity command. `-Play` selects the playback configuration. The
robot, box, and ground use Newton/MJWarp; no particles or MPM solver are created. This is the
rigid-ground stage for `g1_29dof_mpm_moving_patch_object_push_e2e`.

The only command is a robot-frame velocity. There is no hand pose command or hand target: the arms
hold a fixed forward posture and act as a bumper, so walking forward brings the wrists onto the box.
The policy uses no global state: every actor input is proprioceptive, expressed in the pelvis frame,
or a robot-frame command.

## Run

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-Box-Flat-G1-29dof \
  --num_envs 2048 --logger wandb --viz newton_gl --video
```

```bash
uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Push-Box-Flat-G1-29dof-Play \
  --num_envs 1 --viz newton_rtx --video \
  --wandb_run YOUR_PUSH_RUN --wandb_project g1_29dof_object_push_e2e
```

Standard RSL-RL PPO is the default. Add `--agent comp_rsl_rl_cfg_entry_point` for the compositional
PPO config. Both use mirror data augmentation. Checkpoints from earlier versions of this task have a
different observation layout and cannot be loaded.

## Scene

- Robot: `UNITREE_G1_29DOF_BOX_FOOT_CFG_NO_HAND`, 29 actuated joints, no rubber hands and no
  gripper. The distal collider of each `*_wrist_yaw_link` ends 0.042 m beyond the link origin and
  is the pushing surface. The asset references the improved box-foot USDA, so it inherits the
  pelvis-contour to hip-pitch collision exclusions.
- Arm posture: `ARM_PUSH_POSTURE` in `env_cfg/scene_cfg.py` replaces the arm defaults. Both wrist
  tips sit about 0.33 m ahead of the pelvis and 0.07 m above it (about 0.83 m above the ground when
  standing), 0.18–0.20 m either side of the center line. Actions are offsets from this posture.
- Box: 1.0 m long (push direction) × 0.8 m wide × 1.0 m high. Mass is sampled once per environment
  from 5–15 kg. The measured sliding friction on this ground is about 0.35–0.45; changing the box
  material coefficient had almost no effect on it, so it is not randomized.
- Two contact sensors: feet against everything (gait terms), and every robot body filtered
  against the box (separates wrist pushing from leg or torso contact).
- Physics at 200 Hz with two rigid substeps; actions at 50 Hz. Episodes last 12 s.

The box size follows from the arm posture. It must be taller than the wrists, or they pass over
it. A sliding box tips when the push height exceeds `length / (2 × friction)`, so the 1.0 m length
keeps a 0.83 m push stable up to a friction of about 0.6.

## Command

`base_velocity`: forward speed 0.2–0.5 m/s in the robot yaw frame, zero lateral and yaw velocity,
10% standing commands, resampled every 4–6 s. Play uses 0.35 m/s.

## Observations

Every term is an `ObsTerm` with a reflection rule.

| Group | Used by | Contents |
| --- | --- | --- |
| `policy` (10-frame history) | actor | Base angular velocity, projected gravity, joint positions and velocities, previous action, box pose in the pelvis frame (7, with ±2 cm noise) |
| `command` | actor | Forward, lateral, and yaw velocity command (3) |
| `critic` (10-frame history) | critic | Noise-free `policy` terms and the world base orientation |
| `critic_command` | critic | Velocity command plus its world-frame heading target (7) |
| `privileged` (10-frame history) | critic | Base linear velocity, foot height, foot contact, foot forces, foot air time |
| `object_privileged` | critic | Box linear and angular velocity in the pelvis frame (6), log-compressed box contact force on the left wrist, right wrist, and all other bodies (3) |

The velocity command term also stores a world-frame heading quaternion. The actor does not
receive it, because using it requires a global yaw estimate. The box pose is read from simulation
as a stand-in for onboard perception.

## Rewards

Task terms (weight in parentheses):

- `object_progress` (4): box speed along the command divided by the commanded speed, clamped to
  [-1, 1], multiplied by the fraction of wrists touching the box and by `exp(-F / 30 N)`, where
  `F` is the box contact force on every non-wrist body. A stationary box earns nothing, and a box
  driven by the legs, pelvis, or torso earns nothing even when the wrists also touch it.
- `hand_contact` (1): fraction of wrists with more than 1 N of box contact during moving commands,
  with the same non-wrist decay.
- `track_lin_vel_xy` (3), `track_heading` (2), `track_ang_vel_z` (1): base command tracking. The
  linear tolerance is 0.25 m/s, so standing still under a 0.2–0.5 m/s command scores little.

Contact and object penalties:

- `non_hand_contact` (-3): box contact force on any body other than the six wrist links,
  saturating at 50 N.
- `object_tilt` (-10): squared sine of the box tilt.

Posture terms keep the robot balanced on its own feet. A version without `pelvis_lean`,
`feet_trailing`, and the non-wrist decay learned to plank its whole body against the box.

- `joint_deviation` (-1): the locomotion posture term, with arm weights of 0.5 (shoulders, elbows)
  and 0.2 (wrists) so that the arms return to the pushing posture. Waist pitch has weight 0.2.
- `base_height` (-50): squared pelvis-height excursion outside 0.65–0.80 m; free inside the band.
- `torso_lean` and `pelvis_lean` (-10 each): roll and backward lean are penalized; forward lean up
  to 0.35 rad is free.
- `feet_trailing` (-50): squared distance by which either foot trails more than 0.30 m behind the
  pelvis along the heading.

The remaining action-rate, energy, joint-limit, foot-orientation, gait, and foot-clearance terms
match the rigid-ground locomotion task.

## Terminations and resets

Episodes end on time-out, pelvis too low, root tilt beyond 45°, extreme joint targets, nonfinite
state, or box tilt beyond 0.5 rad. At reset the box center is placed 0.95–1.25 m ahead of the
pelvis, which puts its rear face 0.10–0.40 m beyond the wrists, ±0.10 m sideways, with ±15° yaw
relative to the robot heading.

## Logged pushing metrics

`Metrics/base_velocity/*` are episode means, logged at reset:

- `box_speed_ratio`: box speed along the command over the commanded speed, moving commands only.
  1.0 means the box moves at the commanded speed.
- `hand_contact_fraction`: fraction of wrists touching the box, moving commands only.
- `non_hand_contact_force`: box contact force on every non-wrist body [N].

Judge pushing by these metrics and the recorded video, not by episode length: a robot that stands
behind the box also survives every episode.

## Not included yet

No turning commands, external pushes, actuator-gain randomization, or curricula. There is no
termination for a stalled box or for walking away from it.
