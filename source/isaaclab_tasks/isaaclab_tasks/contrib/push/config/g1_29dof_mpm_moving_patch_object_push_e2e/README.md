# G1 box pushing on moving-patch MPM sand, single policy

`IsaacContrib-Push-Box-G1-29dof-MPM-MovingPatch-E2E` is the sand stage of the single-policy
pushing task [`g1_29dof_object_push_e2e`](../g1_29dof_object_push_e2e/README.md). `-Play` selects
the playback configuration. It runs on the recycling moving-patch MPM scene of
`g1_29dof_mpm_moving_patch_object_push`, with the no-hand G1 and standard RSL-RL PPO.

The command, actions, observation layout, arm posture, box size and final mass range, reset ranges,
task and posture rewards, and networks are those of the rigid-ground task, so a rigid-ground checkpoint loads directly and is fine-tuned here.
The pushing MDP functions are imported from that package.

## Run

Fine-tune from a rigid-ground run (`--wandb_project` names the project of the source run):

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-Box-G1-29dof-MPM-MovingPatch-E2E \
  --num_envs 64 --logger wandb --viz newton_gl --video \
  --wandb_run YOUR_RIGID_RUN --wandb_project g1_29dof_object_push_e2e
```

```bash
uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Push-Box-G1-29dof-MPM-MovingPatch-E2E-Play \
  --num_envs 1 --viz newton_gl --video \
  --wandb_run YOUR_SAND_RUN --wandb_project g1_29dof_mpm_moving_patch_object_push_e2e
```

## Differences from the rigid-ground task

| Item | Rigid ground | Sand |
| --- | --- | --- |
| Ground | Rigid plane | 0.25 m of MPM sand in a 2.8 × 2.0 m simulated patch that follows the midpoint of the pelvis and the box |
| Robot and box reset height | Ground plane | Undisturbed sand surface |
| Robot-box contact forces | Contact sensor filtered against the box | Per-body box forces from the coupled environment (`object_contact_force`) |
| Foot contact for gait rewards and critic inputs | Rigid contact sensor | Granular reaction forces from the coupler |
| Foot clearance and stance-foot angle | Flat ground | Height scanner, with the sand depth as the ground offset |
| Extra terminations | — | Box or feet leaving the simulated patch; robot or box leaving the background terrain; implausible robot velocity (solver blow-up) |
| Box mass | 5–15 kg, fixed per environment | Resampled every reset from a curriculum range: 2–5 kg at first, widening to 5–15 kg |
| Action noise | Learned, no lower bound | Standard deviation floored at 0.5 so exploration does not collapse while fine-tuning |
| Step reward | Unbounded | Clamped to ±10 so one blown-up step cannot dominate a batch |

The sand color `(0.32, 0.24, 0.21)` matches the fixed-bed locomotion task `velocity_mpm/g1_29dof_mpm`.
The GL viewer draws particles in that color; pass `--viz newton_gl --video` to record them.

## Box mass curriculum

The mass range moves from 2–5 kg to 5–15 kg with a level in [0, 1]. Each finished episode with a
moving command changes the level by 0.002: up when its mean box speed ratio is above 0.6, down
when below 0.3. Each training process keeps its own level. `Curriculum/box_mass/{level,max_mass,mean_mass}`
log it. A policy only counts as trained once the level is 1. `-Play` uses the full 5–15 kg range.

## Assistive push force

`IsaacContrib-Push-Box-G1-29dof-MPM-MovingPatch-E2E-Assist` is a training-only variant. It adds a
horizontal force on the pelvis along the commanded direction, proportional to the base velocity
error, in the spirit of the assistive wrench in `contrib/mimic`:

`F = beta * min(robot mass * 10 /s * max(commanded speed - base speed, 0), 150 N)`

`beta = clip(1 - s / 0.7, 0, 0.8)` is frozen per environment at reset, where `s` is a running mean
of the episode-mean box speed ratio over finished episodes. The force is gone once the box follows
70% of the command. `Curriculum/assistive_push_force/{beta,similarity,force}` log its state.
`box_speed_ratio` is measured with the force applied, so read it together with `beta`; a policy
only counts as pushing unaided once `beta` is zero. Play with the unassisted `-Play` task.

## Throughput

About 60,000 particles per environment. A local RTX 5090 ran about 140 environment steps per
second at 32 environments, against roughly 90,000 on rigid ground at 2,048 environments.

## Status

The task builds, steps, reflects every observation, and resumes from a rigid-ground checkpoint.
Whether the fine-tuned policy pushes the box on sand has not been established; judge it by
`Metrics/base_velocity/box_speed_ratio`, `hand_contact_fraction`, `non_hand_contact_force`, and
the recorded video. The box resistance on sand has not been measured, and the 0.83 m push height
may tip the box if that resistance is high.
