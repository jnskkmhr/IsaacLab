# G1 object pushing on recycling moving-patch MPM sand

This task uses the ordinary `g1_29dof_mpm_moving_patch` particle recycling implementation.
There is no particle-state cache or CPU tile bookkeeping. A fixed pool of particles covers
one 2 × 2 m active region following the robot–object midpoint, with a 0.2 m boundary band.
The center is quantized at 0.1 m intervals. The complete 2.4 × 2.4 m footprint contains
45,000 particles per environment at the configured depth and resolution.
Departed particles are recycled into newly exposed terrain; previously visited deformation
is not retained. The episode terminates if the cube footprint or either ankle position (with a 0.2 m
foot margin) leaves the active sand region.

MJWarp simulates the robot and a 0.65 m cube weighing 15 kg. Its contacts couple to implicit
MPM through the robot feet and cube. The bed uses 4 cm voxels and a 25 cm particle depth.
The outer physics CUDA graph remains enabled. Closed finger joints are held at fixed targets;
the student controls the 29 body joints.

## Commands and teachers

The student observes a two-dimensional direction in the robot yaw frame. Nonzero commands
are unit vectors sampled within ±45 degrees of forward. The locomotion teacher receives
0.8 m/s forward, zero lateral velocity, and a yaw rate proportional to the direction angle,
clipped to ±0.5 rad/s. Twenty percent of commands are zero: teacher velocity becomes zero,
hand targets return toward neutral, and standing rewards reference the command-start position.
Object-contact reward is disabled for zero commands, which also penalize hand contact.

Hand targets approach the rear cube face as the robot approaches the object. Each target is
projected independently into the closed-hand manipulation workspace in torso coordinates.
The locomotion teacher has a height input but no torso-pitch input. Its height is measured
above the rigid support floor beneath the sand, matching its training reference: the neutral
command is 0.80 m, and the contact command is clamped to 0.75–0.80 m. These heights are
25 cm greater than the corresponding pelvis heights above the initial sand surface.

- Locomotion: `rbfuk3xq/model_9500.pt`, 790 proprioceptive-history inputs plus 8 command inputs,
  producing 15 leg/waist actions.
- Manipulation: `6pbvl2df/model_19999.pt`, 76 inputs and 14 arm actions.
- Student: 102 observations and 29 actions; critic: 111 observations.

Checkpoints reside under `logs/teachers/<run_id>/` and are configured directly in
`agents/compositional_rsl_rl_cfg.py`. For OSMO, supply those directories through the existing
`teacher_checkpoints_url` input so they are copied into the job's `logs/teachers/` directory.

The student acts during collection. Frozen teachers label the states the student visits;
behavior-cloning losses and the complete PPO objective update the student together.
The student starts with action standard deviation 0.1, bounded to 0.05–0.5; the entropy
bonus is disabled. Teacher action-mean losses do not constrain exploration variance,
and unbounded wrist-pitch noise caused extreme-action resets in run `1nr7ez11`.

| Iterations (24 steps each) | Complete PPO loss weight | Each teacher loss weight |
|---|---:|---:|
| 0–300 | 0.0 | 1.0 |
| 300–10,000 | Linear 0.0 → 1.0 | Linear 1.0 → 0.1 |
| 10,000 onward | 1.0 | 0.1 |

## Run

From `/home/jkamohara/isaac/IsaacLab`:

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-Box-G1-29dof-MPM-MovingPatch \
  --agent comp_rsl_rl_cfg_entry_point --num_envs 64 \
  --viz newton_gl --video --logger wandb
```

GL and RTX both render particles and follow the `robot` asset with built-in camera tracking.
The `--video` flag records the selected viewer. Use the `-Play` task and a checkpoint trained
for this student to evaluate it. The former `-ParticleCache` task name no longer describes
this implementation and has been replaced by the name above.

## Validation

```bash
uv run --no-sync python -m pytest source/isaaclab_tasks/test/contrib/test_g1_moving_patch_mpm_push.py -q
```

Before the patch-size reduction, a two-environment GPU smoke run checked observation dimensions, checkpoint loading, direction
mapping, 15 kg object masses, patch recycling, selective reset, and two hybrid PPO/distillation
optimization updates with finite losses. This validates execution, not learned pushing quality.

With the previous 3.4 × 2.4 m patch, an eight-environment CLI training run also completed one 24-step iteration with GL particle
video recording; the camera and particles were checked in the recorded frame. A local
64-environment run exceeded available memory on the shared 32 GB GPU; the reduced patch
was subsequently validated on a 48 GB L40 as described below. Sparse tree reservations are
4,096 leaves, 128 lower nodes and 4 upper nodes per environment; cell capacity remains one
per particle. Newton and Warp dependencies were not modified for this task.

The 2 × 2 m patch passed the GPU integration test, including initial cube/foot coverage,
robot–object midpoint tracking, rigid hand contact, sand feedback, and particle recycling.


On 2026-11-14, `isaac-lab-mpm-push-2m-smoke-1` completed two training iterations
on one L40 with 64 environments, GL particle video, and both frozen teachers.
The first CUDA graph launch and optimization completed without an out-of-memory error.
This checks startup memory and execution; it does not establish long-run stability or pushing quality.
