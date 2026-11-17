# G1 object pushing with locomotion and manipulation teachers

`IsaacContrib-Push-G1-29dof-ObjectPush` trains a floating-base, whole-body G1 with closed articulated hands to
push a rigid box at a commanded planar velocity. `-Play` selects its playback configuration.
The default learner is compositional PPO. Standard RSL-RL source is unchanged.

The student controls the 29 body joints on every training step. Two frozen policies evaluate the
observations collected during that student rollout. PPO optimizes the pushing reward, while two
independent MSE losses match the student's action means to the teachers. This is student-rollout
imitation combined with PPO, not a separate pure DAgger trainer or a high-level action selector.

| Network | Inputs | Outputs / role |
|---|---|---|
| Student actor | 104: joint position 29, joint velocity 29, previous action 29, angular velocity 3, projected gravity 3, relative box pose 7, velocity command 3, pelvis-height command 1 | 29 joint-position actions |
| Critic | Actor inputs + robot linear velocity 3 + box velocity in robot frame 6 | Value |
| Locomotion teacher | 83: joint position 29, joint velocity 29, lower-body previous action 15, angular velocity 3, projected gravity 3, velocity command 3, pelvis-height command 1 | Legs and waist: first 15 student actions |
| Manipulation teacher | 76: arm/waist position 17, velocity 17, arm previous action 14, hand targets 14, measured hand poses 14 | Arms: final 14 student actions |

There is no global position, global orientation, odometry, or teacher output in the student actor's
inputs. The relative box pose comes from simulation as a stand-in for perception. Quaternion order is
XYZW. All groups have one frame of observations, without history.

Training samples forward box commands from 0.1 to 1.0 m/s and enables velocity-command
markers. The waist posture penalty has weight -5.0, and a waist joint-velocity-squared
penalty has weight -0.1, both covering waist yaw, roll, and pitch.

The velocity command `[vx, vy, yaw_rate]` describes box motion: XY velocity in the robot's heading
frame and angular velocity around world z. The same three values go to the locomotion teacher.
The student and locomotion teacher also receive a pelvis-height command of 0.75 m above
flat ground. The height reward uses this same command; change `commands.base_height.height_range`
to select another height within the locomotion teacher's trained 0.60–0.80 m range.
Leg and waist nominal PD gains match `2jiewizp`; arm gains match `6pbvl2df`.
Waist roll/pitch stiffness is 40 N m/rad and damping is 5 N m s/rad.
Manipulation hand targets and measured wrist poses both use `torso_link` coordinates.
Desired wrist SE(3) targets are offset from the nearest vertical face of the box, expressed in the robot's torso-link
frame. Their x-axes point into the box; left/right targets are 20 cm from the face center and 5 cm
below its top. Targets follow the box and can switch faces as the robot moves around it.

The object is a 0.65 × 0.65 × 0.65 m cube weighing 10 kg.
Its center starts 0.575–1.575 m ahead of the robot, preserving the rear-face gap
of 0.25–1.25 m. Hand targets use the closed-hand feasible-pose dataset from the
manipulation teacher. The cube can still tip depending on contact height and friction.

The Unitree source asset has 43 joints. Fourteen finger joints hold closed reset targets
with separate position drives; they are excluded from policy actions, observations, and
extreme-action checks. Closed-hand mesh reach is about 0.145 m along wrist-local +x,
so box-face wrist targets are offset backward by that amount before feasible-pose projection.
The feet use the same 0.17 × 0.06 × 0.01 m box colliders as the previous custom
box-foot asset. The added hand masses and contacts still change robot dynamics.

Box velocity rewards require measured contact with a wrist-yaw or finger link. Each hand
counts once: neither hand gives zero credit, one gives half, and both give full credit.
Contact requires force magnitude above 1 N. Linear-velocity tracking also retains its
hand-proximity weighting. Separate hand-proximity and gait rewards remain active.

`hand_box_contact` rewards measured hand–box contact with weight +1.0: one hand earns 0.5
and both hands earn 1.0 before timestep scaling, even before the box reaches its commanded
velocity. Force above 1 N receives the same credit regardless of magnitude. Knee–box contact
has no explicit penalty and earns no contact bonus. Ground and self contacts are excluded.
Reward tolerances are fixed initially; this task adds no curriculum.

At near-zero commanded box velocity (planar speed and absolute yaw rate below 0.05),
`leg_joint_deviation_l1` penalizes the sum of absolute deviations of the 12 leg joints
from their default standing positions, with weight -1.0. Waist and arm joints are excluded.
The cost is multiplied by `exp(-(root_linear_speed + root_angular_speed)**2)` to relax
standing posture during recovery motion. This discourages stepping in place; training
must verify its effect on stationary behavior and push recovery.

## Train with existing standard PPO teacher checkpoints

Run from `/home/jkamohara/isaac/IsaacLab`. The teacher checkpoints are configured directly in
`TeacherNetworksCfg` in `agents/compositional_rsl_rl_cfg.py`: locomotion run `2jiewizp` at `model_16000.pt` and
closed-hand manipulation run `6pbvl2df` at `model_9500.pt`. Paths resolve from the IsaacLab repository root.

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-ObjectPush \
  --num_envs 1024 --viz newton_gl --logger wandb
```

Teacher hidden layers are `[512, 256, 128]` with ELU and no observation normalization, matching the
configured checkpoints. Edit their `pretrained_cfg.path` fields to use other teacher checkpoints.
The library loads the full teacher state, including its distribution and observation normalizer if
configured. A checkpoint with a different architecture, input order, or normalization needs matching
configuration; equal input dimensions alone do not establish compatibility.

For teachers trained with the new compositional configs, also set these overrides (each teacher can
independently use either checkpoint format):

```bash
agent.networks.locomotion_teacher.pretrained_cfg.state_dict_key=network_state_dict \
agent.networks.locomotion_teacher.pretrained_cfg.prefix=terms.actor. \
agent.networks.manipulation_teacher.pretrained_cfg.state_dict_key=network_state_dict \
agent.networks.manipulation_teacher.pretrained_cfg.prefix=terms.actor.
```

Hand commands use the manipulation teacher's torso-frame feasible-pose dataset. Beyond 0.65 m
from the torso, each hand targets a comfortable approach pose. Between 0.65 m and 0.45 m,
the desired position blends toward the nearest box-face target. Both position and orientation
are then projected to one of 4,096 deterministic dataset poses per hand. Left/right candidates
use matching mirrored dataset rows, but are selected independently. These poses are feasible
for the dataset's robot model; contact with the box can still prevent perfect tracking.

The first 2,000 iterations optimize only teacher action-mean losses. PPO (including critic and
entropy) has zero weight, and the separate mirror loss is disabled. The student still collects
its own rollouts with initial action standard deviation 0.1. A fixed learning rate of 3e-4
avoids PPO's KL adaptation shrinking behavior-cloning updates. PPO is then ramped in as below.

The loss coefficients transition from imitation toward PPO:

| Completed PPO iterations (24 steps each) | PPO loss | Each teacher loss |
|---|---:|---:|
| 0–2,000 | 0.0 | 1.0 |
| 2,000–10,000 | Linear ramp 0.0 → 1.0 | Linear ramp 1.0 → 0.1 |
| 10,000 onward | 1.0 | 0.1 |

`ppo_loss_weight` scales the complete PPO objective: surrogate loss, weighted value
regression, and entropy. The internal value/entropy coefficients are unchanged;
there is no separate actor-versus-critic schedule. The separate mirror loss is disabled. Coefficients do not
measure gradient dominance; monitor the losses and tracking performance while tuning them.

Edit `curriculum_cfg` in `agents/compositional_rsl_rl_cfg.py` to change either
teacher schedule independently. Schedule endpoints use rollout control steps per environment,
not total samples or minibatch updates. A normal resume restores this progress. Loading only
weights with `iteration=False` starts a fresh schedule; older checkpoints infer progress from
the saved iteration. Current configuration determines the schedule values on resume.

Schedules override the static loss weights. For fixed weights, use
`agent.curriculum_cfg=null`, then set `agent.algorithm.ppo_loss_weight=...`,
`agent.loss.locomotion.weight=...`, and `agent.loss.manipulation.weight=...` as desired.
Setting a teacher's scheduled weight to zero also skips its evaluation.
The actor and critic retain symmetry augmentation; teacher supervision uses
original student rollout observations. Teachers remain frozen and do not acquire optimizer state.

## Action and observation compatibility

Student action order is legs (12), waist yaw/roll/pitch (3), left arm (7), right arm (7), explicitly
listed in `env_cfg/action_cfg.py`. Joint targets use `default_position + 0.25 * action`.
Lower-body defaults match the locomotion teacher; arm defaults match the manipulation teacher.
The locomotion observation reconstructs its original arm offsets, so changing the student's default
arm pose does not corrupt the teacher input. The manipulation observation uses its own waist order:
yaw, pitch, roll. Each teacher sees only its own previous-action subset.

Wrist PD gains are doubled relative to the shared full-body asset to match the listed wrist-gain
teacher; other gains, including waist gains controlled by locomotion, retain the shared asset values.
The reaching teacher was trained on a pinned upper-body robot; transferring it to a moving full-body
robot changes the dynamics. This implementation is an initial training setup, not evidence that the
combined policy already pushes successfully. The two teachers may disagree during contact; PPO and
the independent loss weights determine the compromise.

## Resume or play the student

A full compositional checkpoint contains both teachers. Original teacher files are unnecessary when
restoring that checkpoint; disable their initial file loads:

```bash
uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-ObjectPush-Play \
  --checkpoint /absolute/path/to/student/model_19999.pt \
  --num_envs 1 --viz newton_rtx \
  agent.networks.locomotion_teacher.pretrained_cfg=null \
  agent.networks.manipulation_teacher.pretrained_cfg=null \
  "env.video_recorders=[]"


uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-ObjectPush-Play \
  --wandb_run ds3fbub2 \
  --num_envs 1 --viz newton_rtx \
  "env.video_recorders=[]"
```

For training continuation, use the same two `null` overrides with `train --resume --checkpoint ...`.
Do not use those overrides when starting from scratch: that would initialize frozen random teachers.
The exported actor requires only the student observation vector; teacher networks are excluded.

## Validation

Compositional RSL-RL `0.2.0.dev5` at `31f0803` and the SuperMimic RSL-RL revision
`7ba0bf0` are pinned in the workspace dependency files.
Its test suite covers PPO plus multiple losses, frozen targets, action mappings, recurrent students,
distributed gradients, and checkpoint/export behavior.

Task regression tests:

```bash
uv run python -m pytest source/isaaclab_tasks/test/contrib/test_g1_teacher_push.py -q
```

With the current checkpoints, all four task regression tests passed, covering teacher input/action
alignment, torso-relative hand targets and observations, partial resets, and contact reward filtering.
A separate eight-environment smoke run completed two hybrid PPO iterations with both real checkpoints
and symmetry enabled. The student has 104 inputs; the locomotion and manipulation teachers have
83 and 76 inputs respectively. The torso-frame regression also fails when the old pelvis-frame
conversion is substituted. These checks verify integration, not trained-policy performance.

## Sanity check
```bash
uv run isaaclab -p scripts/environments/zero_agent.py \
  --task IsaacContrib-Push-G1-29dof-ObjectPush-Play \
  --checkpoint /absolute/path/to/student/model_19999.pt \
  --num_envs 1 --viz newton_rtx \
```

### Articulated-hand validation before the box-foot substitution

With the original spherical foot colliders, a Newton/MJWarp check with the two frozen teacher action means ran 32 environments
for 1,100 steps (22 s per environment). There were 29 timeouts, three robot-tilt
terminations, zero extreme-action terminations, and zero box-tip terminations.
All 32 environments registered hand–box contact: 3,282 environment-steps with hand
contact and 4,545 with knee contact (counts can overlap). This is not a trained
student evaluation and does not establish hand-only pushing.

Finger position targets remained equal to the closed reset posture throughout.
The fingers are effort-limited PD joints, not welded: contact can deflect them;
the maximum observed joint error was 1.38 rad, including falling episodes.
The asset's source joint limits and positive masses/inertias were checked in its
physics, mujoco, and physx USD variants. Runtime validation used Newton/MJWarp only.


### Zero-command standing

When commanded planar speed and absolute yaw rate are both below 0.05, box-velocity,
hand-contact, and hand-position rewards contribute zero. The manipulation teacher
receives fixed feasible wrist poses in the torso frame, independent of the box.
The locomotion teacher receives the zero velocity command immediately at reset.
Root linear/angular motion is penalized with weight -5, and XY displacement from
the reset position is penalized with weight -10. After a moving interval, the XY
reference is captured when the reward first observes the new stop command; later
stop evaluations retain it. Partial episode resets replace only those references.
These motion penalties are not attenuated by measured velocity.

This changes the training objective and requires retraining. It does not guarantee
that a learned policy stops safely on hardware; deployment needs a separately
validated stop mechanism.
