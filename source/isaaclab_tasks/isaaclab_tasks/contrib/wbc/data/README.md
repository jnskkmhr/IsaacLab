# G1 whole-body pose data

`stance_poses.npz` is the default transition-training dataset: 400 validated poses in four
groups with shared foot positions and orientations. Each episode keeps one stance group.
Half of the 200 original poses use full-range posture proposals; half are nearby variants.
Validated mirrors double the count. Nearby variants differ from their paired original by
at most 0.20 radians RMS joint position, providing candidates for the early curriculum.

`representative_poses.jpg` is a 5×5 preview of the earlier independent-pose generation,
showing squats, waist roll/pitch, and side stretches. It illustrates pose diversity rather
than the fixed-stance groups used by the current training dataset.

## Sampling and validation

Full-range objectives are sampled inside the central 90% of each URDF joint range: midpoint
plus or minus 45% of the full range. Waist categories favor large roll/pitch angles. Nearby
variants perturb a validated posture within those limits. Mink projects objectives into
feasible poses; accepted joints are not uniformly distributed. Both objectives and accepted
joint positions are saved.

Each original and mirror must pass:

- Joint positions inside the same central 90% interval.
- Foot position error below 1 mm and orientation error below 0.005 radians.
- At least 2.5 mm clearance between checked non-adjacent collision pairs. Welded and adjacent
  bodies follow Mink/MuJoCo filtering; feet may contact the ground.
- At least 15 mm projected center-of-mass margin inside the support polygon.
- Minimum singular value above 0.001 for each limb Jacobian, with rotational rows scaled
  by a characteristic length of 0.2 m.

These are endpoint geometry checks, not guarantees of collision-free transitions, torque
feasibility, or dynamic tracking. The generator writes a JSON report with sampling ranges,
rejection statistics, and geometric validation results beside each generated dataset.
These reports are optional local artifacts and are not needed for training.

## Target switches and episodes

Episodes last 20 seconds, with earlier resets for falls or sustained tracking loss.
Every 4–10 seconds, independently per environment, the command selects a different pose
from the same stance group. Only the target changes: physical state, velocity, action
history, and episode time continue uninterrupted. References have zero velocity and represent
desired final postures; no command interpolation or history averaging is applied.

Allowed changes start at 0.25 radians RMS joint difference and increase to 1.0 radians over
24,000 control steps (1,000 default PPO iterations). Tracking-loss termination allows two
seconds after a switch; fall detection remains active. Foot targets use the group's exact
anchors, avoiding changes from numerical IK residuals.

Regularization uses Isaac Lab's joint-velocity, joint-acceleration, joint-torque, action-rate,
joint-limit, and undesired-contact rewards. The contact penalty excludes both ankle-roll
and wrist-yaw links. A three-frame contact sensor history captures transient contacts.

Base-velocity and foot-force perturbations run independently every 5–10 seconds per
environment, without stance gating. Base pushes add uniformly sampled world-frame X/Y
velocity changes within ±1 m/s. Foot pushes apply a one-step world-frame X/Y force within
±100 N to each ankle-roll link, with zero vertical force and torque.

Term configurations live in `config/g1_29dof/env_cfg/`: commands, rewards, observations,
actions, events, terminations, and curricula each have a `@configclass` collection.
The command-specific schema is in `mdp/commands.py`. `mdp/` contains
implementations; `wbc_env_cfg.py` composes the environment.

## Generate a grouped dataset

Run from the repository root:

```bash
uv run --with mink --with 'qpsolvers[quadprog]' python \
  source/isaaclab_tasks/isaaclab_tasks/contrib/wbc/data/generate_dataset.py \
  --urdf /home/jkamohara/isaac/newton-assets/unitree_g1/urdf/g1_29dof_rev_1_0_box_foot_improved_collision.urdf \
  --output /tmp/g1_wbc_poses.npz --poses 200 --stance-groups 4 --seed 44
```

`--poses` counts originals; mirrors double the count. It must be divisible by twice
`--stance-groups`. The URDF SHA-256 is recorded. Use another seed for held-out data.
Omitting `--stance-groups` retains independent-pose generation for visualization.

## Train and play

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-WBC-G1-29dof --num_envs 8192 --viz newton_gl --video
```

The default logger is W&B. Existing experiment names are unchanged:
`g1_29dof_wbc_teacher` and `g1_29dof_wbc_student`, with matching W&B project names.
Training records 20-second clips at 30 FPS every 10,000 control steps. Validate transition
tracking and stationary feet before student distillation.

```bash
uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-WBC-G1-29dof --num_envs 20 --viz newton_gl --video --checkpoint latest
```

## Earlier static-hold experiment

The original 8,192-environment trial trained for 500 iterations in 603 seconds. Its checkpoint
is `logs/rsl_rl/g1_29dof_wbc_teacher/2026-10-02_01-56-59_static_pose_trial/model_499.pt`.
It used one pose per episode, initialized at the reference. Held-out 10-second evaluation
had 93.5% survival, 80.5% survival with both feet within 2 cm throughout, and 6.7 cm mean
body-position error while alive. These results do not demonstrate transitions: when only
targets were switched during subsequent 20-agent playback, 14 robots fell.

The old run's scripts and independent-pose datasets target the previous static-hold
implementation. Transition training uses the grouped dataset and periodic commands above.

## Observation symmetry

Teacher, student, and critic observations declare each term separately with its
mirror rule, following the mimic task. PPO uses the shared `compute_mirrored_states`
callback for observation/action augmentation and mirror loss. Joint channels use
the configured G1 joint order; angular velocity uses axial-vector reflection, while
gravity and linear velocity use polar-vector reflection. Target body terms swap
left/right bodies and reflect their positions and orientations.

Target positions and orientations are now separate terms rather than interleaved
body poses. Observation dimensions remain 138 (student), 167 (teacher), and 201
(critic), but their ordering changed. Retrain policies; previous WBC checkpoints
and their observation normalization statistics are incompatible with this layout.

## Target ghost robot

Playback enables a translucent blue G1 showing the current commanded pelvis and
joint pose. The ghost uses the same visual meshes as the box-foot robot, with no
collision geometry, actuators, or additional simulated articulation. A separate
forward-kinematics model positions the meshes; reset and interval events refresh
all displayed targets. This visualization currently supports `--viz newton_gl`.

The first 20 environment targets are displayed by default. Configure
`env.scene.ghost.max_instances` to change this limit,
`env.scene.ghost.spawn.opacity` for alpha (default `0.25`), and
`env.scene.ghost.spawn.color` for RGB (default `[0.2,0.55,1.0]`). For example,
`'env.scene.ghost.spawn.color=[1.0,0.85,0.1]'` selects yellow. Disable the overlay
with `env.scene.ghost.enabled=false`. Training disables it by default; set
`env.scene.ghost.enabled=true` to include the overlay in training videos.
