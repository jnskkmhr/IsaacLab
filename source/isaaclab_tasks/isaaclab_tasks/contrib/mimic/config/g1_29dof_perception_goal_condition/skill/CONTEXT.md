# Goal-conditioned perceptive Mimic: development context

Updated 2026-11-02 UTC (2026-11-01 America/Los_Angeles).
This is the task handoff requested by the user. Read this file and
[EXPERIMENTS.md](EXPERIMENTS.md) before changing the task or interpreting its runs.
The [task README](../README.md) contains train/play commands and current defaults.

## Checkout, scope, and working preferences

- Checkout: `/home/jkamohara/isaac/tmp/isaac_PR/IsaacLab`.
- Branch: `dokyun/multi-mimic`.
- Push remote: `origin`, `git@github.com:jnskkmhr/IsaacLab.git`.
- Upstream: `https://github.com/isaac-sim/IsaacLab.git`; upstream development is on
  `main`, with the old main archived as `archive/main-2.x`.
- Before this handoff commit, local HEAD and freshly fetched origin branch both
  pointed at `5e2b2b4d1` (`add vibe coded GfR`). Fetch again before later pushes;
  do not overwrite concurrent remote work or force-push.
- Follow `AGENTS.md`, existing code conventions, and the task's `goal_mdp` exports.
  Keep changes small; do not introduce a new general framework for each reward.
- The user explicitly requested no added unit-test files or changelog fragments.
  Use focused temporary checks, real environment smoke runs, and scoped linting.
- Keep generated videos/checkpoints and scratch scripts local unless requested.
  Temporary files under `/tmp` are evidence pointers, not durable dependencies.
- Use uv-managed Python. Sandbox execution on this machine has failed with
  `bwrap: loopback: Failed RTM_NEWADDR`; escalated tool execution was needed.
- For job changes, act on the named workflow, report actual CLI results, and
  distinguish workflow RUNNING from PPO making progress. Never claim a new
  training run includes code written after its immutable source snapshot.

The older `contrib/mimic/skill/CONTEXT.md` describes the general motion/terrain
pipeline as of October 21. Its blanket exclusion of Kimodo and older CLI advice
are historical: the user subsequently explicitly selected Kimodo supplied stairs
for this task. The current task config and this handoff supersede those statements
for this experiment. Do not silently change other Mimic tasks based on this note.

## Research objective and decisions

The larger objective is perceptive skills for skill chaining, eventually robust
whole-body navigation/contact behaviors. Velocity tracking was a learning baseline;
reference-tracking Mimic exposed sensitivity to reference initial pose, velocity,
trajectory timing, and terrain. The user wants a goal policy that can adapt its
start/goal and terrain while retaining natural limb coordination.

Style here means similar local limb motion, cadence, and coordination, not following
an identical world-space root path. Success alone is insufficient if the robot
climbs aggressively, makes rapid steps, or ends in an unnatural pose.

The initial task used `parc/jump_001/motion.npz` and a terminal root-pose goal.
Joint regularization near the supported goal was added to discourage abnormal
terminal postures. The user then selected stair ascent as a simpler experiment.
A single supplied staircase was extended to a four-sided pyramid to avoid falling
off a short isolated stair block and to support four reference-valid approaches.
No descent task is currently intended.

### Relationship to GfR

Sources examined during the discussion:

- [GfR project](https://jiashunwang.github.io/GfR/)
- [GfR paper](https://arxiv.org/html/2602.20375v1)
- [Hybrid-Motion-Imitation implementation](https://github.com/jiashunwang/Hybrid-Motion-Imitation)
- [AMP](https://xbpeng.github.io/projects/AMP/) as a possible future style prior

This implementation is inspired by GfR, not an exact reproduction. Its shared actor
receives no explicit motion frame/phase, and a privileged critic helps training.
Imitation episodes use reference tracking; task episodes use destination rewards.
Sharing a policy can transfer motion structure, but it does not guarantee that
style survives arbitrary starts/goals when task rewards omit style. The linked
implementation identifies itself as unofficial; do not equate its fixed episode
partition/reset choices with every detail of the paper.

The user first tried reducing imitation to zero, saw loss of style, then compared
50% and 20% imitation. The 50% policy looked more promising. Current 50/50 means
**episode sampling probability**, not a weighted blend of two losses at every
step and not an exact fraction of rollout samples. A learned AMP/discriminator
has not been implemented. The new task style reward is a lightweight experiment.

## Current environment contract

Base registration: `IsaacContrib-Mimic-G1-29dof-Perception-Goal-Condition`.
Append `-Play` for deterministic task-mode evaluation.

- Physics step: 0.005 s; decimation 4; control/reference rate 50 Hz.
- Episode timeout: 10 s. Task episodes continue after reference completion.
- Robot: the task's 29-DOF G1 with box feet and ball-hand asset variant.
- Actor: proprioception, goal displacement in the robot root yaw frame,
  sine/cosine of goal heading error, and flattened terrain height scan.
  No reference joints, reference phase, or explicit episode-mode indicator.
- Critic: privileged robot/goal/terrain observations plus reference slots.
  Reference joint positions, phase, and anchor pose slots are **zeroed in task
  episodes**, and populated during imitation. Actor/critic dimensions are retained
  for checkpoint compatibility (285/598 in the evaluated configuration).
- Inference uses the actor's mean action. The critic does not choose actions.
  Aggressive deterministic behavior is not explained away as exploration noise.
- Both networks are MLPs `[512, 256, 128]`, ELU, without observation normalization,
  trained using compositional PPO. The height scan is not a CNN in this task.
- Rollout length 24, default 30,000 iterations, checkpoint interval 500.
- Actor initial action standard deviation 1.0; entropy coefficient 0.005.
  Training samples actions; deterministic playback uses the mean.
- Ghost is disabled. Train videos show the first 50 environments with Newton GL.
  Play uses Newton RTX by default; CLI `--viz` can select the configured backend.
  Camera eye `(3, -4, 2)`, look-at `(0, 0, 0.3)`, asset tracking `robot`, 960x540.

A goal is the **terminal robot root position and yaw**, not terminal joint angles
fed to the actor. It uses the reference endpoint plus the environment's terrain
origin and task goal offsets/noise. Root goal Z follows the endpoint root height.
The final joint posture is encouraged separately by regularization/style rewards.

Goal success requires root XY error <0.35 m, absolute root Z error <0.25 m,
yaw error <0.4 rad, and at least one foot contact above 5 N. Success is latched
within an episode; no dwell time is required. `task_success_rate` excludes
imitation episodes and is not equivalent to reference completion or timeout.

Reference completion and anchor/foot tracking terminations apply only to
imitation. Both episode types retain physical termination checks: root too low
relative to the feet (0.2 m), excessive orientation (1.2 rad), pelvis/torso contact,
and extreme joint-position action. See `env_cfg/termination_cfg.py` for exact terms.

## Data and terrain

Paths below are relative to `contrib/mimic/data/motions/data/kimodo/`:

| Directory | Purpose |
| --- | --- |
| `supplied_stairs` | Original supplied Kimodo staircase motion and provenance |
| `supplied_stairs_pyramid` | Four cardinally rotated 174-frame climbing clips on one pyramid |
| `supplied_stairs_pyramid_stand` | Current four clips with a procedural standing finish |

Current files are `motion_yaw_000.npz`, `motion_yaw_090.npz`,
`motion_yaw_180.npz`, and `motion_yaw_270.npz`. Motion sampling chooses among
these approaches uniformly. The full reference and goal are rotated together;
arbitrary large yaw rotations do not preserve stair/reference compatibility.

All four clips share identical pyramid geometry with a 2.5 m square top landing.
The motion-paired terrain importer deduplicates identical geometry. Many simulated
robots can share the same static terrain patch; that does not imply one simulated
robot. Terrain remains fixed at reset; robot/reference states receive the same
terrain-origin translation. The terrain's ground margin is 2 m to support the full
randomized standing-start footprint. See `contrib/mimic/utils/README.md` for the
mesh/box metadata and importer contract; this task does not rebuild it.

Current reference length: 374 frames at 50 Hz, 7.48 s:

1. Preserve the original 174 climbing poses.
2. Add a two-second procedural transition finishing the left-foot swing, bringing
   the right foot forward, and settling toward the default standing pose.
3. Add a two-second static standing hold with zero reference velocities.

Body states are regenerated by forward kinematics for the ball-hand G1. The final
root is 0.35 m farther forward than the old endpoint; final root Z is about
1.90755 m in the reference frame. The foot box collider is 0.17 x 0.06 x 0.01 m,
with local center `(0.035, 0, -0.0304)`; final foot corners were checked against
the landing with 2 mm clearance. Read each dataset's README/provenance for details.
This finish is synthesized, not new human motion capture. Kinematic validity
and a reference video do not establish dynamic tracking or learned standing.

## Episode sampling and reset curriculum

Imitation-only warmup lasts 120,000 control steps per environment (5,000 PPO
iterations). Imitation probability decreases to 0.5 over the next 120,000 steps.
`common_step_counter` is a single environment control-step count, not a vector,
and is not multiplied by the number of parallel environments. Restoring its
checkpoint value restores this time-based sampling schedule; verify the runner's
checkpoint compatibility when moving between library versions.

Task resets use a **separate per-environment success-based difficulty**, initially
zero. It is not currently saved/restored across newly constructed environments.
After a completed task episode, goal reached without physical termination adds
0.02; otherwise difficulty decreases by 0.05, clamped to [0,1]. Imitation and the
initial reset do not change it. Check `Curriculum/task_resets` alongside success.

At difficulty zero, task starts retain the first reference posture/velocity and
use a goal offset of 0.2 m forward from the reference endpoint. At difficulty one:

- Start offset: -0.9 m along the first reference root's forward yaw axis.
- Start XY noise: uniform +/-0.3 m per axis in that yaw frame.
- Start yaw noise: uniform +/-0.3 rad.
- Joint posture blends toward the robot's default standing pose; reference joint
  velocities blend to zero. Training retains the configured joint position noise.
- Root roll/pitch blend toward upright; reference root velocity blends to zero.
- Task goal offset: +0.75 m along the final reference root's forward yaw axis.
- Goal XY noise: uniform +/-0.35 m in world axes; endpoint root Z is retained.
- Above difficulty 0.8, half of task resets receive gradually increasing root
  velocity noise: local forward/lateral +/-0.15 m/s and yaw +/-0.2 rad/s.
  The other half start at rest at full difficulty.

The offsets fit the current pyramid and standing endpoints. Recheck them for
another dataset. Play disables curriculum and evaluates full standing-start
randomization; it is intentionally harder than difficulty-zero training resets.
Without success curriculum, the legacy time-based noise ramp remains supported.

## Reward ownership and new task style objective

Files of interest:

- `env_cfg/reward_cfg.py`: weights and term parameters.
- `mdp/rewards.py`: imitation masking, goal rewards, `TaskMotionStyle`.
- `mdp/commands.py`: goals, episode modes, motion IDs, success, scheduling.
- `mdp/events.py`: reference/task reset blending and root velocity sampling.
- `mdp/curriculums.py`: reset difficulty and fine-tuning terrain progression.
- `mdp/observations.py`: task-mode reference masking.
- `finetune_env_cfg.py`, `mdp/stair_commands.py`, `mdp/mixed_stair_commands.py`,
  `mdp/stair_terrains.py`, `env_cfg/terrain_cfg.py`: generated-stair variants.

`goal_progress` still exists in the term config but is set to `None` by the
base environment and both fine-tuning environments. The user explicitly wanted
to avoid a direct incentive to rush. Position, fine position, heading, settling,
and supported default-joint regularization remain active. Global reference
tracking rewards are masked off in task episodes. There is no assistive wrench.

New `task_motion_style`, weight **2.0**, applies only during task episodes:

- Joint-position MSE scale: 0.35 rad.
- Joint-velocity MSE scale: 1.5 rad/s.
- Mean ankle/wrist squared position-distance scale: 0.15 m.
- Body positions are measured relative to the root and expressed in its yaw frame.
  Absolute root XYZ and world yaw are not matching targets.
- A reward-owned continuous phase starts from the selected clip's current frame
  and advances at a selected 0.5, 0.75, 1.0, or 1.25 times the recorded rate.
  Each control step chooses the candidate minimizing summed normalized errors.
  Joint velocity targets scale with the candidate playback rate.
- Reward is `exp(-summed_error)`, masked by task mode and reference-motion eligibility.
  Phase updates once per control step, resets independently per environment, and
  clamps within each clip. It does not independently search the whole clip each step.
- Reference joint-velocity RMS below 0.1 rad/s is treated as stationary; such a
  frame earns reward only within the goal position tolerances. At the goal, if
  the final frame is stationary, matching switches to that standing frame.
- No observation or network inputs were changed. The task critic's reference
  slots remain zero. This style term is not an imitation-episode probability.
- Ablation: `env.rewards.task_motion_style.weight=0.0`.

Limitations: finite reference duration/step count still constrain timing. The
phase clamps at clip end; it cannot invent extra climbing cycles for longer
stairs. Goal-triggered switching to the final standing frame is an explicit
exception to bounded phase advancement. Stationary eligibility uses reference
joint speed, not a learned contact/style classifier. The reward has hidden
alignment state while task reference observations remain masked; whether that
helps or complicates learning must be evaluated. No experiment yet establishes
that weight 2.0 preserves style or solves generalization.

## Fine-tuning variants

- `...-Stairs-Finetune`: no motion files; generated ascending pyramid stairs,
  standing resets, task-only rewards. `task_motion_style` is explicitly disabled
  because `StairGoalCommand` has no motion library. Actor/critic are both retained;
  critic reference slots are zero, not removed. Initial learning rate is 1e-4.
- `...-Stairs-Mixed-Finetune`: 50% reference-compatible imitation on the reference
  terrain column; task episodes on generated stairs. Current inherited local
  style reward remains enabled for task episodes because this command loads
  reference clips. World-space reference tracking remains imitation-only.
- Both have `-Play` variants forcing task evaluation, 15 s timeouts, and retain
  post-success settling. Both disable `goal_progress`.
- Generated stairs: 8 m tiles, step depths 0.22/0.28/0.34 m, eight difficulty rows,
  step heights 0.12-0.23 m. Start on flat outer ground, goal on the central top
  landing; root Z is sampled terrain surface Z plus 0.8 m.
- Terrain success progression differs from reset difficulty: goal arrivals promote
  a terrain row; unsuccessful task episodes demote; imitation/initial reset skip.
  The current terrain progression does not use the reset curriculum's additional
  survive-until-episode-end condition.

## Validation and next work

Completed for the task style change:

- Actual motion-library checks: exact local match scores 1.0; global translation
  and yaw preserve it; triple-speed joints at a checked moving frame score about
  0.00274; frozen joint velocities score about 0.617 at that same pose.
  A single frozen pose can get partial reward; these checks do not prove absence
  of all reward exploits. The reference phase keeps advancing.
- Checked imitation masking, stationary-before-goal rejection, supported phase
  bounds, clip boundaries, duplicate calls within a step, and partial reset isolation.
- Three local PPO iterations with 64 environments, loading `l6ddoj98/model_29999.pt`,
  produced finite rewards and completed without integration errors.
- Ruff lint/format and `git diff --check` passed for the style change.
- No permanent unit-test files or changelog fragments were added.

Next: evaluate the newly submitted style run against the reset-curriculum baseline
with matched terrain, robot state, goal, checkpoint age, and deterministic actions.
Measure task success and survival after arrival separately from local style,
ascent duration, and joint velocities. Do not increase speed rewards to fix style.
If bounded finite-clip alignment is the limitation, discuss a local transition
prior/AMP or augmented reference set before adding complexity.

Handoff commit checks: the repository's scoped `uv run --no-sync isaaclab --format`
passed for all staged files after adding the existing OSMO-template YAML exemption
for `osmo/osmo_multi_gpu.yaml` and fixing spelling/header formatting. The changelog
hook was run once and reported pre-existing malformed fragments/missing coverage
across several packages; it was excluded on the final pass to honor the user's
no-fragments instruction. Staged Python/JSON syntax, shell syntax, and diff checks
passed. No claim is made that the branch-wide changelog gate passes.
