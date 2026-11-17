# G1 pushing tasks: agent handoff

Read repository `AGENTS.md` and `skills/env_construction.md` before editing. This
file documents the current checkout as of 2026-11-14; saved run configurations
and checkpoint metadata are authoritative for reproducing historical experiments.
Run the commands below from `/home/jkamohara/isaac/IsaacLab`.

## Code organization and editing rules

Each task owns `agents/`, `env_cfg/`, `mdp/`, optional `util/` and `data/`, its
registration `__init__.py`, and environment/config modules. Keep term configs
standalone; reuse MDP implementations where appropriate. Task-specific joint
lists, body selections and ranges belong in `env_cfg/`, not constant-only MDP
modules. Prefer importing the task's `mdp` namespace and using `mdp.function`.

The two MPM pushing packages now live under `contrib/push/config/`. Their Gym
IDs, checkpoints, W&B project names, reward settings and physics behavior were
preserved by the move. Update Python imports to `isaaclab_tasks.contrib.push.config`;
do not restore a duplicate registration at the former package location.

Do not change rewards, observations, action order, teacher defaults, material
settings or dependencies as part of directory cleanup. Do not edit RSL-RL,
Newton or Warp to make a task configuration change. Keep generated reports and
training logs out of commits. The user's requested handoff filename is `AGENT.md`.

## Task definitions

All paths below are relative to this `push/` directory. Add `-Play` to each Gym
ID for playback, except the manipulation-object diagnostic which is already play-only.

| Package under `config/` | Gym task ID | Purpose |
|---|---|---|
| `g1_29dof_locomotion` | `IsaacContrib-Push-G1-29dof-Locomotion` | Rigid-ground lower-body teacher: 15 leg/waist actions; randomized arms observed by the policy. |
| `g1_29dof_manipulation` | `IsaacContrib-Push-G1-29dof-Manipulation` | Fixed-pelvis upper-body teacher: 14 arm actions, torso-relative wrist SE(3) targets. |
| `g1_29dof_manipulation_object` | `IsaacContrib-Push-G1-29dof-Manipulation-Object-Play` | Upper-body object-interaction diagnostic task; not the locomotion teacher. |
| `g1_29dof_object_push` | `IsaacContrib-Push-G1-29dof-ObjectPush` | Whole-body student pushing on rigid ground with two frozen teachers. |
| `g1_29dof_mpm_object_push` | `IsaacContrib-Push-G1-29dof-MPM-ObjectPush` | Unconfined fixed particle bed; preserves displaced sand and piles within an episode. |
| `g1_29dof_mpm_moving_patch_object_push` | `IsaacContrib-Push-Box-G1-29dof-MPM-MovingPatch` | Recycling moving-patch native MPM; a direction-conditioned whole-body student. |
| `g1_29dof_object_push_e2e` | `IsaacContrib-Push-Box-Flat-G1-29dof` | Rigid-ground single-policy PPO, no teachers, no-hand G1; velocity command only, fixed pushing arm posture, contact-gated box progress. Formerly `g1_29dof`. |
| `g1_29dof_object_push_e2e_v4` | `IsaacContrib-Push-Box-Flat-G1-29dof-V4` | `g1_29dof_object_push_e2e` with one change: the penalty-free foot-trailing distance is 0.25 m instead of 0.30 m. Reuses that package's MDP module. |
| `g1_29dof_mpm_moving_patch_object_push_e2e` | `IsaacContrib-Push-Box-G1-29dof-MPM-MovingPatch-E2E` | Moving-patch sand stage of the single-policy task: same command, observations, actions, and task rewards as `g1_29dof_object_push_e2e`, so its checkpoints load here for fine-tuning. Box mass follows a curriculum (2–5 kg widening to 5–15 kg). `-Assist` adds a training-only assistive push force that retires as the box follows the command. |

`g1_29dof_mpm_object_push_old` (`IsaacContrib-Push-Box-G1-29dof-MPM`) is a separate older
moving-patch MPM implementation, not a rigid-ground task; do not confuse it with either migrated
package or with the `_e2e` packages. Particle NeRD tasks belong to the separate
`IsaacLab-particle-nerd` checkout and use a different material solver.

### Fixed bed versus moving patch

- **Fixed bed:** initial 3 × 2 × 0.3 m sand grid per environment, no retaining
  walls, no recycling or particle caching. Objects have mass 5 kg; each world
  samples a cube, sphere or upright cylinder at scene construction. All particles
  remain simulated, so displaced sand can pile up. Commands track object velocity.
- **Moving patch:** 15 kg, 0.65 m cube; 2 × 2 m active sand patch plus a 0.2 m
  boundary band on each side, 0.25 m depth, 0.04 m voxels. The particle pool follows
  the robot–object midpoint; material leaving the patch is recycled and old tracks
  outside it are lost. There is **no particle-state cache**. The shared flat
  background is 200 × 200 m. Feet/object coverage guards remain enabled.
- Both use MJWarp for rigid bodies and native implicit MPM for sand. Coupling
  includes the feet and object. The scene asset name is `object`. Actual
  object-filtered rigid hand/finger forces determine hand-contact credit; sand
  reaction forces on the feet must not count as hand–object contact.

## Teacher training recipes

Train the two controllers separately before student training. Standard RSL-RL PPO
is the default for these teacher tasks; the student uses compositional PPO through
`--rl_library rsl_rl --agent comp_rsl_rl_cfg_entry_point`.

### Locomotion

The rigid-ground teacher controls six joints per leg plus three waist joints.
The robot observes randomized arm motion, but those 14 arm joints are not policy
actions. The current rigid teacher has a height command and 83 actor inputs;
it is not interchangeable with the MPM teacher below merely because both have
15 actions.

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-Locomotion \
  --num_envs 1024 --viz newton_gl --video --logger wandb
```

The native-MPM lower-body teacher used by the moving-patch student is owned by
`contrib/velocity_mpm/g1_29dof_mpm_moving_patch_lower_body`, outside this directory. It uses
half rigid / half MPM environments, randomized arms, 790 proprioceptive-history
values and an eight-value velocity/heading/height command. Use an even environment
count (64 per GPU in the current training recipe).

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Sand-G1-29dof-MPM-MovingPatch-LowerBody \
  --agent rsl_rl_cfg_entry_point --num_envs 64 \
  --viz newton_gl --video --logger wandb
```

**Height reference matters:** `rbfuk3xq` tracks pelvis height above ray-cast rigid
support beneath the sand, not above the initial particle surface. The moving-patch
push adapter accounts for the 0.25 m sand depth. Do not silently replace this with
foot height or particle-surface height. This teacher has no torso-pitch command.

### Manipulation

The pelvis is pinned; legs are removed from the upper-body asset. The policy
controls 14 arm joints, events randomize waist targets, and the finger actuators
hold the closed posture. The actor has 76 inputs including waist state and both
wrist poses. Commands, measured wrist poses, and tracking rewards use `torso_link`.
Left/right arms sample independent dataset rows and timers during training;
mirrored dataset construction does not mean paired training commands.

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-Manipulation \
  --num_envs 1024 --viz newton_gl --video --logger wandb
```

The closed-hand feasible workspace is in
`config/g1_29dof_manipulation/data/hand_poses_closed.npz`. Regenerate only when
intentionally changing the asset, hand posture, collision filtering or target
sampling recipe:

```bash
uv run --no-sync python \
  source/isaaclab_tasks/isaaclab_tasks/contrib/push/config/g1_29dof_manipulation/util/generate_hand_pose_dataset.py
```

Both teacher families include symmetry augmentation. Inspect their own READMEs
and term configs for current reward and command curricula. Training the current
checkout is not a guarantee of reproducing an older W&B checkpoint exactly.

## Student training recipe

The student acts during rollout; frozen teachers label the visited states.
Behavior-cloning losses supervise student action means: locomotion owns action
columns 0–14 and manipulation owns 15–28. The student controls 29 body joints;
closed fingers are outside its action space. PPO and teacher losses update the
same student, with the **entire PPO objective** scaled together, including the
value loss. Do not separate actor/critic scheduling.

For the moving-patch task:

- Student actor/critic input widths are 102/111. Commands are a robot-yaw-frame
  2D unit direction, or `(0, 0)` for standing. Nonzero commands produce teacher
  forward velocity 0.8 m/s and direction-dependent yaw rate capped at ±0.5 rad/s.
- Zero commands request standing at the stop-entry/reset position with neutral
  hands; they must not attract the robot toward the object. Contact height and
  hand targets transition near the object through the command implementation.
- Teacher checkpoints are configured directly in `agents/compositional_rsl_rl_cfg.py`:
  `logs/teachers/rbfuk3xq/model_9500.pt` and
  `logs/teachers/6pbvl2df/model_19999.pt`.
- Keep teacher joint order, observation history/order/scales, reference joint
  positions, action scale (0.25), and leg/waist versus arm PD gains aligned.
  Standard PPO teacher files use `actor_state_dict` with empty prefix. For a
  compositional teacher, use `network_state_dict` and prefix `terms.actor.`.
- Runner-level loss curriculum: PPO weight zero for 300 iterations, then linear
  ramp to 1 at iteration 10,000; each teacher weight ramps from 1 to 0.1 over the
  same interval. There are 24 environment steps per iteration and 20,000 iterations.
- **Preserve the exploration fix:** entropy coefficient is 0; initial action
  standard deviation is 0.1, bounded to 0.05–0.5. Mean-action cloning does not
  constrain variance. Run `1nr7ez11` collapsed when an entropy bonus of 0.005 and
  unbounded noise produced wrist-pitch extreme actions. Do not disable the
  termination to conceal that failure. A short replay verified the bound removes
  those extreme-action resets; this is not proof of long-run pushing success.

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-Box-G1-29dof-MPM-MovingPatch \
  --agent comp_rsl_rl_cfg_entry_point --num_envs 64 \
  --viz newton_gl --video --logger wandb
```

For fixed-bed MPM, select `IsaacContrib-Push-G1-29dof-MPM-ObjectPush`. Its configured
teachers remain `hpmmpox5/model_5500.pt` and `6pbvl2df/model_9500.pt`, at the paths
in that task's agent config. Its velocity command, material randomization, memory
cost and particle retention differ. Do not substitute the moving-patch teacher
bundle or assume the same environment count fits without validating it.

### Current moving-patch OSMO recipe

Use the existing wrapper to snapshot the checkout. The teacher bundle below
contains `teachers/<run_id>/model_*.pt` and is copied into the worker's `logs/`
directory. There is no student `--wandb_run` here: this trains a fresh student.

```bash
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40-07 \
  --set num_nodes=40 num_gpu=1 \
  --set-string \
    workflow_name=isaac-lab-mpm-moving-patch-push \
    platform=ovx-l40 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/mpm-moving-patch/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    teacher_checkpoints_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/mpm-moving-patch/teachers/rbfuk3xq-9500_6pbvl2df-19999/teacher_bundle/ \
    args="--task IsaacContrib-Push-Box-G1-29dof-MPM-MovingPatch --num_envs 64 --logger wandb --viz newton_gl --video --agent comp_rsl_rl_cfg_entry_point --run_name entropy0_std05_40gpu"
```

For either rigid teacher, the same wrapper can use `num_nodes=4 num_gpu=1` and
its corresponding teacher task with `--num_envs 1024`; omit `teacher_checkpoints_url`
and the compositional agent override when training standard PPO teachers. For the
MPM locomotion teacher use its registered task with 64 environments per GPU.
Pool availability and node exclusions must be checked at submission time.

The moving-patch viewers follow the `robot` asset and render particles; `--video`
records the selected GL viewer. Existing OSMO jobs use uploaded snapshots, so
local edits or this package migration do not update an already-running job.
Submission/startup success is not evidence of policy quality.

## Migration and validation checks

The migration changes only locations and required imports/resource paths; task
IDs and serialized numeric settings should compare equal after normalizing the
package prefix. Keep dataset files intact. The fixed-bed command's workspace path
must resolve to sibling `g1_29dof_manipulation/data/hand_poses_closed.npz`.

Run the existing integration checks separately:

```bash
uv run --no-sync python -m pytest source/isaaclab_tasks/test/contrib/test_g1_fixed_mpm_push.py -q
uv run --no-sync python -m pytest source/isaaclab_tasks/test/contrib/test_g1_moving_patch_mpm_push.py -q
```

These validate contacts, particle state and reset/recycling behavior, not trained
pushing quality. Inspect W&B episode duration, each termination, object progress,
actual hand contacts, teacher losses, and per-joint action noise before claiming
successful learning. `time_out` means reaching the configured episode horizon;
`extreme_action` means an early reset from an excessive commanded joint target.
