# G1 manipulation controller

Run from the IsaacLab checkout. These tasks use Newton/MJWarp and standard RSL-RL PPO.

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-Manipulation --num_envs 1024 --viz newton_gl --logger wandb

uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-Manipulation-Play --num_envs 1 --viz newton_rtx \
  --wandb_run xxx "env.video_recorders=[]"
```

Configuration files are standalone; reusable MDP functions are imported where appropriate.
The control rate is 50 Hz, with a 200 Hz simulation step and two solver substeps.
There is no high-level planner or downstream policy composition in this task.

## Asset and control

The USD variant `source/isaaclab_assets/data/unitree/g1_upper_body_with_hand/g1_upper_body_with_hand.usda`
references the G1 asset with articulated hands. Both leg subtrees and their joints are removed;
the three waist joints remain actuated. The pelvis is fixed 1 m above the ground by the
Newton articulation configuration. The pelvis, torso and head remain physical bodies,
with **31 movable joints**: 14 arm joints controlled by the policy, three waist joints controlled by events,
and 14 finger joints held at closed reset targets. Finger joints are excluded from policy observations and actions. No GPU kernel or physics dependency is modified.

The policy outputs left arm (7), then right arm (7), in the order in `env_cfg/action_cfg.py`.
Targets are `default_joint_position + 0.25 * action` in radians.
Wrist and waist gains are doubled relative to the shared robot configuration; other arm gains are retained.
Finger holding gains and closed targets match the object-push asset. These are effort-limited
position drives, so contacts can deflect the fingers.

Waist position targets are sampled independently per environment every **2–4 seconds**:
yaw ±0.10 rad, roll and pitch ±0.15 rad. Configure these ranges and the interval in
`env_cfg/event_cfg.py`. Targets are absolute angles, held between events and applied
through the waist actuators. Episode resets restore waist positions and targets to zero.
Joint velocity and joint limit penalties apply only to the policy-controlled arm joints.
The end-effector velocity penalty remains commented out for the current training experiment.
Measured joint positions and velocities include the three waist joints; wrist poses and commands are relative to the moving torso.

## Hand commands and observations

Both wrists receive **independent dataset targets every 3–5 seconds**, in **`torso_link`**
coordinates. The controlled points are the wrist-yaw link origins. Poses use
`[x, y, z, qx, qy, qz, qw]`, in meters and XYZW quaternion order.

`env_cfg/commands_cfg.py` selects `DatasetPoseCommand`. Each arm samples its own row
and timer independently for each environment. Both arms resample on episode reset;
afterward, either arm can change its target while the other holds its target.
The dataset contains collision-checked mirrored pairs, but independently combined
left/right targets do not retain that pairwise collision guarantee.
The dataset is loaded once onto the simulation device, not on each reset.

### Feasible pose dataset

`data/hand_poses_closed.npz` contains **231,849 mirrored pose pairs** and their arm joint
configurations. The generator sampled 2,000,000 left-arm configurations using the task's
exported Newton/MJWarp model with the new hand geometry and fingers set to the same closed
posture used during training. It rejects joint-limit violations (0.02 rad margin),
self contacts/clearance below 2 mm, and inward elbows.

The saved left-wrist positions occupy **78,709 cells at 1 cm resolution**, retaining up to
eight configurations per cell. On 12,080 held-out valid poses, nearest saved wrist-position
distances were 3.63 mm median, 7.24 mm at the 95th percentile, and 9.58 mm at the 99th;
**99.17% were within 1 cm**. This measures spatial coverage, not full orientation coverage.
The old `hand_poses.npz` is retained for existing object-push teacher adapters; the manipulation
commands explicitly select the new closed-hand dataset.

The exact right target is the sagittal reflection of the left target. Its FK mismatch
under mirrored joints is at most 0.01 mm for this asset.

The collision check preserves the asset's collision masks and exclusions and uses a
neutral waist. It validates target configurations, not paths between targets or every
possible randomized waist angle. The measured poses remain torso-relative when the waist moves.
The observation/action dimensions are unchanged, but the command distribution is wider than
the former Cartesian box; old pelvis-frame policies still need retraining.

Settings live in `env_cfg/dataset_cfg.py`; generation counts, rejection reasons, model
hash, and coverage statistics are also saved in `data/hand_poses_closed.json`. Regenerate with:

```bash
uv run --no-sync python source/isaaclab_tasks/isaaclab_tasks/contrib/push/config/g1_29dof_manipulation/util/generate_hand_pose_dataset.py
```

This exports the current task's Newton/MJWarp model and runs four CPU sampling workers.
Use `--model /path/to/export.xml` to reuse an export, or `--output /tmp/hand_poses.npz`
for an experiment. The dataset is tracked with Git LFS and included in package data.

The actor has 76 values: 17 relative joint positions, 17 joint velocities scaled by
0.05, previous 14 actions, two commanded wrist poses (14), and two measured wrist
poses (14). The critic receives the same channels without observation noise.

The command targets, measured hand poses, tracking errors, and goal markers all use the
same torso reference. Moving the waist carries the desired hand poses with the torso.
Existing pelvis-frame checkpoints have different input semantics and require retraining;
the object-push task's frozen teacher adapters remain unchanged for those older checkpoints.

Position and orientation have separate exponential tracking rewards for each hand.
Their standard deviations ramp from 0.20 m to 0.05 m and from 0.50 rad to 0.15 rad,
respectively, over 120,000 environment steps. There is no object or locomotion reward.

The elbow-inward penalty (weight -50.0) measures wrist position minus elbow position
in the frame selected by `reference_frame_link`. This task selects `torso_link`,
the body after waist yaw, roll, and pitch, so waist motion alone does not change
the penalty. It penalizes positive y on the left and negative y on the right,
summing the squared inward distances in meters. Elbows level with or outward of
their wrists receive zero penalty. This is a soft reward penalty, not a hard constraint.

## Symmetry and later composition

PPO data augmentation and a mirror loss (coefficient 0.1) are enabled. Both commanded
and measured wrist poses are reflected across the sagittal plane **and swapped left/right**.
Joint positions, velocities and actions use the G1 joint reflection rules.

This task learns reaching on a fixed base. Successful fixed-base reaching alone does
not demonstrate stability under walking or pushing forces. Those
interactions need validation when combining the two controllers later.

## Validation

With waist randomization enabled, four environments completed 260 control steps (5.2 s).
Each environment resampled its waist targets, the waist joints moved, and the pelvis
world position remained fixed. Checks also confirmed that target events do not teleport
joints or overwrite arm targets, partial resets clear only the selected environments'
waist targets, and the policy had 14 actions and 70 observation values before adding waist observations.

Before adding waist randomization, four-environment reset/step checks (260 steps), two PPO iterations with symmetry
augmentation and mirror loss, checkpoint save/reload, and TorchScript export passed.
The fixed pelvis stayed within 0.01 mm of its initial world position. With the waist held at zero, an independent
joint-limited inverse-kinematics check reached 64 sampled wrist goals with maximum
position and orientation residuals below 1e-6 m and 1e-6 rad. This checks sampled
kinematic reachability, not the complete workspace, collision clearance, or trained
policy accuracy. No converged policy is supplied.

With waist joint observations and doubled wrist gains, a four-environment check confirmed
76 actor/critic observations, 14 actions, mirror round-trip consistency, and two PPO
iterations with symmetry augmentation. Shared robot gains were unchanged.

## Compositional PPO option

The standard PPO configuration remains the default. Select the compositional implementation with
`--agent comp_rsl_rl_cfg_entry_point`; it preserves this task's observation/action layout, PPO settings,
and symmetry augmentation:

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-G1-29dof-Manipulation \
  --agent comp_rsl_rl_cfg_entry_point --num_envs 1024 --viz newton_gl
```

Use the same `--agent` override for playback. To use its actor as an object-pushing teacher, select
`state_dict_key=network_state_dict` and `prefix=terms.actor.` in that teacher's `pretrained_cfg`.
See `../g1_29dof_object_push/README.md` for the combined task.

Closed-hand collision checks are recorded in `data/validation_closed.json`; the corresponding
workspace projections are in `data/workspace_closed_xz_yz.png`. Three focused tests passed,
covering wrist-pose reconstruction, independent resampling, fixed finger targets, and symmetry.
A 16-environment local smoke run completed two standard PPO updates (768 transitions).
