## Train
```bash
uv run --with wandb isaaclab train \
    --rl_library rsl_rl \
    --task IsaacContrib-WBC-G1-29dof \
    --num_envs 8192 \
    --viz newton_gl \
    --video
```

## WBC 2.0 terminal-goal control

`IsaacContrib-WBC-PoseGoal-G1-29dof` is the new goal-reaching task; the original WBC
registration above remains the fixed-stance baseline. WBC 2.0 commands pelvis world
xyz/yaw, torso orientation relative to the full pelvis frame, and both wrist SE3 poses
relative to the full measured torso frame. Feet and joint endpoints are not tracked.

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-WBC-PoseGoal-G1-29dof --num_envs 4096
```

The default logger is local TensorBoard. Add `--logger wandb` and set the desired project
if remote experiment logging is wanted. For a bounded pipeline check, use `--num_envs 32
--max_iterations 3`. No stepping competence should be inferred from such a short run.

```bash
uv run isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-WBC-PoseGoal-G1-29dof-Play --num_envs 20 \
  --viz newton_gl --checkpoint latest
```

The default dataset is the existing 400-pose `stance_poses.npz`. Goals are sampled across
all endpoints, independently of initial poses and stance groups. Newton FK recovers the
missing torso pose from legacy data and extracts all eight collision-box corners per foot.
The existing Mink generator now saves `torso_link` too; its joint-limit, collision,
singularity, and support checks remain endpoint checks. No trajectory dataset is required.

Initial states are sampled from endpoints with pelvis height at least 0.55 m. Reset randomizes
world heading and initializes zero velocities. Commands hold for 8–12 seconds and resample
without resetting the robot. Twenty percent hold current x/y and yaw while changing upper-body
posture or height. Navigation offsets grow from a 0.15 m disk and ±0.25 rad yaw to a 1 m disk
and ±1.57 rad over 24,000 control steps; playback uses the final range immediately. Tune via
`env.commands.pose_goal` fields (`initial_distance`, `final_distance`, `initial_yaw_change`,
`final_yaw_change`, `curriculum_steps`, `standing_probability`, `dataset_path`).

The actor has 149 channels, ordered as robot state (ideal-state proprioception
and current parent-relative link state), last action, then
terminal goals. The critic has 159 channels in the same order, with minimum sole
clearance, contact flags, and normal forces included in robot state. Deployment needs pelvis odometry,
velocity estimation, and joint FK. Foot contacts are privileged training data used by
the critic and rewards; the actor does not require contact sensing or estimation. Existing stance actor
checkpoints and normalization statistics are incompatible with this observation layout.

The clearance reward prefers 5 cm minimum sole clearance for moving unloaded feet while
another foot supports. It is capped by foot speed, active for translation or in-place yaw,
and zero near arrival. It specifies no phase, trajectory, or landing position. Slip uses
latest measured contacts rather than dataset labels. Final height and relative torso
orientation have reduced priority in transit. Settled holds require all pose tolerances,
low world pelvis velocity, and low parent-relative motion for one second. Metrics report
per-part errors, clearance, current goal attainment, and completed-goal counts.

External command sources can call the command term's `set_goals` with batched world pelvis
positions/yaw and parent-relative torso/hand poses (normalized XYZW). World positions include
environment origins. The method validates inputs before updating them and disables periodic
resampling for those environments until reset. Convert relative navigation offsets to fixed
world destinations once upstream; torso/hand relative commands remain in their moving parents.
The sampled goal never follows the measured pelvis position after acceptance. Playback shows four goal frames per robot: pelvis destination/heading, relative torso
orientation at the measured torso location, and both torso-relative hand goals. Red/green/blue
axes mean local x/y/z. Hand markers move with the measured torso, matching the reward frames.
Set `env.commands.pose_goal.debug_vis=false` to hide them or `max_visualized_envs` to limit count.
The baseline joint-pose ghost is disabled because feet and final joints are unspecified.

See [wbc_2.0.md](wbc_2.0.md) for the formulation, reward gates, and staged evaluation.
