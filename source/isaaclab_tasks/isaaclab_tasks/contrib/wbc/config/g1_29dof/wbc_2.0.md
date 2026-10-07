# WBC 2.0: pelvis navigation with relative torso and hand goals

## Controller objective

Give the policy a final task-space pose and let it discover how to reach it. The command
is held constant until replaced. The policy chooses the intermediate whole-body motion,
foot placement, contact timing, walking speed, and stopping behavior.

The command separates three objectives: pelvis world position and yaw, torso orientation
relative to the pelvis, and both hand SE3 poses relative to the torso. Feet remain controlled
through joint actions, but their poses and contact timing are not commanded or tracked.
A pelvis destination beyond standing reach should induce stepping; pelvis yaw specifies
lower-body heading, while torso-relative yaw specifies waist twist. Near the destination,
the same controller should settle and maintain all requested quantities.

This preserves the terminal-goal idea in the original
[implementation notes](../../wbc_implementation_idea.md). There is no per-time reference,
reference velocity, trajectory preview, prescribed swing curve, or commanded contact
schedule. Recomputing goal error every control step is feedback on a fixed goal; it does
not change the goal into a trajectory.

The earlier explicit-contact proposal is superseded for this formulation: contacts become
policy decisions mediated by the simulator. This trades precise user control of feet and
support timing for a simpler torso/hand interface and a harder exploration problem.
Keep the current 29-joint PD action contract and a single whole-body policy initially.
The task is local goal reaching with whole-body control on flat ground; obstacle-aware
navigation would additionally require environment observations and collision-aware training.

The initial environment is implemented as `IsaacContrib-WBC-PoseGoal-G1-29dof`, with a
separate `-Play` registration. The original fixed-stance task remains available. The
implementation uses asymmetric PPO, constant terminal goals, independent endpoint
resets, measured-contact step shaping, and a distance/yaw curriculum. Later robustness,
distillation, optional task masks, and alternate hand-frame modes remain future work.
Playback visualizes pelvis, torso, and hand goal frames using measured parents.
Pipeline validation does not establish trained goal-reaching performance.

## Command semantics and frames

Let `W` be the local world/odometry frame, `P` the pelvis link frame, `T` the torso link
frame, and `H` a wrist-yaw link frame. Position is in metres; use normalized XYZW quaternions
or rotation matrices internally. An RPY command interface is acceptable, but orientation
error and frame composition must use rotations rather than subtracting Euler angles.

| Command | Frame and meaning |
| --- | --- |
| `goal_pelvis_pos_w` | Final pelvis x, y, z in W. |
| `goal_pelvis_yaw_w` | Final pelvis heading in W; no commanded pelvis roll/pitch. |
| `goal_torso_quat_p` | Full torso orientation relative to the measured pelvis: roll, pitch, and waist twist. |
| `goal_left_hand_pos_t`, `goal_left_hand_quat_t` | Left wrist SE3 pose expressed relative to the measured torso. |
| `goal_right_hand_pos_t`, `goal_right_hand_quat_t` | Right wrist SE3 pose expressed relative to the measured torso. |

These quantities remain fixed until a new command arrives. Relative torso and hand goals
are fixed in their parent **link frames**, not frozen to those frames' world poses at
command acceptance. The torso frame includes its full roll/pitch/yaw, not just its heading.
Torso position follows pelvis/waist kinematics; it is not independently commanded.

For measured poses, evaluate the relative quantities directly:

```text
R_PT = transpose(R_WP) R_WT
p_TH = transpose(R_WT) (p_WH - p_WT)
R_TH = transpose(R_WT) R_WH
```

Compare these to the constant relative goals. Do not substitute a desired or imagined
parent pose for the measured parent in those errors. For visualization, the equivalent
instantaneous world hand target is `p_WT + R_WT p_TH_goal`, with orientation
`R_WT R_TH_goal`. Its motion follows the robot's torso and is not an externally supplied
trajectory. If the torso deviates while walking, the hands still track their commanded
relationship to that torso.

This separates yaw intent: pelvis yaw 90 degrees with neutral torso-relative orientation
turns the lower body and torso together; unchanged pelvis yaw with torso-relative yaw
30 degrees requests waist twist. Commanding both requests turning and twisting together.
Relative torso orientation uses the full pelvis rotation, so it does not guarantee a
world-upright torso when the pelvis tilts. Reward pelvis roll/pitch stability softly,
without overriding the commanded torso-relative bend. Body heading also need not be the
travel direction: train forward, lateral, and backward approaches if these are desired.

World goals need a pose estimate shared with the command source. If the user sends a
relative pelvis displacement or heading offset, convert it to a world goal **once when
accepting the command**. A zero displacement means hold the position at acceptance.
Do not add it to the moving pelvis each step. Encode pelvis position error in measured
pelvis-heading coordinates and yaw error with angle wrapping; expose measured relative
torso/hand state or their goal errors. Joint encoders and forward kinematics give relative
link poses; pelvis world translation still needs floating-base odometry.

For initial goal sampling, choose a feasible standing posture, use its pelvis height,
compute torso-relative orientation and torso-relative wrist poses, then choose an independent
pelvis x/y destination and yaw. Legacy datasets store pelvis, ankles, and wrists. The command obtains the torso and
other named link transforms by Newton FK from stored joints and pelvis poses using the
same USD as simulation. Newly generated datasets also include `torso_link`. Do not approximate torso
orientation with pelvis orientation. Limit sampled relative goals to reachable combinations
and validate stability of the complete final posture.

Torso-relative hands are a useful default for carrying, reaching around the body, and
keeping arm posture while moving. They do not hold a hand fixed on a world object:
translating or turning the torso moves the world hand target. Later, add an explicit,
observed world-hand mode for stationary object interaction if needed. Do not silently
convert a world hand goal to a torso-relative constant once and assume it remains world-fixed.

## Feasibility during motion

Pelvis z and torso-relative roll/pitch/twist remain terminal goals. A stable final squat
or twist is not automatically a workable walking posture. Allow intermediate deviations:
the policy can travel at a suitable height/orientation, then settle into the final goal.
Keep moderate posture reward in transit and increase precision near arrival, with smooth
state-dependent weighting based on navigation error. The command itself never interpolates.
Keep enough navigation incentive to prevent reaching a distant posture while staying put.

Exact height and relative orientation during walking are a stronger optional requirement;
train and evaluate their feasible ranges before promising it. Relative hand tracking can
remain active throughout, but large reaches and twists also change balance. Evaluate
combined commands rather than assuming independently reachable targets combine safely.

Start with all goals active. If task masks are added for pelvis-only navigation or one-hand
tasks, observe the masks and apply them consistently to observations, rewards, and success.

## Policy inputs and training architecture

The actor receives proprioception, previous action, fixed terminal goals, and current
task-space state or errors. Use estimated linear velocity, angular velocity, projected
gravity, and joint positions/velocities. Foot contact flags are excluded from the actor
because reliable contact estimation is not assumed at deployment.
The policy needs current velocity to learn braking and distinguish motion toward versus
away from the same destination. Observation history can later help infer support
without requiring explicit contact sensing or prescribing contact selection.

The critic can additionally receive simulator base velocity, foot contact flags and forces, center of mass,
actuator state, and other privileged physical quantities. Both actor and critic use the
same fixed goal; neither receives a time-indexed motion reference. Start with an MLP when
the state is sufficiently observed; evaluate history or recurrence for estimator delay
and contact uncertainty.

An asymmetric PPO actor/critic is the simplest first baseline. The original teacher/student
pipeline remains possible: train a teacher with privileged measured state, then distill
a student with deployable observations on student rollouts. A teacher may receive a
terminal joint posture as an auxiliary cue, but it must remain free to move legs en route.
Avoid asking the student to recover an arbitrary hidden posture choice from identical
SE3 goals. Prefer a consistent posture preference or an explicit deployable cue when
that distinction matters. Review teacher quality before starting distillation.

Removing foot goals changes observation dimensions and semantics. Old checkpoint inputs
and normalization statistics need explicit conversion or retraining. The existing stance
policy is a useful benchmark or optional initialization with documented parameter mapping;
it has not already learned navigation simply because the goals can be translated.
Mirror pelvis position/yaw, torso-relative orientation, and torso-relative hand poses;
swap left/right hands and contacts, and reflect polar/axial vectors consistently. Keep each sampled posture and its mirror in the same data split.

## Reward: reach, settle, and hold

Compute dense rewards against the **same goals in their specified frames at every step**.
Split pelvis horizontal position, height, yaw, torso-relative orientation, and each hand's
relative position/orientation so arm accuracy cannot hide a failure to navigate. Allow transient tracking error during
balance adjustments; the policy must be able to shift weight before making progress.

A narrow exponential position reward alone is a poor initial navigation signal: it can
be almost zero far from the destination. Combine broad distance-sensitive reward with
fine near-goal precision. For example, use a smooth distance cost such as
`sqrt(||p-p_goal||^2 + epsilon^2) - epsilon`, with appropriate length normalization,
plus a near-goal position reward. For rotation, use the shortest, sign-invariant SO3 angle.
Reward design must work over the sampled distance range rather than only near a posture.

Optional potential shaping can reward progress with
`gamma * Phi(state_next, goal) - Phi(state, goal)`, where `Phi` is negative weighted goal
error and `gamma` matches training. Handle true terminal states and timeout bootstrapping
consistently. On a goal switch, initialize the potential for the new goal; do not award
progress from comparing errors for different commands. Direct error reward remains the
primary contract; progress alone does not reward staying at the destination.

| Term | Intended behavior |
| --- | --- |
| Pelvis position | Reach world x/y/z; use broad and fine error scales, with final height precision near arrival. |
| Pelvis yaw | Reach the commanded world heading with wrapped angular error, independently of waist twist. |
| Relative torso orientation | Match `R_PT` to its goal with sign-invariant SO3 error; permit transit deviations. |
| Relative hand SE3 | Match each measured `p_TH`, `R_TH` to its fixed goal using the full measured torso frame. |
| Settled goal hold | Reward all active errors within tolerance with small task-space velocity for a sustained dwell period. |
| Foot slip | Penalize tangential velocity of feet with measured ground contact. Permit lifting, swinging, and new landing locations. |
| Foot clearance | Reward moving, non-contact feet for useful sole clearance while navigation is active; no prescribed swing trajectory. |
| Foot scuffing | Penalize poorly cleared moving feet and excessive contact during dragging; use sole geometry and measured state. |
| Impact and stability | Penalize excessive touchdown speed/forces, falls, self-collisions, and undesired body contacts. |
| Actuator regularization | Retain joint-limit, effort, acceleration, and action-rate costs, tuned so moving beats standing still. |

Velocity settling belongs near the goal; a strong global zero-velocity reward competes
with navigation. A relative hand can track perfectly while the robot still moves, so
settled success must also require low pelvis world linear/angular velocity. Measure
relative torso/hand motion as well; relative accuracy alone does not establish arrival. For task masks, goal completion checks only active goals. Contact forces
are measured outcomes rather than desired binary labels. Use force hysteresis/debouncing
if needed; the existing history-max detection can retain a lifted foot's contact briefly.
Slip penalties must use contact information with understood latency.

### Foot-clearance reward

Include a foot-clearance reward from the first stepping experiment. It shapes the policy's
chosen swing motion without commanding foot placement, swing phase, timing, or a trajectory.
Use a constant clearance preference rather than a time-varying foot-height reference.

For each foot `i`, let `h_i` be minimum sole clearance above the ground, `v_xy_i` its
measured horizontal speed, and `m_i` its measured ground-contact flag. One bounded reward is:

```text
speed_gate_i = clamp(v_xy_i / v_scale, 0, 1)
height_score_i = exp(-((h_i - h_clearance) / sigma_clearance)^2)
r_clearance = navigation_gate * support_gate
              * sum_i [(1 - m_i) * speed_gate_i * height_score_i] / 2
```

`navigation_gate` smoothly goes to zero when both pelvis x/y error and wrapped yaw error
are within arrival tolerances. It remains active for in-place heading changes as well as
translation, but not solely for a squat or waist twist. `support_gate` is one when at least
one foot has measured ground contact, zero otherwise; clearance should not reward jumping.
Cap foot-speed weighting so excessive leg speed cannot increase the reward without bound.
Divide by the fixed number of feet, not the number currently airborne.

Start with candidate values `h_clearance = 0.05 m` and `sigma_clearance = 0.03 m`, then
validate them against the actual box-foot geometry. Give this term a modest positive
weight relative to navigation; reaching and stopping must dominate swinging in place.
These values are tuning proposals, not demonstrated settings.

Measure sole clearance from transformed sole geometry or calibrated sole sample points,
including toe and heel; ankle-link height is not sole clearance. For the flat-plane stage,
subtract ground height from the lowest relevant sole point. Future uneven terrain needs
local terrain clearance. Use measured contact with understood threshold/debounce latency;
the existing history-max flag can delay recognizing lift-off and suppress early reward.

The speed/contact gates suppress the reward for a stationary raised foot, and the height
score falls above the preferred clearance instead of rewarding unlimited height. Normal
lift-off and touchdown have lower clearance, so this is a soft reward, not a hard height
constraint or a failure condition. The arrival gate prevents continued clearance rewards
after reaching the navigation goal. Check for hopping, foot waving, slow arrival, and
shuffling just outside the arrival tolerance; bounded shaping still admits these exploits
if its weight overwhelms goal progress and settling.

Keep scuffing, stance slip, impact, and no-flight costs alongside clearance. Avoid an
unconditional air-time bonus. Neither foot must return to its original anchor, and there
is no landing-position error because foot placements are free decisions. This reward
improves the quality of discovered steps; pelvis goal reward and exploration still need
to make taking a step preferable to standing still.

Remove all-leg terminal joint-position tracking from the locomotion objective. It can
fight stepping by rewarding the final stance joints while the legs need to move. A weak
posture regularizer for redundant upper-body configuration may remain if it does not
compete with hand/torso goals. Similarly, remove dataset-contact gating from sliding.
Avoid a generic upright-torso cost that erases commanded bends, squats, and waist twists.

The main reward risk is standing still: survival plus low effort can outweigh weak
far-goal reward. Evaluate this explicitly. Excessive running, overshoot, shuffling,
and never settling are separate failure modes; inspect them rather than solving all
with one large regularization weight.

## Goal sampling, resets, and curriculum

Use the existing Mink pose dataset as a source of feasible pelvis/relative-torso/hand goals,
not as a time-indexed reference. Collision and support checks of a final pose do not
prove the policy can reach it. Stepping-specific motion clips are not required for this
formulation. Begin with a small set of validated endpoints and expand workspace coverage
only after learning reliable transitions.

Reset the physical robot to a stable initial state, then sample a distinct goal. Do not
initialize exactly at the new goal throughout training: that teaches holding rather
than reaching. Placement transforms and environment origins must be applied consistently
to initial state and goals, with partial resets affecting only selected environments.

| Stage | Goal distribution | Evidence required |
| --- | --- | --- |
| 0: baseline | Existing fixed-stance posture changes. | Evaluate the selected stance checkpoint and establish held-out precision and survival. |
| 1: nearby goals | Small pelvis translations/yaw changes and reachable relative torso/hand goals; mix with holds. | Reach and hold through bending or small steps, with no fixed foot constraints. |
| 2: stepping required | Goals beyond the measured fixed-stance reachable region in forward, backward, and lateral directions. | Actual feet relocate, pelvis reaches its position/yaw goal, and the robot stops. |
| 3: turns and posture | Larger translations/yaw changes plus independently varied feasible height, roll/pitch, and hands. | Independent pelvis heading and waist twist, relative hand accuracy, and final posture on held-out combinations. |
| 4: robustness | Goal changes during motion, pushes, estimator error/latency, and later payload variation. | Replan directly toward the new goal, recover, and settle without resets. |
| 5: student | Deployable observations and teacher supervision, if needed. | Review teacher first; demonstrate that the student preserves reach-and-hold performance. |

For early training, use modest goal changes, longer opportunity to reach, and fewer pushes.
Sample some easy standing goals throughout to preserve manipulation precision. To force
stepping, choose pelvis destinations outside verified standing reach rather than issuing
lift commands. A later goal can be sampled after success or after a configured command
hold duration; command replacement changes only the goal, not the physical robot state.
Success-triggered resampling should include a hold interval so it does not skip learning
to stay still.

Keep falls and sustained unsafe contacts as failures. The current 0.5 m pelvis tracking-loss
termination cannot be reused for a pelvis destination metres away: that initial error is
intentional. Use a task horizon and safety failures initially; add a stalled-progress
condition only after allowing normal support transfer and braking. Failure is termination;
a time limit is truncation with the appropriate value bootstrap. Maintain successful
holds within the episode or count them explicitly before sampling the next goal.

## Evaluation and implementation order

Measure success as all active terminal errors inside tolerance with low velocity for a
continuous dwell, not merely passing through the goal. Report success rate, time to reach,
pelvis position/yaw errors, relative torso/hand errors, hold drift, overshoot, falls,
sole clearance, scuffing, stance slip, impacts, and
actuator saturation. Include failures in success statistics instead of averaging only
surviving agents. There is no reference contact timing or foot-placement accuracy metric.

Fix acceptance thresholds before experiments. Candidate starting criteria are pelvis
horizontal error below 5 cm, height error below 3 cm, yaw error below 0.1 rad, torso-relative
orientation error below 0.1 rad, hand-relative position errors below 3 cm and orientation
errors below 0.1 rad, and a 1 s stable hold. Define velocity thresholds for
that hold and tune tolerances to the actual manipulation requirement. These are proposed
criteria, not demonstrated results. Hold out endpoints, offsets, goal combinations,
and mirrors together. Test pelvis-only movement if masks are enabled, squat/waist motion
without unnecessary stepping, destinations requiring steps, arrival and holding, and goal
changes during motion. Explicitly compare pelvis-only yaw, waist-only twist, and simultaneous
turn-and-twist. Verify relative hand errors stay small as world hand poses move with the
torso; report world hand motion separately when assessing a later world-hand mode.

Implement the smallest goal-reaching experiment first:

1. Add `mdp/pose_goal_commands.py` and its pure-data config with pelvis and relative goals,
   one-time placement transforms, named bodies, coherent sampling, and no trajectory clock.
   Preserve the existing fixed-stance command as a baseline.
2. Add goal observations and mirror rules. Bind pelvis, torso, and wrists by name; expose
   pelvis navigation errors and measured torso/hand relative poses or their errors.
3. Add pelvis position/yaw, relative torso/hand, settled-hold, and measured-motion foot-clearance
   rewards. Remove foot-pose and all-leg target-joint rewards for this variant. Gate slip
   with measured contact; verify sole height, yaw-only navigation gating, and arrival gating.
4. Adapt reset and termination terms to independent initial states and goals. The current
   ghost requires a full joint pose; show goal frames, or a clearly identified optional
   IK endpoint visualization, without inventing an intermediate reference.
5. Wire an asymmetric PPO configuration. Validate persistent world goals, relative-command
   acceptance, full-rotation relative frames, independent yaw/twist, mirrored commands,
   partial resets, and near/far reward behavior at their
   owning boundaries. Follow repository test-audit and changelog guidance for source changes.
6. Run a small Newton scene and short PPO trial; compare nearby versus stepping-required
   goals. Expand only after actual reach-and-hold behavior works, not because training runs.

WBC 2.0 terms and symmetry live under `wbc/mdp` without velocity/mimic task imports.
Standard Isaac Lab proprioception and regularization terms remain shared core APIs.
The baseline task retains its existing helpers; no long training or distillation run is
started by this implementation.

## Related work and future scope

[Agility's motor cortex](https://www.agilityrobotics.com/content/training-a-whole-body-control-foundation-model)
provides task-space hand/torso objectives, emphasizes position rather than velocity control,
and discusses stepping to extend reach. Its described training uses time-indexed motion,
so it is inspiration for workspace coverage and the user interface, not our terminal-goal
training formulation.

[KINO](https://arxiv.org/abs/2609.18869) is closer to the desired goal contract: its authors
describe a keyframe-conditioned whole-body policy producing joint actions to reach selected
target poses. This supports researching sparse goals as an interface, but its full-body
keyframes do not establish this pelvis/relative-goal formulation or our proposed rewards.
[ULC](https://arxiv.org/abs/2507.06905) reports unified G1 loco-manipulation with root velocity,
height, torso rotation, and arm joint commands; it supports a unified architecture but
uses a different locomotion command.

[Boston Dynamics/TRI's Atlas article](https://bostondynamics.com/blog/large-behavior-models-atlas-find-new-footing/)
describes MPC-based teleoperation and torso/hand/foot action chunks. Its explicit foot
interface is not required by our goal-reaching design. The
[Lukas Ziegler article](https://ziegler.substack.com/p/ep110-google-deepmind-launches-new)
links to [DeepMind's whole-body VLA announcement](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/),
which does not specify the low-level RL recipe needed here.

With feet uncommanded, the controller cannot guarantee a particular final stance width,
foot placement, gait, or support sequence. If these later matter, add optional terminal
stance constraints while preserving terminal-goal semantics, rather than introducing
per-time motion references. Conflicting or unreachable pelvis/relative-torso/hand goals need explicit
handling by the command source; the policy cannot satisfy arbitrary SE3 combinations.

Future arm/lower-body separation must still account for arm mass, momentum, tracking
error, and payloads. Free-space wrist SE3 accuracy alone does not train forceful object
interaction. Keep those extensions separate from demonstrating terminal-goal locomotion.
