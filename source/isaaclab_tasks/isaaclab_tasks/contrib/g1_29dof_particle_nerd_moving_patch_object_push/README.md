# Particle NeRD moving-patch object pushing

This experimental task uses a recycling particle patch, not particle caching or a fixed sand bed. The robot feet and object interact with the same Particle NeRD update through three coupled rigid bodies per environment. The student controls 29 body joints; the finger joints hold the closed-hand posture.

## Run from IsaacLab-particle-nerd

```bash
cd /path/to/IsaacLab
uv run --with wandb isaaclab train --rl_library rsl_rl   --task IsaacContrib-Push-G1-29dof-ParticleNeRD-MovingPatch-ObjectPush   --num_envs 4 --viz newton_gl --video --logger wandb
```

The task also registers `IsaacContrib-Push-G1-29dof-ParticleNeRD-MovingPatch-ObjectPush-Play`. Use that with the normal `isaaclab play` procedure once a student checkpoint exists. Particle visualization is enabled for both GL and RTX; cameras track `robot`. Training records GL video.

To inspect the two frozen teachers directly, without training a student:

```bash
uv run --no-sync python -m isaaclab_tasks.contrib.g1_29dof_particle_nerd_moving_patch_object_push.util.rollout_teachers   --steps 400 --num-envs 2 --video   --output exports/particle_nerd_object_push/teacher_rollout
```

Environment 0 receives the straight command; environment 1 receives zero, including after automatic resets. The video follows environment 0. Measurements include both environments, reset flags, wrist tracking error, measured object contact forces and particle reaction wrenches. Do not calculate displacement across reset boundaries.

## Command and teacher contracts

Directions are expressed in the robot heading frame. The student observes only the 2D direction, proprioception and object pose relative to the robot. Teacher commands are derived from the current state.

| Direction | Lower-body teacher velocity |
| --- | --- |
| `(1, 0)` | `(0.8, 0, 0)` m/s, m/s, rad/s |
| Unit `(dx, dy)` | `(0.8, 0, clip(atan2(dy, dx), -0.5, 0.5))` |
| `(0, 0)` | Zero velocity, neutral hand targets and upright posture |

Training samples headings within ±45 degrees and zero commands with probability 0.2. A heading-relative nonzero angle is a sustained steering command, not a fixed world-space destination. The commanded world heading is integrated for the locomotion teacher.

The command stores the robot position when zero is sampled. Position/velocity rewards encourage staying there; touching the object is penalized during zero commands. Object progress and turning rewards require measured wrist/finger contact. These are training objectives, not a guarantee that the current frozen teachers remain stationary.

The posture transitions from neutral at robot–object XY distance 0.85 m to contact posture at 0.65 m. Neutral torso height is 0.75 m. Contact height follows object center height plus 0.20 m, clamped to 0.50–0.65 m **above the initial sand surface**, and torso pitch approaches 25 degrees. Contact wrist points are near the rear upper face; their positions are transformed into `torso_link` coordinates and projected onto a sampled feasible wrist workspace. Away from the object, the neutral bent-arm targets are preserved. See `env_cfg/commands_cfg.py` for all parameters.

Pinned teachers in `agents/compositional_rsl_rl_cfg.py`:

- `jnskkmr/g1_29dof_wbc_particle_nerd_mixed_terrain/ltn8eu4n`: `model_2000.pt`, latest available when downloaded. Input 799, output 15; includes pitch and height commands.
- `jnskkmr/g1_29dof_manipulation/6pbvl2df`: `model_19999.pt`. Input 76, output 14; wrist targets and measured wrist poses are relative to `torso_link`.

Checkpoints and source run configs are stored locally under `exports/particle_nerd_object_push/teachers/<run-id>/`. Copy these files to the same relative paths when moving the checkout; they are not included as source artifacts. Lower-body/waist PD gains and joint offsets match the locomotion teacher; arms match the manipulation teacher. The workspace dataset provenance is in `data/README.md`.

The compositional PPO runner clones the two action subsets initially, then ramps the complete PPO objective from iteration 300 to 10,000. Both teacher weights decline from 1.0 to 0.1 during that interval. Symmetry augmentation is enabled. The compositional-rsl-rl revision is `31f080317f519d84327dde401daa66ef2244d53a`, matching the main checkout's multi-teacher implementation. Newton, MuJoCo Warp, Warp and standalone RSL-RL pins are unchanged.

## Geometry and simulation scope

The default object is a 5 kg, 0.65 m cube, spawned 0.9–1.0 m ahead of the robot, on a flat surface. Random sphere/cylinder allocation is not enabled. The simulated patch is 3.4 × 2.4 m with a 0.2 m recycling boundary on each side, sand depth 0.3 m and sampling voxel size 0.04 m. There are no physical containment walls. The episode ends if the object's projected extent leaves the simulated patch. Initial material parameters are fixed; runtime material randomization is not enabled because the neural material features are bound at solver initialization.

The opt-in `primitive_colliders` solver path supports more than the original two foot colliders and uses the existing primitive contact estimator. Its per-world contact preparation is not the optimized two-foot batched path; large-scale throughput has not been benchmarked. The 5 × 5 m virtual feature domain encloses this larger patch; the default for existing tasks remains 3 × 3 m.

The particle model was trained for the existing G1 setup. Applying it to a large cube is an extrapolation; numerical smoke tests do not establish agreement with native MPM object–sand forces.

## Validation and current limitations

- The integration test checks direction mapping, teacher observation sizes, finite simulation, particle reaction on both objects, and isolation of partial resets.
- Both existing fixed-bed Particle NeRD regression variants passed.
- Two real PPO + behavior-cloning updates passed with both teachers loaded and compiled neural inference enabled.
- In the final 8-second two-teacher rollout, the straight-command cube moved approximately 0.42 m forward with no reset in that environment. Hand contact above 1 N was present in 41% of steps; mean wrist target error after 2 seconds was about 4.1 cm.
- Contact remains intermittent. The zero-command environment reset twice. This is not yet a reliable hand-only pushing or stationary controller. The generated rollout is an experimental baseline, not evidence of successful policy training.

Local artifacts: `exports/particle_nerd_object_push/rollout_low/teachers.mp4`, `measurements.json`, and `report.html`.

```bash
uv run --no-sync python -m pytest source/isaaclab_tasks/test/contrib/test_particle_nerd_push.py -q
```

## Rear pushing versus side squeeze probe

`util/compare_grips.py` runs frozen teachers with a timed neutral/reach/hold/walk sequence and records actual contact forces. Example:

```bash
uv run --no-sync python -m isaaclab_tasks.contrib.g1_29dof_particle_nerd_moving_patch_object_push.util.compare_grips --mode side --speed 0.5 --output exports/particle_nerd_grip_comparison/side_test
```

Use `--mode rear` for the rear-contact control and `--mode standing --speed 0` for the unloaded-standing control. Both contact trials use the same 0.70 m object-center distance, 0.60 m pelvis height command above the initial sand surface, and 25-degree torso pitch. This probe overrides commands locally; it does not change training defaults. Targets are sent without workspace projection so that an infeasible side target cannot silently become a rear target. Trials stop before excessive torso tilt or loss of particle coverage; the standing control stops at first robot-object contact.

The five final screening clips and measurements are in `exports/particle_nerd_grip_comparison/report.html`. With the pinned teachers, attempted side squeezing did not establish a sustained grip: hold wrist errors were about 44 cm and mean bilateral inward squeeze was below 0.1 N. The zero-command control drifted about 0.68 m by first object contact at 1.78 s. These results do not establish that side squeezing is mechanically ineffective: the current teacher did not achieve the requested contact geometry. Knee contacts also contaminate the pushing trials. Source-task locomotion validation and feasible side-contact targets are required before drawing a training conclusion.
