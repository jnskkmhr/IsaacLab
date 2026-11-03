# G1 box pushing on flat MPM sand

A single whole-body compositional PPO policy tracks the robot's commanded base velocity and two hand SE(3) targets attached to a dynamic box. This initial task covers forward pushing and stopping on flat sand. It does not specify a global object goal or include slope, dune, or tilting variants.

## Run

Train from scratch:

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Push-Box-G1-29dof-MPM \
  --num_envs 16 --logger wandb --viz newton_gl --video
```

Play a checkpoint **trained for this task**:

```bash
uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Push-Box-G1-29dof-MPM-Play \
  --num_envs 1 --viz newton_rtx --video --video_length 500 \
  --wandb_run YOUR_PUSH_RUN --wandb_project g1_29dof_mpm_object_push
```

The policy has additional manipulation inputs. Existing locomotion checkpoints cannot be loaded unchanged: locomotion warm-starting requires transferring compatible weights and expanding the actor's first layer. That conversion is not implemented by this task. Terrain-state checkpointing and symmetry augmentation are disabled; a hand/object mirroring map would be required before enabling symmetry.

## Commands and observations

- Base command: forward velocity sampled in `[0, 0.4]` m/s, zero lateral/yaw velocity, with 20% standing commands and resampling every 4–6 seconds. Play uses 0.25 m/s.
- Hand commands: left/right contact positions and orientations attached to the rear box face, expressed relative to the robot root. They follow the box continuously, including immediately after reset.
- The actor retains the locomotion `policy` history and the separate `command` group. A new **unstacked** `manipulation` group contains object pose (7), desired hand poses (14), and measured hand poses (14).
- The critic also receives object linear/angular velocity in the robot root frame (6), in addition to the existing locomotion privileged inputs.
- Poses use XYZ position and XYZW quaternion. Object observations use simulation state; a perception pipeline is not part of this implementation.

`commands.py` defines the hand contact frames and target offsets. The contact point is 0.132 m along each rubber-hand link's local x axis, near the front of its mesh. The default hand orientation aligns the hand axes with the box axes. These are configurable hand targets, not independent arm joint commands.

## Scene and physics

- Box: 0.50 × 0.65 × 0.70 m, 5 kg, initially 0.85–0.95 m ahead of the robot along its yaw direction. Its bottom starts 1 cm above the flat initial particle surface. Play uses 0.90 m.
- Robot and box share the MJWarp solver. The box and feet are also collision proxies in the implicit MPM solver, with reaction forces returned to MJWarp.
- The original locomotion USD has visual-only rubber hands. A task-local spawner enables convex collision meshes on those hands before cloning; it does not edit the shared USD asset.
- The robot-centered moving particle patch is enlarged to 3.6 × 2.4 m. A termination checks all box corners against the actual patch center, reserving a 0.30 m interior margin before recycled/boundary particles.
- Episodes terminate on excessive box tilt as well as the inherited robot failures, and time out after 12 seconds.
- Physics runs at 200 Hz, policy at 50 Hz, with CUDA graph capture enabled.

The task reuses the moving-patch implementation. It initially disables material/gain/mass randomization, external pushes, and the locomotion curricula to isolate learning hand contact and pushing. It uses unbiased projected gravity. Training still uses the inherited proprioceptive observation noise.

## Rewards

The locomotion rewards are retained, with arm joint-deviation coefficients set to zero so reaching is not penalized for departing from the nominal arm pose. Leg and waist posture regularization remains active.

| Added term | Weight | Definition |
| --- | ---: | --- |
| Hand position tracking | 5.0 | Mean of `exp(-position_error² / 0.25²)` over the two hands |
| Hand orientation tracking | 1.0 | Mean of `exp(-rotation_angle² / 0.7²)` over the two hands |
| Object planar velocity tracking | 3.0 | `exp(-velocity_error² / 0.25²)`, multiplied by hand-target proximity with a 0.15 m scale |

Object velocity is measured relative to the world and rotated into the robot yaw frame. Its target is the same planar velocity command used for the robot. The proximity factor discourages earning object-motion reward while abandoning the hand targets; it is not a measured hand-contact force.

These are initial training settings. Simulator/PPO smoke checks establish that the task runs; they do not establish successful learned pushing or deployment readiness.
