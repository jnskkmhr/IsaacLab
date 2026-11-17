# G1 lower-body locomotion on native MPM

This task uses the fixed beds, approach platforms, powered trays, robot asset,
material randomization, and MJWarp/implicit-MPM coupling of `g1_29dof_mpm`.
It does not load Particle NeRD or a learned particle model.

The actions, observation order/history, reward weights, and standard PPO architecture
match `g1_29dof_particle_nerd_moving_patch_lower`:

- 15 policy actions: left leg (6), right leg (6), waist yaw/roll/pitch (3), with scale 0.25.
- 14 arm joints follow randomized trapezoidal position trajectories, contributing no policy actions.
- Actor history: angular velocity, biased projected gravity, all 29 relative joint positions,
  all 29 relative joint velocities, previous 15 actions; ten frames = 790 values.
- Critic history: angular velocity, root quaternion, joint positions/velocities, previous actions = 800 values.
- Command: velocity/heading (7) and root height (1). Height is sampled in [0.60, 0.80] m.
- Privileged history: linear velocity, foot heights, contact flags, log-compressed combined
  MPM/MJWarp forces, air times, material placeholder = 160 values.
- PPO inputs: actor 798, critic 968. Only policy-controlled joints receive joint regularization.
- Training retains the reference roll/pitch bias curriculum (0 to ±3° over 120,000 steps).
  Play uses clean projected gravity. Random arm motion and the height command remain active in play.

All six testbeds have train and `-Play` registrations:

| Testbed | Training task |
| --- | --- |
| Flat | `IsaacContrib-Velocity-Sand-Flat-G1-29dof-MPM-Lower` |
| Ascending | `IsaacContrib-Velocity-Sand-Ascending-G1-29dof-MPM-Lower` |
| Descending | `IsaacContrib-Velocity-Sand-Descending-G1-29dof-MPM-Lower` |
| Dune | `IsaacContrib-Velocity-Sand-Dune-G1-29dof-MPM-Lower` |
| Tilting balance | `IsaacContrib-Balance-Sand-Tilting-G1-29dof-MPM-Lower` |
| Tilting walking | `IsaacContrib-Velocity-Sand-Tilting-G1-29dof-MPM-Lower` |

## Terrain references

The foot rewards keep the reference formulas and weights. A mesh-targeted height
scanner sees only the platform/support floor (and tracks the moving tray), excluding
the robot. Clearance adds sand depth over the bed, zero over the approach, and
converts tray-normal depth to vertical depth for a tilted tray. It estimates the
initial surface, not the deformed particle surface.

The root-height command/reward remains relative to the environment origin, exactly
as in the source lower-body task; it is **not terrain-relative on slopes**. Velocity
ranges, spawn/reset locations, episode lengths, and material randomization follow
the fixed-bed task. The generated-terrain level curriculum is inapplicable to these
finite testbeds. Reward/IMU curricula are local to this task.

## Run

Run from `/home/jkamohara/isaac/IsaacLab`:

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Sand-Flat-G1-29dof-MPM-Lower \
  --num_envs 4 --viz newton_gl

uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Sand-Flat-G1-29dof-MPM-Lower-Play \
  --num_envs 1 --viz newton_rtx \
  --checkpoint /absolute/path/to/lower_body_model.pt 'env.video_recorders=[]'
```

Use a checkpoint with this 15-action/history/height-command contract; existing
29-action full-body policies are incompatible. Replace `Flat` with `Ascending`,
`Descending`, or `Dune`, or use a tilting task from the table.
The default fixed beds use 0.02 m voxels and simulate the entire bed, so start with
a small environment count. No caching or moving patch is involved.

## Validation

- All 12 registrations validate; their scene geometry and physics settings match the
  corresponding `g1_29dof_mpm` configuration, apart from the added height scanner.
- Action/observation/reward term configurations were compared against the lower-body
  Particle NeRD reference.
- The CUDA integration test checks 15-action control, independent arm motion, partial
  reset, native sand contact, and floor scans in two environments.
- Two PPO updates completed with two environments at the default 0.02 m voxel size;
  final value/surrogate losses were 4.0970 / -0.0626. This is a training smoke test,
  not evidence of learned locomotion performance.
- Separate reduced-resolution checks covered flat, ascending, descending, dune, and
  tilting scenes. Two moving-tray scans matched the actual tray plane within 1.4e-7 m.

```bash
uv run --no-sync python -m pytest source/isaaclab_tasks/test/contrib/test_g1_mpm_lower.py -q
```
