# G1 29-DoF locomotion on MPM sand

MJWarp integrates the humanoid, an implicit MPM solver integrates a per-environment granular bed,
and a `CouplerProxy` hands the ankle roll links to the MPM solver as lagged colliders. The bed
surface sits at `z = 0`, so the flat-ground reward and termination terms carry over unchanged.

## Scene layout

The robot is spawned on a rigid approach platform and walks in `+x` onto the bed, so a policy
meets the granular transition rather than starting already immersed in it.

```
        rigid approach platform                 granular bed
   x = -4.60 ......................... -1.50 ................... +1.50
   slab top z =  0.00                         surface z = 0.00, depth 0.25
                  ^ robot spawns at x = -3.00, 1.5 m short of the bed edge
```

* The retaining wall on the entry face is replaced by `approach_bank`, an MPM-only collider whose
  top is flush with the bed surface. It still holds the whole particle column back, but presents
  no step for the robot to climb.
* `approach_platform` is the rigid-entry slab the robot actually walks on. Its top is flush with
  the bed surface, so the only step at the boundary is the bed's steady-state sinkage.
* `WALKABLE_XY_BOUNDS` spans platform and bed together; `root_outside_workspace` terminates on it.

Because the feet are proxies, the granular reaction on each foot is the feedback wrench the
coupler returns to the rigid solver. `G1MPMEnv` harvests it once per policy step and derives the
contact flags and the gait timers from it, so the privileged observation group carries the same
terms as the soft-contact task.

## Wandb login
```bash
uv run --with wandb wandb login
```

## Training
```bash
isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Sand-G1-29dof-MPM --num_envs 32 --viz none
uv run --with wandb isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Sand-G1-29dof-MPM --num_envs 16 --viz none
```

## Inference
```bash
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Sand-G1-29dof-MPM-Play --num_envs 4 --viz newton_gl --wandb_run 4blqnxbi
uv run --with wandb --extra ovrtx isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Sand-G1-29dof-MPM-Play --num_envs 4 --viz newton_rtx --wandb_run 4blqnxbi
```

## Tuning notes

* `FOOT_CONTACT_MARGIN` (default three quarters of an MPM voxel) inflates the *sole* collision
  surfaces for the granular solver. It is scoped to the proxied ankle roll links through the
  spawner's per-pattern `collision_props`, never applied to the whole robot: Newton sums both
  shapes' margins, and with self-collisions enabled a whole-robot margin holds every non-adjacent
  link pair apart by twice the value, which locks up legs whose shin and sole clear each other by
  only 0.02 m in the nominal stance. Implicit MPM sees a collider only through its occupancy of the
  background grid, so a sole thinner than a voxel carries no load and the robot sinks through the
  bed. Refining `MPM_VOXEL_SIZE` without keeping the margin at roughly a voxel brings the sinking
  back. A full-voxel margin traps particles inside the sole, which a coupled entry cannot project
  out, and the world diverges into `NaN`; the three-quarter value is that stability limit.
* `grid_padding` must stay `0`: a padded sparse grid is not rebuildable, which disables CUDA-graph
  capture and overruns the node capacity. The `50` used by the standalone Newton examples applies
  to their fixed grid only.
* `critical_fraction` is the voxel fill fraction below which the yield surface collapses. The bed
  is sampled at a spacing that does not divide the voxel size, so its cells straddle the `0.5` of
  the standalone G1 example and half the bed turns into a frictionless fluid; keep it at `0.0`.
* `proxy_mass_scale` (default 26.6) is the effective-mass ratio the feet present to the granular
  solver, taken from `coupling_relaxation` in the standalone Newton G1 sand example. Lower it and
  the robot sinks; raise it and the bed behaves rigidly.
* The sparse-grid capacities (`mpm_*_per_world`) are sized for the default 3.0 x 3.0 x 0.25 m bed
  at a 0.05 m voxel size. Enlarging the bed or refining the voxels requires raising them.
* `default_shape_cfg` carries `ke=160000.0, kd=1100.0`, matching the rigid and soft-contact G1
  tasks. Newton's defaults (`ke=2.5e3`, `kd=1e2`) reach MJWarp through `convert_solref` because
  `use_mujoco_contacts=False`, and leave the rigid approach platform far too compliant for a 32 kg
  humanoid to walk on.
* `foot_contact_force_threshold` (default 5 N) sets how much granular reaction a foot must carry
  to count as in contact, and therefore drives every gait reward.
* `terrain_material_parameters` is a zero placeholder that keeps the privileged observation layout
  identical to `g1_29dof_soft`; there is no single per-foot stiffness in the MPM bed.
