# G1 29-DoF locomotion on MPM sand

MJWarp integrates the humanoid, an implicit MPM solver integrates a per-environment granular bed,
and a `CouplerProxy` hands the ankle roll links to the MPM solver as lagged colliders. The bed
surface sits at `z = 0`, so the flat-ground reward and termination terms carry over unchanged.

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

* `FOOT_CONTACT_MARGIN` (default three quarters of an MPM voxel) inflates the robot collision
  surfaces for the granular solver. Implicit MPM sees a collider only through its occupancy of the
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
* `foot_contact_force_threshold` (default 5 N) sets how much granular reaction a foot must carry
  to count as in contact, and therefore drives every gait reward.
* `terrain_material_parameters` is a zero placeholder that keeps the privileged observation layout
  identical to `g1_29dof_soft`; there is no single per-foot stiffness in the MPM bed.
