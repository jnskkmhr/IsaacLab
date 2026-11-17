# G1 29-DoF locomotion on MPM sand

MJWarp integrates the humanoid, an implicit MPM solver integrates a per-environment granular bed,
and a `CouplerProxy` hands the ankle roll links to the MPM solver as lagged colliders. Flat sand starts at `z = 0`; generated scenes shape both the particle layer and its support mesh.

## Scene variants

Each scene declares its own particle-bed dimensions in `env_cfg/scene_cfg.py` through
`_sand_cfg(size=(length, width, depth), ...)`. Support colliders, approach placement,
robot spawn, terrain-generator footprint and tilting-tray dimensions are derived from
that scene's particle bounds. Resizing the balance tray does not resize the flat bed.
The particle spawner computes its radius from the grid; solver capacities use the actual
particle count. Runtime bounds checks and tray-relative height rewards read the active
scene geometry.

All entries use the same `agents/compositional_ppo_cfg.py:G1MPMPPORunnerCfg`. Add `-Play` to
any training ID below for playback with randomization disabled.

The runner uses compositional PPO with independent actor and critic MLPs. Separate-head ordinary
RSL-RL checkpoints remain loadable when their network and observation layouts match. The optional
terrain curriculum interface is disabled for these fixed-bed scenes.

| Scene class | Sand dimensions (m) | Shape | Training ID |
| --- | --- | --- | --- |
| `G1MPMSceneCfg` | 6 × 2 × 0.3 | Flat | `IsaacContrib-Velocity-Sand-G1-29dof-MPM` |
| `G1RoughMPMSceneCfg` | 6 × 2 × 0.3 | Ascending X ramp, slope +0.4 | `IsaacContrib-Velocity-Sand-Rough-G1-29dof-MPM` |
| `G1DescendingMPMSceneCfg` | 6 × 2 × 0.3 | Descending X ramp, slope −0.4 | `IsaacContrib-Velocity-Sand-Descending-G1-29dof-MPM` |
| `G1DuneMPMSceneCfg` | 6 × 2 × 0.3 | Two sinusoidal ridges, 0.3 m trough-to-crest | `IsaacContrib-Velocity-Sand-Dune-G1-29dof-MPM` |
| `G1TiltingMPMSceneCfg` | 3 × 2 × 0.3 | Balance tray, ±15° roll/pitch | `IsaacContrib-Balance-Sand-G1-29dof-MPM-Tilting` |
| `G1WalkingTiltingMPMSceneCfg` | 4 × 4 × 0.3 | Walking tray, ±10° roll/pitch | `IsaacContrib-Velocity-Sand-Tilting-G1-29dof-MPM` |

For example, the walking tray's independent size is declared as:

```python
@configclass
class G1WalkingTiltingMPMSceneCfg(G1TiltingMPMSceneCfg):
    sand = _sand_cfg(size=(4.0, 4.0, 0.3))
```

The fixed-bed scenes start G1 on a rigid approach platform and walk in +X onto the
sand. The platform top is 5 cm below the initial sand surface, matching the settled
default sand. The sand entrance is at world Z = 0; descending terrain falls below
that height. Dunes are cosine ridges along X, constant across Y, with a level entrance.
Both rigid and MPM support meshes follow the generated initial surface 0.3 m below it.

The tilting scenes start G1 in the sand, with no approach platform. Their physical
roll/pitch joints drive the tray; particle motion remains solver-owned. `util/terrain.py`
contains terrain/particle spawners, and `util/tilting_testbed.py` contains the tray spawner
and motion configuration. The previous root-level helper module paths have moved.

These geometry variants retain the existing rewards and policy inputs. Some inherited
foot-clearance/touchdown terms still use a flat-ground reference; adding an entrypoint
does not establish that those rewards are suitable for training on every new shape.

### Walking on the tilting tray

Training samples forward/backward and lateral speed in ±0.5 m/s and yaw rate in
±0.5 rad/s. Playback defaults to forward motion at 0.5 m/s, with no standing-command
fraction. To walk laterally with an existing checkpoint:

```bash
uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Sand-Tilting-G1-29dof-MPM-Play \
  --num_envs 1 --viz newton_rtx \
  --wandb_run 876gqr97 --wandb_project g1_29dof_soft_flat_ppo \
  'env.commands.base_velocity.ranges.lin_vel_x=[0.0,0.0]' \
  'env.commands.base_velocity.ranges.lin_vel_y=[0.3,0.3]'
```

For turning in place, set both linear ranges to `[0.0,0.0]` and
`env.commands.base_velocity.ranges.ang_vel_z=[0.3,0.3]`. The finite tray still terminates
an episode when the robot base approaches its edge; the walking task does not steer
back toward the tray center automatically.

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

### Tilting testbed

`IsaacContrib-Balance-Sand-G1-29dof-MPM-Tilting` starts G1 in the sand with a zero
velocity command. Its `-Play` variant disables randomization. The stationary
approach platform is replaced by a closed tray attached to roll and pitch joints.

```bash
uv run --extra rsl-rl isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Balance-Sand-G1-29dof-MPM-Tilting --num_envs 4 --viz none

uv run --extra rsl-rl isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Balance-Sand-G1-29dof-MPM-Tilting-Play --num_envs 1 \
  --viz newton_gl --checkpoint /path/to/model.pt
```

Configure `testbed_motion` in `G1TiltingMPMEnvCfg`: roll/pitch ranges are in radians,
defaulting to ±15° for balance and ±10° for walking, with 12 s and 17 s periods. The tray holds level for 2 s after
reset, then ramps into the oscillation over 3 s. Mechanical joint limits are ±45°.
For example, `roll_range=(math.radians(-5), math.radians(10))` sets the roll target
range to −5° through +10°. Each target range must include zero and lie strictly
inside the mechanical limits (`scene.testbed.spawn.joint_limit_deg`).

`testbed_motion.debug_vis=True` draws two arrows at a tray corner: blue follows
the actual tray normal; green points opposite gravity (world-up with default gravity).
The angle between them shows the tray's tilt. Set it to `False` to hide
the arrows. The tray geometry and actuator configuration live in
`env_cfg/scene_cfg.py:G1TiltingMPMSceneCfg`.

Targets update at 50 Hz; MJWarp integrates the driven joints at 400 Hz and MPM
updates at 200 Hz with the default settings. The tray is a dynamic proxy collider
in MPM. Its actual angle can differ from its target under load. No particle or
tray poses are overwritten during stepping; episode resets restore the tray,
robot, particles, and coupling history together. Bounds and base-height rewards
use the rotating tray frame; the height reference is the nominal sand surface,
not the instantaneous deformed particle surface.

### Fixed-bed settings

* `MPM_VOXEL_SIZE` defaults to 0.04 m. `FOOT_CONTACT_MARGIN` is 0.01875 m,
  applied only to the proxied ankle-roll links through the robot spawner's per-pattern
  collision settings. Static MPM retaining colliders use `MPM_COLLIDER_MARGIN=0.0`.
* `grid_padding` must stay `0`: a padded sparse grid is not rebuildable, which disables CUDA-graph
  capture and overruns the node capacity. The `50` used by the standalone Newton examples applies
  to their fixed grid only.
* `critical_fraction` is the voxel fill fraction below which the yield surface collapses. The bed
  is sampled at a spacing that does not divide the voxel size, so its cells straddle the `0.5` of
  the standalone G1 example and half the bed turns into a frictionless fluid; keep it at `0.0`.
* `proxy_mass_scale` defaults to 1.0 and scales the effective mass of the robot proxies
  seen by the particle solver.
* `mpm_active_cell_count_per_world=None` computes active-cell capacity from the scene's
  actual particle-grid dimensions. Sparse-grid node counts also grow with particle count;
  the square walking tray has larger minimum node budgets than the balance tray.
* `default_shape_cfg` carries `ke=160000.0, kd=1100.0`, matching the rigid and soft-contact G1
  tasks. Newton's defaults (`ke=2.5e3`, `kd=1e2`) reach MJWarp through `convert_solref` because
  `use_mujoco_contacts=False`, and leave the rigid approach platform far too compliant for a 32 kg
  humanoid to walk on.
* `foot_contact_force_threshold` (default 5 N) sets how much granular reaction a foot must carry
  to count as in contact, and therefore drives every gait reward.
* `terrain_material_parameters` is a zero placeholder that keeps the privileged observation layout
  identical to `g1_29dof_soft`; there is no single per-foot stiffness in the MPM bed.
