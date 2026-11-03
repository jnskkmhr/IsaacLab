# G1 moving-patch MPM with particle caching

This experimental task replaces `g1_29dof_mpm_moving_patch_beta`. Its registered
IDs end in `MovingPatch-ParticleCache` and `MovingPatch-ParticleCache-Play`.
The original `g1_29dof_mpm_moving_patch` task is unchanged by this implementation.

New tiles start from a **pre-relaxed material template**, including elastic strain,
plastic volume ratio, stress and deformation transform. Previously visited tiles
restore their own complete particle history. GL and RTX can render a continuous
isosurface of both loaded and saved particles. Multiple tracked bodies within
one environment load the union of their surrounding tiles; their rigid bodies
participate in two-way MJWarp–MPM coupling.

## Source and reuse

The paged-sand example is on Felix Meyer's `fmeyer/mpm-paged-sand` branch,
[PR #14](https://github.com/NVIDIA-dev/devtech-newton-examples/pull/14), rather
than the repository's `main` branch at the time of this implementation.
The reference revision is `3e4f29ecf369bf66aeb93e73c9b4f9bd59bf6e7f`.

* `util/felix/paged_sand_isosurface.py` contains his tiled marching-cubes,
  particle splatting, surface normals, seam/skirt kernels and the GPU `IsoCache` mesh pool.
* `util/felix/settled_template.py` adapts his offline PIC relaxation and
  frictionless mirror-wall construction. Fresh tiles mirror particle positions
  and constitutive tensors in alternating X/Y tiles.
* The task retains its GPU particle-state store and per-environment ownership.
  It does **not** replace this with Felix's single-scene host chunk store.

The reference files retain their Apache-2.0 attribution and license. Local
adaptations and their reasons are listed in `util/felix/README.md`.

## Run

```bash
uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Sand-G1-29dof-MPM-MovingPatch-ParticleCache-Play \
  --num_envs 1 --viz newton_rtx \
  --wandb_run tyiby5k6 --wandb_project g1_29dof_soft_flat_ppo
```

Use `newton_gl` for OpenGL. The training ID omits `-Play`. Both use the existing
observation/action layout, with a separate `g1_29dof_mpm_moving_patch_particle_cache`
experiment name. The PLAY configuration has its own terrain and depth overrides.
Camera tracking uses the core viewer's `origin_type="asset"` and
`origin_track_path="robot"`. The same surface path is used for offline video.

## Template relaxation

A small material tile is simulated under gravity at startup. Acceptance requires
both a low maximum particle speed and a stable mean particle height over a
consecutive window. These are numerical tolerances, **not exact equilibrium**:

```python
terrain = cfg.scene.terrain.moving_patch_terrain
cache = terrain.particle_cache
cache.settle_dt = 0.005
cache.settle_max_steps = 500
cache.settle_speed = 0.03  # maximum particle speed, m/s
cache.settle_quiet_steps = 25
cache.settle_height_tolerance = 0.001  # mean-height variation over that window, m
cache.max_material_templates = 16
```

The task's jittered initial samples are contained inside the template box. A
normal-velocity guard extends Felix's floor guard to the four mirror walls,
preventing particles that escape Newton's finite collider search distance from
being accepted into a template. A collapsed or nonconverged template raises an
error instead of being silently used. Fresh tiles copy the relaxed material
state, with zero translational velocity and APIC velocity gradient, as in the
reference's pristine-tile initialization.

Templates are keyed by all available constitutive material parameters, particle
mass and initial plastic volume ratio. Worlds with identical materials share a
cached CPU template but receive independent GPU copies. Reset-time material
randomization builds a new template when necessary; it never reuses a template
from a different material. **Continuous per-episode material randomization can
therefore make resets expensive.** For fixed-material experiments, disable
`events.mpm_material` and `events.mpm_material_log` in the task configuration (or
use `env.events.mpm_material=null env.events.mpm_material_log=null` on the CLI).

Template relaxation is performed on a flat support surface. The resulting
positions are mapped under the generated terrain height. Rough terrain,
material changes and external loads can still cause physical rearrangement;
pre-relaxation does not guarantee static equilibrium on a slope. The configured
particle depth describes the initial sampled bed; compaction may lower the
relaxed sand surface relative to that height.

## Multiple rigid objects

The rigid solver and its MPM proxy mapping include all rigid bodies beneath
`/World/envs/env_[^/]+/`, including the robot's links and additional objects.
Additional bodies that need their own moving particle region must be declared
explicitly. Overlapping regions share particles within one environment; regions
in different environments never share particle histories.

For example, add a standard `RigidObjectCfg` to the scene:

```python
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils

cfg.scene.box = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Box",
    spawn=sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 0.0, 0.4)),
)
cfg.events.reset_box = EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={"asset_cfg": SceneEntityCfg("box"), "pose_range": {}, "velocity_range": {}},
)
terrain.tracked_body_patterns = [r"/World/envs/env_[^/]+/Box"]
cache.simulation_capacity_multiplier = 4
```

Patterns match Newton's full body labels and must resolve in every environment.
They can also select robot hands or bodies from an additional articulation. Size
`simulated_terrain_size` to cover the object's contact footprint and leave a
boundary at least two MPM voxels wide. The fixed slot budget must cover the union
of all regions, including disjoint objects and displaced material. Insufficient
capacity raises before recycling any particle state. New objects also need
appropriate reset events, observations and rewards for the desired manipulation
task; these are not inferred from their presence in the scene.

The current implementation keeps these tracked regions active. It does not
implement Felix's rigid-island sleep/wake scheduler or cache rigid-body state.

## Particle storage and reset behavior

```python
cache.tile_size = 0.2
cache.simulation_capacity_multiplier = 2  # increase for additional regions
cache.max_cached_particles_per_env = 500_000
cache.max_visited_tiles_per_env = 4096
```

Each departed particle retains position, velocity, APIC velocity gradient,
elastic strain, plastic volume ratio, stress, deformation transform, dynamic
mass, radius and flags: approximately 184 bytes per record. Boundary particles
have zero mass and velocity while inactive, but their original physical state
is retained separately and restored on reactivation. Loading a saved tile
consumes its saved records. An emptied visited tile remains empty.

A reset clears only that environment's visited tiles and saved particles. Other
environments are unaffected, including when their world coordinates overlap.
Saved history lives in memory and does not survive process exit. Frozen and
unloaded particles do not evolve while inactive. Material coefficients are
per-environment model parameters; changing them mid-episode while retaining old
history is unsupported.

Tile dictionaries and free lists remain on the CPU; physical particle records
stay on the GPU. The coupled solver is captured separately with
`env.sim.physics.capture_solver=true` (default for proxy coupling).
The manager-wide `use_cuda_graph` remains false so host paging runs before every
physics step and fixed-boundary restoration runs afterwards. Reset randomizers
update existing GPU buffers; a changed buffer address or material-dependent
solver branch invalidates the captured solver. Set `capture_solver=false` for an
eager comparison. The experimental ADMM and direct-coupling presets remain eager.
The task requires sparse MPM,
`separate_worlds=True` and `strain_basis="P0"`. The default saved-record pool costs
about 92 MB per environment, plus simulation slots, frozen state and solver
storage. There is no claim of improved throughput or large-scale training
scalability over the original moving-patch task.

## Surface rendering

```python
cache.render_mode = "isosurface"  # default; alternatively "particles"
cache.surface_resolution = 0.02  # approximate grid spacing, m; use 0.04 for cheaper rendering
cache.surface_cache_mb = 64  # lazy GPU mesh pool per visible world
```

GL and RTX reconstruct loaded, frozen and saved particles for the visible
worlds. Surface resolution changes only rendering, not particle positions,
colliders or contact forces. The splat weight accounts for particle volume
relative to the surface grid cell volume. Saved deformations remain visible
outside the current active region; a reset removes the selected world's old
surface history. Kit currently requires `render_mode="particles"`.

The GL/RTX background visual follows the estimated relaxed sand height, with openings
for live and previously visited tiles. Saved particle tiles are also excluded
from that background, so it cannot cover footprints. The rigid support meshes
remain at their original physical depth beneath the sand. This changes rendering
only. Live surfaces, saved surfaces and the background share the terrain's color
texture, world-XY UV scale, fallback color and roughness. Surface reconstruction
keeps its B-spline width at least as large as the particle spacing when the display
grid is refined, avoiding gaps caused solely by a finer visual grid.
The untouched sand height is estimated from the upper particle layer of each
world's relaxed template; between worlds, the nearest environment origin selects
the material's height offset. This avoids a raised background around compacted
sand. Newly deposited particles outside previously loaded tiles are drawn above
that untouched bed. The estimate does not move particles or rigid support meshes.
At the outer isosurface edge, a narrow display band blends vertex heights and
normals into the exact background triangles. Its width is three display cells or
one particle spacing, whichever is larger, capped at one cache tile (6 cm with
the default display settings). Internal tile joins and deformation farther inside
the particle region are unchanged. This removes sharp edge shading and small
height gaps without adjusting particle states or collision geometry.

Both Newton viewers start with `show_particles=False` and `show_isosurface=True`.
Open **Sand visualization** in the GUI to toggle **Particles** and **Isosurface**
independently. To inspect particle motion without the surface obscuring it, enable
Particles and disable Isosurface. The same configuration flags apply to offline
video rendering. With `render_mode="particles"`, enable `show_particles=True`;
the task does not reconstruct isosurfaces in that mode.
Particle view includes unloaded particles at their saved positions and radii,
so deformation remains visible after the moving patch leaves. These display
buffers refresh only when the saved records change; they do not reactivate
cached particles in physics. Reloading consumes the saved records, and resetting
a world clears its displayed particle history. The particle toggle controls both
live and cached particles.

Loaded tiles, including their fixed boundary, are extracted each rendered frame
to keep the live/boundary seam consistent. Unloaded tiles are baked into
Felix's GPU mesh pool once, then reused until their particle records or adjacent
tile availability changes. Reloading removes the corresponding baked mesh;
selective reset clears only that world's mesh history. Only saved particles next
to the loaded patch are gathered for live extraction. Mesh-pool exhaustion raises
an error asking for a larger `surface_cache_mb`, rather than silently removing
footprints. GL/RTX tasks
do not author a second particle-cloud visual on the borrowed USD stage; Kit
retains its authored particle visuals.

The pinned Newton revision includes Felix's
[inactive-particle state fix (#4807)](https://github.com/newton-physics/newton/pull/4807)
on top of the existing triangle–box ground-contact fix. It preserves stress,
elastic strain and plastic volume ratio for inactive or massless particles.
It does not alter the triangle–box contact calculation.

## Validation

```bash
uv run --no-sync python -m pytest -q \
  source/isaaclab_tasks/test/contrib/test_mpm_particle_cache.py
```

The GPU integration test runs two worlds, each containing a robot and a separate
falling box. It checks nonzero sand reaction forces, supported box positions,
exact restoration of all ten fields after travel and return, selective reset
isolation, and finite, indexed isosurface geometry including saved tiles. It uses
coarse particles to keep the integration test small; it is not a physics
accuracy or training benchmark.

Local checks on 2026-11-02 (RTX 5090, Warp `1.19.0.dev20261008`) also produced
headless GL and RTX images. With the normal 0.04 m MPM voxel size and 0.25 m
initial bed depth, an unloaded-bed comparison over one simulated second measured
mean particle-height decreases of **23.6 mm from zero material state** and
**5.1 mm from the pre-relaxed template**. This reduced continued settling by about
78%; it did not eliminate it. Both runs used the same task settings with the
robot raised clear of the sand and material randomization disabled. A separate
two-world reset check passed with independently randomized material parameters.
A 50-step playback of local checkpoint `tyiby5k6/model_19999.pt` also completed
with finite policy observations and rewards while recording the RTX surface;
6,324 particle records were cached at the end. This is an integration check,
not evidence of long-horizon policy stability.
