# G1 MPM with GPU body-following activity regions

The Flat/Dune/Ascending/Descending tasks use finite, fully resident sand beds.
The Generator task instead uses a cached moving particle patch, described below. Newton selects activity around
tracked bodies on the GPU, then applies its native sleeping and compact working-set logic.
It does not import another task's environment or term configurations. Scene, action,
observation, reward, event, curriculum, physics, and region settings live in `env_cfg/`;
agent settings live in `agents/`. Reusable MDP functions may come from the rigid G1 task.

## Run

Run from the experimental checkout, whose environment uses the modified local Newton:

```bash
cd /home/jkamohara/isaac/experimental/IsaacLab
uv run --no-sync --extra ovrtx isaaclab play \
  --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Sand-Dune-G1-29dof-MPM-MovingPatch-Play \
  --num_envs 1 \
  --checkpoint logs/experimental_mpm/5f1xyc21/model_39998.pt \
  --viz newton_rtx "env.video_recorders=[]"
```

Replace `Dune` with `Flat`, `Ascending`, or `Descending` for the other registered
geometries. Omit `-Play` for the corresponding training configuration. These are
finite-bed experimental tasks; rough-surface reward height references retain the
fixed-bed baseline and need evaluation before a training comparison.

## Generated terrain with a cached moving particle patch

```bash
uv run --no-sync --extra ovrtx isaaclab play \
  --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Generator-G1-29dof-MPM \
  --num_envs 1 \
  --checkpoint logs/experimental_mpm/5f1xyc21/model_39998.pt \
  --viz newton_rtx "env.video_recorders=[]"
```

Append `-Play` to disable reset randomization and use playback commands.

This variant loads and unloads particle tiles as the robot moves, unlike the finite
Flat/Dune/Ascending/Descending beds below. It ports the main checkout's
`g1_29dof_mpm_moving_patch_particle_cache` cache and Felix-derived settled template.
Newton's body-following activity selection operates inside the loaded particle patch.

| Quantity | Default |
|---|---|
| Generated terrain | One 20 × 20 m tile; wave, rough, slope, or flat, selected with seed 42 |
| Dynamic particle patch | 1.2 × 1.2 m, follows pelvis |
| Boundary particle band | 0.2 m on each side; full loaded patch 1.6 × 1.6 m |
| Patch movement increment | 0.2 m |
| Sand depth | 0.3 m above matching MJWarp/MPM support meshes |
| Simulation slots | 411,008 including reserve slots at current sampling settings |
| Saved particle capacity | 2,000,000 records; overflow raises rather than losing deformation |
| Supported environment count | 1 for this experimental cache adapter |

Departing particles retain position, velocity, constitutive history, mass, and radius
in GPU cache buffers. Revisiting terrain restores that material state. New tiles use
one settled reference template; there is no per-reset or per-tile settling simulation.
Episode reset clears that environment's cached history. The isosurface displays loaded
sand, follows patch changes, and excludes unused reserve slots. No retaining walls or
approach platform are authored.

Crossing the local isosurface/particle-patch boundary does **not** terminate the episode.
Only the outer generated terrain boundary is used for workspace termination; normal
fall, timeout, and divergence conditions remain. This is finite generated terrain,
not infinite procedural terrain.

`env_cfg/terrain_cfg.py` owns terrain selection. `env_cfg/paged_patch_cfg.py` owns patch
size and cache settings. The scene uses `TerrainMPMGridCfg.paged_patch` to distinguish
this variant from the fixed beds. The reference cache uses CPU tile bookkeeping and
small index/count transfers outside captured physics; particle/material arrays remain
on the GPU. It is not an all-GPU paging implementation. Physics graph buffers remain
stable during patch movement. Training rewards retain the experimental fixed-bed
height references and are not yet a matched rough-terrain training recipe.

Validation: moved the robot to X=4 m beyond the old 4 × 4 m bed, observed patch center
follow without changing the episode counter, returned and recovered a marked particle's
position and plastic volume ratio exactly, then continued physics. This passed for
both normal and Play task IDs. Renderer regressions also exclude parked reserve slots.

## What moves

The region moves; the particle storage and terrain geometry do not.

| Item | Behavior |
|---|---|
| Tracked region | World-axis-aligned box following the pelvis; default half sizes `(0.8, 0.8, 0.8)` m and center offset `(0, 0, -0.5)` m from the pelvis |
| Inside a region | Newton's particle motion, force, stress, and collider activity tests select active patches |
| Stationary support | Newton adds neighboring support patches around active patches; this halo can extend beyond the region |
| Outside all regions | Collider-driven requests are filtered; moving/changing material is retained by default until it becomes quiet |
| Sleeping particles | Full material state remains in GPU memory, excluded from the MPM evaluation working set |
| Overlapping regions | Union of regions; no duplicate particle evaluations |
| Different environments | Body and particle world IDs keep their activity regions isolated |
| Episode reset | Restore the initial particle positions, velocities, material history, and activity state from GPU buffers; no settling simulation |

The initial snapshot uses one second of sand-only simulation. It is a relaxed initial
state, not a guarantee of complete equilibrium. Its remaining active particles stay
active after restoration and are reclassified on the next step. Material randomization
reuses this snapshot as an approximation; changing bed geometry requires a new snapshot.

## Multiple tracked objects

Edit `env_cfg/moving_patch_cfg.py`. Each body expression can match one or many bodies.
For example, after adding a rigid object named `Box` to the scene:

```python
regions = [
    BodyRegionCfg(),  # Robot pelvis in every environment
    BodyRegionCfg(
        body_path=r"/World/envs/env_[^/]+/Box(?:/.*)?",
        half_extents=(0.5, 0.5, 0.5),
        offset=(0.0, 0.0, -0.2),
    ),
]
```

Expressions match Newton body labels, not visual mesh labels. An unmatched expression
raises an error. Adding a region only changes activity selection: the object must also
be configured as a collider/proxy in the coupled physics configuration for it to exert
forces on sand. Multiple region/body inputs are covered by a captured-GPU Newton test;
the G1 playback smoke test uses one tracked pelvis per environment.

`keep_material_motion=False` forces requests outside the regions to sleep, apart from
the support halo. This can freeze moving sand and changes the simulation approximation.
For an explicit comparison:

```bash
# Append to the play command:
"env.moving_patch.keep_material_motion=false"
```

`working_set_capacity=None` reserves enough evaluation capacity for the entire bed so
initial relaxation, resets, and distant moving material cannot overflow it. Only
active/support particles are selected. A smaller explicit capacity is an optimization
limit and raises on overflow. This implementation does not bound total particle memory,
page particle state to CPU, recycle particles, or extend the finite bed. It is therefore
not yet a replacement for the main checkout's unbounded/recycling moving patch.

## Implementation and validation

- Newton `particle_regions.py`: region filtering on GPU before support-patch dilation.
- Newton `SolverImplicitMPM.set_particle_active_regions()`: configure live body transforms,
  world IDs, region sizes, and offsets before capture.
- `util/moving_regions.py`: resolve task body expressions, copy current coupled body poses
  inside the captured physics step, and configure the compact working set.
- `util/settled_bed.py`: one-time preparation and snapshot restoration.

Validated locally with 11 native activity tests, one- and two-environment policy playback,
selective snapshot restoration without an MPM step, and 150 control steps of RTX playback
with particle activity colors. The native region test covers moving inputs during graph
replay, overlapping regions, world isolation, and the optional hard cutoff. These checks
establish functionality; they do not establish a speedup over the original moving patch.


## Sand rendering

The moving-region task displays an isosurface by default in Newton GL and RTX.
The particle cloud is hidden. In **Sand visualization**, **Isosurface** toggles the
surface and **Awake + support particles** toggles the filtered debug particle glyphs.
The standard viewer particle checkbox controls that same filtered display.

The surface uses Newton's native `ParticleSurface` extractor. Moving regions are
reconstructed, while unchanged sleeping regions retain their existing mesh. Particles
near region boundaries contribute to reconstruction on both sides, so there is no
separate background sand mesh to stitch. Rigid retaining walls and platforms remain.

GPU kernels classify 0.5 m XY tiles using native activity and particle movement. Newly
sleeping tiles are refreshed once; waking tiles leave the cached mesh immediately.
Particles outside the initial display tile map remain live rather than disappearing.
Reset refreshes the surface from restored particle state without recapturing physics.

Particle gathering, classification, extraction, and mesh compaction run on the GPU.
Each display update reads five control integers to obtain exact mesh sizes and decide
whether to submit the cached mesh again. Particle positions and mesh geometry stay on
the GPU. This small synchronization avoids uploading large unused mesh buffers; this
is not a zero-CPU or zero-readback renderer.

Settings are in `env_cfg/surface_cfg.py`. Surface spacing is 0.025 m, independent of the
physics voxel size. Fixed allocation capacities are checked on the GPU rather than
silently truncating geometry. The viewer receives only the populated buffer ranges.
Optional awake/support particle glyphs remain expensive for large particle counts.

The environment also defers guarded MPM status checks to the control-step boundary,
instead of synchronizing after every physics substep. Faults are checked before a
transition is returned; reset paths still validate status before clearing it. Solver
parameters and the physical integration are unchanged.

### Performance measurement

Earlier timings may have been affected by another GPU process. A subsequent sequential
comparison found no competing compute job (desktop and OBS remained active). Both runs
used one Dune environment, `5f1xyc21/model_39998.pt`, RTX headless capture at 800×600,
195 warmup steps, and 30 measured steps with explicit GPU synchronization.

| Mean time | Previous whole-bed GPU surface | Incremental cached surface |
|---|---:|---:|
| Simulation/policy control step | 69.4 ms | 65.1 ms |
| Surface update and RGB capture | 30.9 ms | 21.1 ms |
| Combined | 100.3 ms | 86.2 ms |

This single sequential pair measured 14% lower combined latency. It includes the
control-step status-check optimization; it is not an isolated renderer-only ablation.
It is not a comparison against the main checkout or proof that surfaces are faster than
particle rendering. Most bed particles were still moving/active at this early interval;
longer trajectories and other workloads can have different cache reuse.

### Validation

- GPU regression: material translation and reset change the visible surface through graph
  replay; a warm update rejects array readback; an empty field clears stale triangles.
- Live/cached geometry matches full extraction through deformation, sleep/wake transitions,
  reset, and movement outside the initial display tile map.
- GL/RTX policy smoke checks: surface-only, particles-only, both, hidden, and reset captures.
- The debug glyph count matched all 366,156 awake/support particles at the sampled step.
- Sand reset preserved robot joint positions and velocities exactly.

Run the focused regression with:

```bash
uv run --no-sync --with pytest python -m pytest source/isaaclab_tasks/test/contrib/test_mpm_surface.py -q
```
