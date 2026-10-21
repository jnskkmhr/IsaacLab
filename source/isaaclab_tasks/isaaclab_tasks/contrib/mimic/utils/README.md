# Motion-paired terrain

Runtime code lives here, next to `mdp/`. Offline generation and retargeting stay
under `multi_task_locomotion`. The existing `mdp/motion_loader.py` reads motion
states; these utilities read the paired terrain and build the shared scene.

## NPZ fields

Motion states and terrain use the **same reference coordinate frame**, in meters.
The usual motion fields (`fps`, joint/body states and names) remain required.

| Field | Shape | Meaning |
|---|---|---|
| `terrain_position` | `[N, 3]` | Box center positions in the reference frame [m] |
| `terrain_dimension` | `[N, 3]` | Full box sizes along the box's local X/Y/Z axes [m] |
| `terrain_quat` | `[N, 4]` | Box orientations, **xyzw**; omitted means identity |
| `terrain_vertices` | `[V, 3]` | Optional triangle mesh vertices in the reference frame [m] |
| `terrain_faces` | `[F, 3]` | Triangle vertex indices |
| `ground_height` | scalar | Optional ground top height in the reference frame [m] |
| `terrain_id` | scalar string | Optional identifier; reusing it requires identical geometry |

Supply box positions and dimensions together, and mesh vertices and faces together.
Boxes and meshes can coexist. `terrain_quat` is always xyzw, independent of the
motion body's `quaternion_order` metadata. A terrain-only ground plane must have an
explicit `ground_height`.

For existing exports, the reader accepts `terrain_boxes[:, :3]` as positions,
`terrain_boxes[:, 3:]` as full dimensions, and `terrain_box_quat` as orientations,
with a migration warning. New files should use the three separate fields above.
Do not store both schemas in one file. Existing generated files are not rewritten
by runtime loading.

## Reference playback

The existing player automatically loads terrain from the NPZ:

```bash
uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/data/motions/play_motion_file.py \
  --motion_file /absolute/path/jump/motion.npz --video --headless
```

The default `--terrain_mode auto` loads paired terrain when metadata is present
and uses the default plane otherwise. `--terrain_mode motion` requires terrain;
`--terrain_mode scene` deliberately uses the plane instead. Video and interactive
playback use the same geometry and reference translation.

## Standard RL training and policy playback

In `config/g1_29dof/env_cfg/commands_cfg.py`, set the motion paths and terrain mode:

```python
motion = mimic_mdp.MotionCommandCfg(
    # Keep the existing robot, tracked-body, and other command settings.
    motion_files=["/absolute/path/jump/motion.npz", "/absolute/path/vault/motion.npz"],
    terrain_mode="auto",
)
```

Both rigid G1 task IDs use `MotionTrackingEnv`, which selects terrain immediately
before constructing the scene, **after CLI overrides**. No call to a helper is
needed in the training or playback script. The default mode is `auto`:

- `auto`: load NPZ terrain when all clips contain it; retain the scene plane when
  none do. Mixed collections raise an error: add `ground_height` to flat clips.
- `motion`: require terrain metadata in every clip.
- `scene`: keep the configured scene terrain, ignoring terrain in the NPZ files.

The same fields can be overridden from the command line. Quote the complete list
argument and each path inside it:

```bash
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Mimic-G1-29dof-v0 \
  "env.commands.motion.motion_files=['/absolute/path/jump/motion.npz']" \
  env.commands.motion.terrain_mode=auto

uv run isaaclab play --rl_library rsl_rl --task IsaacContrib-Mimic-G1-29dof-Play-v0 \
  --checkpoint /absolute/path/model.pt \
  "env.commands.motion.motion_files=['/absolute/path/jump/motion.npz']" \
  env.commands.motion.terrain_mode=auto
```

Use references generated for the selected robot asset, with named joints and bodies.
Motion FPS must match `1 / (cfg.sim.dt * cfg.decimation)`. The default G1 task is
60 Hz. For existing 50 Hz clips, append `env.sim.dt=0.005` with the default
decimation of 4. The standalone reference player reads FPS directly from the NPZ.
The runtime does not relabel or resample frames.

The environment disables reference XY/yaw randomization for paired terrain.
Other robot-state perturbations remain configured as before. The soft-terrain
variants keep their existing environment entry point and terrain configuration.

For custom Python environments that directly instantiate `ManagerBasedRLEnv`,
`configure_motion_paired_terrain(cfg, motion_files)` is still available. Call it after
configuration overrides and before creating the environment. Set
`cfg.scene.terrain.patch_margin` and `patch_spacing` afterward to customize the
space around each patch and between patches [m].

## Reset and geometry alignment

1. At scene construction, load each distinct terrain once and place its patch in a
   nonoverlapping grid. Bounds include the full motion and terrain geometry.
2. At reset, the motion command samples a clip and reference time.
3. The importer assigns that clip's patch translation to the environment origin.
4. Root reset, reference bodies, observations, and tracking targets use that same
   translation. Terrain geometry stays fixed throughout training.

Clips with identical geometry in the same reference frame share a patch. Robot
collision isolation still comes from the scene's environment filtering. This is
shared static geometry, not one terrain object per robot environment.

An optional ground slab is bounded to each patch and has its top at
`ground_height`. There is **no implicit plane at Z=0** that would fill jump gaps.
A terrain curriculum cannot move an environment to an unrelated patch. Changing
reference XY/yaw offsets independently of the terrain raises an error.

With `terrain_mode="auto"`, files without terrain metadata keep the existing flat-ground setup.
This integration does not change rewards, observations, or the motion tracker.
Contact-aware training still requires motion references and task rewards suitable
for the chosen robot and terrain.
