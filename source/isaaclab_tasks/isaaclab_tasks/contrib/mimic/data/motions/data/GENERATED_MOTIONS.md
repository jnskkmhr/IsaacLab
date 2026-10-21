# Generated G1 motions with terrain

19 current non-Kimodo clips: 5 PARC jumps, 4 Deep Whole-Body Parkour clips,
7 OmniRetarget climbs, and 3 CLAW crouch/crawl clips. PARC uses the corrected
`parc_anatomical_arms_v2` exports. The duplicate ball-hand crawl is included once,
as `claw/00543_hand_crawl/motion.npz`. No Kimodo data or archived variants are included.

Each `motion.npz` contains the motion and its terrain. Box metadata uses
`terrain_position`, `terrain_dimension` (full sizes in meters), and `terrain_quat`
(xyzw). Mesh metadata uses `terrain_vertices` and `terrain_faces`. Existing ground
heights are retained; CLAW and OmniRetarget explicitly use the original viewer's
Z=0 ground plane. No joint/body states or velocities were changed or resampled.

All clips are **50 Hz**, with 29 joints and 35 named bodies, exported for
`UNITREE_G1_29DOF_BALL_HAND_CFG`. This import preserves that asset metadata; it does
not change the robot configured by the task/player. For RL with decimation 4,
use `env.sim.dt=0.005`. Reference playback reads the file's FPS automatically.

From the repository root:

```bash
uv run python source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/data/motions/play_motion_file.py \
  --motion_file source/isaaclab_tasks/isaaclab_tasks/contrib/mimic/data/motions/data/parc/jump_005/motion.npz \
  --terrain_mode motion
```

In `commands_cfg.py`, choose clips with paths such as:

```python
motion_files=[
    str(Path(MOTION_TRACKING_DATA_DIR) / "motions/data/parc/jump_005/motion.npz"),
    str(Path(MOTION_TRACKING_DATA_DIR) / "motions/data/deep_wholebody_parkour/roll_vault/motion.npz"),
],
terrain_mode="auto",
```

`generated_motion_manifest.json` records source paths, checksums, metadata changes,
and loader validation. Each clip also has `provenance.json` and its original
`source_validation.json` when available. Source reports can describe an earlier
retargeting stage at a different FPS; the installed NPZ and manifest specify the
runtime FPS. Loader validation does not establish successful policy tracking;
existing contact/retargeting limitations remain in the source validation reports.

| Motion | Frames | First-to-last frame duration [s] |
|---|---:|---:|
| [claw/00452_crouch/motion.npz](claw/00452_crouch/motion.npz) | 281 | 5.60 |
| [claw/00543_hand_crawl/motion.npz](claw/00543_hand_crawl/motion.npz) | 312 | 6.22 |
| [claw/02067_elbow_crawl/motion.npz](claw/02067_elbow_crawl/motion.npz) | 483 | 9.64 |
| [omniretarget/climb_06/motion.npz](omniretarget/climb_06/motion.npz) | 602 | 12.02 |
| [omniretarget/climb_08/motion.npz](omniretarget/climb_08/motion.npz) | 572 | 11.42 |
| [omniretarget/climb_14/motion.npz](omniretarget/climb_14/motion.npz) | 454 | 9.06 |
| [omniretarget/climb_16/motion.npz](omniretarget/climb_16/motion.npz) | 284 | 5.66 |
| [omniretarget/climb_20/motion.npz](omniretarget/climb_20/motion.npz) | 604 | 12.06 |
| [omniretarget/climb_26/motion.npz](omniretarget/climb_26/motion.npz) | 316 | 6.30 |
| [omniretarget/climb_28/motion.npz](omniretarget/climb_28/motion.npz) | 367 | 7.32 |
| [deep_wholebody_parkour/dive_roll/motion.npz](deep_wholebody_parkour/dive_roll/motion.npz) | 417 | 8.32 |
| [deep_wholebody_parkour/jump_sit/motion.npz](deep_wholebody_parkour/jump_sit/motion.npz) | 537 | 10.72 |
| [deep_wholebody_parkour/kneel_climb/motion.npz](deep_wholebody_parkour/kneel_climb/motion.npz) | 554 | 11.06 |
| [deep_wholebody_parkour/roll_vault/motion.npz](deep_wholebody_parkour/roll_vault/motion.npz) | 523 | 10.44 |
| [parc/jump_001/motion.npz](parc/jump_001/motion.npz) | 182 | 3.62 |
| [parc/jump_002/motion.npz](parc/jump_002/motion.npz) | 517 | 10.32 |
| [parc/jump_003/motion.npz](parc/jump_003/motion.npz) | 224 | 4.46 |
| [parc/jump_004/motion.npz](parc/jump_004/motion.npz) | 241 | 4.80 |
| [parc/jump_005/motion.npz](parc/jump_005/motion.npz) | 326 | 6.50 |
