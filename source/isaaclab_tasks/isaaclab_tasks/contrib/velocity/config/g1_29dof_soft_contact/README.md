# G1 Newton RFT locomotion

The flat and rough task configurations are ported from
`~/isaac/experiment/ral-gm-loco-lab-3.0`, branch `ral_gm_loco_lab_3.0`.
The reference rough policy is W&B run `lpwxe5i0` in `g1_29dof_contact_benchmark`.
The source configuration port was compared against that worktree; this does not
claim identical trajectories across different Newton/MJWarp dependency revisions.

## Task layout

`flat_env_cfg.py` and `rough_env_cfg.py` assemble the built-in `ManagerBasedRLEnv`.
All term configurations live in `env_cfg/`. Rough terms inherit only from this
task's own flat terms. Reusable reward/observation helpers remain in `mdp/`.
No particle simulation or MPM environment class is used.

| Task | Actor input | PPO |
|---|---|---|
| `IsaacContrib-Velocity-Flat-G1-29dof-Soft` | Policy history including command history: 1000 values | Reference flat PPO |
| `IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Symmetry` | Policy history 930 + current command 7 | Augmentation and mirror loss 0.1 |
| `IsaacContrib-Velocity-Rough-G1-29dof-Soft` | Policy history 930 + current command 7 | Augmentation and mirror loss 0.1 |

Existing `-Play` IDs retain the corresponding training observation layout and
use one environment by default. Training uses 24 rollout steps, 20,000 PPO
iterations, actor/critic MLPs `[512, 256, 128]`, and no observation normalization.
Default runner configurations are in `agents/rsl_rl_ppo_cfg.py`.

## Terrain and measurements

The rough generator uses 10 rows and 10 columns of 8 m tiles, with equal shares
of waves, random roughness, uphill slopes, downhill slopes, and flat terrain.
The complete generator is in `env_cfg/terrain_cfg.py`. Terrain difficulty advances
with walking distance. Root resets sample the terrain height at the randomized XY.

Base-height reward targets 0.75 m above the queried terrain surface. Foot height
uses the sole point 0.03539 m below the ankle frame and the surface ray beneath each
foot. The three scanners query the specified surface mesh, rather than arbitrary
rigid colliders. Other reward weights, events, and command curricula follow the
reference Newton task. Cameras track the robot with eye `(4, -4, 3)` and look-at
`(0, 0, 0)`, using the same 1280 × 720 window as the reference.

The rough surface is non-colliding. Even columns have rigid support at the surface;
odd columns lower the rigid support from 0 to 0.12 m across the ten difficulty rows.
RFT penetration is measured against the upper material surface, independently of
that buried support. RFT contact timing uses a 40 N threshold and no EMA filtering;
the rigid contact sensor uses a 1 N timing threshold, as in the reference.

## Play the rough policy

Run from `/home/jkamohara/isaac/IsaacLab`:

```bash
uv run --with wandb isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Rough-G1-29dof-Soft-Play \
  --num_envs 1 --viz newton_rtx \
  --wandb_run lpwxe5i0 --wandb_project g1_29dof_contact_benchmark \
  "env.video_recorders=[]"
```

The RSS benchmark entry points are under
`~/isaac/IsaacLab/examples/RSS/locomotion`. Evaluation must use one
explicit IsaacLab checkout for all policies and record that checkout's revision.
