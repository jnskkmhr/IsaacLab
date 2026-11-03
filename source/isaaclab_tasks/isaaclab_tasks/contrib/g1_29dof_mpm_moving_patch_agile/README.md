# Agile G1 on moving-patch and mixed terrain

Both variants use the reference `g1_29dof_mpm_moving_patch` runtime for
MJWarp–MPM proxy coupling, particle allocation/recycling, material randomization,
and terrain generation. The mixed variant also reuses its rigid/MPM world
assignment, reset handling, and per-world foot-margin assignment.

| Variant | Training task ID | Ground assignment |
| --- | --- | --- |
| Moving patch | `IsaacContrib-Velocity-Sand-G1-29dof-MPM-MovingPatch-Agile` | MPM in every environment |
| Mixed moving patch | `IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-Agile-MixedTerrain` | Half MPM, half rigid |

Append `-Play` for playback. Mixed terrain requires an even environment count of
at least two. Both tasks default to the compositional FlashSAC agent; its explicit
selector is `--agent rsl_rl_flash_sac_cfg_entry_point`.

## Configuration

The full action, command, curriculum, event, observation, reward, termination,
scene, terrain, and physics declarations live in this task's `env_cfg/` directory.
The base environment derives directly from `ManagerBasedRLEnvCfg`; the mixed and
play configurations extend this task's own base configuration. Shared runtime
functions are imported from the reference tasks instead of duplicating solver code.

The robot is the rigid Agile task's `G1_29DOF_BeyondMimic_delay`: convex-hull foot
colliders and delayed implicit actuators. The policy controls 15 joints, including
waist yaw; upper-body joints retain randomized targets. Observation groups are
`policy` (270), `critic` (85), and `height_scan` (9, history length 1). The height
scan feeds only the critic. Learner settings match the rigid Agile FlashSAC config,
including symmetry augmentation and replay reward relabeling.

All 26 current rigid Agile reward terms and weights are retained. The additional
`stance_foot_angle_penalty` uses the reference MPM reward with weight **-8.0**,
ankle-roll bodies, `height_scanner`, and zero angle tolerance.

Foot collider margins are **0.016 m in rigid environments** and
**0.5 × voxel_size in MPM environments** (0.020 m at the default 0.04 m voxel size).
The latter is recomputed after configuration overrides. Other shapes retain the
reference moving-patch solver's zero default margin. The robot-wide zero-margin
override is disabled so it cannot erase these per-world foot settings.

The coupled solver does not expose ordinary contact sensors. The task records
MJWarp contact forces plus MPM feedback in articulation body order at each 0.005 s
physics step. Agile's slip, jumping, and loading rewards use this history. The
loading reward retains five samples, including the sample before the current
0.02 s control interval. Partial resets clear only the selected environments'
force history and particle state. Only the ankle bodies are proxied into MPM;
torso/pelvis forces therefore come from rigid contacts, with a height-based fall
check retained for the particle bed.

Ray casts intersect the rigid support mesh beneath the particle bed. The task
adds particle depth only in MPM environments to recover the **initial sand
surface** for height observations and base-height checks. These measurements do
not track deformation of the sand surface. The stance-foot-angle reward retains
the reference MPM implementation. Workspace and nonfinite-state terminations are
also retained from the moving-patch task.

Defaults: 64 environments, 200 Hz physics, 50 Hz control, 30-second episodes,
and 100,000 FlashSAC training iterations. Delayed actuators run outside the
captured solver graph so their command delays advance once per physics step.

## Train from scratch

```bash
uv run --extra wandb --extra video isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-Agile-MixedTerrain \
  --agent rsl_rl_flash_sac_cfg_entry_point \
  --num_envs 64 --logger wandb --viz newton_gl --video --video_length 200
```

For MPM-only training, replace the task ID with the moving-patch ID above.
No checkpoint or resume argument is needed for a fresh run.

```bash
cd ~/isaac/IsaacLab
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-dev-rtx6000-01 \
  --set num_nodes=1 num_gpu=1 \
  --set-string \
    workflow_name=isaac_flash_sac_g1_mpm_agile \
    platform=x86-6000 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/mpm-moving-patch-agile/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-Agile-MixedTerrain --agent rsl_rl_flash_sac_cfg_entry_point --num_envs 64 --logger wandb --viz newton_gl --video --video_length 200"
```

The wrapper uploads the current working-tree snapshot, including new task files.
It does not require pushing a branch first.

## Validation

Two-environment CUDA checks passed for both terrain variants: finite observations
and rewards, 200 Hz force history, timeout final observations, and partial-reset
isolation. The mixed check also verified convex foot geometry, actual per-world
foot margins, and 16 FlashSAC iterations with symmetry augmentation. Solver and
MPM material configuration parity with the reference task was checked separately.
These are integration checks, not a learning-convergence evaluation.
