# Flat RFT with standard G1 feet and zero contact margins

Standalone manager-based Newton/MJWarp task based on the saved training configuration of
[`kyfj909z`](https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_contact_benchmark/runs/kyfj909z).

| Setting | This task |
|---|---|
| Robot asset | `UNITREE_G1_29DOF_CFG`: standard G1 with four sphere colliders per foot |
| Robot gains and initial joint pose | Original `kyfj909z` values, including original waist gains |
| Terrain | Flat tiles; 50% rigid columns, 50% RFT columns |
| Rigid-column foot collider margin | 0 m |
| RFT-column foot collider margin | 0 m |
| Hard support under RFT | Original depth curriculum from 0 to −0.12 m |
| Analytical RFT surface | World Z = 0 m; original force model and intrusion depth |
| Actor observations | Original term-major history of 10 samples, including command history; 1,000 values |
| Actions | 29 joint-position actions, scale 0.25 |
| PPO | Original `[512, 256, 128]` ELU actor/critic; no symmetry augmentation or mirror loss |

The margin event runs at startup and selects **foot shapes only** using each environment's
assigned terrain column. Terrain curriculum changes the row/depth, not that column.
Small environment counts may not divide exactly 50/50 under the original terrain importer.
The event changes Newton collision geometry against the hard floor; it does **not** add a
2 cm layer to the analytical RFT surface. Collider gaps remain at the original solver setting.

Foot pitch is penalized only during stance; constant pitch and pitch-difference penalties
are removed. The corrected 3D RFT force and wrench implementation is used.
Other reward, observation, event, command, termination, curriculum, solver and PPO configurations
are preserved. In particular, this task retains the original tracking-std curriculum
schedule rather than changing it as part of the asset/margin experiment. Current installed
library versions are used; this is not a reconstruction of the run's historical dependencies.

From `/home/jkamohara/isaac/IsaacLab`:

```bash
uv run --no-sync --with wandb isaaclab train \
  --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Improvement \
  --num_envs 1024 \
  --logger wandb --wandb_project g1_29dof_contact_benchmark \
  --run_name newton_soft_flat_rft_fixed_standard_g1 \
  --viz newton_gl --video
```

The `-Play` task retains the same terrain and observation definitions. No existing tasks are changed.

Validation: four environments reset and stepped for 100 policy steps; actual Newton shape and
MJWarp geometry margins were inspected for both terrain types. Observations and rewards remained
finite, and margins persisted through reset. A 16-environment, two-iteration PPO smoke run passed.
These checks establish configuration and runtime validity, not improved locomotion performance.

## USD foot-collider and pitch-reward ablations

The following training variants start from `w69gnak5` and use
`g1_29dof_rev_1_0.usd` (not `.usda`). Append the suffix to
`IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Improvement`:

| Suffix | Enabled ankle-roll colliders per foot | Pitch rewards |
|---|---|---|
| `-FourSphere-AlwaysPitch` | Four corner spheres; box and toe sphere disabled | `feet_pitch` and `feet_pitch_diff`, each −4; stance-only term disabled |
| `-FourSphere-StancePitch` | Four corner spheres; box and toe sphere disabled | Reference stance-only `feet_pitch_contact`, −4 |
| `-Box-AlwaysPitch` | Existing USD box; all five spheres disabled | `feet_pitch` and `feet_pitch_diff`, each −4; stance-only term disabled |

All three omit `foot_contact_margin`. The shared USD file is unchanged; the
spawn configuration makes the geometry editable and selects its enabled colliders.
The four spheres have 5 mm radius. The existing box is 0.15 × 0.06 × 0.015 m,
centered at (0.035, 0, −0.016) m in the ankle-roll frame. Its dimensions and position
are preserved rather than adjusted to the spheres' contact height.

These selections affect rigid collision geometry. The analytical RFT integration
box in `env_cfg/action_cfg.py` remains unchanged, as do observations, other rewards,
randomization, curricula, and PPO settings. The baseline task remains unchanged.
