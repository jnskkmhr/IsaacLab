# Flat RFT locomotion aligned with moving-patch MPM

Task: `IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-MPMAlign`

This standalone manager-based task uses Newton MJWarp rigid dynamics and the
analytical `3D-warp` RFT callback. It creates **no MPM particles**. The robot is
`UNITREE_G1_29DOF_BOX_FOOT_CFG`, including the USDA pelvis-contour/hip-pitch
collision exclusions. Rigid foot and ground collider margins are zero.

## Alignment and deliberate differences

Reference: `contrib/velocity_mpm/g1_29dof_mpm_moving_patch`, as inspected on 2026-11-14.

| Component | Configuration |
| --- | --- |
| Actor / critic proprioception | Same terms, joint order, scales, noise settings, mirrors, and 10-step histories |
| Commands | Same seven-component command in its own observation group, without history |
| Privileged observations | Same 160-dimensional history layout; rigid/RFT contact sources replace coupled MPM contact buffers; material placeholder remains zero |
| Rewards | Same 24 active terms and weights, including foot distance, foot sliding, and stance-only pitch |
| Height reference | World `z=0`; base target 0.75 m with weight −10; foot clearance target 0.10 m plus the 0.03539 m ankle-origin offset; no scanner or sand-depth correction |
| Robot events | Same friction, added torso mass, joint reset, gain randomization, pushes, and root velocity ranges; root reset ignores buried support height |
| Material events | Existing flat RFT friction, internal-friction/stiffness, density, and packing sampling replace MPM particle-material randomization |
| Curriculum | Same command, ±3° gravity bias, tracking standard deviation, and linear tracking weight ramps |
| Terrain | Existing flat 50/50 rigid/RFT column layout; progression lowers soft-column rigid support from 0 to −0.12 m; analytical RFT surface stays at world `z=0` |
| Contact state | RFT/rigid authoritative-solver selection replaces coupled MPM buffers; contact detection and force-observation thresholds are 5 N |
| PPO | Reference MLP actor/critic, 24-step rollout, 20,000 iterations, and data augmentation; mirror loss explicitly enabled at 0.1 per the requested training setup |

Actor input is 937 values (`policy=930`, `command=7`); critic input is 1,107
(`critic=940`, `command=7`, `privileged=160`). There are 29 joint actions; the RFT
callback has zero policy-action dimensions and an identity mirror rule.

Contact-dependent rewards retain the reference formulas. The local air-time
adapter preserves separate linear/angular command gates, and sliding/no-flight
adapters use force-norm contact detection. Reusable RFT landing and stance-pitch
terms provide the corresponding contact timing and forces.

The standard MPM PPO config currently enables augmentation without mirror loss;
the explicit 0.1 mirror loss is an additional setting, not an exact PPO-config
copy. Newton solver settings and RFT material parameters follow the flat RFT
task, rather than the MPM coupler.

## Train

From `/home/jkamohara/isaac/IsaacLab`:

```bash
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40-07 \
  --set num_nodes=1 num_gpu=4 num_cpu=32 memory=192 \
  --set-string \
    workflow_name=isaac-lab-g1-rft-mpm-align \
    platform=ovx-l40 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/contact-benchmark/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-MPMAlign --num_envs 1024 --seed 42 --logger wandb --wandb_project g1_29dof_contact_benchmark --run_name rft_flat_mpm_align_box --viz newton_gl --video"
```

Training starts from scratch. The launcher snapshots the checkout and uploads
assets by content hash, including the updated USDA collision filters.

## Validation

Local checks exercised two PPO updates with symmetry, reset and step execution,
finite rewards/observations, both rigid and RFT forces, collision filters in all
16 clones, zero foot margins, and the world-height reward formula. Configuration
comparison verified actor/critic/command parity, all reward weights, robot event
ranges, and the shared curriculum parameters against the reference.

Generated validation records live under `logs/rft_mpm_align_validation/` and may
not be available in a fresh checkout. Smoke tests validate execution, not gait
quality after training.

## Rough RFT variant

Task: `IsaacContrib-Velocity-Rough-G1-29dof-SoftContact-MPMAlign` (also `-Play`).
`rough_env_cfg.py` composes the rough terms defined in `env_cfg/`. Flat task
configuration remains unchanged, including the recipe used by run `u4za5id6`.

- The generator has 10 rows and 10 columns of 8 × 8 m tiles: waves,
  random roughness, positive/inverted pyramid slopes, and flat terrain, each 20%.
  Shape ranges match moving-patch MPM. Both aligned rough tasks use the same
  generator configuration and seed; random rough tiles are not asserted to be
  exact duplicates of adjacent tiles.
- Even columns keep rigid support at the visible rough surface. Odd columns
  place rigid support below that surface, increasing from 0 to 0.12 m across rows.
  Terrain progression therefore increases both roughness and RFT support depth.
- The RFT callback queries the rough surface mesh. The surface itself has no
  rigid collider; the separate support mesh supplies rigid contact.
- Root reset samples the visible surface at the sampled XY. The 0.75 m base
  target and per-foot clearance use surface raycasts, excluding buried support.
  Stance pitch uses the plane fitted to the pelvis scanner's hits.
- The actor/critic/command layout, all 24 reward weights, robot randomization,
  command/reward/bias ramps, PPO, augmentation and 0.1 mirror loss stay aligned
  with the flat task. Only the privileged foot-height measurement changes to
  height above the local rough surface. No height map is added to the actor.

Ready-to-submit rough command (four GPUs, 1,024 environments per GPU):

```bash
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40-07 \
  --set num_nodes=1 num_gpu=4 num_cpu=32 memory=192 \
  --set-string \
    workflow_name=isaac-lab-g1-rft-rough-mpm-align \
    platform=ovx-l40 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/contact-benchmark/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Rough-G1-29dof-SoftContact-MPMAlign --num_envs 1024 --seed 42 --logger wandb --wandb_project g1_29dof_contact_benchmark --run_name rft_rough_mpm_align_box --viz newton_gl --video"
```

The rough task passed a 20-environment, 120-step reset/contact/height check and a
two-update PPO smoke test. These checks establish execution, not trained gait quality.
