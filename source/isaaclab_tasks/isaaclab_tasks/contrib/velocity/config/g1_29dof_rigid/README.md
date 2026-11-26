# Rigid locomotion aligned with the RFT/MPM training terms

Standalone manager-based Newton MJWarp tasks using `UNITREE_G1_29DOF_BOX_FOOT_CFG`
with the pelvis-contour/hip-pitch collision exclusions:

- `IsaacContrib-Velocity-Flat-G1-29dof-Rigid`
- `IsaacContrib-Velocity-Rough-G1-29dof-Rigid`
- Both also have a `-Play` registration.

Term configs live in `env_cfg/`; the top-level environment configs only compose
them and define simulation/viewer settings. Reusable rigid contact, terrain and
height-query implementations come from the existing rigid and shared velocity
MDP modules. Config classes do not inherit from another task package.

## Alignment

Reference: `g1_29dof_soft_contact` (flat run `u4za5id6`).

| Component | Rigid variant |
| --- | --- |
| Robot, gains, actions | Same filtered box-foot asset, 29 joint actions, scale 0.25 |
| Physics | Same MJWarp settings; zero foot/ground margins; no RFT callback or MPM particles |
| Observations | Same 930 policy, 940 critic, 7 command, 160 privileged layout; same histories/noise/mirrors |
| Foot contact | Built-in ContactSensor; 5 N force-norm threshold |
| Force observations | Same signed log compression, 5 N component threshold, ±1000 N clipping |
| Contact rewards | Same formulas/weights, using rigid sensor stance, air time, first contact and forces |
| All rewards | Same 24 terms and weights, including stance-only pitch, foot distance and sliding |
| Events | Same robot randomization, resets and pushes; remove RFT-only material events |
| Curriculum | Same command, gravity-bias, reward-std and weight ramps; terrain progression only on rough terrain |
| PPO | Same network and optimization, augmentation and mirror loss coefficient 0.1 |
| Video/viewer | Same camera setup and video schedule; no visible-environment restriction added |

Flat terrain is entirely rigid at world z=0; base target is 0.75 m. Rough terrain
uses the same 10 × 10, 8 × 8 m tile generator as aligned RFT rough: 20% each waves,
random roughness, positive/inverted pyramid slopes and flat ground. Every tile
has rigid collision at its surface. Terrain difficulty progresses across rows.
Rough root resets and foot/base height terms query the surface; stance pitch uses
the scanned terrain plane. The actor does not receive a terrain height map.

## Training

From `/home/jkamohara/isaac/IsaacLab`, four GPUs with W&B and video:

```bash
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40-07 \
  --set num_nodes=1 num_gpu=4 num_cpu=32 memory=192 \
  --set-string \
    workflow_name=isaac-lab-g1-rigid-rough-mpm-align \
    platform=ovx-l40 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/contact-benchmark/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Rough-G1-29dof-Rigid --num_envs 1024 --seed 42 --logger wandb --wandb_project g1_29dof_contact_benchmark --run_name rigid_rough_mpm_align_box --viz newton_gl --video"
```

For flat training replace `Rough` with `Flat` in the task ID and use a distinct
flat workflow/run name. Both start from scratch unless explicitly given a checkpoint.

## Validation

Both variants passed 120 steps with 20 environments, finite observations/rewards,
resets, contact forces, zero foot margins, and the appropriate base-height reference.
Both passed two PPO updates with symmetry enabled. Configuration checks compared
all reward weights, proprioceptive/command observations, rough generator and
robot settings against the aligned RFT task. Local validation artifacts are in
`logs/contact_mpm_align_rough_validation/`; smoke tests do not establish final gait quality.
