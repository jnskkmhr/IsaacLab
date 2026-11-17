# Flat RFT: tyiby5k6 reward and randomization recipe

Standalone variant of `g1_29dof_soft_contact_improvement`, using the saved
[tyiby5k6 run configuration](https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_soft_flat_ppo/runs/tyiby5k6)
for rewards, curricula, randomization events and symmetry augmentation.

Task IDs:

- `IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Improvement-Tyiby5k6`
- `IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Improvement-Tyiby5k6-Play`

## Changes from the improvement task

| Setting | Improvement | This variant |
|---|---|---|
| Air-time reward weight | 0.5 | 2.0 |
| No-flight penalty weight / velocity threshold | -1 / 1 m/s | -2 / 1.5 m/s |
| Hip-yaw posture weights, walking and running | 0.15 | 0.08 |
| Foot pitch penalties | Stance-only contact pitch, weight -4 | MPM-style stance-only squared pitch, weight -8, ankle-roll links |
| Velocity and heading reward standard deviations | Step change to 0.25 at 480,000 steps | Linear 1 to sqrt(0.1), steps 0–360,000 |
| Linear velocity / heading reward weights | Constant 4 | Linear 4 to 8, steps 0–360,000 |
| RFT stiffness randomization | [0.1, 0.9] | [0.2, 0.9] |
| Symmetry data augmentation | Disabled | Enabled |
| Mirror loss | Disabled | Disabled, coefficient 0 (as saved in tyiby5k6) |

All rewards except foot pitch, seven curriculum terms and nine original randomization
events retain the saved run's parameters. Foot pitch follows the moving-patch MPM
task: weight -8, zero tolerance, measured on ankle-roll links throughout contact.
The flat surface uses a vertical normal; swing feet contribute zero. The three
original pitch terms are removed. Action-rate penalties select policy action
channels, preserving the saved joint ordering.

## Preserved improvement settings

Both tasks now use the standard G1 asset with four sphere colliders per foot.
PD gains, zero foot collision margins on both terrain types,
flat rigid/RFT terrain mixture, floor-depth curriculum geometry, fixed 3D RFT
implementation, contact thresholds, solver, viewer and PPO hyperparameters are
unchanged. The additional startup event that sets zero margins is retained.

Observations are deliberately unchanged: the command remains in the ten-step
history, with WXYZ quaternion observations and 1,000 actor inputs. This differs
from tyiby5k6's standalone command observation and 937 actor inputs. Reflection
metadata handles the improvement task's actual layout; it does not change values,
noise, ordering or history. This is a fresh-training ablation, not a checkpoint
resume or a complete reproduction of the old run.

## Launch

```bash
bash osmo/run_multinode.sh osmo/osmo_multi_gpu.yaml \
  --pool isaac-lab-l40-07 --set num_nodes=1 num_gpu=4 \
  --set-string platform=ovx-l40 \
    workflow_name=isaac-lab-g1-rft-tyiby5k6 \
    output_url=swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/jkamohara/contact-benchmark/runs \
    wandb_credential=wb-auth wandb_key=wb_api_key \
    args="--task IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Improvement-Tyiby5k6 --num_envs 1024 --logger wandb --wandb_project g1_29dof_contact_benchmark --run_name newton_soft_flat_tyiby5k6_recipe --viz newton_gl --video"
```

Four GPUs, 1,024 environments per GPU, 20,000 PPO iterations, W&B logging and video.
