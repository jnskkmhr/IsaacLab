# G1 rigid locomotion on PhysX

Ports the standard PPO task from `ral26_gm_loco`, commit
`c9e988831ddd747153bc36c7a512b818498156ff`, directory
`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1_29dof`.
This task uses Isaac Sim PhysX and `ManagerBasedRLEnv`. It has no RFT forces or MPM particles.
All term configurations are owned by this directory; the reference custom MDP formulas are
ported locally so changes to the Newton benchmark do not change this baseline.

## Tasks

| Terrain | Training ID | Inference ID |
|---|---|---|
| Flat | `IsaacContrib-Velocity-Flat-G1-29dof-Rigid-PhysX` | `IsaacContrib-Velocity-Flat-G1-29dof-Rigid-PhysX-Play` |
| Rough | `IsaacContrib-Velocity-Rough-G1-29dof-Rigid-PhysX` | `IsaacContrib-Velocity-Rough-G1-29dof-Rigid-PhysX-Play` |

## Preserved settings

| Setting | Value |
|---|---|
| Timing | Physics 200 Hz, policy 50 Hz, 20-second training episodes; seed 42 |
| Robot controller | Original BeyondMimic gains, armatures, effort/velocity limits, initial pose; 29 joint-position actions scaled by 0.25 |
| Flat actor observations | 1000 values: 10 samples per term, including seven-component velocity/heading commands |
| Flat critic observations | 1010 proprioceptive/command values + 150 privileged values |
| Rough observations | Actor 2870; critic 2880 + 150 privileged values; 187 terrain rays per frame |
| Observation noise | Disabled by the reference history-group configuration; noise ranges remain defined |
| Rewards | All 26 original terms and weights, including foot separation -2, air time +0.5, sliding -0.25, touchdown pitch -4, base-height target 0.75 m |
| Height references | Original world-z base-height/foot-clearance formulas, including on rough terrain; no adaptive surface-height substitution |
| Commands | Original stages at 240000 and 360000 policy steps |
| Tracking curriculum | Standard deviation switches to 0.25 after 480000 steps; not a linear ramp |
| Randomization | Original material/mass/gain/reset/push settings; flat and rough retain their different joint-reset ranges |
| Rough terrain | Original 10 × 20 tiles, 8 × 8 m each: stairs, inverted stairs, boxes, random roughness, slopes, inverted slopes |
| PPO | Standard PPO, [512, 256, 128] ELU actor/critic; no normalization or symmetry loss; 24 rollout steps; 30000 flat / 20000 rough iterations |

Lab 3.0 adaptation uses explicit Torch views of simulation data, the renamed contact-force
buffers, and actuator effort telemetry. Simulation quaternions remain xyzw; network-facing
quaternions remain wxyz as in the original. Full-reset command sampling accepts Lab 3.0's
slice indices. The constant 0.1 m swing command supports equal sampling bounds.

## Requested asset change

The task uses the shared `UNITREE_G1_29DOF_BOX_FOOT_CFG`, also selected for the
soft-contact PhysX and teacher/student tasks. It loads
`g1_29dof_rev_1_0_box_foot_improved_collision/g1_29dof_rev_1_0_box_foot_improved_collision.usda`,
including the pelvis-contour / hip-pitch collision filtering fix.

This changes the collision asset from the original recipe; it is not merely a change
from binary USD to ASCII USD. Use the same asset for training and evaluation.

## Run

From `/home/jkamohara/isaac/experiment/ral-gm-loco-lab-3.0`, use the workspace's PhysX/Kit
USD environment setup (also documented by `g1_29dof_soft_contact_physx`):

```bash
export OMNI_KIT_ACCEPT_EULA=YES
kit_usd_root=$(uv run --no-sync python -c 'import pathlib,sys; print(next((pathlib.Path(sys.prefix)/"lib/python3.12/site-packages/isaacsim/extscache").glob("omni.usd.libs-*")))')
export PYTHONPATH="$kit_usd_root${PYTHONPATH:+:$PYTHONPATH}"
python_lib_dir=$(uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export LD_LIBRARY_PATH="$kit_usd_root/bin:$python_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv run --no-sync isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Flat-G1-29dof-Rigid-PhysX \
  --num_envs 4096 --logger wandb --video
```

Use `--viz kit` for an interactive viewer. Replace `Flat` with `Rough` for generated terrain.
No OSMO job is submitted by adding this task.

## Validation

The recipe audit and PPO checks below preceded the switch to the shared box-foot asset.
The box-foot asset separately passed four-environment reset/step checks.

- Compared 1309 reference leaf values in term configs, actuator/robot configs, and explicit
  PhysX defaults; all match. Function/class paths and the added history-order field are
  excluded; the intended asset/path changes are checked separately.
- Flat and rough each ran four PhysX environments for 110 steps with three explicit resets;
  all observations and rewards remained finite. Rough smoke used a smaller 2 × 6 tile grid.
- Standard PPO completed three flat training iterations with eight environments.
- Formatting and lint checks passed for the new source files.

These are migration and runtime checks, not a claim of reproduced learning curves or trained gait quality.
