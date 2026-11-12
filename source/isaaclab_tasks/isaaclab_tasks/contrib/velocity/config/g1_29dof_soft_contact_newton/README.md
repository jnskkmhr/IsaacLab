# G1 flat RFT with Newton

Standalone Newton counterpart of the PhysX run
[hiwy2c7j](https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_contact_benchmark/runs/hiwy2c7j).
Task: `IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Newton`.

The task owns its scene, action, observation, reward, event, curriculum,
termination and PPO configs. Reusable MDP functions and the analytical RFT
implementation are shared. There are no MPM particles.

| Setting | Reference and Newton task |
|---|---|
| Physics/control period | 0.005 / 0.020 s |
| Robot | Original G1 29-DoF USD and reference gains/armature |
| Terrain | Equal rigid and RFT columns; rigid support under RFT progresses from z=0 to -0.12 m |
| RFT | 3D Warp model, no EMA, 40 N contact threshold |
| Rigid contact sensor | 0.005 s updates, 1 N threshold, history 3 |
| Observations | Policy 1000, critic 1010 + privileged 160; 10-frame histories including commands |
| Quaternion observations | WXYZ, preserving the reference input layout |
| PPO | Vanilla PPO, MLP [512,256,128], 24 steps, 5 epochs, 4 minibatches; no mirror loss |
| Training population | 4 GPUs × 1024 environments = 4096 |
| Camera | eye=(4,-4,3), lookat=(0,0,0), focal length 12, tracks robot; all environments visible |

Newton uses MJWarp for rigid dynamics and analytical RFT wrenches on the feet.
The simulation time step matches the reference; Newton uses two rigid-solver
substeps and capacity for 300 contacts / 1000 constraints per robot. PhysX
contact response and material-combination rules cannot be reproduced exactly by
changing task parameters. No equality of trained gait or trajectories is assumed.

```bash
uv run --with wandb isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Newton \
  --num_envs 1024 --seed 42 --logger wandb \
  --log_project_name g1_29dof_contact_benchmark \
  --run_name newton_soft_flat_hiwy2c7j --viz newton_gl --video
```

See `osmo/osmo_contact_benchmark.md` for the four-GPU command. Training starts
from scratch, as did the reference run. No checkpoint is loaded.

## Current-command symmetry variant

Task: `IsaacContrib-Velocity-Flat-G1-29dof-SoftContact-Newton-Symmetry`.
This ablation keeps the reference physics, terrain, rewards, events, curricula,
noise settings, camera and PPO hyperparameters. Only these settings change:

- Remove velocity/heading commands from policy and critic histories. Proprioception
  retains ten frames (policy 930 values, critic 940 values).
- Add a separate `command` group with the current seven values and no history.
  Actor inputs are `policy + command` (937); critic inputs are
  `critic + command + privileged` (1107).
- Enable left/right data augmentation and mirror loss with coefficient 0.1.
  Quaternion observations retain the reference WXYZ order and use matching mirror
  rules. Feet are swapped and vector/joint signs are reflected in their declared order.

Start this variant from scratch; the reference checkpoint has different input sizes.
The baseline task ID continues to use command histories and no symmetry.

Validation: three local PPO iterations with 32 environments completed with finite
symmetry loss and a 50-frame GL video. Config comparison confirmed all non-observation
MDP settings are unchanged; quaternion reflection was checked against `S @ R @ S`.


## Rough PPO baseline

The rough task and its rigid comparison are documented in
[the rough baseline setup](../g1_29dof_rigid_newton/README.md).
The flat tasks retain the `binkrmei` recipe.
