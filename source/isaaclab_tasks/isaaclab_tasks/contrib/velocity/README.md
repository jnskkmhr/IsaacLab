# Locomotion experiments on Lab 3.0

Branch: `ral_gm_loco_lab_3.0` in `jnskkmhr/IsaacLab`.

Only `velocity` is retained under `isaaclab_tasks/contrib`. Other contributed task families (MPM, particle caching, mimic, pushing, manipulation, etc.) are removed from this branch. Core tasks and backend packages remain available.

| Experiment | Configuration | Backend |
|---|---|---|
| PhysX reference flat RFT | [g1_29dof_soft_contact_physx](config/g1_29dof_soft_contact_physx/README.md) | Isaac Sim PhysX + analytical RFT |
| Newton flat / rough RFT | `config/g1_29dof_soft_contact` | Newton MJWarp + analytical RFT |
| Rigid flat / rough | `config/g1_29dof_rigid` | Newton MJWarp |

The PhysX recipe matches the task and PPO settings from `4blqnxbi`; the Lab/Isaac Sim/software stack is newer and the 4096-environment batch is distributed across four GPUs. See the linked task README for installation, Kit USD library setup, video recording and OSMO launch commands.

Shared analytical RFT code now lives in `velocity/soft_contact`. Terrain reset/query helpers and shared agent configurations also live under `velocity`, so retained task imports do not require the removed MPM or mimic task directories. Existing tests for removed task families are retained as historical source and are not runnable in this reduced branch.

No checkpoints, training logs, benchmark outputs or credentials are included.

Validation: the PhysX recipe audit matches 825 reference settings; G1 rigid, soft-contact, PhysX and perception configuration/agent imports pass. The retained `g1_29dof_tree` variant has a pre-existing configuration error: its disabled contact sensor conflicts with the rigid parent config. It is not one of the benchmark training tasks.
