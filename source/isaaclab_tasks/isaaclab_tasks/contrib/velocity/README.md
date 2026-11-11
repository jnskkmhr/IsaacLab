# Locomotion experiments on Lab 3.0

Branch: `ral_gm_loco_lab_3.0` in `jnskkmhr/IsaacLab`.

Original upstream Isaac Lab contributed environments are preserved. Only the custom G1 MPM, moving-patch/particle-cache, MPM pushing, and mimic/SuperMimic task families are removed. The custom locomotion experiments remain under `velocity`; core tasks and backend packages remain available.

| Experiment | Configuration | Backend |
|---|---|---|
| PhysX reference flat RFT | [g1_29dof_soft_contact_physx](config/g1_29dof_soft_contact_physx/README.md) | Isaac Sim PhysX + analytical RFT |
| Newton flat / rough RFT | `config/g1_29dof_soft_contact` | Newton MJWarp + analytical RFT |
| Rigid flat / rough | `config/g1_29dof_rigid` | Newton MJWarp |

The PhysX recipe matches the task and PPO settings from `4blqnxbi`; the Lab/Isaac Sim/software stack is newer and the 4096-environment batch is distributed across four GPUs. See the linked task README for installation, Kit USD library setup, video recording and OSMO launch commands.

Shared analytical RFT code lives in `contrib/soft_contact`. Terrain reset/query helpers and shared agent configurations also live under `velocity`, so retained task imports do not require the removed MPM or mimic task directories. Tests belonging to removed custom task families remain as historical source; the original upstream task tests retain their environments.

No checkpoints, training logs, benchmark outputs or credentials are included.

Validation: the PhysX recipe audit matches 825 reference settings; G1 rigid, soft-contact, PhysX and perception configuration/agent imports pass. The retained `g1_29dof_tree` variant has a pre-existing configuration error: its disabled contact sensor conflicts with the rigid parent config. It is not one of the benchmark training tasks.
