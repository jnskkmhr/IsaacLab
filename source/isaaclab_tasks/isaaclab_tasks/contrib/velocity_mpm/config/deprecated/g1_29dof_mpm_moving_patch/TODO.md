Thing to figure out

## Motion difference

### 1. Exessive arm swing
P1: https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_mpm_mixed_terrain/runs/876gqr97
vs
P2: https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_mpm_mixed_terrain/runs/gzi8enaw


| Reward setting | 876gqr97 | gzi8enaw |
|---|---:|---:|
| Roll/pitch angular-velocity penalty (ang_vel_xy_l2) | −0.05 | −0.2 |
| Standing contact-break penalty (break_contact) | −0.5 | Removed |
| Standing joint-deviation coefficient: hip pitch, knee, ankle pitch/roll | 0.02 | 0.5 |
| Standing joint-deviation coefficient: hip roll | 0.3 | 0.5 |
| Standing joint-deviation coefficient: hip yaw | 0.08 | 0.5 |

**Being sensitive to ang vel xy penalty?**
* P1 produces constant stepping during stance -> due to relaxed joint_deviation
* P2 produces higher arm swing -> could be due to higher ang_vel_xy penalty (swing arm minimizes centroidal angular momentum ?)

### 2. Weaker hip yaw motion and lower foot clearance during turning
P0: https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_mpm_mixed_terrain/runs/5f1xyc21 (finetuned policy)
P1: https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_mpm_mixed_terrain/runs/876gqr97
vs
P2: https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_mpm_mixed_terrain/runs/gzi8enaw

**Being sensitive to ang vel xy penalty?**
* Exessive torso ang vel may prohibit hip yaw motion as it produces bigger torso oscillation


### 3. Heel liftup
P1: https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_mpm_mixed_terrain/runs/gzi8enaw
vs
P2: https://forge.coreweave.com/wandb/jnskkmr/g1_29dof_mpm_mixed_terrain/runs/7cawp916

**Being sensitive to torso height target?**
* Higher torso target may shifts COM to forward
