# G1 locomotion on a fixed sand bed

This manager-based task starts a G1 robot on a rigid approach platform next to a fixed MPM sand bed. MJWarp simulates the robot and rigid colliders. Implicit MPM simulates the sand, and proxy coupling transfers sand reaction forces to the robot.

The task uses the contributed G1 box-foot asset. Observations and rewards include robot foot contact forces from the rigid platform and the MPM sand. Reset events sample particle material parameters with uniform or log-uniform distributions. Symmetry augmentation uses the shared `isaaclab_contrib.mdp.symmetry` API and task-local G1 joint mappings.

```bash
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Sand-G1-29dof-MPM --num_envs 2
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Sand-G1-29dof-MPM-Play --num_envs 1 --wandb_run <run-id> --viz newton_rtx
```

The environment configuration is `g1_mpm_env_cfg.py`. Scene geometry and particle material defaults are in `env_cfg/scene_cfg.py`; material sampling ranges are in `env_cfg/event_cfg.py`. The robot's initial position is on the rigid approach platform.

Coupled material updates require Newton's particle-material synchronization fix. Multi-environment MPM requires the Warp multi-environment cell-lookup fix. Until compatible upstream dependency pins are available, use the dependency versions described in the draft PR validation notes.
