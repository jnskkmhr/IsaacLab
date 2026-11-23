* Replaced ``isaaclab_tasks.contrib.particle_nerd_local`` with a self-contained
  :mod:`isaaclab_tasks.contrib.particle_nerd` package (``network/``, ``modules/``, ``nerd_solver.py``,
  ``nerd_solver_cfg.py``, ``nerd_wrench_solver.py``, ``nerd_manager.py``, ``nerd_kernels.py``) and renamed its
  classes to ``ParticleNeRDNetwork``, ``ParticleNeRDSolver``, ``ParticleNeRDSolverCfg``,
  ``NewtonParticleNeRDManager`` and ``NewtonParticleNeRDCouplerManager``. Task names are unchanged.
  Existing gated-update checkpoints load through ``ParticleNeRDNetwork.load_checkpoint``; checkpoints trained with a removed
  option (for example the ballistic update) are rejected with an error.
* Moved the superseded ``particle_nerd``, ``particle_world`` and ``particle_world_batch`` packages to
  ``isaaclab_tasks.contrib.legacy``. Update imports to ``isaaclab_tasks.contrib.legacy.<package>``.
