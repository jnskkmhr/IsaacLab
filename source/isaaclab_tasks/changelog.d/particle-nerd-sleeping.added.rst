* Added particle sleeping to the Particle NeRD solver through
  :attr:`~isaaclab_tasks.contrib.particle_nerd.ParticleNeRDSolverCfg.sleep_speed`: a particle whose predicted
  move in a step is slower than this speed keeps its position and gets zero velocity. It is off by default;
  0.01 m/s is the value recommended by the benchmark.
