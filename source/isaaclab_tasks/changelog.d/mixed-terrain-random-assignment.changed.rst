* Added single-environment MPM playback and seeded random rigid/MPM membership for larger mixed-terrain batches.
  The environment configuration exposes ``mpm_ratio`` to control the MPM sampling probability.
  Particle resets, material randomization, terrain placement, and solver capacity follow the sampled membership.
  Rigid-only batches advance the MJWarp entry without proxy connections.
