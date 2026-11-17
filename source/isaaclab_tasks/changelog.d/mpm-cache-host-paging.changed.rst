* Reduced the host-side cost of paging in the moving-patch MPM particle cache. Free saved-record ids are kept in a pool
  instead of a list of ``max_cached_particles_per_env`` entries that was copied for every environment on each paging
  event, planning no longer rebuilds the Python tile sets when no patch window moved, and departing particles are
  grouped by tile in one pass instead of copying each tile's page list per particle. Particle state is unchanged;
  paging spikes that grew with the number of environments are much smaller.
* Allowed ``particle_cache.simulation_capacity_multiplier`` to be a fractional number of at least 1. Reserve slots are
  allocated in whole template tiles, so fewer slots reduce memory and the particle solver's per-step cost on flat
  terrain; running out of slots still raises before any particle is recycled.
