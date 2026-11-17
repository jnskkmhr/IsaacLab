* Added ``particle_cache.gpu_resident`` (default ``False``) and ``particle_cache.capture_step`` to the moving-patch MPM
  particle cache. With ``gpu_resident`` the tile bookkeeping runs on the GPU (a per-environment hash table of tiles,
  atomic slot and record allocators, fixed-size kernels), so the cache update has no host synchronization and the
  whole physics step, cache included, can be recorded in one CUDA graph like the recycling-only task. Particle state
  after every paging step matches the host-planned cache exactly. Capacity errors are reported at the next control
  step instead of before any particle is recycled.
