* Added experimental G1 particle-cache tasks with material-specific pre-relaxed
  particle templates, GL/RTX isosurfaces, and tracked regions for multiple rigid
  bodies with two-way MPM coupling. Adapted Felix Meyer's paged-sand example.
* Renamed the experimental ``g1_29dof_mpm_moving_patch_beta`` package to
  ``g1_29dof_mpm_moving_patch_particle_cache``. Its task IDs now end in
  ``MovingPatch-ParticleCache`` and ``MovingPatch-ParticleCache-Play``.
* Capture coupled GPU solver steps separately from CPU particle paging, reuse
  unloaded terrain meshes with Felix Meyer's GPU mesh pool, and honor RTX
  particle visibility in interactive and offline rendering.
* Keep unused particle slots outside the terrain grid so MPM quadrature does
  not traverse thousands of coincident inactive particles in a sand cell.
* Hide individual particles by default and provide independent particle and
  isosurface GUI controls in the moving-patch GL and RTX viewers.
* Preserve particle-spawner visibility to avoid an extra USD cloud bypassing
  the viewer controls, and refine the default surface grid to 2 cm.
* Match live and cached sand surfaces to the background terrain material, draw
  untouched terrain at the estimated relaxed-template height with openings for particle history, and
  retain physical reconstruction width on finer visual grids.
* Draw unloaded cached particles in GL/RTX particle view, retaining their saved
  deformation and honoring the particle visibility toggle and world resets.
* Blend the outer isosurface rim into the background mesh heights and normals
  to reduce visible seams without changing particle states or contact geometry.
