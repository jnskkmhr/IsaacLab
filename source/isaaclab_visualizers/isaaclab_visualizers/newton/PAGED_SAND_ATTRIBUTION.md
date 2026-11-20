# Reference code attribution

Source: [NVIDIA-dev/devtech-newton-examples](https://github.com/NVIDIA-dev/devtech-newton-examples/tree/fmeyer/mpm-paged-sand),
branch `fmeyer/mpm-paged-sand`, commit `3e4f29ecf369bf66aeb93e73c9b4f9bd59bf6e7f`.
Author of the paged-sand example: Felix Meyer. The original SPDX copyright is
The Newton Developers; the source license is Apache-2.0 (included as `PAGED_SAND_LICENSE`).

`paged_sand_isosurface.py` was copied from `examples/mpm/paged_sand_isosurface.py`.
Local changes expose the already-supported particle-volume splat weight through
`extract`, raise instead of silently truncating an oversized mesh after resizing,
and exempt the reference's grid-coordinate names from the local ambiguous-name
lint rule. Rendering adaptations are described below; the mesh-pool algorithm
is retained.

The IsaacLab adapter in `cached_particle_surface.py` adds independent worlds, variable
surface resolution and vertical origin translation. It extracts the loaded tiles every
frame, including their boundary particles. A tile that leaves the pool is baked once into
a mesh from its saved particles, following the example's baked chunks, but stored in
fixed-size slots grouped into separately drawn pages instead of the example's block pool.
The adapter does not copy the example's robot controller, single-scene chunk store or
sleep/wake scheduler.

The B-spline splat accepts a physical particle spacing so refining the display
grid does not narrow the reconstruction kernel below the particle spacing. The
default preserves the reference grid-relative kernel. The adapter supplies the
shared terrain texture/UVs and an untouched-terrain visual with openings for
loaded tiles; neighboring untouched tiles provide the reference's edge closure.
An additional field-fill kernel retains untouched sand beneath particles deposited
into tiles that have never been loaded, including the shared nodes of adjacent tiles.
The adapter estimates untouched sand height from the relaxed template's upper
particle layer, so compacted templates do not appear inside an elevated background.
These adjustments affect display meshes only.
The adapter also applies a narrow outer-edge blend (`blend_surface_boundary`),
matching the displayed background triangles and normals after extraction.

The adapter exposes a smoothing multiplier for the existing physical B-spline
width (default 1.5) and samples the surface on an approximately 15 mm grid. Particle
volume weights are unchanged.
