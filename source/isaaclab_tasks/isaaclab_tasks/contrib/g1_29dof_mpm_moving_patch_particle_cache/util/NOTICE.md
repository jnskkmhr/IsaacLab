# Paged-sand code attribution

Source: [NVIDIA-dev/devtech-newton-examples](https://github.com/NVIDIA-dev/devtech-newton-examples/tree/fmeyer/mpm-paged-sand),
branch `fmeyer/mpm-paged-sand`, commit `3e4f29ecf369bf66aeb93e73c9b4f9bd59bf6e7f`.
Author of the paged-sand example: Felix Meyer. The original SPDX copyright is
The Newton Developers; the source license is Apache-2.0 (included as
[LICENSE.paged-sand](LICENSE.paged-sand)). This notice and that license apply to
`paged_sand_isosurface.py` and `settled_template.py`; other utility modules retain
their own SPDX license declarations.

`paged_sand_isosurface.py` was copied from `examples/mpm/paged_sand_isosurface.py`.
Local changes expose the already-supported particle-volume splat weight through
`extract`, raise instead of silently truncating an oversized mesh after resizing,
and exempt the reference's grid-coordinate names from the local ambiguous-name
lint rule. Rendering adaptations are described below; the mesh-pool algorithm
is retained.

`settled_template.py` adapts `settle_template` and the floor guard from
`examples/mpm/example_mpm_paged_sand.py`. It accepts the task's sampled positions
and material, bounds initial vertical jitter, extends the normal floor guard to
the four mirror walls, and validates speed, mean-height drift and retained bed
depth. It adds bounded material-keyed reuse and per-world GPU copies. Alternating
tile reflections and tensor conjugation follow the reference's `Template` and
`mirror_tensor` implementation in the task's `util/kernel.py`.

The IsaacLab adapter in `util/surface.py` adds independent worlds, saved-particle
rendering, variable surface resolution and vertical origin translation. It now
uses the reference `IsoCache` GPU mesh pool for unloaded tiles, with revision-based
invalidation and per-world reset handling. Pool exhaustion raises rather than
evicting visible terrain history. It does
not copy the example's robot controller, single-scene chunk store or sleep/wake
scheduler. Those have different ownership and lifecycle assumptions from the
vectorized IsaacLab environment.

The B-spline splat accepts a physical particle spacing so refining the display
grid does not narrow the reconstruction kernel below the particle spacing. The
default preserves the reference grid-relative kernel. The adapter supplies the
shared terrain texture/UVs and an untouched-terrain visual with openings for
particle history; neighboring untouched tiles provide the reference's edge closure.
An additional field-fill kernel retains untouched sand beneath particles deposited
into tiles that have never been loaded, including the shared nodes of adjacent tiles.
The adapter estimates untouched sand height from the relaxed template's upper
particle layer, so compacted templates do not appear inside an elevated background.
These adjustments affect display meshes only.
The adapter also applies a narrow outer-edge blend in `util/surface_kernel.py`,
matching the displayed background triangles and normals after extraction. This
is an IsaacLab rendering adaptation, not a change to Felix's particle simulation.
