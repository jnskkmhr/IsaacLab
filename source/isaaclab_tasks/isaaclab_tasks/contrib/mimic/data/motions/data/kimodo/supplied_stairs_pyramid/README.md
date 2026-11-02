# Kimodo pyramid stairs

Four cardinal approaches derived from `../supplied_stairs/motion.npz`, matching
`kimodo_pyramid_stairs_reference.mp4` in the repository root. Each clip contains
174 frames at 50 Hz. Joint positions, joint velocities, and timing are unchanged.

The pyramid preserves all six original west-facing stair fronts and surface
heights, with a 2.5 m square top landing. For each original box, define its west
edge `x_i = position_x - dimension_x / 2` and top `h_i = position_z + dimension_z / 2`.
The pyramid center is `c = (x_last + 1.25, original_center_y, 0)`. Each replacement
box has center `(c_x, c_y, h_i / 2)` and dimensions
`(2 * (c_x - x_i), 2 * (c_x - x_i), h_i)`.

The four files rotate the reference by 0, 90, 180, or 270 degrees about this
center. World positions, orientations, linear/angular velocities, foot corners,
and hand sphere centers rotate together. Terrain metadata is identical in all
four files, so the paired-terrain importer shares one static mesh. The existing
uniform motion sampler selects the approach at each reset, in both imitation
and task episodes. The goal is the selected clip's final root pose.

`provenance.json` records the source hash, rotation center, and step geometry.
The original NPZ remains unchanged. The reference ends while moving and does
not demonstrate a sustained standing finish. Kinematic replay checks alignment;
it does not establish dynamic tracking performance.
