Added GPU-captured sand isosurface rendering to the experimental G1 moving-region MPM task,
with retained sleeping-region meshes, compact live meshes, optional awake/support particle
glyphs in Newton GL and RTX, and reset-aware refresh. Particle and mesh data remain on GPU;
five control integers are read for exact viewer buffer sizes and cache refresh. Guarded
MPM status checks run at control-step boundaries without changing solver parameters.
