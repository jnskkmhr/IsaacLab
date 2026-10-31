# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU spatial hash with environment-keyed cells and instance-owned storage.

Fixed grid dimensions wrap cell coordinates; distance checks reject aliases.
No host grid descriptor is read during graph replay. Coordinates remain local
to each simulation world and are never translated for neighbor lookup.
"""

import warp as wp


@wp.func
def cell_key(x: int, y: int, z: int, world: int, dim: int):
    a = ((x % dim) + dim) % dim
    b = ((y % dim) + dim) % dim
    c = ((z % dim) + dim) % dim
    return world * dim * dim * dim + (a * dim + b) * dim + c


@wp.kernel
def keys(
    x: wp.array(dtype=wp.vec3),
    particles: int,
    width: float,
    dim: int,
    key: wp.array(dtype=int),
    ids: wp.array(dtype=int),
):
    i = wp.tid()
    p = x[i] / width
    key[i] = cell_key(int(wp.floor(p[0])), int(wp.floor(p[1])), int(wp.floor(p[2])), i // particles, dim)
    ids[i] = i


@wp.kernel
def cell_ranges(key: wp.array(dtype=int), count: int, starts: wp.array(dtype=int), ends: wp.array(dtype=int)):
    i = wp.tid()
    c = key[i]
    if i == 0:
        starts[c] = 0
    elif key[i - 1] != c:
        starts[c] = i
    if i == count - 1:
        ends[c] = count
    elif key[i + 1] != c:
        ends[c] = i + 1


class Grid:
    """One GPU radix sort for all particle worlds, with 32³ buckets per world."""

    def __init__(self, environments, particles, width, device):
        self.environments, self.particles, self.width, self.device = environments, particles, width, device
        self.count = environments * particles
        self.dim = 32
        cells = environments * self.dim**3
        if cells >= 2**31 or self.count >= 2**31:
            raise ValueError("Grid keys and particle indices require signed 32-bit capacity")
        # Warp radix_sort_pairs uses the second half as scratch storage.
        self.keys = wp.empty(2 * self.count, dtype=int, device=device)
        self.ids = wp.empty(2 * self.count, dtype=int, device=device)
        self.starts = wp.zeros(cells, dtype=int, device=device)
        self.ends = wp.zeros(cells, dtype=int, device=device)

    def build(self, position):
        wp.launch(
            keys, self.count, [position, self.particles, self.width, self.dim, self.keys, self.ids], device=self.device
        )
        wp.utils.radix_sort_pairs(self.keys, self.ids, self.count)
        self.starts.zero_()
        self.ends.zero_()
        wp.launch(cell_ranges, self.count, [self.keys, self.count, self.starts, self.ends], device=self.device)
