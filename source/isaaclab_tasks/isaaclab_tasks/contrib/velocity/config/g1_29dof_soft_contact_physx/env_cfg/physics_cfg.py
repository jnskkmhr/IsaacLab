# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_physx.physics import PhysxCfg

from isaaclab.utils import configclass


@configclass
class G1PhysicsCfg(PhysxCfg):
    enable_ccd = False
    solver_type = 1
    gpu_heap_capacity = 67108864
    enable_stabilization = False
    gpu_max_num_partitions = 8
    gpu_collision_stack_size = 67108864
    gpu_temp_buffer_capacity = 16777216
    bounce_threshold_velocity = 0.5
    friction_offset_threshold = 0.04
    gpu_max_particle_contacts = 1048576
    gpu_max_rigid_patch_count = 327680
    gpu_max_soft_body_contacts = 1048576
    enable_enhanced_determinism = False
    gpu_max_rigid_contact_count = 8388608
    max_position_iteration_count = 255
    max_velocity_iteration_count = 255
    min_position_iteration_count = 1
    min_velocity_iteration_count = 0
    friction_correlation_distance = 0.025
    gpu_found_lost_pairs_capacity = 2097152
    solve_articulation_contact_last = False
    gpu_total_aggregate_pairs_capacity = 2097152
    enable_external_forces_every_iteration = True
    gpu_found_lost_aggregate_pairs_capacity = 33554432
