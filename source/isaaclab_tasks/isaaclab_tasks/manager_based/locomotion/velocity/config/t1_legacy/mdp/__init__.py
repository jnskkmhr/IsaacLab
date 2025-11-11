from .terrain.terrain_cfg import *
from .termination.termination import root_height_below_minimum_adaptive
from .curriculums.curriculums import update_terrain_stiffness, terrain_ground_level
from .events.events import *
from .observations.observations import (
    clock, 

    foot_height, 
    foot_air_time, 
    foot_contact, 
    foot_contact_forces,

    hard_contact_forces, 
    foot_hard_contact_forces, 
    soft_contact_forces,
)
from .reward.reward import (
    feet_air_time_positive_biped, 
    feet_clearance, 
    # feet_swing_height, 
    soft_landing,

    track_foot_height, 
    reward_foot_distance,

    reward_feet_pitch, 
    reward_feet_roll,

    reward_feet_contact_number,
    reward_feet_contact_number_soft,

    feet_air_time_positive_biped_soft, 
    feet_slide_soft,
)