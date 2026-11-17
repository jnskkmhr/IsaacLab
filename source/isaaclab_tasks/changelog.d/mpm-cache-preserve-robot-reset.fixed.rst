Preserve authored robot joint positions and velocities while preparing or restoring the
cached initial MPM bed in the experimental G1 fixed-bed and moving-region tasks.
Exclude joint-reset flags when clearing coupled-solver history so MJWarp does not replace
the configured crouched pose with model defaults and launch the robot from the approach platform.
