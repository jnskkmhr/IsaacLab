# Kimodo pyramid stairs with a standing finish

Each clip preserves the original 174 climbing poses from `../supplied_stairs_pyramid`.
A two-second procedural transition finishes the left-foot swing, brings the right
foot forward, and settles toward the task's default standing posture. A two-second
static hold follows. Total: 374 frames at 50 Hz (7.48 seconds).

Leg inverse kinematics preserves the support foot during each step and places both
foot box colliders on the top landing, with 2 mm initial clearance. Body poses are
regenerated using forward kinematics from the task's ball-hand G1 USD. Derived
velocities, foot corners, hand centers, and support labels are extended consistently;
hold velocities are zero. Original velocities are preserved before the splice frame.

The final root is 0.35 m farther forward than the original endpoint, with height
adjusted to the standing pose. Because commands use the final root pose, the task
destination also changes. Existing goal offsets and randomization still fit inside
the landing. The four cardinal approaches share unchanged pyramid terrain metadata.

This is a synthesized finishing segment, not additional captured human motion.
Kinematic checks and reference replay validate geometry; they do not establish
that a trained policy will dynamically track the transition or hold its pose.
`provenance.json` records the source hash, timing, and final pose.
