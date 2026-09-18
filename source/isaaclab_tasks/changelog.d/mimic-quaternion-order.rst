Fixed mimic motion loading and identity rotations to use xyzw quaternions at runtime.
Legacy untagged NPZ references were interpreted as wxyz. Newly converted CSV and Xsens
references retained xyzw and recorded a scalar ``quaternion_order`` field. This metadata
took precedence over the fallback order. For existing untagged xyzw files, set
``motion_quaternion_order="xyzw"`` in the command configuration, or pass
``--quaternion_order xyzw`` to the replay script. Original motion files were not rewritten.
