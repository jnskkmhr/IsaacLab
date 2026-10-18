```bash
uv run --frozen --with mink --with daqp --with loop-rate-limiters \
python scripts/tools/mimic/retarget_cmu_long_jump.py \
--bvh /tmp/g1-longjump-research/83_43.bvh \
--gmr_repo /tmp/g1-longjump-research/GMR \
--robot_xml /home/jkamohara/isaac/newton-assets/unitree_g1/mjcf/g1_29dof_rev_1_0_box_foot.xml \
--output /tmp/cmu_83_43.npz \\
--hold_seconds 5
```