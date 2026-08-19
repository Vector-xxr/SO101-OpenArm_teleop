"""Attribution for vendored robot assets."""

OPENARM = """
OpenArm MuJoCo model (v1 single arm)
Copyright 2025 Enactic, Inc.
Apache License 2.0
Source: vr_teleop/openarm_teleop/openarm_mujoco/v1/
Upstream: https://github.com/enactic/openarm

Collision STLs were not present in that tree and are omitted.
TCP sites ee_native / ee_aligned were added for SO101-umi teleop.
Actuators changed from torque motors to position actuators for kinematic IK.
"""

SO101 = """
SO-101 URDF so101_new_calib.urdf
TheRobotStudio / Hugging Face SO-ARM100
https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
Generated with onshape-to-robot.
Visual STL meshes are not vendored (FK does not need them).
"""
