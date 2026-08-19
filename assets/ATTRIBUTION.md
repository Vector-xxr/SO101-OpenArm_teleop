# OpenArm v1 single-arm MuJoCo model
# Copyright 2025 Enactic, Inc.  Apache-2.0
# Vendored from: /home/vector/work/vr_teleop/openarm_teleop/openarm_mujoco/v1
# Upstream: https://github.com/enactic/openarm
#
# Local edits for SO101-umi:
# - collision STL assets/geoms dropped (files not in that tree)
# - TCP sites ee_native / ee_aligned added on openarm_link7
# - torque motors replaced with position actuators
#
# SO-101 URDF
# TheRobotStudio so101_new_calib.urdf
# https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
# Visual STLs not vendored; numpy FK only needs joint origins.
