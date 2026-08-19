"""SO-101 leader → relative EE pose → OpenArm mink IK teleop."""

__version__ = "0.1.0"

BODY_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT = "gripper"
ALL_JOINTS = BODY_JOINTS + (GRIPPER_JOINT,)
