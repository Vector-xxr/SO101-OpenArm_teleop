"""Gripper TCP frame convention and R_align presets.

Required teleop convention (both leader FK output after align, and OpenArm
``ee_aligned`` site):

    +X  forward  — out of the gripper, towards the object
    +Y  left
    +Z  up

Native frames
-------------
SO-101 ``gripper_frame_link`` (TheRobotStudio ``so101_new_calib.urdf``)
    Fixed joint from ``gripper_link``:
      xyz = (-0.0079, -0.000218, -0.09813),  rpy = (0, π, 0)
    Ry(π) maps parent -Z onto gripper_frame +Z. The TCP origin sits ~10 cm
    along the jaws, so **native +Z is forward (towards the object)**.
    Native +Y = gripper_link +Y (finger-opening / "left" for the stock jaw).
    Native +X = remaining right-handed axis (= "down" when Z=forward, Y=left).

OpenArm v1 ``ee_native`` (identity wrt ``openarm_link7``)
    Fingers attach at z≈0.115 m and slide along ±Y, so
      **native +Z = forward**, **native +Y = left**, **native +X = down**.
    Site ``ee_aligned`` bakes Ry(-90°) into the MJCF so Mink tracks the
    required convention directly. ``openarm_align: baked`` means we do not
    apply a second R_align on the follower.

R_align
-------
To send native Z-forward / Y-left into +X-forward / +Y-left / +Z-up:

    columns of R_align are aligned axes expressed in the native TCP:

        X_aligned (forward) = native +Z
        Y_aligned (left)    = native +Y
        Z_aligned (up)      = native -X

    R_align = Ry(-90°) =

        [[ 0,  0, -1],
         [ 0,  1,  0],
         [ 1,  0,  0]]

    T_aligned = T_native @ R_align

If a robot's native TCP already matches the required convention, use
``identity``. Override with ``--so101-align`` / config ``so101_align``.
"""

from __future__ import annotations

import numpy as np

# Ry(-90°): native Z-forward Y-left → +X forward +Y left +Z up
R_ALIGN_RY_NEG90 = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

R_IDENTITY = np.eye(3, dtype=np.float64)


def _rotx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _roty(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


_PRESETS: dict[str, np.ndarray] = {
    "identity": R_IDENTITY,
    "baked": R_IDENTITY,  # already in MJCF
    "ry-90": R_ALIGN_RY_NEG90,
    # SO-101 physical gripper: forward agrees with OpenArm, while left/up
    # are reversed. Keep +X and flip +Y/+Z with a local Rx(pi).
    "ry-90-rx180": R_ALIGN_RY_NEG90 @ _rotx(np.pi),
    "ry90": _roty(np.pi / 2),
    "rx90": _rotx(np.pi / 2),
    "rx-90": _rotx(-np.pi / 2),
    "rx180": _rotx(np.pi),
    "rz90": _rotz(np.pi / 2),
    "rz-90": _rotz(-np.pi / 2),
    "ry180": _roty(np.pi),
}


def rotation_from_preset(name: str) -> np.ndarray:
    key = (name or "identity").strip().lower()
    if key not in _PRESETS:
        raise ValueError(f"unknown align preset {name!r}; try {sorted(_PRESETS)}")
    return _PRESETS[key].copy()


def se3_from_R_p(R: np.ndarray, p: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    if p is not None:
        T[:3, 3] = np.asarray(p, dtype=np.float64).reshape(3)
    return T


def describe_axes(R: np.ndarray) -> str:
    labels = ("X", "Y", "Z")
    world = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
    axes = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    parts = []
    for i, lab in enumerate(labels):
        col = R[:, i]
        j = int(np.argmax(axes @ col))
        parts.append(f"{lab}->{world[j]}")
    return " ".join(parts)
