"""SO-101 forward kinematics from ``so101_new_calib.urdf`` (numpy, no Placo).

Body joints are in **degrees** (LeRobot ``use_degrees=True``). The gripper
joint is NOT part of the TCP chain — do not deg2rad it with the arm.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import BODY_JOINTS


def _rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    """URDF rpy: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = [float(x) for x in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _origin_T(elem: ET.Element | None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if elem is None:
        return T
    xyz = np.fromstring(elem.attrib.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(elem.attrib.get("rpy", "0 0 0"), sep=" ")
    T[:3, :3] = _rpy_to_R(rpy)
    T[:3, 3] = xyz
    return T


@dataclass
class _Joint:
    name: str
    jtype: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


def _parse_joints(urdf_path: Path) -> dict[str, _Joint]:
    tree = ET.parse(urdf_path)
    joints: dict[str, _Joint] = {}
    for j in tree.getroot().findall("joint"):
        origin = _origin_T(j.find("origin"))
        axis_el = j.find("axis")
        axis = np.array([0.0, 0.0, 1.0])
        if axis_el is not None:
            axis = np.fromstring(axis_el.attrib.get("xyz", "0 0 1"), sep=" ")
            n = float(np.linalg.norm(axis)) or 1.0
            axis = axis / n
        parent = j.find("parent").attrib["link"]
        child = j.find("child").attrib["link"]
        joints[j.attrib["name"]] = _Joint(
            name=j.attrib["name"],
            jtype=j.attrib.get("type", "fixed"),
            parent=parent,
            child=child,
            origin=origin,
            axis=axis.astype(np.float64),
        )
    return joints


def _axis_angle_T(axis: np.ndarray, q: float) -> np.ndarray:
    x, y, z = axis
    c, s = np.cos(q), np.sin(q)
    C = 1.0 - c
    R = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


class SO101FK:
    """FK of ``gripper_frame_link`` from 5 body joints (degrees)."""

    CHAIN = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper_frame_joint",
    )

    def __init__(self, urdf_path: str | Path, ee_link: str = "gripper_frame_link"):
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        self.ee_link = ee_link
        self.joints = _parse_joints(self.urdf_path)
        missing = [n for n in self.CHAIN if n not in self.joints]
        if missing:
            raise KeyError(f"URDF missing joints {missing} in {self.urdf_path}")

    def forward(
        self,
        joint_deg: dict[str, float] | np.ndarray,
        wrist_roll_offset_deg: float = 0.0,
    ) -> np.ndarray:
        """Return 4x4 T_base_ee in the URDF native ``gripper_frame_link``.

        ``joint_deg`` is a mapping of body joint name → degrees, or an array
        in ``BODY_JOINTS`` order. Gripper values, if present, are ignored.
        """
        if isinstance(joint_deg, dict):
            q = {k: float(joint_deg[k]) for k in BODY_JOINTS}
        else:
            arr = np.asarray(joint_deg, dtype=np.float64).reshape(-1)
            q = {name: float(arr[i]) for i, name in enumerate(BODY_JOINTS)}
        q["wrist_roll"] = q["wrist_roll"] + float(wrist_roll_offset_deg)

        T = np.eye(4, dtype=np.float64)
        for name in self.CHAIN:
            j = self.joints[name]
            T = T @ j.origin
            if j.jtype in ("revolute", "continuous"):
                T = T @ _axis_angle_T(j.axis, np.deg2rad(q[name]))
            elif j.jtype == "prismatic":
                t = np.eye(4)
                t[:3, 3] = j.axis * np.deg2rad(q[name])  # unused on SO-101 TCP
                T = T @ t
        return T
