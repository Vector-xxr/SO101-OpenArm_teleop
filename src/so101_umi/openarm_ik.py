"""OpenArm bimanual-URDF mink IK with only one arm enabled."""

from __future__ import annotations

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import mink
from mink import SE3

from .frames import rotation_from_preset
from .se3 import T_to_mink_wxyz_xyz


def _load_openarm_model(model_path: Path, description_root: str | Path | None) -> mujoco.MjModel:
    """Load MJCF directly or adapt the requested URDF in memory for MuJoCo.

    The source URDF is never modified. Its package URIs are resolved in a
    temporary copy and COLLADA visuals are removed because MuJoCo does not
    decode DAE; the collision STL geometry remains visible.
    """
    if model_path.suffix.lower() != ".urdf":
        return mujoco.MjModel.from_xml_path(model_path.as_posix())
    if description_root is None:
        raise ValueError("openarm_description_root is required for package:// URDF meshes")

    tree = ET.parse(model_path)
    root = tree.getroot()
    package_root = Path(description_root).resolve().as_posix()
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename", "")
        mesh.set(
            "filename",
            filename.replace("package://openarm_description", package_root),
        )
    mujoco_options = ET.SubElement(root, "mujoco")
    ET.SubElement(
        mujoco_options,
        "compiler",
        {"balanceinertia": "true", "discardvisual": "true", "fusestatic": "false"},
    )
    with tempfile.NamedTemporaryFile("wb", suffix=".urdf") as temporary:
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        temporary.flush()
        return mujoco.MjModel.from_xml_path(temporary.name)


class OpenArmMinkIK:
    def __init__(
        self,
        mjcf_path: str | Path,
        ee_frame: str = "openarm_right_link7",
        ee_frame_type: str = "body",
        arm_side: str = "right",
        tcp_offset_m: float = 0.2343,
        tcp_align: str = "ry-90",
        description_root: str | Path | None = None,
        position_cost: float = 50.0,
        orientation_cost: float = 0.15,
        posture_cost: float = 0.02,
        damping_cost: float = 0.2,
        ik_dt: float = 0.005,
        ik_iters: int = 8,
        ik_damping: float = 0.08,
        solver: str = "daqp",
    ):
        self.mjcf_path = Path(mjcf_path)
        self.model = _load_openarm_model(self.mjcf_path, description_root)
        self.configuration = mink.Configuration(self.model)
        self.data = self.configuration.data
        if arm_side not in {"left", "right"}:
            raise ValueError(f"arm_side must be left or right, got {arm_side!r}")
        self.arm_side = arm_side
        self.ee_frame = ee_frame
        self.ee_frame_type = ee_frame_type
        self.ik_dt = float(ik_dt)
        self.ik_iters = int(ik_iters)
        self.ik_damping = float(ik_damping)
        self.solver = solver
        self.T_frame_tcp = np.eye(4, dtype=np.float64)
        self.T_frame_tcp[:3, 3] = [0.0, 0.0, float(tcp_offset_m)]
        self.T_frame_tcp[:3, :3] = rotation_from_preset(tcp_align)
        self.T_tcp_frame = np.linalg.inv(self.T_frame_tcp)

        self.ee_task = mink.FrameTask(
            frame_name=ee_frame,
            frame_type=ee_frame_type,
            position_cost=float(position_cost),
            orientation_cost=float(orientation_cost),
            lm_damping=2.0,
        )
        self.posture_task = mink.PostureTask(self.model, cost=float(posture_cost))
        damp = float(damping_cost)
        self.tasks = [self.ee_task, self.posture_task]
        if damp > 0.0:
            self.damping_task = mink.DampingTask(self.model, cost=damp)
            self.tasks.append(self.damping_task)
        # Kinematic teleop: joint limits only. VelocityLimit + torque actuators
        # previously produced NaN ctrl and blew the arm apart in the viewer.
        self.limits = [mink.ConfigurationLimit(self.model)]

        frame_types = {
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "site": mujoco.mjtObj.mjOBJ_SITE,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
        }
        if ee_frame_type not in frame_types:
            raise ValueError(f"unsupported ee_frame_type {ee_frame_type!r}")
        self.ee_frame_id = mujoco.mj_name2id(self.model, frame_types[ee_frame_type], ee_frame)
        if self.ee_frame_id < 0:
            raise ValueError(f"{ee_frame_type} {ee_frame!r} not in {self.mjcf_path}")

        arm_joints = [f"openarm_{arm_side}_joint{i}" for i in range(1, 8)]
        finger_joints = [
            f"openarm_{arm_side}_finger_joint1",
            f"openarm_{arm_side}_finger_joint2",
        ]
        frozen_side = "left" if arm_side == "right" else "right"
        frozen_joints = [f"openarm_{frozen_side}_joint{i}" for i in range(1, 8)]
        frozen_joints += [
            f"openarm_{frozen_side}_finger_joint1",
            f"openarm_{frozen_side}_finger_joint2",
        ]
        self.arm_qadr = [self.model.joint(name).qposadr[0] for name in arm_joints]
        self.finger_qadr = [self.model.joint(name).qposadr[0] for name in finger_joints]
        self.frozen_qadr = [self.model.joint(name).qposadr[0] for name in frozen_joints]

        self.reset_home()

    def reset_home(self) -> None:
        key = "home"
        try:
            self.configuration.update_from_keyframe(key)
        except Exception:
            mujoco.mj_resetData(self.model, self.data)
            self.configuration.update(self.data.qpos)
        self.frozen_qpos = self.data.qpos[self.frozen_qadr].copy()
        self.posture_task.set_target_from_configuration(self.configuration)
        mujoco.mj_forward(self.model, self.data)

    def ee_pose(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        T = np.eye(4, dtype=np.float64)
        if self.ee_frame_type == "body":
            T[:3, 3] = self.data.xpos[self.ee_frame_id].copy()
            T[:3, :3] = self.data.xmat[self.ee_frame_id].reshape(3, 3).copy()
        elif self.ee_frame_type == "site":
            T[:3, 3] = self.data.site_xpos[self.ee_frame_id].copy()
            T[:3, :3] = self.data.site_xmat[self.ee_frame_id].reshape(3, 3).copy()
        else:
            T[:3, 3] = self.data.geom_xpos[self.ee_frame_id].copy()
            T[:3, :3] = self.data.geom_xmat[self.ee_frame_id].reshape(3, 3).copy()
        return T @ self.T_frame_tcp

    def solve(self, T_cmd: np.ndarray) -> np.ndarray:
        if not np.isfinite(T_cmd).all():
            return np.zeros(6)
        q_before = self.data.qpos.copy()
        # FrameTask regulates link7. Convert the standardized virtual TCP
        # command back to the native link7 frame without editing the URDF.
        T_frame_cmd = T_cmd @ self.T_tcp_frame
        wxyz_xyz = T_to_mink_wxyz_xyz(T_frame_cmd)
        if not np.isfinite(wxyz_xyz).all():
            return np.zeros(6)
        target = SE3(wxyz_xyz=wxyz_xyz)
        self.ee_task.set_target(target)

        last_err = None
        for _ in range(self.ik_iters):
            try:
                vel = mink.solve_ik(
                    self.configuration,
                    self.tasks,
                    self.ik_dt,
                    self.solver,
                    damping=self.ik_damping,
                    limits=self.limits,
                    safety_break=False,
                )
            except mink.NoSolutionFound:
                break
            if not np.isfinite(vel).all():
                self.configuration.update(q_before)
                break
            self.configuration.integrate_inplace(vel, self.ik_dt)
            # The opposite arm is explicitly pinned at its startup posture.
            self.data.qpos[self.frozen_qadr] = self.frozen_qpos
            self.configuration.update(self.data.qpos)
            if not np.isfinite(self.data.qpos).all():
                self.configuration.update(q_before)
                break
            last_err = self.ee_task.compute_error(self.configuration)
            if last_err is not None and np.linalg.norm(last_err[:3]) < 1e-4:
                break
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            self.configuration.update(q_before)
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
        return last_err if last_err is not None and np.isfinite(last_err).all() else np.zeros(6)

    def set_gripper(self, open_frac: float, open_m: float = 0.044) -> None:
        width = float(np.clip(open_frac, 0.0, 1.0)) * float(open_m)
        for adr in self.finger_qadr:
            self.data.qpos[adr] = width
        self.configuration.update(self.data.qpos)

    def push_ctrl(self) -> None:
        """Kinematic display only: zero actuator cmd so the viewer cannot explode."""
        if self.data.ctrl.size:
            self.data.ctrl[:] = 0.0
        self.data.qvel[:] = 0.0

    def pos_err(self, T_cmd: np.ndarray) -> float:
        T = self.ee_pose()
        return float(np.linalg.norm(T[:3, 3] - T_cmd[:3, 3]))
