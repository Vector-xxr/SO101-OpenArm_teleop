"""OpenArm single-arm mink IK + MuJoCo kinematic display."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

import mink
from mink import SE3

from .se3 import T_to_mink_wxyz_xyz, split_T


ARM_JOINTS = [f"openarm_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["openarm_finger_joint1", "openarm_finger_joint2"]


class OpenArmMinkIK:
    def __init__(
        self,
        mjcf_path: str | Path,
        ee_frame: str = "ee_aligned",
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
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path.as_posix())
        self.configuration = mink.Configuration(self.model)
        self.data = self.configuration.data
        self.ee_frame = ee_frame
        self.ik_dt = float(ik_dt)
        self.ik_iters = int(ik_iters)
        self.ik_damping = float(ik_damping)
        self.solver = solver

        self.ee_task = mink.FrameTask(
            frame_name=ee_frame,
            frame_type="site",
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

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_frame)
        if self.ee_site_id < 0:
            raise ValueError(f"site {ee_frame!r} not in {self.mjcf_path}")
        self.target_body = "ee_target"
        try:
            self.mocap_id = self.model.body(self.target_body).mocapid[0]
        except Exception:
            self.mocap_id = -1

        self.arm_qadr = [self.model.joint(n).qposadr[0] for n in ARM_JOINTS]
        self.finger_qadr = [self.model.joint(n).qposadr[0] for n in FINGER_JOINTS]
        self.act_ids = {
            n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in (
                "joint1_ctrl",
                "joint2_ctrl",
                "joint3_ctrl",
                "joint4_ctrl",
                "joint5_ctrl",
                "joint6_ctrl",
                "joint7_ctrl",
                "finger_ctrl",
            )
        }

        self.reset_home()

    def reset_home(self) -> None:
        key = "home"
        try:
            self.configuration.update_from_keyframe(key)
        except Exception:
            mujoco.mj_resetData(self.model, self.data)
            self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        mujoco.mj_forward(self.model, self.data)

    def ee_pose(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = self.data.site_xpos[self.ee_site_id].copy()
        T[:3, :3] = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        return T

    def solve(self, T_cmd: np.ndarray) -> np.ndarray:
        if not np.isfinite(T_cmd).all():
            return np.zeros(6)
        q_before = self.data.qpos.copy()
        wxyz_xyz = T_to_mink_wxyz_xyz(T_cmd)
        if not np.isfinite(wxyz_xyz).all():
            return np.zeros(6)
        target = SE3(wxyz_xyz=wxyz_xyz)
        self.ee_task.set_target(target)
        if self.mocap_id >= 0:
            R, p = split_T(T_cmd)
            self.data.mocap_pos[self.mocap_id] = p
            self.data.mocap_quat[self.mocap_id] = wxyz_xyz[:4]

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
        self.data.ctrl[:] = 0.0
        self.data.qvel[:] = 0.0

    def pos_err(self, T_cmd: np.ndarray) -> float:
        T = self.ee_pose()
        return float(np.linalg.norm(T[:3, 3] - T_cmd[:3, 3]))
