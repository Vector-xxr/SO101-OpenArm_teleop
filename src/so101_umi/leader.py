"""SO-101 leader reader.

Hardware path talks to Feetech STS3215 over USB (LeRobot bus). We only
``sync_read(Present_Position)`` — never ``Goal_Position``.

``--sim-leader`` synthesises a dummy arm so MuJoCo still runs without USB.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import ALL_JOINTS, BODY_JOINTS, GRIPPER_JOINT
from .serial_util import default_leader_port

# Approximate ticks→deg when no LeRobot calibration JSON is present.
_TICKS_MID = 2048
_TICKS_RES = 4096


def _ticks_to_deg(ticks: int) -> float:
    return (float(ticks) - _TICKS_MID) * 360.0 / _TICKS_RES


def _ticks_to_gripper_pct(ticks: int) -> float:
    return float(np.clip(ticks / 4095.0 * 100.0, 0.0, 100.0))


class BaseLeader:
    def get_action(self) -> dict[str, float]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class SimLeader(BaseLeader):
    """Dummy joints. Default: slow sinusoid on shoulder_pan so IK is visible."""

    def __init__(
        self,
        rest: dict[str, float] | None = None,
        motion: bool = True,
        pan_amp_deg: float = 25.0,
        pan_hz: float = 0.15,
    ):
        self.rest = rest or {
            "shoulder_pan": 0.0,
            "shoulder_lift": 20.0,
            "elbow_flex": -40.0,
            "wrist_flex": 40.0,
            "wrist_roll": 0.0,
            "gripper": 40.0,
        }
        self.motion = motion
        self.pan_amp_deg = pan_amp_deg
        self.pan_hz = pan_hz
        self.t0 = time.perf_counter()

    def get_action(self) -> dict[str, float]:
        q = dict(self.rest)
        if self.motion:
            t = time.perf_counter() - self.t0
            q["shoulder_pan"] = self.rest["shoulder_pan"] + self.pan_amp_deg * math.sin(
                2.0 * math.pi * self.pan_hz * t
            )
            q["wrist_flex"] = self.rest["wrist_flex"] + 12.0 * math.sin(
                2.0 * math.pi * self.pan_hz * t * 0.7
            )
            g = 50.0 + 40.0 * math.sin(2.0 * math.pi * 0.2 * t)
            q["gripper"] = float(np.clip(g, 0.0, 100.0))
        return {f"{k}.pos": float(q[k]) for k in ALL_JOINTS}


class HardwareLeader(BaseLeader):
    """Read-only Feetech bus via LeRobot (optional dependency)."""

    def __init__(
        self,
        port: str,
        calibration_path: str | Path | None = None,
        use_degrees: bool = True,
    ):
        try:
            from lerobot.motors import Motor, MotorCalibration, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus
        except ImportError as e:
            raise ImportError(
                "Hardware leader needs LeRobot + scservo_sdk. "
                "conda activate lerobot, or pip install lerobot. "
                "Use --sim-leader without USB."
            ) from e

        self.port = port
        norm_body = MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
        motors = {
            "shoulder_pan": Motor(1, "sts3215", norm_body),
            "shoulder_lift": Motor(2, "sts3215", norm_body),
            "elbow_flex": Motor(3, "sts3215", norm_body),
            "wrist_flex": Motor(4, "sts3215", norm_body),
            "wrist_roll": Motor(5, "sts3215", norm_body),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }
        calib = None
        if calibration_path:
            calib = _load_lerobot_calibration(Path(calibration_path))
        else:
            # Dummy span so DEGREES normalisation works without a file.
            calib = {
                name: MotorCalibration(
                    id=motors[name].id,
                    drive_mode=0,
                    homing_offset=0,
                    range_min=0,
                    range_max=4095,
                )
                for name in motors
            }
            print(
                f"[leader] no calibration JSON; using mid=2048 ticks as 0 deg on {port}. "
                "Pass --calibration for true LeRobot offsets."
            )
        self._has_gripper = True
        self._fallback_gripper_pct = 50.0
        self.bus = FeetechMotorsBus(port=port, motors=motors, calibration=calib)
        self._use_dummy_calib = calibration_path is None
        try:
            self.bus.connect(handshake=True)
        except RuntimeError as exc:
            message = str(exc)
            only_gripper_missing = (
                "Missing motor IDs:" in message
                and "- 6 (expected model:" in message
                and all(f"- {motor_id} (expected model:" not in message for motor_id in range(1, 6))
            )
            if not only_gripper_missing:
                raise
            try:
                self.bus.disconnect(disable_torque=False)
            except Exception:
                pass
            body_motors = {name: motor for name, motor in motors.items() if name != GRIPPER_JOINT}
            body_calib = {name: value for name, value in calib.items() if name != GRIPPER_JOINT}
            self.bus = FeetechMotorsBus(
                port=port,
                motors=body_motors,
                calibration=body_calib,
            )
            self.bus.connect(handshake=True)
            self._has_gripper = False
            print(
                "[leader] warning: motor ID 6 (gripper) is missing; "
                "continuing with arm IDs 1-5 and holding simulated gripper at 50%."
            )
        print(f"[leader] connected (read-only) on {port}")

    def get_action(self) -> dict[str, float]:
        raw = self.bus.sync_read("Present_Position", normalize=False)
        if self._use_dummy_calib:
            out = {}
            for name in BODY_JOINTS:
                out[f"{name}.pos"] = _ticks_to_deg(int(raw[name]))
            out[f"{GRIPPER_JOINT}.pos"] = (
                _ticks_to_gripper_pct(int(raw[GRIPPER_JOINT]))
                if self._has_gripper
                else self._fallback_gripper_pct
            )
            return out
        pos = self.bus.sync_read("Present_Position", normalize=True)
        out = {f"{name}.pos": float(pos[name]) for name in BODY_JOINTS}
        out[f"{GRIPPER_JOINT}.pos"] = (
            float(pos[GRIPPER_JOINT]) if self._has_gripper else self._fallback_gripper_pct
        )
        return out

    def close(self) -> None:
        try:
            # Do not disable torque — this port might be a follower if miswired.
            self.bus.disconnect(disable_torque=False)
        except TypeError:
            self.bus.disconnect()
        print("[leader] disconnected")


def _load_lerobot_calibration(path: Path) -> dict[str, Any]:
    from lerobot.motors import MotorCalibration

    data = json.loads(path.read_text())
    out = {}
    for name, rec in data.items():
        out[name] = MotorCalibration(
            id=int(rec["id"]),
            drive_mode=int(rec.get("drive_mode", 0)),
            homing_offset=int(rec.get("homing_offset", 0)),
            range_min=int(rec["range_min"]),
            range_max=int(rec["range_max"]),
        )
    return out


def action_to_body_deg(action: dict[str, float]) -> dict[str, float]:
    return {name: float(action[f"{name}.pos"]) for name in BODY_JOINTS}


def action_gripper_pct(action: dict[str, float]) -> float:
    return float(np.clip(action.get(f"{GRIPPER_JOINT}.pos", 0.0), 0.0, 100.0))


def open_leader(
    *,
    sim: bool,
    port: str | None,
    calibration: str | None,
    sim_motion: bool = True,
) -> BaseLeader:
    if sim:
        print("[leader] --sim-leader dummy joints (no USB writes)")
        return SimLeader(motion=sim_motion)
    p = port or default_leader_port()
    return HardwareLeader(port=p, calibration_path=calibration)
