"""Load YAML config + CLI overrides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_PKG_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TeleopConfig:
    leader_port: str
    scale: float = 2.0
    fps: float = 50.0
    latch_on_start: bool = False
    wrist_roll_offset_deg: float = 0.0
    position_cost: float = 50.0
    orientation_cost: float = 0.15
    posture_cost: float = 0.02
    damping_cost: float = 0.2
    max_ee_step_m: float = 0.008
    max_ee_step_rad: float = 0.06
    ik_iters: int = 8
    ik_dt: float = 0.005
    ik_damping: float = 0.08
    joint_lpf_hz: float = 8.0
    deadband_m: float = 0.003
    deadband_rad: float = 0.03
    so101_align: str = "ry-90"
    openarm_align: str = "ry-90"
    arm_side: str = "right"
    ee_frame: str = "openarm_right_link7"
    ee_frame_type: str = "body"
    tcp_offset_m: float = 0.2343
    gripper_open_m: float = 0.044
    openarm_mjcf: str = "/home/vector/work/openarm_mujoco/openarm_mujoco/v1/scene.xml"
    openarm_description_root: str = "/home/vector/work/openarm_teleop/openarm_description"
    so101_urdf: str = "assets/so101/so101_new_calib.urdf"
    so101_ee_link: str = "gripper_frame_link"

    def resolve(self, root: Path | None = None) -> "TeleopConfig":
        root = root or _PKG_ROOT
        self.openarm_mjcf = str((root / self.openarm_mjcf).resolve()) if not Path(self.openarm_mjcf).is_absolute() else self.openarm_mjcf
        self.openarm_description_root = (
            str((root / self.openarm_description_root).resolve())
            if not Path(self.openarm_description_root).is_absolute()
            else self.openarm_description_root
        )
        self.so101_urdf = str((root / self.so101_urdf).resolve()) if not Path(self.so101_urdf).is_absolute() else self.so101_urdf
        return self


def load_config(path: str | Path | None = None) -> TeleopConfig:
    cfg_path = Path(path) if path else _PKG_ROOT / "config" / "default.yaml"
    raw: dict = {}
    if cfg_path.is_file():
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    known = {f.name for f in TeleopConfig.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in raw.items() if k in known}
    if "leader_port" not in kwargs:
        kwargs["leader_port"] = "/dev/ttyACM0"
    return TeleopConfig(**kwargs)
