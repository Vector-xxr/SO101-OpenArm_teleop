"""SO-101 leader → relative EE → OpenArm mink IK teleop loop."""

from __future__ import annotations

import argparse
import time

import numpy as np

from .config import TeleopConfig, load_config
from .filter import JointEMA
from .frames import describe_axes, rotation_from_preset
from .leader import ALL_JOINTS, action_gripper_pct, action_to_body_deg, open_leader
from .openarm_ik import OpenArmMinkIK
from .se3 import apply_align, clip_step, hold_if_small, overlay_relative, split_T
from .serial_util import print_serial_devices
from .so101_fk import SO101FK


def busy_wait(dt: float) -> None:
    end = time.perf_counter() + dt
    while time.perf_counter() < end:
        pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SO-101 leader relative-EE teleop onto OpenArm (mink IK / MuJoCo)"
    )
    p.add_argument("--config", default=None, help="YAML config path")
    p.add_argument("--leader-port", default=None, help="Feetech USB port / by-id path")
    p.add_argument("--sim-leader", action="store_true", help="Dummy joints, no USB")
    p.add_argument("--sim-static", action="store_true", help="Dummy joints without sinusoid")
    p.add_argument("--calibration", default=None, help="LeRobot calibration JSON")
    p.add_argument("--scale", type=float, default=None, help="Workspace scale S (default 2)")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--wrist-roll-offset-deg", type=float, default=None)
    p.add_argument("--pos-cost", type=float, default=None)
    p.add_argument("--ori-cost", type=float, default=None)
    p.add_argument("--max-ee-step", type=float, default=None)
    p.add_argument("--joint-lpf-hz", type=float, default=None, help="Leader joint EMA cutoff (0=off)")
    p.add_argument("--deadband-mm", type=float, default=None, help="EE hold deadband in mm")
    p.add_argument("--so101-align", default=None, help="R_align preset for SO-101 TCP")
    p.add_argument("--ee-frame", default=None, help="OpenArm site (default ee_aligned)")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--steps", type=int, default=0, help="Finite steps then exit (0=forever)")
    p.add_argument("--no-latch", action="store_true", help="Skip latch (overlay vs identity)")
    p.add_argument("--print-frames", action="store_true")
    return p.parse_args(argv)


def _apply_cli(cfg: TeleopConfig, args: argparse.Namespace) -> TeleopConfig:
    if args.leader_port:
        cfg.leader_port = args.leader_port
    if args.scale is not None:
        cfg.scale = args.scale
    if args.fps is not None:
        cfg.fps = args.fps
    if args.wrist_roll_offset_deg is not None:
        cfg.wrist_roll_offset_deg = args.wrist_roll_offset_deg
    if args.pos_cost is not None:
        cfg.position_cost = args.pos_cost
    if args.ori_cost is not None:
        cfg.orientation_cost = args.ori_cost
    if args.max_ee_step is not None:
        cfg.max_ee_step_m = args.max_ee_step
    if args.joint_lpf_hz is not None:
        cfg.joint_lpf_hz = args.joint_lpf_hz
    if args.deadband_mm is not None:
        cfg.deadband_m = args.deadband_mm / 1000.0
    if args.so101_align:
        cfg.so101_align = args.so101_align
    if args.ee_frame:
        cfg.ee_frame = args.ee_frame
    if args.no_latch:
        cfg.latch_on_start = False
    return cfg


def _smooth_action(action: dict[str, float], ema: JointEMA) -> dict[str, float]:
    raw = np.array([float(action[f"{name}.pos"]) for name in ALL_JOINTS], dtype=np.float64)
    filt = ema.step(raw)
    return {f"{name}.pos": float(filt[i]) for i, name in enumerate(ALL_JOINTS)}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _apply_cli(load_config(args.config), args).resolve()
    dt_loop = 1.0 / max(cfg.fps, 1.0)

    print_serial_devices()
    print(f"[cfg] port={cfg.leader_port}  scale={cfg.scale}  fps={cfg.fps}")
    print(f"[cfg] so101_align={cfg.so101_align}  ee_frame={cfg.ee_frame}  ori_cost={cfg.orientation_cost}")
    print(
        f"[cfg] joint_lpf={cfg.joint_lpf_hz}Hz  deadband={cfg.deadband_m*1000:.1f}mm/"
        f"{np.rad2deg(cfg.deadband_rad):.1f}deg  damping={cfg.damping_cost}"
    )
    if not args.sim_leader:
        print(
            "[cfg] two USB buses: ACM1/5C82107971 is the leader (default); "
            "ACM0/5C82107039 is the parked follower. Frozen MuJoCo → wrong port."
        )

    R_align = rotation_from_preset(cfg.so101_align)
    fk = SO101FK(cfg.so101_urdf, ee_link=cfg.so101_ee_link)
    ik = OpenArmMinkIK(
        cfg.openarm_mjcf,
        ee_frame=cfg.ee_frame,
        position_cost=cfg.position_cost,
        orientation_cost=cfg.orientation_cost,
        posture_cost=cfg.posture_cost,
        damping_cost=cfg.damping_cost,
        ik_dt=cfg.ik_dt,
        ik_iters=cfg.ik_iters,
        ik_damping=cfg.ik_damping,
    )

    T_home = ik.ee_pose()
    print(f"[openarm] home TCP pos={np.round(T_home[:3, 3], 4)}  {describe_axes(T_home[:3, :3])}")

    leader = open_leader(
        sim=args.sim_leader,
        port=cfg.leader_port,
        calibration=args.calibration,
        sim_motion=not args.sim_static,
    )

    T_so101_ref = None
    T_cmd_prev = T_home.copy()
    viewer = None
    n = 0
    q0: np.ndarray | None = None
    joint_ema = JointEMA(cfg.joint_lpf_hz, cfg.fps, dim=len(ALL_JOINTS))

    try:
        action = _smooth_action(leader.get_action(), joint_ema)
        T_native = fk.forward(action_to_body_deg(action), cfg.wrist_roll_offset_deg)
        T_aligned0 = apply_align(T_native, R_align)
        Rn, _ = split_T(T_native)
        Ra, _ = split_T(T_aligned0)
        print(f"[so101] native TCP  pos={np.round(T_native[:3, 3], 4)}  {describe_axes(Rn)}")
        print(f"[so101] aligned TCP pos={np.round(T_aligned0[:3, 3], 4)}  {describe_axes(Ra)}")
        print("[so101] R_align preset", cfg.so101_align, "→ +X forward, +Y left, +Z up")

        if cfg.latch_on_start:
            T_so101_ref = T_native.copy()
            print("[latch] stored T_so101(0) and OpenArm home TCP")
        else:
            T_so101_ref = np.eye(4)

        if not args.no_viewer:
            try:
                import mujoco.viewer

                viewer = mujoco.viewer.launch_passive(ik.model, ik.data)
            except Exception as e:
                print(f"[viewer] disabled ({e})")
                viewer = None

        while True:
            t0 = time.perf_counter()
            action = _smooth_action(leader.get_action(), joint_ema)
            T_now = fk.forward(action_to_body_deg(action), cfg.wrist_roll_offset_deg)
            _, T_cmd = overlay_relative(
                T_now, T_so101_ref, T_home, scale=cfg.scale, R_align=R_align
            )
            T_cmd = hold_if_small(T_cmd_prev, T_cmd, cfg.deadband_m, cfg.deadband_rad)
            T_cmd = clip_step(T_cmd_prev, T_cmd, cfg.max_ee_step_m, cfg.max_ee_step_rad)
            err = ik.solve(T_cmd)
            ik.set_gripper(action_gripper_pct(action) / 100.0, cfg.gripper_open_m)
            ik.push_ctrl()
            T_cmd_prev = T_cmd
            n += 1
            q_body = np.array(list(action_to_body_deg(action).values()), dtype=np.float64)
            if q0 is None:
                q0 = q_body.copy()
            q_max_delta = float(np.max(np.abs(q_body - q0)))
            if n % int(max(cfg.fps, 1)) == 0:
                pos_e = ik.pos_err(T_cmd)
                g = action_gripper_pct(action)
                p_now = T_now[:3, 3]
                print(
                    f"[loop] n={n} ee={np.round(T_cmd[:3, 3], 3)} "
                    f"so101={np.round(p_now, 3)} "
                    f"qdeg={np.round(q_body, 1)} "
                    f"dq={q_max_delta:.1f}deg "
                    f"pos_err={pos_e*1000:.1f}mm gripper={g:.0f} "
                    f"dt={(time.perf_counter()-t0)*1e3:.1f}ms"
                )
                if n >= int(cfg.fps) * 4 and q_max_delta < 2.0 and not args.sim_leader:
                    print(
                        "[hint] SO-101 joints barely moved — this port is likely the parked follower. "
                        "Ctrl+C and pass the other --leader-port "
                        "(try ...5C82107971... / /dev/ttyACM1)."
                    )
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
            if args.steps and n >= args.steps:
                print(f"[done] {n} steps")
                break
            busy_wait(max(0.0, dt_loop - (time.perf_counter() - t0)))
    finally:
        if viewer is not None:
            viewer.close()
        leader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
