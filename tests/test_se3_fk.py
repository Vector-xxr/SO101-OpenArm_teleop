"""Unit tests that do not need USB or a MuJoCo viewer."""

from pathlib import Path

import numpy as np

from so101_umi.frames import R_ALIGN_RY_NEG90, describe_axes
from so101_umi.filter import JointEMA
from so101_umi.se3 import clip_step, hold_if_small, make_T, minus, overlay_relative, plus
from so101_umi.so101_fk import SO101FK

ROOT = Path(__file__).resolve().parents[1]


def test_relative_zero():
    T0 = make_T(np.eye(3), np.array([0.1, 0.2, 0.3]))
    T_rel = minus(T0, T0)
    np.testing.assert_allclose(T_rel, np.eye(4), atol=1e-9)
    T_cmd = plus(T0, T_rel, scale=2.0)
    np.testing.assert_allclose(T_cmd, T0, atol=1e-9)


def test_relative_translation_scale():
    T0 = make_T(np.eye(3), np.zeros(3))
    T1 = make_T(np.eye(3), np.array([0.10, 0, 0]))
    home = make_T(np.eye(3), np.array([0.3, 0, 0.4]))
    _, T_cmd = overlay_relative(T1, T0, home, scale=2.0)
    np.testing.assert_allclose(T_cmd[:3, 3], np.array([0.5, 0, 0.4]), atol=1e-9)


def test_r_align_columns():
    # T_aligned = T_native @ R_align. Native +Z is aligned +X.
    z_n = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(R_ALIGN_RY_NEG90.T @ z_n, np.array([1.0, 0.0, 0.0]), atol=1e-9)
    y_n = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(R_ALIGN_RY_NEG90.T @ y_n, np.array([0.0, 1.0, 0.0]), atol=1e-9)


def test_so101_fk_zero_finite():
    urdf = ROOT / "assets/so101/so101_new_calib.urdf"
    fk = SO101FK(urdf)
    T = fk.forward({k: 0.0 for k in (
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"
    )})
    p = T[:3, 3]
    assert np.isfinite(p).all()
    assert 0.05 < np.linalg.norm(p) < 0.5
    print("zero-config p", p, describe_axes(T[:3, :3]))


def test_clip_step():
    T0 = np.eye(4)
    T1 = make_T(np.eye(3), np.array([1.0, 0, 0]))
    Tc = clip_step(T0, T1, max_pos=0.02, max_rot=0.1)
    np.testing.assert_allclose(Tc[:3, 3], np.array([0.02, 0, 0]), atol=1e-9)


def test_hold_if_small():
    T0 = make_T(np.eye(3), np.array([0.3, 0.1, 0.4]))
    T1 = make_T(np.eye(3), np.array([0.301, 0.1, 0.4]))
    held = hold_if_small(T0, T1, deadband_m=0.003, deadband_rad=0.03)
    np.testing.assert_allclose(held, T0)
    T2 = make_T(np.eye(3), np.array([0.32, 0.1, 0.4]))
    moved = hold_if_small(T0, T2, deadband_m=0.003, deadband_rad=0.03)
    np.testing.assert_allclose(moved, T2)


def test_joint_ema_moves_toward_target():
    ema = JointEMA(cutoff_hz=8.0, fps=50.0, dim=2)
    y0 = ema.step(np.array([0.0, 0.0]))
    y1 = ema.step(np.array([10.0, 0.0]))
    assert y1[0] > y0[0]
    assert y1[0] < 10.0


def test_clip_rotation_does_not_drift():
    """Regression: feedback through clip_step must stay on SO(3)."""
    a = 0.8
    R_target = np.array(
        [
            [np.cos(a), -np.sin(a), 0.0],
            [np.sin(a), np.cos(a), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    prev = np.eye(4)
    target = make_T(R_target, np.zeros(3))
    for _ in range(5000):
        prev = clip_step(prev, target, max_pos=0.01, max_rot=0.06)
    np.testing.assert_allclose(prev[:3, :3].T @ prev[:3, :3], np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(prev[:3, :3]), 1.0, atol=1e-10)
    assert np.isfinite(prev).all()


if __name__ == "__main__":
    test_relative_zero()
    test_relative_translation_scale()
    test_r_align_columns()
    test_so101_fk_zero_finite()
    test_clip_step()
    test_hold_if_small()
    test_joint_ema_moves_toward_target()
    test_clip_rotation_does_not_drift()
    print("ok")
