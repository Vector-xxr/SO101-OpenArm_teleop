"""SE(3) helpers: relative overlay (⊖ / ⊕), scale, and R_align.

Matches umi-vista replay.py:

    T_rel = T(t) ⊖ T(0)   p: p(t)-p(0),  R: R(t) @ R(0).T
    T_cmd = T_home ⊕ T_rel  p: p_home + S * p_rel,  R: R_rel @ R_home
"""

from __future__ import annotations

import numpy as np

from .frames import rotation_from_preset, se3_from_R_p


def project_rotation(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3).

    Repeated relative-pose composition otherwise accumulates tiny scale/shear
    errors. Since clip_step feeds its output back on every control tick, those
    errors can grow until the matrix overflows.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(R).all():
        return np.eye(3, dtype=np.float64)
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0.0:
        U[:, -1] *= -1.0
        Rp = U @ Vt
    return Rp


def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = project_rotation(R)
    T[:3, 3] = np.asarray(p, dtype=np.float64).reshape(3)
    return T


def split_T(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float64)
    return T[:3, :3].copy(), T[:3, 3].copy()


def minus(T_now: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
    """T_now ⊖ T_ref in the world/base frame (pos subtract, R_now @ R_ref.T)."""
    R_n, p_n = split_T(T_now)
    R_0, p_0 = split_T(T_ref)
    return make_T(R_n @ R_0.T, p_n - p_0)


def plus(T_home: np.ndarray, T_rel: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """T_home ⊕ (S * T_rel). Scale applies to translation only."""
    R_h, p_h = split_T(T_home)
    R_r, p_r = split_T(T_rel)
    return make_T(R_r @ R_h, p_h + float(scale) * p_r)


def apply_align(T_native: np.ndarray, R_align: np.ndarray) -> np.ndarray:
    """Right-multiply a TCP-frame correction: T_aligned = T_native @ R_align."""
    return np.asarray(T_native, dtype=np.float64) @ se3_from_R_p(R_align)


def overlay_relative(
    T_now: np.ndarray,
    T_ref: np.ndarray,
    T_home: np.ndarray,
    scale: float,
    R_align: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Latch-relative overlay.

    If ``R_align`` is given it is applied to both leader poses before ⊖
    (``T_cmd = T_home ⊕ (S * R_align * T_rel)`` with R_align acting on the
    leader TCP). OpenArm home is assumed already expressed in the same
    convention as the FrameTask site (``ee_aligned``).
    """
    T_now = np.asarray(T_now, dtype=np.float64)
    T_ref = np.asarray(T_ref, dtype=np.float64)
    if R_align is not None:
        T_now = apply_align(T_now, R_align)
        T_ref = apply_align(T_ref, R_align)
    T_rel = minus(T_now, T_ref)
    T_cmd = plus(T_home, T_rel, scale=scale)
    return T_rel, T_cmd


def hold_if_small(
    T_prev: np.ndarray,
    T_next: np.ndarray,
    deadband_m: float,
    deadband_rad: float,
) -> np.ndarray:
    """Keep the previous pose when the increment is below encoder-noise scale."""
    if not np.isfinite(T_next).all():
        return np.asarray(T_prev, dtype=np.float64).copy()
    if deadband_m <= 0.0 and deadband_rad <= 0.0:
        return T_next
    R0, p0 = split_T(T_prev)
    R1, p1 = split_T(T_next)
    if not np.isfinite(R0).all() or not np.isfinite(R1).all():
        return np.asarray(T_prev, dtype=np.float64).copy()
    dp = float(np.linalg.norm(p1 - p0))
    dR = R1 @ R0.T
    if not np.isfinite(dR).all():
        return np.asarray(T_prev, dtype=np.float64).copy()
    cos_th = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
    th = float(np.arccos(cos_th))
    if dp < deadband_m and th < deadband_rad:
        return np.asarray(T_prev, dtype=np.float64).copy()
    return T_next


def clip_step(
    T_prev: np.ndarray,
    T_next: np.ndarray,
    max_pos: float,
    max_rot: float,
) -> np.ndarray:
    """Limit one control-tick of EE motion (position m, rotation rad)."""
    if not np.isfinite(T_next).all() or not np.isfinite(T_prev).all():
        return np.asarray(T_prev, dtype=np.float64).copy()
    R0, p0 = split_T(T_prev)
    R1, p1 = split_T(T_next)
    R0 = project_rotation(R0)
    R1 = project_rotation(R1)
    dp = p1 - p0
    n = float(np.linalg.norm(dp))
    if n > max_pos > 0:
        dp = dp * (max_pos / n)
    dR = R1 @ R0.T
    # axis-angle from rotation matrix
    cos_th = np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)
    th = float(np.arccos(cos_th))
    if th > max_rot > 1e-9:
        axis_hat = (dR - dR.T) / (2.0 * np.sin(th) + 1e-12)
        axis = np.array([axis_hat[2, 1], axis_hat[0, 2], axis_hat[1, 0]])
        an = float(np.linalg.norm(axis))
        if an > 1e-9:
            axis = axis / an
            k = max_rot
            K = np.array(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ]
            )
            dR = np.eye(3) + np.sin(k) * K + (1.0 - np.cos(k)) * (K @ K)
        else:
            dR = np.eye(3)
    return make_T(dR @ R0, p0 + dp)


def T_to_mink_wxyz_xyz(T: np.ndarray) -> np.ndarray:
    """MuJoCo/mink quaternion order wxyz + translation."""
    R, p = split_T(T)
    # Shepperd / standard quat from rotation (w, x, y, z)
    t = float(np.trace(R))
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return np.concatenate([q, p])


def rotation_preset(name: str) -> np.ndarray:
    return rotation_from_preset(name)
