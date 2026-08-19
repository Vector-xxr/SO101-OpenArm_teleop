"""Realtime filters for leader joints / SE3 commands."""

from __future__ import annotations

import math

import numpy as np


def ema_alpha(cutoff_hz: float, fps: float) -> float:
    """First-order EMA coefficient for a given cutoff and sample rate."""
    if cutoff_hz <= 0.0 or fps <= 0.0:
        return 1.0
    dt = 1.0 / fps
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return float(dt / (tau + dt))


class JointEMA:
    """Independent EMA on each joint (degrees / gripper percent)."""

    def __init__(self, cutoff_hz: float, fps: float, dim: int = 6):
        self.alpha = ema_alpha(cutoff_hz, fps)
        self.y: np.ndarray | None = None
        self.dim = dim

    def reset(self) -> None:
        self.y = None

    def step(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if self.y is None:
            self.y = x.copy()
            return self.y.copy()
        a = self.alpha
        self.y = self.y + a * (x - self.y)
        return self.y.copy()
