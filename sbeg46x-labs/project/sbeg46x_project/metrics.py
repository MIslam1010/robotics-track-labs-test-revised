"""Metrics used by the acceptance tests and the report."""
from __future__ import annotations

import numpy as np


def rms(x, axis=0):
    x = np.asarray(x, float)
    return np.sqrt(np.mean(x**2, axis=axis))


def window(t, t0, t1=np.inf):
    t = np.asarray(t, float)
    return (t >= t0) & (t < t1)


def max_abs(x, mask=None) -> float:
    x = np.abs(np.asarray(x, float))
    return float(np.max(x if mask is None else x[mask])) if (mask is None or np.any(mask)) else 0.0


def finite_diff(t, x) -> np.ndarray:
    return np.gradient(np.asarray(x, float), np.asarray(t, float))


def wrap_phase(p):
    """Wrap a phase given as a fraction of the cycle to [-0.5, 0.5)."""
    return (np.asarray(p, float) + 0.5) % 1.0 - 0.5
