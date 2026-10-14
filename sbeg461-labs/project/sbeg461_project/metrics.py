"""Metrics used in the acceptance tests and the report."""
from __future__ import annotations

import numpy as np


def rms(x, axis=0):
    x = np.asarray(x, float)
    return np.sqrt(np.mean(x**2, axis=axis))


def tracking_metrics(sim: dict, ref, skip: float = 0.0) -> dict:
    """Joint tracking metrics after time `skip` (s): RMS and max error per joint (rad),
    peak |tau| per joint (N m), saturation fraction, control effort integral of tau^T tau dt."""
    t = sim["t"]
    m = t >= skip
    qd = np.array([ref(tt)[0] for tt in t[m]])
    e = qd - sim["q"][m]
    dt = t[1] - t[0]
    return dict(rms_err=rms(e), max_err=np.max(np.abs(e), axis=0),
                peak_tau=np.max(np.abs(sim["tau"][m]), axis=0),
                sat_frac=float(np.mean(sim["sat"][m])),
                effort=float(np.sum(sim["tau"][m] ** 2) * dt))


def rad2deg(x):
    return np.degrees(x)
