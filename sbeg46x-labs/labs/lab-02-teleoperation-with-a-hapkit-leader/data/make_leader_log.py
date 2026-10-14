"""Regenerate the SYNTHETIC Hapkit leader log used as the simulation fallback of Lab 2.

leader_log.csv: t (s), x_m (m, handle displacement), clutch (1 = clutch pedal pressed = follower decoupled).
It imitates a laparoscopic insertion with indexing: three forward strokes of the handle, two clutched returns,
a 9 Hz physiological-tremor component (0.3 mm amplitude) and 0.02 mm sensor noise.  The Hapkit station writes
the same three columns, so your own recording can replace this file.
Run:  python make_leader_log.py   (deterministic, seed 2)
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DT, T_END = 0.001, 12.0


def minjerk(t, t0, t1, x0, x1):
    s = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return x0 + (x1 - x0) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def voluntary(t):
    """Piecewise minimum-jerk handle motion (m)."""
    x = np.full_like(t, -0.030)
    segs = [(0.5, 3.5, -0.030, 0.030), (4.0, 5.0, 0.030, -0.030), (5.5, 8.5, -0.030, 0.030),
            (9.0, 10.0, 0.030, -0.030), (10.5, 11.5, -0.030, 0.000)]
    for t0, t1, x0, x1 in segs:
        m = t >= t0
        x[m] = minjerk(t[m], t0, t1, x0, x1)
    return x


def clutch(t):
    return (((t >= 3.8) & (t < 5.2)) | ((t >= 8.8) & (t < 10.2))).astype(int)


def make():
    t = np.round(np.arange(0.0, T_END + DT / 2, DT), 6)
    rng = np.random.default_rng(2)
    x = voluntary(t) + 0.0003 * np.sin(2 * np.pi * 9.0 * t) + 0.00002 * rng.standard_normal(t.size)
    data = np.column_stack([t, x, clutch(t)])
    np.savetxt(HERE / "leader_log.csv", data, delimiter=",", header="t,x_m,clutch", comments="",
               fmt=["%.3f", "%.7f", "%d"])
    return data


if __name__ == "__main__":
    d = make()
    print("wrote leader_log.csv", d.shape)
