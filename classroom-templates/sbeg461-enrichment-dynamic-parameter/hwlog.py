"""Hardware-log helpers: identical to ``write_log``, ``load_log``, ``resample`` and ``normalised_rms`` of the
SBEG 461 project package (``sbeg461_project/hardware.py``), copied so that this folder is self-contained.

Log format (one row per sample, header required, SI units: s, rad, A):

    t, cmd_1, ..., cmd_n, meas_1, ..., meas_n [, cur_1, ..., cur_n] [, force]

* ``cmd_j``  -- commanded joint position, as sent to the servo
* ``meas_j`` -- measured joint position (servo encoder)
* ``cur_j``  -- measured servo current (A); joint torque = k_j * cur_j with the torque constant k_j
               (N m/A, gear ratio included) from the robot's commissioning sheet

The device clock does not tick at exactly the nominal rate (jitter, an occasional repeated time stamp), so
logs are always resampled onto a uniform grid before they are filtered or differentiated.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

def write_log(path, t, cmd, meas, cur=None, force=None) -> None:
    """Write a log in the format above (used for tests and for logging simulated runs)."""
    t = np.asarray(t, float)
    cmd = np.asarray(cmd, float).reshape(len(t), -1)
    meas = np.asarray(meas, float).reshape(len(t), -1)
    n = cmd.shape[1]
    cols = [t[:, None], cmd, meas]
    header = ["t"] + [f"cmd_{j + 1}" for j in range(n)] + [f"meas_{j + 1}" for j in range(n)]
    if cur is not None:
        cols.append(np.asarray(cur, float).reshape(len(t), -1))
        header += [f"cur_{j + 1}" for j in range(n)]
    if force is not None:
        cols.append(np.asarray(force, float).reshape(len(t), 1))
        header += ["force"]
    data = np.hstack(cols)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(data.tolist())


def load_log(path) -> dict:
    """Read a hardware log. Returns a dict with ``t`` (N,), ``cmd`` and ``meas`` (N, n), and
    ``cur`` (N, n) and ``force`` (N,) when present. Rows are sorted by time; duplicate time
    stamps are dropped (they break the interpolation)."""
    path = Path(path)
    with open(path, newline="") as fh:
        header = [h.strip() for h in next(csv.reader(fh))]
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    col = {h: i for i, h in enumerate(header)}
    if "t" not in col:
        raise ValueError(f"{path}: column 't' missing")
    order = np.argsort(data[:, col["t"]], kind="stable")
    data = data[order]
    _, keep = np.unique(data[:, col["t"]], return_index=True)
    data = data[np.sort(keep)]

    def group(prefix):
        names = sorted((h for h in header if h.startswith(prefix + "_")),
                       key=lambda h: int(h.split("_")[1]))
        return data[:, [col[h] for h in names]] if names else None

    out = dict(t=data[:, col["t"]], cmd=group("cmd"), meas=group("meas"))
    if out["cmd"] is None or out["meas"] is None or out["cmd"].shape != out["meas"].shape:
        raise ValueError(f"{path}: need matching cmd_j and meas_j columns")
    cur = group("cur")
    if cur is not None:
        out["cur"] = cur
    if "force" in col:
        out["force"] = data[:, col["force"]]
    return out


def resample(t_src, x_src, t_dst) -> np.ndarray:
    """Linear interpolation of x_src(t_src) (shape (N,) or (N, n)) onto t_dst. Points of t_dst
    outside [t_src[0], t_src[-1]] are returned as NaN, so they cannot hide in an RMS value."""
    t_src = np.asarray(t_src, float)
    x = np.asarray(x_src, float)
    t_dst = np.asarray(t_dst, float)
    one_d = x.ndim == 1
    x = x.reshape(len(t_src), -1)
    y = np.column_stack([np.interp(t_dst, t_src, x[:, j]) for j in range(x.shape[1])])
    y[(t_dst < t_src[0]) | (t_dst > t_src[-1])] = np.nan
    return y[:, 0] if one_d else y


def rms(x, axis=0):
    """Root-mean-square along ``axis``."""
    return np.sqrt(np.mean(np.square(np.asarray(x, float)), axis=axis))


def normalised_rms(predicted, measured) -> float:
    """RMS(predicted - measured) / RMS(measured). 0 is perfect; 1 means the prediction is no better than
    predicting zero."""
    p = np.asarray(predicted, float).ravel()
    y = np.asarray(measured, float).ravel()
    return float(rms(p - y) / rms(y))
