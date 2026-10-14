"""Helpers for the required hardware run: read a logged CSV, align it with a simulation, and compute
the sim-vs-hardware metrics used in the hardware acceptance tests HWk-1 ... HWk-4.

Log format (one row per sample, header required, SI units: s, rad or m, A or N):

    t, cmd_1, ..., cmd_n, meas_1, ..., meas_n [, cur_1, ..., cur_n] [, force]

* ``cmd_j``  -- commanded joint position (arm) or handle position (Hapkit), as sent to the device
* ``meas_j`` -- measured position (servo encoder or Hapkit sensor)
* ``cur_j``  -- optional measured servo current (A), or commanded Hapkit motor output
* ``force``  -- optional estimated handle force (N, Hapkit)

The device clock does not tick at exactly 1 kHz, so every comparison first resamples the log onto
the simulation time grid by linear interpolation (``resample``). All thresholds are illustrative.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .metrics import rms


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


def estimate_delay(t, a, b, max_delay: float = 0.2) -> float:
    """Time delay (s) of signal b relative to a (b(t) ~ a(t - d)), from the peak of the normalised
    cross-correlation on a uniform grid, searched within +-max_delay. Positive d means b lags a."""
    t = np.asarray(t, float)
    a = np.asarray(a, float).ravel() - np.mean(a)
    b = np.asarray(b, float).ravel() - np.mean(b)
    dt = float(np.median(np.diff(t)))
    kmax = int(round(max_delay / dt))
    lags = np.arange(-kmax, kmax + 1)
    n = len(a)
    score = []
    for k in lags:  # normalised (Pearson) correlation over the overlapping samples only
        x, y = a[max(0, -k):n - max(0, k)], b[max(0, k):n - max(0, -k)]
        score.append(np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else -np.inf)
    return float(lags[int(np.argmax(score))] * dt)


def compare(log: dict, sim: dict | None = None, *, skip: float = 0.0,
            align_delay: bool = False) -> dict:
    """Sim-vs-hardware metrics after time ``skip`` (s).

    Always: ``rms_cmd_meas`` and ``max_cmd_meas`` (per joint) -- tracking error of the device,
    and ``delay`` -- lag of meas behind cmd (s).
    With ``sim`` (a ``dyn.simulate`` result, or any dict with ``t`` and ``q``): the log is
    resampled onto the simulation grid and ``rms_sim_meas``, ``max_sim_meas`` (per joint) and
    ``rms_cmd_sim`` (the simulated tracking error, for the same commanded signal) are added.
    ``align_delay=True`` shifts the measured signal by the estimated delay before comparing
    (report both numbers if you use it, and say why)."""
    t, cmd, meas = log["t"], log["cmd"], log["meas"]
    m = t >= skip
    e = cmd[m] - meas[m]
    out = dict(rms_cmd_meas=rms(e), max_cmd_meas=np.max(np.abs(e), axis=0),
               delay=estimate_delay(t[m], cmd[m, 0], meas[m, 0]))
    if sim is None:
        return out
    ts = np.asarray(sim["t"], float)
    shift = out["delay"] if align_delay else 0.0
    meas_s = resample(t - shift, meas, ts)
    cmd_s = resample(t, cmd, ts)
    q_sim = np.asarray(sim["q"], float).reshape(len(ts), -1)
    ms = (ts >= skip) & np.all(np.isfinite(meas_s), axis=1) & np.all(np.isfinite(cmd_s), axis=1)
    if not np.any(ms):
        raise ValueError("log and simulation do not overlap in time")
    d = q_sim[ms] - meas_s[ms]
    out.update(rms_sim_meas=rms(d), max_sim_meas=np.max(np.abs(d), axis=0),
               rms_cmd_sim=rms(cmd_s[ms] - q_sim[ms]), n_samples=int(np.sum(ms)))
    return out


def normalised_rms(predicted, measured) -> float:
    """RMS(predicted - measured) / RMS(measured): e.g. model-predicted gravity torque against
    torque estimated from servo current (HW2-2, HW4-2, HW5-2). 0 is perfect; 1 means the
    prediction is no better than predicting zero."""
    p = np.asarray(predicted, float).ravel()
    y = np.asarray(measured, float).ravel()
    return float(rms(p - y) / rms(y))


def torque_constant(current, tau_model) -> float:
    """Least-squares torque-per-ampere k (N m/A, gear ratio included) with tau_model ~ k * current,
    from static holding poses. Approach every pose from both sides and average the two currents
    so that static friction cancels to first order."""
    i = np.asarray(current, float).ravel()
    tau = np.asarray(tau_model, float).ravel()
    return float(np.dot(i, tau) / np.dot(i, i))


def rendered_stiffness(force, displacement) -> tuple:
    """Least-squares stiffness K (N/m) and offset from static push data f = K x + f0 (HW1-1, HW6-1).
    Returns (K, f0)."""
    x = np.asarray(displacement, float).ravel()
    f = np.asarray(force, float).ravel()
    A = np.column_stack([x, np.ones_like(x)])
    K, f0 = np.linalg.lstsq(A, f, rcond=None)[0]
    return float(K), float(f0)


def within_limits(values, limit, fraction: float = 1.0) -> bool:
    """True if |values| <= fraction * limit everywhere (per column if limit is a vector), e.g.
    servo current <= 60 % of rated current, or joint speed <= 50 % of rated speed (HWk-3)."""
    v = np.abs(np.asarray(values, float))
    return bool(np.all(v <= fraction * np.asarray(limit, float)))


def stop_time(t, qdot, t_trigger: float, threshold: float) -> float:
    """Time (s) from the stop trigger until |qdot| stays below ``threshold`` for the rest of the
    log (all joints). Returns inf if it never settles (HWk-4)."""
    t = np.asarray(t, float)
    v = np.max(np.abs(np.asarray(qdot, float).reshape(len(t), -1)), axis=1)
    after = t >= t_trigger
    above = np.nonzero(after & (v >= threshold))[0]
    if len(above) == 0:
        return 0.0
    k = above[-1] + 1
    return float(t[k] - t_trigger) if k < len(t) else float("inf")
