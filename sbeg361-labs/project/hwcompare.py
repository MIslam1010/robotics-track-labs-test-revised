"""hwcompare.py - compare a hardware joint log from the course arm with simulation.

Required hardware run (project brief Sec. 4.1, scenario sheets "Hardware task").
During each supervised session log, at a fixed rate (>= 20 Hz), the joint angles you
COMMAND (from your simulation / resolved-rate code) and the joint angles the servos
REPORT, in a CSV file with the header

    t, cmd_1, ..., cmd_n, meas_1, ..., meas_n[, stop]

t in seconds, angles in DEGREES (what the servo SDKs report), `stop` = 0 before and 1
from the instant the stop / e-stop command is given (optional column, needed for HW-6).

This module computes the common hardware acceptance tests HW-2 ... HW-6:
  HW-2  joint tracking   RMS / max |cmd - meas| per joint
  HW-3  task tracking    tool-point error |p(cmd) - p(meas)| via PoE FK (MR Eq. 4.14)
  HW-4  joint-limit margin of the MEASURED joints
  HW-5  peak MEASURED joint speed vs. the allowed (reduced) speed: <= 25 % of rated
        in session H1, <= 50 % of rated otherwise (safety checklist rule 3)
  HW-6  safe stop: time to halt and joint motion after the stop command
Thresholds are the defaults of the brief; the instructor may adjust them after the arms
are commissioned (announced by Week 10).

Command line:
    python hwcompare.py log.csv                     # joint-space tests
    python hwcompare.py log.csv --model desktop6r   # + task-space test with a PoE model
    python hwcompare.py log.csv --rated 160 160 160 160 160 160 --session H1   # HW-5 at 25 %
    python hwcompare.py --demo demo_log.csv         # write a synthetic example log
Use your own course-arm PoE model (Lab 5) in Python:  hw_report(log, Slist=..., M=...).
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from kintools import fk

DEFAULTS = dict(joint_rms_deg=2.0, joint_max_deg=5.0, task_rms_mm=5.0, task_max_mm=10.0,
                limit_margin_deg=10.0, speed_fraction=0.5, stop_time_s=0.5,
                stop_motion_deg=5.0, still_deg_s=1.0)

# Allowed fraction of the rated joint speed per project session (safety checklist rule 3,
# "Course resources: Course robot models and hardware safety checklist"): <= 25 % throughout
# session H1 (and on the first run of any new motion), <= 50 % otherwise.
SESSION_SPEED_FRACTION = {"H1": 0.25, "H2": 0.5}
DEFAULT_SESSION = "H2"


def speed_fraction(session=DEFAULT_SESSION) -> float:
    """Allowed fraction of the rated speed for HW-5 in `session` ('H1' or 'H2')."""
    try:
        return SESSION_SPEED_FRACTION[str(session).upper()]
    except KeyError:
        raise ValueError(f"unknown session {session!r}; use 'H1' or 'H2'") from None


# ----------------------------------------------------------------------------
# Log input / output
# ----------------------------------------------------------------------------

def write_log(path, t, cmd, meas, stop=None):
    """Write a joint log (angles in degrees) in the format described above."""
    t = np.asarray(t, float)
    cmd, meas = np.atleast_2d(cmd), np.atleast_2d(meas)
    n = cmd.shape[1]
    header = ["t"] + [f"cmd_{i+1}" for i in range(n)] + [f"meas_{i+1}" for i in range(n)]
    if stop is not None:
        header.append("stop")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for k in range(len(t)):
            row = [f"{t[k]:.4f}"] + [f"{x:.4f}" for x in cmd[k]] + [f"{x:.4f}" for x in meas[k]]
            if stop is not None:
                row.append(int(stop[k]))
            w.writerow(row)


def read_log(path) -> dict:
    """Read a joint log. Returns dict t [N], cmd [N x n], meas [N x n] (deg), stop [N] or None."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    data = np.array([[float(x) for x in r] for r in rows[1:] if r], float)
    col = {h: i for i, h in enumerate(header)}
    cmd_cols = sorted((h for h in header if h.startswith("cmd_")), key=lambda h: int(h[4:]))
    meas_cols = sorted((h for h in header if h.startswith("meas_")), key=lambda h: int(h[5:]))
    if len(cmd_cols) != len(meas_cols) or not cmd_cols:
        raise ValueError("log needs matching cmd_i and meas_i columns")
    return dict(t=data[:, col["t"]],
                cmd=data[:, [col[h] for h in cmd_cols]],
                meas=data[:, [col[h] for h in meas_cols]],
                stop=data[:, col["stop"]] if "stop" in col else None)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def joint_errors(cmd, meas):
    """Per-joint RMS and maximum absolute error (same units as the inputs)."""
    e = np.asarray(meas, float) - np.asarray(cmd, float)
    return np.sqrt(np.mean(e ** 2, axis=0)), np.max(np.abs(e), axis=0)


def task_errors(Slist, M, cmd_deg, meas_deg, p_b=(0, 0, 0)):
    """Tool-point position error |p(cmd) - p(meas)| [m] along the log, using space-form
    PoE FK (MR Eq. 4.14). p_b is the tool point in {b}. Returns (rms, max, errors)."""
    pb = np.r_[np.asarray(p_b, float), 1.0]
    err = []
    for qc, qm in zip(np.deg2rad(cmd_deg), np.deg2rad(meas_deg)):
        pc = (fk(Slist, M, qc) @ pb)[:3]
        pm = (fk(Slist, M, qm) @ pb)[:3]
        err.append(np.linalg.norm(pc - pm))
    err = np.array(err)
    return float(np.sqrt(np.mean(err ** 2))), float(err.max()), err


def limit_margins(meas_deg, limits_deg):
    """Smallest distance [deg] of each measured joint from its lower/upper limit."""
    lim = np.asarray(limits_deg, float)
    m = np.asarray(meas_deg, float)
    return np.minimum(m.min(axis=0) - lim[:, 0], lim[:, 1] - m.max(axis=0))


def joint_speeds(t, q):
    """Finite-difference joint speeds (units of q per second)."""
    return np.gradient(np.asarray(q, float), np.asarray(t, float), axis=0)


def stop_check(t, meas_deg, stop, still_deg_s=DEFAULTS["still_deg_s"]):
    """Safe-stop test. Returns (time from stop command until every joint speed stays
    below `still_deg_s`, maximum joint motion after the command [deg])."""
    t = np.asarray(t, float)
    stop = np.asarray(stop) > 0.5
    if not stop.any():
        raise ValueError("no stop command in the log")
    k0 = int(np.argmax(stop))
    qd = np.abs(joint_speeds(t, meas_deg)).max(axis=1)
    moving = np.nonzero(qd[k0:] > still_deg_s)[0]
    t_halt = 0.0 if moving.size == 0 else t[k0 + moving[-1]] - t[k0]
    if moving.size and k0 + moving[-1] == len(t) - 1:
        t_halt = np.inf                      # still moving at the end of the log
    drift = np.max(np.abs(np.asarray(meas_deg)[k0:] - np.asarray(meas_deg)[k0]))
    return float(t_halt), float(drift)


def hw_report(log, limits_deg=None, rated_speed_deg_s=None, Slist=None, M=None,
              p_b=(0, 0, 0), thresholds=None, session=DEFAULT_SESSION) -> dict:
    """Common hardware acceptance tests. Returns {test_id: (value, passed)};
    tests whose inputs are not given are left out. `session` ('H1' or 'H2') sets the
    HW-5 speed limit (25 % of rated in H1, 50 % otherwise); an explicit
    thresholds['speed_fraction'] overrides it."""
    th = dict(DEFAULTS, speed_fraction=speed_fraction(session), **(thresholds or {}))
    t, cmd, meas, stop = log["t"], log["cmd"], log["meas"], log.get("stop")
    if stop is not None and np.any(stop > 0.5):
        k0 = int(np.argmax(stop > 0.5))      # tracking is judged before the stop command
    else:
        k0 = len(t)
    rms, mx = joint_errors(cmd[:k0], meas[:k0])
    res = {"HW-2 joint RMS [deg]": (float(rms.max()), bool(rms.max() <= th["joint_rms_deg"])),
           "HW-2 joint max [deg]": (float(mx.max()), bool(mx.max() <= th["joint_max_deg"]))}
    if Slist is not None and M is not None:
        trms, tmax, _ = task_errors(Slist, M, cmd[:k0], meas[:k0], p_b)
        res["HW-3 task RMS [mm]"] = (1e3 * trms, bool(1e3 * trms <= th["task_rms_mm"]))
        res["HW-3 task max [mm]"] = (1e3 * tmax, bool(1e3 * tmax <= th["task_max_mm"]))
    if limits_deg is not None:
        marg = float(limit_margins(meas, limits_deg).min())
        res["HW-4 limit margin [deg]"] = (marg, bool(marg >= th["limit_margin_deg"]))
    if rated_speed_deg_s is not None:
        ratio = np.abs(joint_speeds(t, meas)).max(axis=0) / np.asarray(rated_speed_deg_s, float)
        res["HW-5 speed / rated"] = (float(ratio.max()), bool(ratio.max() <= th["speed_fraction"]))
    if stop is not None and np.any(stop > 0.5):
        t_halt, drift = stop_check(t, meas, stop, th["still_deg_s"])
        res["HW-6 halt time [s]"] = (t_halt, bool(t_halt <= th["stop_time_s"]))
        res["HW-6 motion after stop [deg]"] = (drift, bool(drift <= th["stop_motion_deg"]))
    return res


# ----------------------------------------------------------------------------
# Synthetic example (for testing the pipeline before the first session)
# ----------------------------------------------------------------------------

def synthetic_log(n=6, T=10.0, rate=50.0, lag_s=0.04, noise_deg=0.3, t_stop=8.0, seed=0):
    """A smooth commanded motion and a 'measured' copy with servo lag, noise and a stop
    command at t_stop after which the measured joints decelerate to rest."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T, 1.0 / rate)
    amp = np.linspace(20, 10, n)
    cmd = amp * np.sin(2 * np.pi * t[:, None] / T)
    meas = amp * np.sin(2 * np.pi * (t[:, None] - lag_s) / T) + rng.normal(0, noise_deg, cmd.shape)
    stop = (t >= t_stop).astype(int)
    k0 = int(np.argmax(stop))
    tau = 0.08                                     # deceleration time constant [s]
    v0 = joint_speeds(t, meas[:, :])[k0 - 1]
    dt = t[k0:] - t[k0]
    meas[k0:] = meas[k0 - 1] + v0 * tau * (1 - np.exp(-dt / tau))[:, None]
    cmd[k0:] = cmd[k0 - 1]
    return dict(t=t, cmd=cmd, meas=meas, stop=stop)


def _print(res):
    for k, (v, ok) in res.items():
        print(f"  {k:32s} {v:10.3f}   {'PASS' if ok else 'FAIL'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", help="CSV joint log (or output file with --demo)")
    ap.add_argument("--demo", action="store_true", help="write a synthetic example log")
    ap.add_argument("--limits", type=float, nargs="+", metavar="DEG",
                    help="joint limits lo1 hi1 lo2 hi2 ... in degrees")
    ap.add_argument("--rated", type=float, nargs="+", metavar="DEG_S",
                    help="rated joint speeds in deg/s (one per joint)")
    ap.add_argument("--session", choices=["H1", "H2"], default=DEFAULT_SESSION,
                    help="project session: HW-5 allows 25 %% of rated speed in H1, "
                         "50 %% otherwise (default %(default)s)")
    ap.add_argument("--model", choices=["desktop6r"],
                    help="PoE model for the task-space test (replace by your course-arm model)")
    a = ap.parse_args(argv)
    if a.demo:
        lg = synthetic_log()
        write_log(a.log, lg["t"], lg["cmd"], lg["meas"], lg["stop"])
        print(f"wrote {a.log}")
        return 0
    lg = read_log(a.log)
    lim = None if a.limits is None else np.reshape(a.limits, (-1, 2))
    Slist = M = None
    if a.model == "desktop6r":
        from scenarios import desktop_6r
        r = desktop_6r(tool=0.0)
        Slist, M = r.Slist, r.M
    res = hw_report(lg, limits_deg=lim, rated_speed_deg_s=a.rated, Slist=Slist, M=M,
                    session=a.session)
    print(f"{a.log}: {lg['cmd'].shape[1]} joints, {len(lg['t'])} samples, session {a.session} "
          f"(HW-5 limit {100 * speed_fraction(a.session):.0f} % of rated)")
    _print(res)
    return 0 if all(ok for _, ok in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
