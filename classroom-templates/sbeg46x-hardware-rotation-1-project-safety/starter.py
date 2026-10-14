"""SBEG 46X Hardware rotation 1 (Week 8) -- project safety induction and a supervised teleoperation trial
on a Hapkit pair: pre-flight record, dry run, unit identification by free decay, watchdog check, trial
analysis (tracking, contact, channel energy, limits), simulation versus hardware, safe stop, IMU at rest.
STARTER CODE -- you edit this file. Write every function marked TODO (A1 ... D2); the helpers above
Part A are provided (do not change them). Run the checks with:  LAB_IMPL=starter python -m pytest -q test_lab.py
It becomes the module `hwrun` of your library sbeg46x_medrob.

Conventions (Labs 2-3 and the project starter):
  device       m_h x'' + b_h x' + k_g x = F_motor + F_ext   (x positive away from the operator, 1 kHz firmware loop)
  follower PD  F_c = Kp (x_d - x_s) - Kd v_s,   x_d = alpha x_m (one tick old: the station link has >= 1 sample delay)
  leader       F_m = beta F_c is the force of the teleoperator ON the leader; the leader motor applies -F_m
  tissue       f_env = k (x_s - x_w) + b v_s for x_s > x_w, rendered in the follower firmware (motor outputs F_c - f_env)
  channel      E_ch(t) = integral (F_m v_m - F_c v_d) dt >= 0 for a passive channel (Lab 3), v_d = dx_d/dt
Logs use the project format: t, cmd_1 (= x_d), meas_1 (= x_s), cur_1 (follower motor output), force (= F_m),
aux_x_m, aux_F_c, aux_f_env (SI units).  All parameter values are illustrative.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit  # noqa: F401  (for fit_free_decay)

from course_models import COURSE_ARM_SAFETY, HAPTIC_SAFETY, IMU, handle_model, hapkit  # noqa: F401

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# =====================================================================  provided helpers (same in starter.py)
def load_task(path=DATA / "hw1_task.json") -> dict:
    """Station settings and test definitions (data/hw1_task.json)."""
    with open(path) as fh:
        return json.load(fh)


def load_record(path=DATA / "session_record_template.json") -> dict:
    """A session record (JSON); the template is in data/."""
    with open(path) as fh:
        return json.load(fh)


def write_log(path, t, cmd, meas, cur=None, force=None, aux=None) -> None:
    """Write a log in the project format: t, cmd_1, meas_1[, cur_1][, force][, aux_<name> ...]."""
    cols = [np.asarray(t, float), np.asarray(cmd, float), np.asarray(meas, float)]
    header = ["t", "cmd_1", "meas_1"]
    if cur is not None:
        cols.append(np.asarray(cur, float))
        header.append("cur_1")
    if force is not None:
        cols.append(np.asarray(force, float))
        header.append("force")
    for name, val in (aux or {}).items():
        cols.append(np.asarray(val, float))
        header.append(f"aux_{name}")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(np.column_stack(cols).tolist())


def load_log(path) -> dict:
    """Read a log: dict with t, cmd, meas (1-D arrays), cur and force if present, and aux = {name: array}.
    Rows are sorted by time and duplicate time stamps dropped (the station clock is not exactly 1 kHz)."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = DATA / path
    with open(path, newline="") as fh:
        header = [h.strip() for h in next(csv.reader(fh))]
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    data = data[np.argsort(data[:, 0], kind="stable")]
    _, keep = np.unique(data[:, 0], return_index=True)
    data = data[np.sort(keep)]
    col = {h: i for i, h in enumerate(header)}
    out = dict(t=data[:, col["t"]], cmd=data[:, col["cmd_1"]], meas=data[:, col["meas_1"]])
    if "cur_1" in col:
        out["cur"] = data[:, col["cur_1"]]
    if "force" in col:
        out["force"] = data[:, col["force"]]
    out["aux"] = {h[4:]: data[:, col[h]] for h in header if h.startswith("aux_")}
    return out


def minjerk(t, t0, T, x0, x1):
    """Minimum-jerk move from x0 (at t0) to x1 (at t0 + T), constant outside."""
    s = np.clip((np.asarray(t, float) - t0) / T, 0.0, 1.0)
    return x0 + (x1 - x0) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def intent(task: dict, t):
    """The operator's intended leader position for the trial: reach, hold in the tissue, withdraw."""
    tr = task["trial"]
    up = minjerk(t, tr["t_on"], tr["t_reach"], tr["x_start"], tr["x_goal"])
    down = minjerk(t, tr["t_hold_end"], tr["t_back"], 0.0, tr["x_start"] - tr["x_goal"])
    return up + down


def tissue_force(x, v, env: dict) -> float:
    """Lab 2 tissue wall rendered in the follower firmware: resisting force (N, positive opposes +x)."""
    d = x - env["x_wall"]
    return float(env["k"] * d + env["b"] * v) if d > 0 else 0.0


class Device:
    """One Hapkit-type device at handle level, with the firmware output stage (2 N saturation, latched
    +-40 mm watchdog, optional fade to zero after a stop trigger) and an optional Coulomb friction f_c
    (smoothed over 1 mm/s). Same structure as the project starter's devices.Hapkit (f_c = 0 there)."""

    def __init__(self, m_h=0.05705, b_h=0.15, k_g=1.2111, T=1e-3, x_res=2e-5, force_max=2.0, x_range=0.040,
                 f_c=0.0, x=0.0, v=0.0):
        self.m_h, self.b_h, self.k_g, self.T, self.x_res = m_h, b_h, k_g, T, x_res
        self.force_max, self.x_range, self.f_c = force_max, x_range, f_c
        self.x, self.v, self.F_motor, self.tripped, self.sat_count = x, v, 0.0, False, 0

    def measure(self) -> float:
        return float(np.round(self.x / self.x_res) * self.x_res)

    def command(self, F, scale=1.0) -> float:
        if abs(self.x) > self.x_range:
            self.tripped = True
        if self.tripped:
            self.F_motor = 0.0
            return 0.0
        if abs(F) > self.force_max:
            self.sat_count += 1
        self.F_motor = float(np.clip(F, -self.force_max, self.force_max)) * scale
        return self.F_motor

    def advance(self, dt, F_ext=0.0, m_extra=0.0):
        fr = self.f_c * math.tanh(self.v / 1e-3)
        a = (self.F_motor - self.b_h * self.v - self.k_g * self.x - fr + F_ext) / (self.m_h + m_extra)
        self.v += a * dt                     # semi-implicit Euler, as in the project starter
        self.x += self.v * dt


def device_from(params: dict, **kw) -> Device:
    """A Device from a dict with keys m_h, b_h, k_g (and optionally f_c, x_res, force_max, x_range)."""
    keys = ("m_h", "b_h", "k_g", "x_res", "force_max", "x_range", "f_c")
    d = {k: params[k] for k in keys if k in params}
    d.update(kw)
    return Device(**d)


class VelocityEstimator:
    """Firmware velocity: backward difference of the quantised position, first-order low-pass at fc Hz."""

    def __init__(self, T=1e-3, fc=50.0):
        self.T, self.alpha, self.prev, self.v = T, 1.0 - math.exp(-2 * math.pi * fc * T), None, 0.0

    def __call__(self, x):
        if self.prev is not None:
            self.v += self.alpha * ((x - self.prev) / self.T - self.v)
        self.prev = x
        return self.v


def simulate_pp(task: dict, leader=None, follower=None, operator=None, T_end=None, t_stop=None, x_int=None):
    """Position-position teleoperation of a Hapkit pair WITHOUT delay (the station link adds one 1 ms sample),
    the operator's hand on the leader (spring-damper towards the intended position, its mass carried by the
    handle) and the tissue wall rendered in the follower firmware.  t_stop: stop button pressed at this time
    (both motor outputs fade linearly to zero over task['stop']['fade_s'], starting one tick later).
    Returns dict of arrays t, x_m, x_s (true), xm_meas, xs_meas, x_d, F_m (force on the leader), F_c,
    f_env, F_s (follower motor output), E_ch, and the saturation counts and watchdog flags."""
    h, sub = task["dt"], task["substeps"]
    pp, env = task["pp"], task["tissue"]
    op = operator or task["operator_nominal"]
    L = leader or device_from(task["device_nominal"])
    F = follower or device_from(task["device_nominal"])
    T_end = task["trial"]["t_end"] if T_end is None else T_end
    n = int(round(T_end / h))
    t = np.arange(n) * h
    xi = intent(task, t) if x_int is None else np.asarray(x_int, float)
    L.x = L.v = F.v = 0.0
    F.x = pp["alpha"] * L.x
    est_m, est_s = VelocityEstimator(h), VelocityEstimator(h)
    keys = "x_m x_s xm_meas xs_meas x_d F_m F_c f_env F_s E_ch".split()
    out = {k: np.zeros(n) for k in keys}
    pkt_x, pkt_F = L.measure(), 0.0          # packets in flight (one sample)
    x_d_prev, E = None, 0.0
    fade = task["stop"]["fade_s"]
    for k in range(n):
        tk = t[k]
        xm, xs = L.measure(), F.measure()
        vm_est, vs_est = est_m(xm), est_s(xs)
        f_env = tissue_force(xs, vs_est, env)
        x_d = pp["alpha"] * pkt_x                               # follower side: received leader position
        dx = 0.0 if x_d_prev is None else x_d - x_d_prev
        x_d_prev = x_d
        F_c = pp["Kp"] * (x_d - xs) - pp["Kd"] * vs_est
        F_m = pp["beta"] * pkt_F                                # leader side: received follower force
        pkt_x, pkt_F = xm, F_c                                  # send (arrive next tick)
        scale = 1.0
        if t_stop is not None and tk > t_stop:
            scale = max(0.0, 1.0 - (tk - t_stop - h) / fade) if tk - t_stop > h else 1.0
        F_m_app = -L.command(-F_m, scale)
        F.command(F_c - f_env, scale)
        E += (F_m_app * L.v - F_c * dx / h) * h
        for key, val in zip(keys, (L.x, F.x, xm, xs, x_d, F_m_app, F_c, f_env, F.F_motor, E)):
            out[key][k] = val
        for _ in range(sub):
            Fh = op["k"] * (xi[k] - L.x) - op["b"] * L.v
            L.advance(h / sub, F_ext=Fh, m_extra=op["m"])
            F.advance(h / sub)
    out.update(t=t, x_int=xi, sat_m=L.sat_count, sat_s=F.sat_count, tripped_m=L.tripped, tripped_s=F.tripped)
    return out


def simulate_follower(t_log, x_d_log, Kp, Kd, dev: Device, f_env_log=None, h=1e-3, substeps=5) -> dict:
    """Replay the FOLLOWER device alone, driven by the logged set-point x_d(t) through the follower PD and minus
    the logged rendered tissue force (the 'same command' simulation of project HW1-2). Uniform 1 kHz grid."""
    t_log = np.asarray(t_log, float)
    t = np.arange(t_log[0], t_log[-1] + 0.5 * h, h)
    r = np.interp(t, t_log, x_d_log)
    fe = np.zeros_like(t) if f_env_log is None else np.interp(t, t_log, f_env_log)
    dev.x, dev.v = float(r[0]), 0.0
    est = VelocityEstimator(h)
    x = np.zeros_like(t)
    for k in range(len(t)):
        xm = dev.measure()
        dev.command(Kp * (r[k] - xm) - Kd * est(xm) - fe[k])
        x[k] = dev.x
        for _ in range(substeps):
            dev.advance(h / substeps)
    return dict(t=t, x=x)


def simulate_free_decay(dev: Device, K, x0, t_release, T_end, h=1e-3, substeps=5) -> dict:
    """Free-decay test: the firmware renders a spring F = -K x_meas (held for one period); the handle is held
    at x0 until t_release, then released with nobody's hand on it."""
    n = int(round(T_end / h))
    t = np.arange(n) * h
    dev.x, dev.v = x0, 0.0
    x, F = np.zeros(n), np.zeros(n)
    for k in range(n):
        xm = dev.measure()
        dev.command(-K * xm)
        x[k], F[k] = xm, dev.F_motor
        for _ in range(substeps):
            if t[k] < t_release:
                dev.x, dev.v = x0, 0.0
            else:
                dev.advance(h / substeps)
    return dict(t=t, x=x, F=F)


def rms(x) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x**2)))


# =====================================================================  Part A: pre-flight and dry run
RULES = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "I")


def _num(d: dict, key: str, default: float) -> float:
    """d[key] as a float, or `default` when the key is missing or None (an unfilled record field)."""
    v = (d or {}).get(key)
    return default if v is None else float(v)


def preflight(record: dict) -> list:
    """Check a session record against the hardware safety checklist (project brief section 5.2; Course resources:
    Course robot models and hardware safety checklist). Returns a list of violations, each a string that starts
    with the rule label ('A1: ...', 'B4: ...', 'I: ...' for the derated current); [] means the session may start.

    Haptic station (every record): A5/B1 induction of every member who touches a device, supervisor present,
    at least two team members (never alone); A7 stop/e-stop tested; B2 dry run done, predicted peak force
    <= 2 N and peak |x| <= 40 mm written on the record; B3 force_max <= 2 N, watchdog enabled at <= 40 mm and
    on loop overrun, only the operator's hand on the handle; B4 delay emulated in software, watchdog and force
    limit never disabled, position-position ('pp') only with zero delay, a named stop holder who is not the
    leader operator; B5 cables tidy, no loose items, devices fixed to the bench; B6 IMU low voltage (<= 5 V),
    strapped over clothing; B7 CSV logging and a printed session log.
    Arm station (record['arm'] not None): A1 software-limit margin >= 10 deg, A2 keep-out radius >= reach + 0.3 m
    and taped, A3 speed fraction <= 0.25 on a first run (<= 0.50 otherwise), A4 payload <= 0.125 kg, A6 nobody
    inside the keep-out zone while the arm can move, A7 e-stop tested, B5 soft mat under the workspace,
    I servo current fraction <= 0.60."""
    # TODO A1: write this function (see the docstring and the lab handout, Part A).
    raise NotImplementedError("preflight")


def contact_equilibrium(task: dict, x_m: float) -> dict:
    """Static P-P teleoperation at leader position x_m (no motion, no delay): follower PD force balances the
    tissue spring and the follower's gravity stiffness,  Kp(alpha x_m - x_s) = k_e (x_s - x_w)^+ + k_g x_s.
    Returns x_s, delta (penetration, >= 0), F_c, F_m = beta F_c, and k_felt = dF_m/dx_m (channel only,
    in contact) = alpha beta Kp (k_e + k_g) / (Kp + k_e + k_g)."""
    # TODO A2: write this function (see the docstring and the lab handout, Part A).
    raise NotImplementedError("contact_equilibrium")


def dry_run(task: dict) -> dict:
    """Dry run (rule B2): simulate the trial with the nominal devices and the nominal operator (simulate_pp)
    and return the numbers to write on the session log: peak |F_m| (leader), peak |F_s| (follower motor),
    peak_force (the larger), peak |x_m| and |x_s|, peak_x (the larger), saturation counts, min E_ch, and
    ok = peak_force <= dry_run_force_fraction * 2 N and peak_x <= x_abs_max and no saturation, no trip."""
    # TODO A3: write this function (see the docstring and the lab handout, Part A).
    raise NotImplementedError("dry_run")


# =====================================================================  Part B: bring-up and identification
def _extrema(tt, y, sign):
    """Indices of the local maxima of sign*y. Runs of equal samples (quantised plateaus) count as one point; a run
    is a maximum if both neighbouring runs are lower; its middle sample is returned. Maxima closer than 20 ms to
    the previous one are dropped."""
    z = sign * np.asarray(y, float)
    start = np.r_[0, np.nonzero(np.diff(z) != 0)[0] + 1]         # first sample of each run
    end = np.r_[start[1:], len(z)]
    v = z[start]
    k = np.nonzero((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]))[0] + 1
    idx = (start[k] + end[k] - 1) // 2
    keep = [int(idx[0])]
    for i in idx[1:]:
        if tt[i] - tt[keep[-1]] > 0.02:
            keep.append(int(i))
    return np.array(keep)


def log_decrement(t, x, t0: float, t_end: float = None, n_max: int = 6) -> dict:
    """Hand estimate from a free-decay record released at t0 (window t0 <= t <= t_end). Find the peaks and the
    troughs of x (a sample larger, or smaller, than its neighbours; a quantised plateau counts as one sample, its
    middle; extrema at least 20 ms apart). Pair each of the first n_max peaks with the trough that follows it; the
    amplitude of cycle k is a_k = (peak_k - trough_k) / 2, which does not depend on the offset of x.
    Damped period = mean spacing of those peaks; log decrement delta = mean of ln(a_k / a_{k+1});
    zeta = delta / sqrt(4 pi^2 + delta^2); wd = 2 pi / period; wn = wd / sqrt(1 - zeta^2).
    Returns dict(zeta, wd, wn, n_peaks)."""
    # TODO B1: write this function (see the docstring and the lab handout, Part B).
    raise NotImplementedError("log_decrement")


def fit_free_decay(t, x, K: float, k_g: float, T: float, t0: float, t_end: float = None) -> dict:
    """Least-squares fit of x = A exp(-s tau) cos(wd tau + phi) + c (tau = t - t0, window t0 <= t <= t_end) to a
    free-decay record taken while the firmware renders a spring K. Then wn = sqrt(s^2 + wd^2), zeta = s / wn, m_h = (K + k_g) / wn^2 and
    b_h = 2 zeta wn m_h + K T / 2: the sampled, held spring acts as a negative damper of about K T / 2
    (Week 4; Colgate and Brown 1994), so the decay shows b_h - K T / 2. Start the fit from log_decrement.
    Returns dict(m_h, b_h, wn, zeta, rms)."""
    # TODO B2: write this function (see the docstring and the lab handout, Part B).
    raise NotImplementedError("fit_free_decay")


def watchdog_check(t, x, F, x_range: float) -> dict:
    """Watchdog test log (handle pushed out slowly by hand while the firmware renders a spring): t_cross = first
    time |x| > x_range; t_zero = first time at or after t_cross from which the motor output stays exactly 0 to
    the end; delay = t_zero - t_cross; latched = the output stays 0 even after the handle is back inside.
    Returns dict(t_cross, t_zero, delay, latched)."""
    # TODO B3: write this function (see the docstring and the lab handout, Part B).
    raise NotImplementedError("watchdog_check")


# =====================================================================  Part C: the teleoperation trial
def resample(t_src, x_src, t_dst) -> np.ndarray:
    """Linear interpolation of x_src(t_src) onto t_dst; NaN outside [t_src[0], t_src[-1]]."""
    # TODO C1: write this function (see the docstring and the lab handout, Part C).
    raise NotImplementedError("resample")


def channel_energy(t, F_m, x_m, F_c, x_d) -> np.ndarray:
    """Energy (J) that has entered the teleoperation channel from the logged port signals:
    E_ch[k] = sum_{i <= k} (F_m[i] v_m[i] - F_c[i] v_d[i]) dt[i], with backward differences
    v[i] = (x[i] - x[i-1]) / dt[i], dt[i] = t[i] - t[i-1] (v[0] = 0). A passive channel keeps E_ch >= 0."""
    # TODO C2: write this function (see the docstring and the lab handout, Part C).
    raise NotImplementedError("channel_energy")


def trial_metrics(log: dict, task: dict) -> dict:
    """Metrics of a logged trial (project format). On a uniform 1 kHz grid (resample) compute:
    rms_free  -- RMS of x_s - x_d in the 'free' window (follower tracking in free motion, m)
    x_m_hold, x_s_hold, delta_hold, F_m_hold -- means in the 'hold' window; delta = x_s - x_w
    F_m_pred  -- contact_equilibrium(task, x_m_hold)['F_m'];  transparency_error = F_m_hold / F_m_pred - 1
    k_felt    -- least-squares slope of F_m against x_m over the samples in contact (x_s > x_w + contact_margin)
                 that move slowly (|v_m| < slow_speed, v_m by central differences)
    peak_F_m, peak_F_s (from cur), peak_x (max |x_m|, |x_s|), n_sat (samples with |force| or |cur| >= 2 N - 1e-9),
    E_min     -- minimum of channel_energy on the ORIGINAL log samples
    ok        -- peak forces <= 2 N with n_sat = 0, peak_x <= x_abs_max and E_min >= E_min limit."""
    # TODO C3: write this function (see the docstring and the lab handout, Part C).
    raise NotImplementedError("trial_metrics")


def sim_vs_hw(log: dict, task: dict, dev: Device, skip: float = 0.5) -> dict:
    """Simulation versus hardware for the follower (project HW1-2): replay the follower with simulate_follower
    driven by the logged x_d (cmd) and the logged tissue force (aux f_env), with the device model `dev`; resample
    the measured x_s onto the replay grid and return dict(rms, max) of (x_sim - x_meas) for t >= skip, and
    ok = rms <= rms_sim_meas."""
    # TODO C4: write this function (see the docstring and the lab handout, Part C).
    raise NotImplementedError("sim_vs_hw")


# =====================================================================  Part D: safe stop and the IMU at rest
def fade_time(t, F, t_trigger: float, threshold: float) -> float:
    """Time from the stop trigger until |F| stays below threshold for the rest of the log (F may be a 2-D array:
    all columns). 0 if already below; inf if it never settles."""
    # TODO D1: write this function (see the docstring and the lab handout, Part D).
    raise NotImplementedError("fade_time")


def imu_rest_stats(t, gyro, ax, ay, t_rest: float) -> dict:
    """IMU module at rest (B6 bring-up; project HW3-1/HW6-1): from the samples with t < t_rest
    bias = mean gyro (rad/s), noise = standard deviation of the gyro (rad/s, ddof = 1),
    drift_deg = |bias| * t_rest in degrees (the angle error after t_rest if the bias is not removed),
    tilt_deg = atan2(-mean ay, mean ax) in degrees (accelerometer tilt of the mounting, ax = g cos, ay = -g sin);
    and from the samples with t >= t_rest, residual_deg = the angle obtained by integrating (gyro - bias) with
    the sample period (rectangle rule) over that interval, in degrees (it should be close to 0)."""
    # TODO D2: write this function (see the docstring and the lab handout, Part D).
    raise NotImplementedError("imu_rest_stats")
