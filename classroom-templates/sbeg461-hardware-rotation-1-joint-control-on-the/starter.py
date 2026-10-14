"""SBEG 461 Hardware rotation 1 (Week 8) -- joint control on the course arm: the PID step response of the
servo loop, gravity compensation, and logging.  STARTER: complete every function marked TODO.

These functions become `sbeg461_dynamics/hwrun.py` in your cumulative library.  The project
(`project/starter/sbeg461_project/hardware.py`) uses the same log format and the same metrics, so the
code you write today is the code you use in the two project hardware sessions (Weeks 10-13).

What runs on the arm.  The course arm has position-controlled smart servos (project brief Sec. 5.1).  The
station streams a joint-angle command every 20 ms (50 Hz).  Inside each servo a firmware loop at 1 kHz is
a PD controller of exactly the Lab 5 kind (MR Eq. 11.23 with Ki = 0, derivative on the measured velocity),
acting on one joint of the arm after a bus/firmware delay d:

    M_eff q'' + b q' + g_j(q) = tau,    tau = sat( Kp (q_cmd(t - d) - q) - Kd q' )         (MR Eq. 11.21)

M_eff = M_jj(q) + the apparent rotor inertia G^2 I_rotor of the geared servo (MR Sec. 8.9.2).  You cannot
change Kp and Kd (they are in the firmware), but you can identify them, and you can wrap your own loop
around the servo at 50 Hz: gravity feedforward through the position command (MR Eq. 11.38 realised with
a position servo) and an integral outer loop (the I of PID, MR Eq. 11.23).

The PID code of Lab 5 (`pd_gains`, `pd_gravity_error`, `JointPID`, `simulate_joint`) is given below,
unchanged, and is used by your functions.  All device numbers are ILLUSTRATIVE VALUES; identify them on
your own unit.

Notation: Modern Robotics (MR).  Units: SI (s, rad, rad/s, N m, A) unless a name ends in _deg.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import modern_robotics as mr
import numpy as np

import course_arm as _ca

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DT = 0.02                                    # station stream / logger period (s), 50 Hz
H = 0.001                                    # servo firmware period (s), 1 kHz
ENC_RES = 2 * np.pi / 4096                   # 12-bit encoder (rad)
NOMINAL_SERVO = dict(wn=25.0, zeta=0.9, delay=0.02)   # the nominal servo model of the course (Lab 5; not identified)
MODES = ("none", "ff", "int", "ff+int")      # hold-test modes (Part D)


# =============================================================================================
# Provided: model, task, logs
# =============================================================================================
def course_arm_model(payload: float = 0.0) -> dict:
    """The course arm (the lab's own copy of the shared course-arm model, see course_arm.py)."""
    return _ca.course_arm(payload=payload)


def inverse_dynamics(model: dict, q, dq, ddq, gravity: bool = True) -> np.ndarray:
    """modern_robotics RNEA (MR Eqs. 8.50-8.54) for a model dict; gravity=False sets g = 0."""
    g = model["g"] if gravity else np.zeros(3)
    return np.asarray(mr.InverseDynamics(np.asarray(q, float), np.asarray(dq, float),
                                         np.asarray(ddq, float), g, np.zeros(6),
                                         model["Mlist"], model["Glist"], model["Slist"]))


def gravity_torques(model: dict, q) -> np.ndarray:
    """g(theta): RNEA with zero velocity and acceleration (MR Eq. 8.1).  q shape (n,) or (N, n)."""
    q = np.asarray(q, float)
    if q.ndim == 1:
        return inverse_dynamics(model, q, np.zeros_like(q), np.zeros_like(q))
    return np.array([gravity_torques(model, qk) for qk in q])


def mass_diagonal(model: dict, q) -> np.ndarray:
    """diag M(theta) of the links (MR Eq. 8.1; MassMatrix of modern_robotics), shape (n,)."""
    return np.diag(mr.MassMatrix(np.asarray(q, float), model["Mlist"], model["Glist"], model["Slist"])).copy()


def load_task(name: str = "rotation1_task.json") -> dict:
    """Task data: poses, limits, nominal servo, step test, static poses, hold test, move and stop tests."""
    with open(DATA / name) as fh:
        return json.load(fh)


def write_log(path, t, cmd, meas=None, cur=None) -> None:
    """Write a CSV in the project log format (project starter `hardware.py`):
    t, cmd_1..cmd_n[, meas_1..meas_n][, cur_1..cur_n]  (s, rad, A)."""
    t = np.asarray(t, float)
    cmd = np.asarray(cmd, float).reshape(len(t), -1)
    n = cmd.shape[1]
    cols, header = [t[:, None], cmd], ["t"] + [f"cmd_{j + 1}" for j in range(n)]
    if meas is not None:
        cols.append(np.asarray(meas, float).reshape(len(t), -1))
        header += [f"meas_{j + 1}" for j in range(n)]
    if cur is not None:
        cols.append(np.asarray(cur, float).reshape(len(t), -1))
        header += [f"cur_{j + 1}" for j in range(n)]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(np.round(np.hstack(cols), 7).tolist())


def load_log(path) -> dict:
    """Read a log written by `write_log` or by the station logger.  Returns t (N,), cmd, meas,
    cur (N, n) (None when absent) and any extra columns (e.g. `pose`, `dir`, `mode`) as 1-D arrays."""
    with open(path, newline="") as fh:
        header = [h.strip() for h in next(csv.reader(fh))]
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    col = {h: i for i, h in enumerate(header)}

    def group(prefix):
        names = sorted((h for h in header if h.startswith(prefix + "_")),
                       key=lambda h: int(h.split("_")[1]))
        return data[:, [col[h] for h in names]] if names else None

    out = dict(t=data[:, col["t"]] if "t" in col else None,
               cmd=group("cmd"), meas=group("meas"), cur=group("cur"))
    for h in header:
        if "_" not in h and h != "t":
            out[h] = data[:, col[h]]
    return out


def log_window(log: dict, t0: float, t1: float, mode=None) -> dict:
    """The rows of a log with t0 <= t <= t1 (and, for the hold log, mode == mode)."""
    m = (log["t"] >= t0 - 1e-9) & (log["t"] <= t1 + 1e-9)
    if mode is not None:
        m &= log["mode"] == mode
    return {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in log.items()}


def step_command(pose, joints, t_steps, amplitude_deg: float, duration: float, dt: float = DT):
    """Step-test command: hold `pose` (n,), then add `amplitude_deg` to joint j (1-based) from time
    t_steps[k] on, for each joint of `joints`.  Returns (t (N,), cmd (N, n))."""
    t = np.arange(int(round(duration / dt)) + 1) * dt
    cmd = np.repeat(np.asarray(pose, float)[None], len(t), 0)
    for j, ts in zip(joints, t_steps):
        cmd[t >= ts - 1e-9, j - 1] += np.deg2rad(amplitude_deg)
    return t, cmd


def smooth_move(q0, q1, T: float, dt: float = DT, dwell: float = 0.5, back: bool = True):
    """The station move routine: rest at q0 for `dwell`, a rest-to-rest quintic to q1 in T s
    (s = 10 r^3 - 15 r^4 + 6 r^5, r = t/T; MR Sec. 9.2.2.1, taught in Week 10), rest for 2 dwell, and
    (if back) the same move back and a final dwell.  Returns (t (N,), cmd (N, n))."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    nd, nT = int(round(dwell / dt)), int(round(T / dt))
    r = np.arange(nT) / nT
    s = 10 * r**3 - 15 * r**4 + 6 * r**5
    seg = [np.repeat(q0[None], nd, 0), q0 + np.outer(s, q1 - q0), np.repeat(q1[None], 2 * nd, 0)]
    if back:
        seg += [q1 + np.outer(s, q0 - q1), np.repeat(q0[None], nd + 1, 0)]
    else:
        seg[-1] = np.repeat(q1[None], 2 * nd + 1, 0)
    cmd = np.vstack(seg)
    return np.arange(len(cmd)) * dt, cmd


def check_command(t, q, model: dict, speed_fraction: float, margin_deg: float = 10.0,
                  tol: float = 0.01) -> dict:
    """Safety check of a command table (project brief Sec. 5.2, rules A1 and A3): joint limits with a
    margin, peak finite-difference speed against speed_fraction * rated speed, rest at both ends.
    (A step command fails the speed test by construction: for steps use the dry-run speed instead.)"""
    t = np.asarray(t, float)
    q = np.asarray(q, float)
    lim = np.asarray(model["limits"], float)
    m = np.deg2rad(margin_deg)
    within = bool(np.all(q >= lim[:, 0] + m) and np.all(q <= lim[:, 1] - m))
    v = np.diff(q, axis=0) / np.diff(t)[:, None]
    ratio = float(np.max(np.abs(v) / (speed_fraction * np.asarray(model["qdot_max"], float))))
    steps = np.abs(np.diff(q, axis=0)).max(axis=1)
    start_rest, end_rest = bool(steps[0] < 1e-3), bool(steps[-1] < 1e-3)
    speed_ok = ratio <= 1.0 + tol
    return dict(ok=within and speed_ok and start_rest and end_rest, within_limits=within,
                peak_speed_ratio=ratio, speed_ok=speed_ok, starts_at_rest=start_rest,
                ends_at_rest=end_rest)


def tracking_metrics(t, cmd, meas, max_lag: float = 0.2) -> dict:
    """RMS and max |meas - cmd| per joint, the lag d (multiple of the sample time) that minimises the
    RMS of meas(t) - cmd(t - d), and the RMS after removing it."""
    t = np.asarray(t, float)
    cmd = np.asarray(cmd, float).reshape(len(t), -1)
    meas = np.asarray(meas, float).reshape(len(t), -1)
    e = meas - cmd
    dt = float(np.median(np.diff(t)))
    best = (np.inf, 0)
    for k in range(int(round(max_lag / dt)) + 1):
        ek = meas[k:] - cmd[:len(t) - k]
        r = float(np.sqrt(np.mean(ek**2)))
        if r < best[0] - 1e-15:
            best = (r, k)
    k = best[1]
    ek = meas[k:] - cmd[:len(t) - k]
    return dict(rms=np.sqrt(np.mean(e**2, axis=0)), max_abs=np.max(np.abs(e), axis=0), lag=k * dt,
                rms_after_lag=np.sqrt(np.mean(ek**2, axis=0)))


def average_approaches(log: dict) -> dict:
    """Static-pose log (columns pose, dir, cmd_j, meas_j, cur_j; each pose approached from below,
    dir = +1, and from above, dir = -1): average the two approaches of every pose, so that static
    friction cancels to first order.  Returns pose ids and the averaged cmd, meas, cur (P, n)."""
    poses = np.unique(log["pose"]).astype(int)
    out = {k: [] for k in ("cmd", "meas", "cur")}
    for p in poses:
        m = log["pose"] == p
        for k in out:
            out[k].append(log[k][m].mean(axis=0))
    return dict(pose=poses, **{k: np.array(v) for k, v in out.items()})


def step_response(tau, wn: float, zeta: float) -> np.ndarray:
    """Unit step response of q_dd + 2 zeta wn q_d + wn^2 q = wn^2 u (Astrom-Murray Eq. 6.22, k = 1) at
    times tau >= 0 after the start of motion (0 for tau < 0)."""
    tau = np.asarray(tau, float)
    x = np.clip(tau, 0.0, None)
    if zeta < 1.0 - 1e-6:
        wd = wn * np.sqrt(1 - zeta**2)
        y = 1 - np.exp(-zeta * wn * x) * (np.cos(wd * x) + zeta / np.sqrt(1 - zeta**2) * np.sin(wd * x))
    elif zeta > 1.0 + 1e-6:
        a = wn * (zeta + np.sqrt(zeta**2 - 1)); b = wn * (zeta - np.sqrt(zeta**2 - 1))
        y = 1 - (a * np.exp(-b * x) - b * np.exp(-a * x)) / (a - b)
    else:
        y = 1 - np.exp(-wn * x) * (1 + wn * x)
    return np.where(tau > 0, y, 0.0)


def normalised_rms(predicted, measured) -> float:
    """RMS(predicted - measured) / RMS(measured) (project `hardware.normalised_rms`; the HWk-2 gravity
    check passes at <= 0.3)."""
    p = np.asarray(predicted, float).ravel()
    y = np.asarray(measured, float).ravel()
    return float(np.sqrt(np.mean((p - y) ** 2)) / np.sqrt(np.mean(y**2)))


# =============================================================================================
# Provided: the Lab 5 PID code, unchanged (MR Sec. 11.4.1)
# =============================================================================================
def plant_gravity(plant: dict, q):
    """g(q) = g_cos cos q + g_sin sin q (N m); the torque needed to hold the joint still at q."""
    return plant["g_cos"] * np.cos(q) + plant["g_sin"] * np.sin(q)


def plant_rhs(plant: dict, x, tau: float, gravity: bool = True) -> np.ndarray:
    """State derivative of x = (q, q') under torque tau:  q'' = (tau - b q' - g(q)) / M."""
    q, dq = x
    g = plant_gravity(plant, q) if gravity else 0.0
    return np.array([dq, (tau - plant["b"] * dq - g) / plant["M"]])


def rk4_step(f, x, h: float) -> np.ndarray:
    """One classical Runge-Kutta step of x' = f(x) (the input is held constant over the step)."""
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def pd_gains(M: float, b: float, zeta: float, wn: float):
    """PD gains for the setpoint error dynamics  M e'' + (b + Kd) e' + Kp e = 0  (MR Eq. 11.25) with the
    target zeta and wn (MR Eq. 11.26):  Kp = M wn^2,  Kd = 2 zeta wn M - b.  Returns (Kp, Kd)."""
    return M * wn**2, 2.0 * zeta * wn * M - b


def pd_gravity_error(plant: dict, kp: float, qd: float) -> float:
    """Steady-state error of PD setpoint control under gravity (MR Eq. 11.27): at rest Kp e = g(q) with
    q = qd - e, solved by Newton's method from e = g(qd)/Kp.  Returns e (rad)."""
    e = float(plant_gravity(plant, qd)) / kp
    for _ in range(50):
        q = qd - e
        f = kp * e - float(plant_gravity(plant, q))
        fp = kp + (-plant["g_cos"] * math.sin(q) + plant["g_sin"] * math.cos(q))
        step = f / fp
        e -= step
        if abs(step) < 1e-14:
            break
    return e


class JointPID:
    """Discrete PID joint controller of Lab 5 (MR Eq. 11.23, pseudocode MR Fig. 11.15) with an optional
    output limit and integrator anti-windup (MR p. 427; Astrom-Murray Sec. 10.4).

        ctrl = JointPID(kp, ki, kd, dt, tau_max=None, anti_windup=True)
        ctrl.reset(q0)
        u = ctrl.update(qd, qd_dot, q, qdot=None, tau_ff=0.0)

    e = qd - q, e_dot = qd_dot - qdot (qdot differenced from q if None), eint += e dt,
    u = Kp e + Kd e_dot + Ki eint + tau_ff, clipped to +-tau_max; while clipped with e of the sign of u the
    integral is frozen.  The output is a torque in Lab 5; in the outer loop of Part D it is a position
    correction (rad), and tau_max is then the correction clamp."""

    def __init__(self, kp: float, ki: float, kd: float, dt: float, tau_max: float | None = None,
                 anti_windup: bool = True):
        self.kp, self.ki, self.kd, self.dt = float(kp), float(ki), float(kd), float(dt)
        self.tau_max = None if tau_max is None else float(tau_max)
        self.anti_windup = bool(anti_windup)
        self.reset()

    def reset(self, q0: float | None = None) -> None:
        self.eint = 0.0
        self.q_prev = None if q0 is None else float(q0)
        self.terms = dict(P=0.0, I=0.0, D=0.0)
        self.saturated = False

    def update(self, qd: float, qd_dot: float, q: float, qdot: float | None = None, tau_ff: float = 0.0) -> float:
        q = float(q)
        if qdot is None:
            qdot = 0.0 if self.q_prev is None else (q - self.q_prev) / self.dt
        self.q_prev = q
        e = qd - q
        edot = qd_dot - qdot
        eint_new = self.eint + e * self.dt
        v = self.kp * e + self.kd * edot + self.ki * eint_new + tau_ff
        tau = v
        self.saturated = False
        if self.tau_max is not None and abs(v) > self.tau_max:
            self.saturated = True
            tau = math.copysign(self.tau_max, v)
            if self.anti_windup and e * v > 0:
                eint_new = self.eint
                v = self.kp * e + self.kd * edot + self.ki * eint_new + tau_ff
                tau = float(np.clip(v, -self.tau_max, self.tau_max))
        self.eint = eint_new
        self.terms = dict(P=self.kp * e, I=self.ki * self.eint, D=self.kd * edot)
        return float(tau)


def simulate_joint(plant: dict, ctrl: JointPID, t, qd, qd_dot, q0: float, dq0: float = 0.0,
                   substeps: int = 4, velocity: str = "measured", enc_res: float | None = None,
                   tau_ff=None, gravity: bool = True) -> dict:
    """Lab 5 closed-loop simulation of  tau = M q'' + b q' + g(q)  (MR Eq. 11.21) under a sampled
    controller on the grid t (dt = t[1] - t[0] = ctrl.dt); torque clipped to plant["tau_max"]; RK4.
    Returns dict(t, q, dq, tau, e)."""
    t = np.asarray(t, float)
    qd = np.asarray(qd, float)
    qd_dot = np.asarray(qd_dot, float)
    n = len(t)
    dt = t[1] - t[0]
    if abs(dt - ctrl.dt) > 1e-12:
        raise ValueError("controller dt must equal the time-grid spacing")
    ctrl.reset(q0)
    x = np.array([q0, dq0], float)
    Q, DQ, TAU = np.empty(n), np.empty(n), np.empty(n)
    h = dt / substeps
    lim = plant.get("tau_max")
    for k in range(n):
        Q[k], DQ[k] = x
        qm = x[0] if enc_res is None else enc_res * np.round(x[0] / enc_res)
        vm = x[1] if velocity == "measured" else None
        ff = 0.0 if tau_ff is None else float(tau_ff[k])
        tau = ctrl.update(qd[k], qd_dot[k], qm, vm, ff)
        if lim is not None:
            tau = float(np.clip(tau, -lim, lim))
        TAU[k] = tau
        if k + 1 < n:
            f = lambda s, tau=tau: plant_rhs(plant, s, tau, gravity)  # noqa: E731
            for _ in range(substeps):
                x = rk4_step(f, x, h)
    return dict(t=t, q=Q, dq=DQ, tau=TAU, e=qd - Q)


# =============================================================================================
# Provided: one course-arm joint as a Lab 5 plant, the servo model of the whole arm, log replay
# =============================================================================================
def joint_plant(model: dict, q_lock, joint: int, M_eff: float, b: float = 0.0, tau_max=None) -> dict:
    """Lab 5 plant dict of course-arm joint `joint` (0-based) with the other joints locked at q_lock:
    g(q) = g_cos cos q + g_sin sin q is exact for a horizontal joint axis (the distal links turn rigidly
    about it), with g_cos = g_j(q_j = 0) and g_sin = g_j(q_j = pi/2) from the RNEA."""
    q = np.asarray(q_lock, float).copy()
    q[joint] = 0.0
    gc = float(gravity_torques(model, q)[joint])
    q[joint] = np.pi / 2
    gs = float(gravity_torques(model, q)[joint])
    return dict(M=float(M_eff), b=float(b), g_cos=gc, g_sin=gs, tau_max=tau_max,
                locked_q=np.asarray(q_lock, float), joint_index=joint + 1)


def nominal_servo(model: dict, q, task: dict) -> dict:
    """Servo dict of the NOMINAL servo model at pose q (before any identification): for every joint
    M_eff = M_jj(q) + armature_j (task["nominal_servo"]), and (Kp, Kd) = pd_gains(M_eff, b, zeta, wn) with
    the nominal zeta = 0.9, wn = 25 rad/s.  Keys: kp, kd, M_eff, b (n,), delay (s)."""
    ns = task["nominal_servo"]
    M_eff = mass_diagonal(model, q) + np.asarray(ns["armature_kgm2"], float)
    b = np.full(len(M_eff), float(ns["b"]))
    kp, kd = pd_gains(M_eff, b, ns["zeta"], ns["wn"])
    return dict(kp=kp, kd=kd, M_eff=M_eff, b=b, delay=float(ns["delay_s"]))


def make_servo(base: dict, fits: dict) -> dict:
    """Copy of the servo dict `base` with the entries of the joints in `fits` replaced:
    fits = {joint (0-based): dict(kp=..., kd=..., M_eff=...[, b=...][, delay=...])} (the result of
    `servo_from_fit` plus the fitted delay).  The delay of the arm is the mean of the fitted delays."""
    s = {k: (np.array(v, float) if k != "delay" else float(v)) for k, v in base.items()}
    delays = []
    for j, f in fits.items():
        for k in ("kp", "kd", "M_eff", "b"):
            if k in f:
                s[k][j] = float(f[k])
        if "delay" in f:
            delays.append(float(f["delay"]))
    if delays:
        s["delay"] = float(np.mean(delays))
    return s


class ServoArm:
    """Simulation of the course arm under its joint servos (the simulation fallback and the "sim" of the
    sim-vs-hardware comparison).  The joints in `joints` (0-based) move; the others are held rigidly at
    their command.  Every moving joint j is the Lab 5 plant

        M_eff_j q_j'' + b_j q_j' + g_j(q) = tau_j,   tau_j = Kp_j (c_j(t - delay) - q_j) - Kd_j q_j'

    (a JointPID with Ki = 0, derivative on the measured velocity, at the firmware period h; optional
    output limit tau_limit (n,)), with g(q) of the whole arm from the RNEA, evaluated once per station
    period dt and held (the coupling through M is neglected: MR Sec. 11.4.2.1, decentralised control).
    Semi-implicit Euler at h.  Station interface (the same methods as the station's hardware object):

        arm.reset(q_cmd0)        settle at rest under the command q_cmd0 (with the gravity sag)
        q, tau = arm.read()      encoder angles (n,) and servo torques (n,); with kt (n,), currents tau/kt
        arm.send(q_cmd)          new command, applied after `delay`
        arm.wait()               advance one station period dt
        res = arm.run(t, cmd)    open-loop run of a command table: meas, q, qd, tau (N, n) and the
                                 peaks |q'|, |tau| (n,) over all firmware steps (peak_qd, peak_tau)."""

    def __init__(self, model: dict, servo: dict, joints=(1, 2), dt: float = DT, h: float = H,
                 enc_res: float = ENC_RES, tau_limit=None, kt=None):
        self.model, self.joints = model, list(joints)
        self.kp, self.kd = np.asarray(servo["kp"], float), np.asarray(servo["kd"], float)
        self.M_eff, self.b = np.asarray(servo["M_eff"], float), np.asarray(servo["b"], float)
        self.delay = float(servo["delay"])
        self.dt, self.h, self.enc_res = float(dt), float(h), float(enc_res)
        self.nsub = int(round(dt / h))
        self.tau_limit = None if tau_limit is None else np.asarray(tau_limit, float)
        self.kt = None if kt is None else np.asarray(kt, float)
        self.reset(np.zeros(len(self.kp)))

    def reset(self, q_cmd0) -> None:
        c = np.asarray(q_cmd0, float).copy()
        q = c.copy()
        for _ in range(60):                               # static sag: Kp (c - q) = g(q)
            g = gravity_torques(self.model, q)
            q[self.joints] = c[self.joints] - g[self.joints] / self.kp[self.joints]
        self.q, self.qd, self.t = q, np.zeros_like(q), 0.0
        self.queue = [(-np.inf, c)]
        self.tau = gravity_torques(self.model, q)
        self.pids = {j: JointPID(self.kp[j], 0.0, self.kd[j], self.h, None) for j in self.joints}
        for j in self.joints:
            self.pids[j].reset(q[j])
        self.peak_qd = np.zeros_like(q)
        self.peak_tau = np.abs(self.tau)

    def read(self):
        q = self.q if not self.enc_res else np.round(self.q / self.enc_res) * self.enc_res
        out = self.tau if self.kt is None else self.tau / self.kt
        return q.copy(), out.copy()

    def send(self, q_cmd) -> None:
        self.queue.append((self.t + self.delay, np.asarray(q_cmd, float).copy()))

    def _active(self):
        c = self.queue[0][1]
        for ta, ck in self.queue:
            if ta <= self.t + 1e-9:
                c = ck
            else:
                break
        return c

    def wait(self) -> None:
        g = gravity_torques(self.model, self.q)
        J = self.joints
        for _ in range(self.nsub):
            c = self._active()
            tau = g.copy()                                # held joints carry their gravity torque
            for j in J:
                u = self.pids[j].update(c[j], 0.0, self.q[j], self.qd[j])
                if self.tau_limit is not None:
                    u = float(np.clip(u, -self.tau_limit[j], self.tau_limit[j]))
                tau[j] = u
            for j in range(len(c)):
                if j not in J:
                    self.q[j] = c[j]
            qdd = (tau[J] - self.b[J] * self.qd[J] - g[J]) / self.M_eff[J]
            self.qd[J] = self.qd[J] + self.h * qdd
            self.q[J] = self.q[J] + self.h * self.qd[J]
            self.t += self.h
            self.tau = tau
            self.peak_qd = np.maximum(self.peak_qd, np.abs(self.qd))
            self.peak_tau = np.maximum(self.peak_tau, np.abs(tau))
        while len(self.queue) > 1 and self.queue[1][0] <= self.t + 1e-9:
            self.queue.pop(0)

    def run(self, t, cmd) -> dict:
        t = np.asarray(t, float)
        cmd = np.asarray(cmd, float).reshape(len(t), -1)
        self.reset(cmd[0])
        N, n = cmd.shape
        meas, Q, QD, TAU = (np.empty((N, n)) for _ in range(4))
        for k in range(N):
            meas[k], _ = self.read()
            Q[k], QD[k], TAU[k] = self.q, self.qd, self.tau
            self.send(cmd[k])
            if k + 1 < N:
                self.wait()
        return dict(meas=meas, q=Q, qd=QD, tau=TAU, peak_qd=self.peak_qd.copy(), peak_tau=self.peak_tau.copy())


class ReplayIO:
    """Replays a logged run through the station interface (read / send / wait): read() returns the
    logged meas and cur of the current row, send() records the command, wait() moves to the next row.
    Your station loop run on a ReplayIO of a station log must reproduce that log's cmd column."""

    def __init__(self, log: dict):
        self.meas, self.cur = np.asarray(log["meas"], float), log.get("cur")
        self.k, self.sent = 0, []

    def read(self):
        c = None if self.cur is None else np.asarray(self.cur[self.k], float)
        return self.meas[self.k].copy(), c

    def send(self, q_cmd) -> None:
        self.sent.append(np.asarray(q_cmd, float).copy())

    def wait(self) -> None:
        self.k = min(self.k + 1, len(self.meas) - 1)


# =============================================================================================
# Part A -- pre-flight: units, the dry run and the session record
# =============================================================================================
def servo_to_model(q_servo_deg, offset_deg) -> np.ndarray:
    """Servo reading (degrees, servo zero) -> model angle (rad, PoE zero of the course robot models):
    q_model = q_servo - offset, converted to radians."""
    # TODO: implement (see the docstring and the lab handout). Hint: np.deg2rad(q_servo - offset)
    raise NotImplementedError


def model_to_servo(q_model, offset_deg) -> np.ndarray:
    """Inverse of `servo_to_model`: model angle (rad) -> servo command (degrees)."""
    # TODO: implement (see the docstring and the lab handout). Hint: np.rad2deg(q_model) + offset
    raise NotImplementedError


def dry_run(model: dict, t, cmd, servo: dict, joints=(1, 2), speed_fraction: float = 0.25,
            torque_fraction: float = 0.6) -> dict:
    """Dry-run prediction written on the session log before the arm is powered (brief Sec. 5.2, rule B2):
    run the command table cmd (N, n) on the time grid t through `ServoArm(model, servo, joints,
    enc_res=0)` (no torque limit: we want to see whether the command would need more), and return
      * meas (N, n): the predicted encoder angles;
      * peak_qd (n,), peak_tau (n,): the peaks of |q'| and |servo torque| over all firmware steps;
      * speed_ratio (n,): peak_qd / (speed_fraction * qdot_max);
      * torque_ratio (n,): peak_tau / (torque_fraction * tau_max);
      * ok: every ratio <= 1.
    A step of amplitude A asks for a torque spike of about Kp A on top of the gravity torque."""
    # TODO: implement (see the docstring and the lab handout). Hint: ServoArm(model, servo, joints, enc_res=0.0).run(t, cmd) gives meas, peak_qd, peak_tau; divide by the derated limits
    raise NotImplementedError


PAYLOAD_MAX_KG = _ca.COURSE_ARM_SAFETY["payload_max_kg"]   # 0.125 kg (brief Sec. 5.2, rule A4)
OUTER_KI_MAX = 4.0                                          # 1/s, approved outer-loop integral gain
OUTER_CLAMP_MAX_DEG = 5.0                                   # deg, approved correction clamp

# labels = the rule labels of project brief Sec. 5.2 (A1-A7 course arm, B1-B6 session rules)
PREFLIGHT_RULES = (
    "A5: every member who touches the arm has a signed induction",
    "A7: a TA or the instructor is named as supervisor",
    "A7: the e-stop was tested at the start of the session",
    "B4: a named member holds the e-stop",
    "A2/A6: the keep-out zone is taped and clear",
    "B5: the workspace is ready (mat, payload fixed, cables tidy)",
    "A1: every commanded angle is >= 10 deg inside the hardware limits",
    "A3/B2: the dry run predicts peak speeds <= the required fraction of rated speed",
    "B2: the dry run predicts peak torques <= 60 % of rated and is written on the log",
    "A (current): the servo current limit is <= 60 % of rated",
    "A4: the payload is <= 0.125 kg",
    "B2 (station loop): outer-loop integral gain <= 4 1/s and correction clamp <= 5 deg",
)


def preflight(record: dict) -> list:
    """Checks a session record (see data/session_record_template.json) against the hardware safety
    checklist of project brief Sec. 5.2 and returns the list of FAILED rules (strings of
    PREFLIGHT_RULES, in that order); an empty list means the session may start.

    Record keys: members_at_arm (list of names), inducted (list of names), supervisor (str),
    estop_tested (bool), estop_holder (str, must be one of members_at_arm), keepout_clear (bool),
    workspace_ready (bool), first_run (bool), limits_ok (bool: check_command(...)["within_limits"] of every
    command file), margin_deg (number, the margin used), dry_run_speed_fraction (number, the speed
    fraction of the dry run), dry_run_speed_ratio, dry_run_torque_ratio (numbers, the largest over all
    command files), current_limit_fraction (number), payload_kg (number, weighed payload at the tool
    point), outer_ki (1/s) and outer_clamp_deg (deg) of the station loop (0 if it is not used).
    The required speed fraction is 0.25 for a first run of a motion and 0.5 otherwise; a dry run made at
    a HIGHER fraction than required does not count.  A missing key (or None) counts as a failure."""
    # TODO: implement (see the docstring and the lab handout). Hint: one test per rule, in the order of PREFLIGHT_RULES; treat a missing key or None as a failure
    raise NotImplementedError


# =============================================================================================
# Part B -- the step response of the servo's PD loop
# =============================================================================================
def zeta_wn_from_step(overshoot: float, t_peak: float):
    """Hand method (Astrom-Murray Table 6.1; MR p. 415): from the fractional overshoot
    Mp = exp(-pi zeta/sqrt(1-zeta^2)) and the peak time t_peak = pi/(wn sqrt(1-zeta^2)) (measured from the
    START OF MOTION, i.e. after the delay) return (zeta, wn).  Requires 0 < overshoot < 1."""
    # TODO: implement (see the docstring and the lab handout). Hint: invert Mp = exp(-pi zeta / sqrt(1 - zeta^2)) and t_peak = pi / (wn sqrt(1 - zeta^2))
    raise NotImplementedError


def step_features(t, cmd, meas, threshold: float = 0.1) -> dict:
    """Features of ONE step in one joint signal (1-D arrays).
      * t_step: first sample time where cmd differs from cmd[0];  A: cmd[-1] - cmd[0];
      * y0, y_ss: mean of meas before t_step, and mean over the last 0.3 s;
      * t_move: first time (linear interpolation between samples) at which (meas - y0)/(y_ss - y0)
        reaches `threshold` ... used only as a rough delay estimate;
      * overshoot: (max(meas) - y_ss)/(y_ss - y0) for a positive step (min for a negative one);
      * t_peak_abs: time of that extreme sample."""
    # TODO: implement (see the docstring and the lab handout). Hint: index k0 of the first changed command sample; normalise z = (meas - y0) / (y_ss - y0)
    raise NotImplementedError


def identify_servo(t, cmd, meas, window: float = 1.0) -> dict:
    """Least-squares fit of (wn, zeta, delay) of the servo loop to ONE step (1-D arrays) on
    [t_step - 0.1, t_step + window]:  meas(t) ~ y0 + A_fit * step_response(t - t_step - delay).
    y0 and A_fit (which absorbs the change of sag) are solved linearly for every trial (wn, zeta, delay).
    Start from the hand method (`step_features` + `zeta_wn_from_step`) and refine with
    scipy.optimize.least_squares (bounds wn in [2, 200], zeta in [0.05, 3], delay in [0, 0.2]).
    Returns dict(wn, zeta, delay, y0, A_fit, rms) with rms the fit residual (rad)."""
    # TODO: implement (see the docstring and the lab handout). Hint: inner linear least squares for (y0, A_fit) with np.linalg.lstsq; outer scipy.optimize.least_squares over (wn, zeta, delay)
    raise NotImplementedError


def servo_from_fit(wn: float, zeta: float, K: float, b: float = 0.0, M_link: float | None = None) -> dict:
    """The servo's PD loop from its step response and its static stiffness (Lab 5 `pd_gains` backwards).
    A PD loop on  M_eff q'' + b q' + g = tau  has  wn^2 = Kp / M_eff  and  2 zeta wn = (b + Kd) / M_eff
    (MR Eq. 11.26), and its static stiffness is Kp (sag Kp e = g, MR Eq. 11.27).  So, with K from the static
    poses (Part C):  kp = K,  M_eff = K / wn^2,  kd = 2 zeta wn M_eff - b.
    If M_link (the link inertia M_jj of the model) is given, also armature = M_eff - M_link: the apparent
    rotor inertia G^2 I_rotor of MR Sec. 8.9.2 (else None).  Returns dict(kp, kd, M_eff, armature)."""
    # TODO: implement (see the docstring and the lab handout). Hint: kp = K, M_eff = K / wn^2, kd = 2 zeta wn M_eff - b (MR Eq. 11.26 backwards)
    raise NotImplementedError


def predict_step(plant: dict, kp: float, kd: float, t, cmd, delay: float, h: float = H,
                 enc_res: float | None = None) -> np.ndarray:
    """Predicted response of ONE joint to its command cmd (1-D, on the station grid t) with the Lab 5 code:
      * fine grid tf = t[0] + h * (0, 1, ..., round((t[-1] - t[0]) / h)) and the delayed zero-order-hold
        command c(tf) = cmd[k], k the last sample with t[k] <= tf - delay (cmd[0] before the first one);
      * start at rest on the sag of the first command: q0 = cmd[0] - pd_gravity_error(plant, kp, cmd[0]);
      * ctrl = JointPID(kp, 0, kd, h) (the servo firmware, Ki = 0), and
        simulate_joint(plant, ctrl, tf, c, zeros, q0, substeps=1) -- qd_dot = 0 makes the derivative act on
        the measured velocity, as in the firmware; the torque is clipped to plant["tau_max"];
      * resample q onto t (np.interp) and, if enc_res is given, round to the encoder resolution.
    Returns the predicted measurement (N,)."""
    # TODO: implement (see the docstring and the lab handout). Hint: fine grid at h, delayed zero-order hold of cmd (np.searchsorted), q0 from pd_gravity_error, then simulate_joint with JointPID(kp, 0, kd, h)
    raise NotImplementedError


# =============================================================================================
# Part C -- static poses: torque constant, loop stiffness, gravity check
# =============================================================================================
def torque_constant(current, tau_model) -> float:
    """Least-squares k (N m/A, gear ratio included) with tau_model ~ k * current (one joint, or all
    entries pooled).  Same definition as project `hardware.torque_constant`."""
    # TODO: implement (see the docstring and the lab handout). Hint: least squares through the origin: k = (i . tau) / (i . i)
    raise NotImplementedError


def loop_stiffness(sag, tau_model) -> float:
    """Least-squares stiffness K (N m/rad) of one joint's servo loop from static poses:
    sag = cmd - meas ~ tau_model / K, i.e. 1/K = sum(tau*sag)/sum(tau^2).  For a PD servo, K = Kp."""
    # TODO: implement (see the docstring and the lab handout). Hint: 1/K = sum(tau sag) / sum(tau^2)
    raise NotImplementedError


def gravity_check(static: dict, model: dict, joints=(1, 2, 3), fit=(0, 2, 4), check=(1, 3, 5)) -> dict:
    """Gravity check of HW2-2(b)/HW3-2 type on the averaged static poses (`average_approaches`).
    For each joint index j in `joints` (0-based; default joints 2, 3 and 4, the gravity-loaded ones):
      * the model gravity torque at the MEASURED pose, g_j(q_meas);
      * k_j = torque_constant(cur_j[fit], g_j[fit]);   tau_cur_j = k_j * cur_j;
      * nrms_j = normalised_rms(g_j[check], tau_cur_j[check]) on the check poses only;
      * K_j = loop_stiffness(cmd_j - meas_j, g_j) on all poses.
    Returns dict(k, nrms, K) of arrays over `joints`, and g (P, n), tau_cur (P, len(joints))."""
    # TODO: implement (see the docstring and the lab handout). Hint: g at the MEASURED poses (gravity_torques), then torque_constant, normalised_rms and loop_stiffness per joint
    raise NotImplementedError


# =============================================================================================
# Part D -- gravity compensation, the station loop, sim vs hardware, safe stop
# =============================================================================================
def gravity_comp_command(model: dict, q_des, K, joints=(1, 2)) -> np.ndarray:
    """Gravity compensation through the position command: a PD servo at rest sags by g/Kp (MR Eq. 11.27),
    so command  q_cmd_j = q_des_j + g_j(q_des) / K_j  for every joint j in `joints` (0-based; K (n,) the
    loop stiffness of Part C); the other joints get q_des.  For a servo Kp (q_cmd - q) - Kd q' this is
    exactly PD control plus the gravity feedforward g(q_des) (MR Eq. 11.38 with Ki = 0, model at q_des)."""
    # TODO: implement (see the docstring and the lab handout). Hint: q_des plus g_j(q_des) / K_j for the joints in `joints`
    raise NotImplementedError


def station_hold(io, q_des, n: int, mode: str, model: dict | None = None, K=None, ki: float = 3.0,
                 clamp: float = np.deg2rad(5.0), t_on: float = 0.5, t_ramp: float = 0.5, joints=(1, 2),
                 dt: float = DT) -> dict:
    """The station loop of the hold test (50 Hz, n samples) on a servo interface `io` (ServoArm,
    ReplayIO or the station's hardware object: read() -> (q, cur), send(q_cmd), wait()).
    At sample k (t_k = k dt): q, cur = io.read(); the command starts as q_des; from t_on on, add
      * mode "ff" or "ff+int": the gravity offset of `gravity_comp_command(model, q_des, K, joints)` - q_des,
        multiplied by the ramp factor min(1, (t_k - t_on) / t_ramp) (bumpless switch-on);
      * mode "int" (from t_on) or "ff+int" (from t_on + t_ramp, when the feedforward is fully on, so that the
        integrator does not also integrate the sag that the feedforward is removing): for each joint j in
        `joints`, the outer-loop correction u_j = JointPID(0, ki, 0, dt, tau_max=clamp).update(q_des_j, 0, q_j)
        (one controller per joint, created at the start; the clamp is the anti-windup limit of Lab 5): an
        integral of the error q_des - q;
    then io.send(cmd) and, except after the last sample, io.wait().
    Mode "none" sends q_des throughout; any other mode string raises ValueError.  Returns dict(t (n,),
    cmd, meas, cur (n, 6)) with meas, cur as read (cur is NaN if io gives None)."""
    # TODO: implement (see the docstring and the lab handout). Hint: loop k = 0..n-1: read, build the command (ramped offset, integral corrections with one JointPID(0, ki, 0, dt, tau_max=clamp) per joint), send, wait
    raise NotImplementedError


def compare_runs(t, cmd, meas, sim, skip: float = 0.0) -> dict:
    """Sim-vs-hardware metrics after time `skip` (project brief Sec. 5.4), all per joint (rad):
    rms_cmd_meas, max_cmd_meas (tracking error of the device), rms_sim_meas, max_sim_meas (model error:
    simulation of the SAME command against the measurement), and lag (s) of meas behind cmd from
    `tracking_metrics`.  All arrays are on the same time grid t."""
    # TODO: implement (see the docstring and the lab handout). Hint: mask t >= skip; RMS and max of meas - cmd and of meas - sim per column; lag from tracking_metrics
    raise NotImplementedError


def stop_time(t, q, t_trigger: float, threshold: float = np.deg2rad(2.0), window: float = 0.1) -> float:
    """Safe-stop time (HWk-4): time from `t_trigger` until the joint speed, estimated by a central
    difference over `window` s (q(t + w/2) - q(t - w/2))/w to suppress encoder steps, stays below
    `threshold` (rad/s, all joints) for the rest of the log.  Returns inf if it never settles.
    The speed estimate is assigned to the centre sample; samples within w/2 of the ends are ignored."""
    # TODO: implement (see the docstring and the lab handout). Hint: central-difference speed over the window, last sample above the threshold after t_trigger
    raise NotImplementedError
