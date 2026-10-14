"""Device models: the Hapkit-type 1-DOF haptic device, the operator's hand, the course-arm tip axis
(position-controlled servos) and a low-cost IMU. All values are ILLUSTRATIVE VALUES.

Haptic device (two per station). The numbers are the SBEG 461 Lab 8 (Week 9) reference values of the
Hapkit-type device in Course resources ("Course robot models and hardware safety checklist",
model ``hapkit``): handle mass m_h = 0.05705 kg (handle + reflected rotor), physical damping
b_h = 0.15 N s/m, gravity stiffness k_g = 1.2111 N/m, loop period T = 1 ms, position resolution
20 um, handle force saturated at 2 N in the firmware, watchdog outside +-40 mm. Re-identify m_h and
b_h on your own unit (free-decay test) in hardware session 1.

Device convention: x (m) is the handle position, positive away from the operator. A force F (N) is
positive in +x. The firmware runs once per T: it reads the quantised position, runs the controller,
saturates the force at +-2 N and applies it with a zero-order hold until the next tick.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HAPKIT = dict(m_h=0.05705, b_h=0.15, k_g=1.2111, T=1e-3, x_res=2e-5, force_max=2.0,
              x_range=0.040, k_t=0.01, i_max=1.5)
"""Hapkit-type device data (SBEG 461 Lab 8 reference values; Course resources ``hapkit``)."""

OPERATOR = dict(m=0.10, b=2.0, k=200.0)
"""Operator hand on a handle (the relaxed pinch grip of Labs 2-3, illustrative): mass 0.10 kg,
damping 2 N s/m, grip stiffness 200 N/m toward the intended position. A passive hand adds damping;
a released handle is the worst case."""


def quantise(x, res):
    return np.round(np.asarray(x, float) / res) * res


@dataclass
class Hapkit:
    """One Hapkit-type device: m x'' = F_motor - b_h x' - k_g x + F_ext (MR §11.7 notation: the
    device is the "plant" of an impedance controller). ``F_motor`` is the saturated firmware output.
    The watchdog sets the motor output to zero (latched) if |x| > x_range."""
    m_h: float = HAPKIT["m_h"]
    b_h: float = HAPKIT["b_h"]
    k_g: float = HAPKIT["k_g"]
    T: float = HAPKIT["T"]
    x_res: float = HAPKIT["x_res"]
    force_max: float = HAPKIT["force_max"]
    x_range: float = HAPKIT["x_range"]
    x: float = 0.0
    v: float = 0.0
    F_motor: float = 0.0
    tripped: bool = False
    sat_count: int = 0

    def measure(self) -> float:
        return float(quantise(self.x, self.x_res))

    def command(self, F: float) -> float:
        """Firmware output stage: saturation at +-force_max and the watchdog."""
        if abs(self.x) > self.x_range:
            self.tripped = True
        if self.tripped:
            self.F_motor = 0.0
            return 0.0
        if abs(F) > self.force_max:
            self.sat_count += 1
        self.F_motor = float(np.clip(F, -self.force_max, self.force_max))
        return self.F_motor

    def advance(self, dt: float, F_ext: float = 0.0, m_extra: float = 0.0) -> None:
        a = (self.F_motor - self.b_h * self.v - self.k_g * self.x + F_ext) / (self.m_h + m_extra)
        self.v += a * dt                     # semi-implicit Euler
        self.x += self.v * dt


@dataclass
class ServoAxis:
    """Tool-tip motion of the course arm along one task axis (the approach or insertion axis), as
    seen through the position-controlled servos: x'' = wn^2 (x_c - x) - 2 zeta wn x', with the
    command rate-limited to v_max and delayed by ``delay``. This 1-D model is used for the teleoperation
    stability tests (Scenarios 4 and 5); the full 6-joint kinematics are in ``arm.py``. Defaults are
    ILLUSTRATIVE (identify them in hardware session 1). ``x_res`` is the tip-position resolution
    that the 12-bit encoders give at about 0.2 m."""
    wn: float = 30.0
    zeta: float = 0.8
    delay: float = 0.010
    v_max: float = 0.050
    x_res: float = 1e-4
    x: float = 0.0
    v: float = 0.0
    x_cmd: float = 0.0
    _xl: float | None = None
    _buf: list = field(default_factory=list)

    def measure(self) -> float:
        return float(quantise(self.x, self.x_res))

    def command(self, x_c: float, T: float = 1e-3) -> None:
        k = int(round(self.delay / T))
        self._buf.append(float(x_c))
        x_c = self._buf.pop(0) if len(self._buf) > k else self._buf[0]
        if self._xl is None:
            self._xl = self.x
        self._xl += float(np.clip(x_c - self._xl, -self.v_max * T, self.v_max * T))
        self.x_cmd = self._xl

    def advance(self, dt: float, F_ext: float = 0.0, m_extra: float = 0.0) -> None:
        a = self.wn**2 * (self.x_cmd - self.x) - 2 * self.zeta * self.wn * self.v
        self.v += a * dt
        self.x += self.v * dt


class Operator:
    """Operator's hand: F_h = k (x_int(t) - x) - b x' while holding, 0 after ``release_time``; its
    mass is carried by the handle while holding. ``x_int(t)`` is the intended handle position."""

    def __init__(self, x_int, k=OPERATOR["k"], b=OPERATOR["b"], m=OPERATOR["m"], release_time=np.inf):
        self.x_int, self.k, self.b, self.m, self.release_time = x_int, k, b, m, release_time

    def holding(self, t) -> bool:
        return t < self.release_time

    def force(self, t, x, v) -> float:
        if not self.holding(t):
            return 0.0
        return float(self.k * (self.x_int(t) - x) - self.b * v)

    def mass(self, t) -> float:
        return self.m if self.holding(t) else 0.0


class LossOfResistanceOperator(Operator):
    """Operator who pushes the needle at constant intended speed and stops when they FEEL the loss of
    resistance: when the force on their hand F_m drops by more than ``drop`` (N) below its maximum of
    the last ``window`` seconds (and that maximum exceeded ``f_min``), they freeze their intended
    position at the current handle position after a reaction time ``t_react`` (s). The detection is
    armed only once the handle has passed ``x_arm`` (the operator knows the approximate depth of the
    ligament from imaging, so the skin "pop" is ignored). Before that, the intended position ramps
    from x0 at ``speed`` up to x_max. ``t_freeze`` records when it happened."""

    def __init__(self, x0, x_max, speed, t0=0.2, drop=0.3, window=0.1, f_min=0.5, t_react=0.25,
                 x_arm=-np.inf, h=HAPKIT["T"], **kw):
        self._ramp = ramp_to(x0, x_max, speed, t0)
        super().__init__(self._intent, **kw)
        self.drop, self.f_min, self.t_react, self.x_arm = drop, f_min, t_react, x_arm
        self.n_win = max(1, int(round(window / h)))
        self.hist = []
        self.t_detect = np.inf
        self.t_freeze = np.inf
        self.x_hold = None
        self._x_now = x0

    def _intent(self, t):
        return self.x_hold if self.x_hold is not None else self._ramp(t)

    def observe(self, t, F_m, x_m):
        self._x_now = x_m
        self.hist.append(F_m)
        if len(self.hist) > self.n_win:
            self.hist.pop(0)
        fmax = max(self.hist)
        if (self.t_detect == np.inf and x_m >= self.x_arm and fmax > self.f_min
                and F_m < fmax - self.drop):
            self.t_detect = t
        if self.x_hold is None and t >= self.t_detect + self.t_react:
            self.x_hold = x_m
            self.t_freeze = t


def ramp_to(x0, x1, speed, t0=0.0, hold_after=None):
    """Intended position: hold x0 until t0, then move at constant ``speed`` (m/s) to x1 and hold."""
    T = abs(x1 - x0) / speed

    def f(t):
        if t <= t0:
            return x0
        s = min((t - t0) / T, 1.0)
        return x0 + s * (x1 - x0)
    return f


def min_jerk(x0, x1, T, t0=0.0):
    """Minimum-jerk intended position from x0 to x1 in T seconds, starting at t0."""
    def f(t):
        s = np.clip((t - t0) / T, 0.0, 1.0)
        return x0 + (x1 - x0) * (10 * s**3 - 15 * s**4 + 6 * s**5)
    return f


class VelocityEstimator:
    """Firmware velocity estimate: backward difference of the quantised position, low-pass filtered
    (first order, cut-off ``fc`` Hz). This is the baseline estimator; Scenario 3 asks for better."""

    def __init__(self, T=HAPKIT["T"], fc=50.0):
        self.T, self.alpha = T, 1.0 - np.exp(-2 * np.pi * fc * T)
        self.prev, self.v = None, 0.0

    def __call__(self, x) -> float:
        if self.prev is not None:
            self.v += self.alpha * ((x - self.prev) / self.T - self.v)
        self.prev = x
        return self.v


# ------------------------------------------------------------------------------------- IMU model
IMU = dict(gyro_noise=np.radians(0.5), gyro_bias=np.radians(2.0), acc_noise=0.15, rate=200.0)
"""Low-cost 6-axis IMU (illustrative): gyro white noise 0.5 deg/s (1 sigma per sample at 200 Hz),
constant gyro bias 2 deg/s, accelerometer noise 0.15 m/s^2 per axis, 200 Hz."""


def imu_from_angle(t, theta, r_imu=0.0, seed=0, gyro_noise=IMU["gyro_noise"],
                   gyro_bias=IMU["gyro_bias"], acc_noise=IMU["acc_noise"], g=9.81):
    """Synthetic planar IMU readings for a segment that rotates by theta(t) (rad) about a horizontal
    axis, with the IMU mounted r_imu (m) from the axis along the segment, which hangs down at theta=0.
    Returns dict(gyro (rad/s), ax, ay (m/s^2, sensor x along the segment toward the joint, y
    perpendicular), theta, omega). The accelerometer measures specific force a - g in sensor axes,
    so at rest ax = g cos(theta), ay = -g sin(theta) (tilt from atan2(-ay, ax)); motion adds the
    centripetal and tangential terms r w^2 and r alpha."""
    t = np.asarray(t, float)
    th = np.asarray(theta, float)
    w = np.gradient(th, t)
    al = np.gradient(w, t)
    rng = np.random.default_rng(seed)
    gyro = w + gyro_bias + gyro_noise * rng.standard_normal(len(t))
    ax = g * np.cos(th) + r_imu * w**2 + acc_noise * rng.standard_normal(len(t))
    ay = -g * np.sin(th) - r_imu * al + acc_noise * rng.standard_normal(len(t))
    return dict(t=t, gyro=gyro, ax=ax, ay=ay, theta=th, omega=w)


# ------------------------------------------------------------------------- single-device simulator
def simulate_device(controller, T_end, user=None, x0=0.0, v0=0.0, dev=None, substeps=5, sensors=None,
                    h=HAPKIT["T"]):
    """Sampled-data simulation of ONE Hapkit-type device with a hand on it (Scenarios 3 and 6).

    controller(t, x_meas, sens) -> F  is called once per tick T = 1 ms with the quantised handle
                 position and ``sens`` = sensors(t, x, v) (e.g. an IMU sample, or None); its output is
                 saturated at 2 N and held (zero-order hold); the watchdog cuts it outside +-40 mm.
    user         object with force(t, x, v) and mass(t) (an ``Operator`` or a patient/wearer model)
    Returns dict of arrays t, x, v, x_meas, F (applied motor force), F_cmd (before saturation),
    F_h (user force) and scalars sat (saturation count), tripped, watchdog_t."""
    D = dev or Hapkit()
    D.x, D.v = x0, v0
    n = int(round(T_end / h))
    t = np.arange(n) * h
    out = {k: np.zeros(n) for k in ("x", "v", "x_meas", "F", "F_cmd", "F_h")}
    wd = np.nan
    for k in range(n):
        tk = t[k]
        xm = D.measure()
        sens = sensors(tk, D.x, D.v) if sensors is not None else None
        F_cmd = float(controller(tk, xm, sens))
        F = D.command(F_cmd)
        F_h = user.force(tk, D.x, D.v) if user is not None else 0.0
        for key, val in zip(("x", "v", "x_meas", "F", "F_cmd", "F_h"), (D.x, D.v, xm, F, F_cmd, F_h)):
            out[key][k] = val
        if D.tripped and np.isnan(wd):
            wd = tk
        m_u = user.mass(tk) if user is not None else 0.0
        for _ in range(substeps):
            fu = user.force(tk, D.x, D.v) if user is not None else 0.0
            D.advance(h / substeps, F_ext=fu, m_extra=m_u)
    out.update(t=t, sat=D.sat_count, tripped=D.tripped, watchdog_t=wd)
    return out
