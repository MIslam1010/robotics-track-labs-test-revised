"""SBEG 461 Lab 5 -- error dynamics and PID control of a single joint; ROS 2 controller node.
STARTER CODE. Complete every function marked TODO (keep the names and signatures).
Run the auto-checks with:  LAB_IMPL=starter python -m pytest -q test_lab.py
See lab_handout.md for the checkpoints.

Notation follows Modern Robotics (MR) Ch. 11: joint error  theta_e = theta_d - theta  (MR p. 407), unit error
response (MR Sec. 11.2.1), standard second-order form  e'' + 2 zeta wn e' + wn^2 e = 0  (MR Eq. 11.8).
In code q = theta, qd = theta_d, e = theta_e.  Units: SI (s, rad, rad/s, N m).

Parts
  A. Error dynamics: zeta and wn, characteristic roots and stability, the unit error response, overshoot and
     2 % settling time, measured metrics of a simulated response            MR Sec. 11.2, pp. 407-415
  B. Motion control with velocity inputs: P, PI and feedforward + PI control of a joint whose servo is in
     velocity mode (stepper or velocity-mode amplifier)                    MR Sec. 11.3.1, Eqs. 11.11-11.15
  C. PID control with torque inputs: gain design for a target zeta and wn, the integral-gain bound, the PD
     steady-state error under gravity, a discrete PID controller (MR Fig. 11.15 pseudocode) with torque
     saturation and anti-windup, and a single-joint simulator of  tau = M q'' + b q' + g(q)  (MR Eq. 11.21)
                                                                           MR Sec. 11.4.1.1, pp. 424-429
  D. The ROS 2 node (lab/ros2/) reuses JointPID unchanged; nothing to write here.

These functions become `sbeg461_dynamics/control.py` in your cumulative library.  Lab 6 (computed torque vs
PID with model error, Week 6) adds gravity compensation and computed torque as separate controller laws next
to JointPID; the project's control.py has its own implementations.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ==========================================================================================
# Provided (do not edit): plants, reference trajectories, RK4 integrator
# ==========================================================================================
def load_plant(name: str) -> dict:
    """A single-joint plant from data/joint_plants.json:  tau = M q'' + b q' + g(q)  (MR Eq. 11.21) with
    g(q) = g_cos cos q + g_sin sin q.  Keys: M, b, g_cos, g_sin, tau_max (None = no limit), qdot_max,
    q_limits, description, source.  Names: arm_j1, arm_j2 (course arm, illustrative inertias, derated
    limits), knee (prosthesis), eod_wrist (EOD arm), mr_link (MR Sec. 11.4.1, p. 424)."""
    plants = json.loads((DATA / "joint_plants.json").read_text())
    return dict(plants[name])


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


def time_grid(T: float, dt: float) -> np.ndarray:
    """0, dt, 2 dt, ..., T (inclusive, rounded so that the last sample is T)."""
    n = int(round(T / dt))
    return np.arange(n + 1) * dt


def step_reference(t, q0: float, q1: float, t_step: float = 0.0):
    """Setpoint reference: qd = q0 before t_step, q1 from t_step on; qd' = 0."""
    t = np.asarray(t, float)
    qd = np.where(t >= t_step, q1, q0).astype(float)
    return qd, np.zeros_like(qd)


def ramp_reference(t, q0: float, rate: float):
    """Constant-velocity reference qd = q0 + rate t, qd' = rate (MR Fig. 11.8)."""
    t = np.asarray(t, float)
    return q0 + rate * t, np.full_like(t, rate)


def load_knee_swing():
    """Illustrative knee swing-phase reference (data/knee_swing.csv): t, qd, qd', qd'' (1 ms grid)."""
    with open(DATA / "knee_swing.csv") as fh:
        rows = list(csv.reader(fh))[1:]
    a = np.array(rows, float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


# ==========================================================================================
# Part A -- error dynamics (MR Sec. 11.2)
# ==========================================================================================
def second_order_params(a1: float, a0: float):
    """Damping ratio and natural frequency of  e'' + a1 e' + a0 e = 0  compared with MR Eq. 11.8:
    a1 = 2 zeta wn, a0 = wn^2.  Requires a0 > 0.  Returns (zeta, wn)."""
    # TODO (second_order_params): see the docstring; MR reference in the module header.
    raise NotImplementedError("TODO: second_order_params")


def char_roots(coeffs) -> np.ndarray:
    """Roots of the characteristic polynomial  c0 s^p + c1 s^(p-1) + ... + cp  (MR Eq. 11.3), coefficients
    highest power first (numpy.roots convention)."""
    # TODO (char_roots): see the docstring; MR reference in the module header.
    raise NotImplementedError("TODO: char_roots")


def is_stable(coeffs, tol: float = 1e-12) -> bool:
    """True if every root of the characteristic polynomial has a negative real part (MR p. 410)."""
    # TODO (is_stable): see the docstring; MR reference in the module header.
    raise NotImplementedError("TODO: is_stable")


def unit_error_response(zeta: float, wn: float, t):
    """Unit error response e(t) of MR Eq. 11.8 with e(0) = 1, e'(0) = 0 (MR p. 413):
      zeta > 1:  c1 e^{s1 t} + c2 e^{s2 t},  s1,2 = -zeta wn +- wn sqrt(zeta^2 - 1),
                 c1 = 1/2 + zeta / (2 sqrt(zeta^2 - 1)),  c2 = 1/2 - zeta / (2 sqrt(zeta^2 - 1));
      zeta = 1:  (1 + wn t) e^{-wn t};
      zeta < 1:  (cos wd t + zeta / sqrt(1 - zeta^2) sin wd t) e^{-zeta wn t},  wd = wn sqrt(1 - zeta^2).
    Treat |zeta - 1| < 1e-9 as critical damping.  t may be a scalar or an array."""
    # TODO (unit_error_response): see the docstring; MR reference in the module header.
    raise NotImplementedError("TODO: unit_error_response")


def overshoot_from_zeta(zeta: float) -> float:
    """Percent overshoot of the second-order unit error response, exp(-pi zeta / sqrt(1 - zeta^2)) x 100
    for 0 <= zeta < 1 and 0 for zeta >= 1 (MR p. 415)."""
    # TODO (overshoot_from_zeta): see the docstring; MR reference in the module header.
    raise NotImplementedError("TODO: overshoot_from_zeta")


def zeta_from_overshoot(overshoot_pct: float) -> float:
    """Inverse of overshoot_from_zeta for 0 < overshoot < 100 %:  zeta = -ln(OS) / sqrt(pi^2 + ln^2(OS)),
    OS = overshoot_pct / 100."""
    # TODO (zeta_from_overshoot): see the docstring; MR reference in the module header.
    raise NotImplementedError("TODO: zeta_from_overshoot")


def step_metrics(t, e, ess: float | None = None, band: float = 0.02) -> dict:
    """Measured metrics of an error response e(t) (MR Sec. 11.2.1, p. 408):
      ess            steady-state error (default: the last sample e[-1]);
      overshoot      |e_min - ess| / |e(0) - ess| x 100 if the response crosses ess, else 0, where e_min is
                     the extreme value on the far side of ess (use the normalised d = (e - ess)/(e(0) - ess):
                     overshoot = max(0, -min d) x 100);
      settling_time  the first sample time T such that |e(t) - ess| <= band |e(0) - ess| for all t >= T
                     (nan if the last sample is still outside the band).
    Returns dict(ess=..., overshoot=..., settling_time=...)."""
    # TODO (step_metrics): see the docstring; MR reference in the module header.
    raise NotImplementedError("TODO: step_metrics")


# ==========================================================================================
# Part B -- motion control with velocity inputs (MR Sec. 11.3.1)
# ==========================================================================================
def pi_velocity_gains(zeta: float, wn: float):
    """Gains of the velocity-input PI controller (MR Eq. 11.13) giving the error dynamics
    e'' + Kp e' + Ki e = 0 (MR Eq. 11.14) the target zeta and wn:  Kp = 2 zeta wn, Ki = wn^2.  Returns (Kp, Ki)."""
    # TODO (pi_velocity_gains): see the docstring; MR reference in the module header.
    raise NotImplementedError("TODO: pi_velocity_gains")


def simulate_velocity_joint(t, qd, qd_dot, kp: float, ki: float = 0.0, q0: float = 0.0,
                            feedforward: bool = False, qdot_max: float | None = None) -> dict:
    """Discrete-time simulation of a joint with a velocity input (the servo realises the commanded velocity
    exactly, MR Sec. 11.3).  At every sample k (dt = t[1] - t[0]):
        e_k = qd_k - q_k ;  E_k = E_{k-1} + e_k dt  (E_{-1} = 0) ;
        u_k = [qd_dot_k if feedforward] + kp e_k + ki E_k            (MR Eqs. 11.12, 11.13, 11.15)
        u_k is clipped to +-qdot_max if given; while clipped (|u| at the limit and the error has the sign of u)
        the integral is NOT updated (E_k = E_{k-1}; simple anti-windup);
        q_{k+1} = q_k + u_k dt.
    Returns dict(q, u, e) arrays of len(t) (u and e at every sample; q[0] = q0)."""
    # TODO (simulate_velocity_joint): see the docstring; MR reference in the module header.
    raise NotImplementedError("TODO: simulate_velocity_joint")


# ==========================================================================================
# Part C -- PID control with torque inputs (MR Sec. 11.4.1.1)
# ==========================================================================================
def pd_gains(M: float, b: float, zeta: float, wn: float):
    """PD gains for the setpoint error dynamics  M e'' + (b + Kd) e' + Kp e = 0  (MR Eq. 11.25) with the
    target zeta and wn (MR Eq. 11.26):  Kp = M wn^2,  Kd = 2 zeta wn M - b.  Returns (Kp, Kd)."""
    # TODO (pd_gains): see the docstring; MR reference in the module header.
    raise NotImplementedError("TODO: pd_gains")


def ki_max(M: float, b: float, kp: float, kd: float) -> float:
    """Upper bound on Ki for stability of the PID third-order error dynamics (MR Eq. 11.30, p. 427):
    0 < Ki < (b + Kd) Kp / M."""
    # TODO (ki_max): see the docstring; MR reference in the module header.
    raise NotImplementedError("TODO: ki_max")


def pid_char_poly(M: float, b: float, kp: float, ki: float, kd: float) -> np.ndarray:
    """Coefficients (highest power first) of MR Eq. 11.30:  s^3 + (b + Kd)/M s^2 + Kp/M s + Ki/M.
    With ki = 0 return the PD polynomial  s^2 + (b + Kd)/M s + Kp/M  (MR Eq. 11.26)."""
    # TODO (pid_char_poly): see the docstring; MR reference in the module header.
    raise NotImplementedError("TODO: pid_char_poly")


def pd_gravity_error(plant: dict, kp: float, qd: float) -> float:
    """Steady-state error of PD setpoint control under gravity (MR Eq. 11.27): at rest Kp e = g(q) with
    q = qd - e.  Solve  f(e) = Kp e - g(qd - e) = 0  for e by Newton's method from e = g(qd)/Kp
    (f'(e) = Kp + g'(qd - e), g'(q) = -g_cos sin q + g_sin cos q).  Returns e (rad)."""
    # TODO (pd_gravity_error): see the docstring; MR reference in the module header.
    raise NotImplementedError("TODO: pd_gravity_error")


class JointPID:
    """Discrete PID joint controller (MR Eq. 11.23, pseudocode MR Fig. 11.15) with an optional torque limit
    and integrator anti-windup (MR p. 427; Astrom-Murray Sec. 10.4).

        ctrl = JointPID(kp, ki, kd, dt, tau_max=None, anti_windup=True)
        ctrl.reset(q0)                       # error integral = 0, previous angle = q0
        tau = ctrl.update(qd, qd_dot, q, qdot=None, tau_ff=0.0)

    update() at every servo cycle:
        if qdot is None: qdot = (q - q_prev) / dt   (0 at the first call after reset without q0)
        q_prev = q
        e = qd - q ;  e_dot = qd_dot - qdot
        eint_new = eint + e dt
        v = Kp e + Kd e_dot + Ki eint_new + tau_ff
        tau = clip(v, -tau_max, tau_max)  (no clipping if tau_max is None)
        anti-windup: if anti_windup and v was clipped and e has the sign of v, keep eint (do not integrate)
                     and recompute tau = clip(Kp e + Kd e_dot + Ki eint + tau_ff); otherwise eint = eint_new.
    Attributes: kp, ki, kd, dt, tau_max, anti_windup, eint, q_prev, terms (dict P, I, D of the last call),
    saturated (bool, last call).  Gains may be changed between calls (ROS parameters do this)."""

    def __init__(self, kp: float, ki: float, kd: float, dt: float, tau_max: float | None = None,
                 anti_windup: bool = True):
        self.kp, self.ki, self.kd, self.dt = float(kp), float(ki), float(kd), float(dt)
        self.tau_max = None if tau_max is None else float(tau_max)
        self.anti_windup = bool(anti_windup)
        self.reset()

    def reset(self, q0: float | None = None) -> None:
        # TODO (reset): see the docstring; MR reference in the module header.
        raise NotImplementedError("TODO: reset")

    def update(self, qd: float, qd_dot: float, q: float, qdot: float | None = None, tau_ff: float = 0.0) -> float:
        # TODO (update): see the docstring; MR reference in the module header.
        raise NotImplementedError("TODO: update")


def simulate_joint(plant: dict, ctrl: JointPID, t, qd, qd_dot, q0: float, dq0: float = 0.0,
                   substeps: int = 4, velocity: str = "measured", enc_res: float | None = None,
                   tau_ff=None, gravity: bool = True) -> dict:
    """Closed-loop simulation of  tau = M q'' + b q' + g(q)  (MR Eq. 11.21) under a sampled controller.
    t is the controller time grid (dt = t[1] - t[0] must equal ctrl.dt).  ctrl.reset(q0) first; then at every
    sample k:
        q_meas = q_k rounded to a multiple of enc_res (if given; an encoder), qdot_meas = dq_k if
        velocity == "measured", else None (the controller differences q_meas, MR Fig. 11.15);
        tau_k = ctrl.update(qd[k], qd_dot[k], q_meas, qdot_meas, tau_ff[k] if tau_ff is not None else 0);
        the actuator applies tau_k clipped to +-plant["tau_max"] (if not None), held constant until t_{k+1};
        integrate (q, q') from t_k to t_{k+1} with `substeps` rk4_step calls of plant_rhs(..., gravity).
    Returns dict(t, q, dq, tau, e) arrays of len(t) (tau = applied torque, e = qd - q)."""
    # TODO (simulate_joint): see the docstring; MR reference in the module header.
    raise NotImplementedError("TODO: simulate_joint")
