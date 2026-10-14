"""SBEG 461 Lab 8 -- impedance control on a Hapkit-type haptic device and a simulated rehabilitation robot.
STARTER CODE. Complete every function marked TODO (keep the names and signatures).
Run the auto-checks with:  LAB_IMPL=starter python -m pytest -q test_lab.py
See lab_handout.md for the checkpoints.

Notation follows Modern Robotics (MR) Ch. 11.  Rendered impedance (MR Eq. 11.62):  m x'' + b x' + k x = f;
task-space impedance  M x'' + B x' + K x = f_ext  (MR Eq. 11.64); impedance-control law (MR Eq. 11.65);
admittance-control law  x''_d = M^-1 (f_ext - B x' - K x)  (MR Eq. 11.66); low-level torque control by motor
current, tau = k_t I through a transmission (MR Sec. 11.8).  Units: SI (s, m, kg, N, rad, N m).

Sign conventions used everywhere in this file
  * x is the handle (or hand) position, F_motor the force the device applies TO the user's hand,
    f_hand the force the hand applies TO the device.  Device:  m_h x'' + b_h x' + k_g x = F_motor + f_hand.
  * A rendered spring-damper about x0 applies F_motor = -K (x - x0) - B (v - v0).
  * Rehabilitation robot:  M(q) q'' + c(q, q') = tau + J^T(q) f_ext   (f_ext = force of the hand on the robot;
    MR Eq. 11.48 writes the same thing with F_tip = -f_ext).

Parts
  A. The haptic device as a 1-DOF plant: handle mass from MR's dynamics, gravity stiffness, the
     force -> motor current chain (MR Sec. 11.8), free-decay identification of m_h and b_h
  B. Rendering virtual springs, dampers and walls at 1 kHz: sampled-data simulation, static push test,
     the exact stability limit of the sampled loop (eigenvalues) and the first-order delay estimate,
     energy (passivity) check
  C. The rehabilitation robot: apparent (task-space) inertia (MR Eq. 8.90), impedance control without
     inertia shaping (MR Eq. 11.65 with M = 0) and admittance control (MR Eq. 11.66 + inverse dynamics)

These functions become `sbeg461_dynamics/interaction.py` in your cumulative library; the project
scenarios S1 (virtual tunnel) and S6 (casualty-handling arm) and the haptic-device hardware runs use them.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm

import course_models as cm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ==========================================================================================
# Provided (do not edit): data files, device parameters, rehab-robot kinematics and dynamics
# ==========================================================================================
def load_device() -> dict:
    """Nominal device and loop parameters from data/device.json (illustrative values): T (s), x_res (m),
    force_max (N), k_t (N m/A), i_max (A) and the nominal identified m_h, b_h, k_g used in Parts B and C."""
    return json.loads((DATA / "device.json").read_text())


def load_free_decay(K: int) -> tuple[np.ndarray, np.ndarray]:
    """A logged free-decay test: the handle is held at x0 and released while the device renders a spring
    K (N/m) with B = 0 at 1 kHz.  Returns (t, x) in s and m (x quantised to the sensor resolution)."""
    t, x = [], []
    with open(DATA / f"free_decay_K{K}.csv") as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["t"]))
            x.append(float(row["x"]))
    return np.array(t), np.array(x)


def rk4_step(f, x, h: float) -> np.ndarray:
    """One classical Runge-Kutta step of x' = f(x) (inputs held constant over the step)."""
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


REHAB = cm.rehab_2r()
_L1, _L2 = REHAB["params"]["L"]


def _rehab_inertial():
    """(m1, c1, I1, m2, c2, I2) of the rehab robot from its Glist / Mlist (I about the CoM, z axis)."""
    G1, G2 = REHAB["Glist"]
    return G1[3, 3], _L1 / 2, G1[2, 2], G2[3, 3], _L2 / 2, G2[2, 2]


def rehab_fk(q) -> np.ndarray:
    """Hand (end-effector) position p = (x, y) in {s} of the planar rehab robot (MR Example 4.1 style)."""
    q1, q2 = q
    return np.array([_L1 * math.cos(q1) + _L2 * math.cos(q1 + q2), _L1 * math.sin(q1) + _L2 * math.sin(q1 + q2)])


def rehab_jacobian(q) -> np.ndarray:
    """2 x 2 Jacobian of the hand position, p' = J(q) q'  (linear-velocity rows of J_s for a planar arm)."""
    q1, q2 = q
    s1, c1, s12, c12 = math.sin(q1), math.cos(q1), math.sin(q1 + q2), math.cos(q1 + q2)
    return np.array([[-_L1 * s1 - _L2 * s12, -_L2 * s12], [_L1 * c1 + _L2 * c12, _L2 * c12]])


def rehab_jacobian_dot(q, dq) -> np.ndarray:
    """Time derivative J'(q, q') of rehab_jacobian."""
    q1, q2 = q
    w1, w12 = dq[0], dq[0] + dq[1]
    s1, c1, s12, c12 = math.sin(q1), math.cos(q1), math.sin(q1 + q2), math.cos(q1 + q2)
    return np.array([[-_L1 * c1 * w1 - _L2 * c12 * w12, -_L2 * c12 * w12],
                     [-_L1 * s1 * w1 - _L2 * s12 * w12, -_L2 * s12 * w12]])


def rehab_ik(p) -> np.ndarray:
    """Elbow-positive (theta_2 > 0) inverse kinematics of the hand position p = (x, y)."""
    x, y = p
    c2 = (x * x + y * y - _L1**2 - _L2**2) / (2 * _L1 * _L2)
    q2 = math.acos(max(-1.0, min(1.0, c2)))
    q1 = math.atan2(y, x) - math.atan2(_L2 * math.sin(q2), _L1 + _L2 * math.cos(q2))
    return np.array([q1, q2])


def rehab_dynamics(q, dq) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form M(q) and c(q, q') of the rehab robot (horizontal plane, g(q) = 0); identical to
    modern_robotics MassMatrix / VelQuadraticForces of course_models.rehab_2r() (checked by the tests)."""
    m1, c1, I1, m2, c2, I2 = _rehab_inertial()
    cq2, sq2 = math.cos(q[1]), math.sin(q[1])
    M11 = I1 + m1 * c1**2 + I2 + m2 * (_L1**2 + c2**2 + 2 * _L1 * c2 * cq2)
    M12 = I2 + m2 * (c2**2 + _L1 * c2 * cq2)
    M22 = I2 + m2 * c2**2
    h = m2 * _L1 * c2 * sq2
    c = np.array([-h * (2 * dq[0] * dq[1] + dq[1] ** 2), h * dq[0] ** 2])
    return np.array([[M11, M12], [M12, M22]]), c


def simulate_rehab(controller, q0, T_end: float, dt: float = 0.001, f_ext=None, dq0=(0.0, 0.0),
                   substeps: int = 4, force_lag_hz: float | None = None) -> dict:
    """Simulate the rehab robot  M q'' + c = tau + J^T f_ext  (f_ext = force of the hand ON the robot).

    controller(t, q, dq, f_meas) -> tau (2,) is called every dt (zero-order hold); tau is clipped to tau_max.
    f_ext(t, p, v) -> (2,) hand force, evaluated continuously inside the RK4 steps (None = no hand).
    f_meas is the force-sensor reading at the sample: f_ext itself, or a first-order low-pass filtered
    version with cutoff force_lag_hz (a sensor/filter lag; MR p. 447 recommends low-pass filtering).
    Returns t, q, dq, p (hand position), v (hand velocity), tau, f (true hand force at the samples)."""
    f_ext = f_ext or (lambda t, p, v: np.zeros(2))
    tau_max = REHAB["tau_max"]
    n = int(round(T_end / dt))
    t = np.arange(n + 1) * dt
    X = np.zeros((n + 1, 4))
    X[0] = np.r_[q0, dq0]
    TAU, F = np.zeros((n + 1, 2)), np.zeros((n + 1, 2))
    f_filt = None
    alpha = None if force_lag_hz is None else 1.0 - math.exp(-2 * math.pi * force_lag_hz * dt)
    for k in range(n + 1):
        q, dq = X[k, :2], X[k, 2:]
        p, v = rehab_fk(q), rehab_jacobian(q) @ dq
        fk = np.asarray(f_ext(t[k], p, v), float)
        F[k] = fk
        if alpha is None:
            f_meas = fk
        else:
            f_filt = fk.copy() if f_filt is None else f_filt + alpha * (fk - f_filt)
            f_meas = f_filt
        tau = np.clip(np.asarray(controller(t[k], q, dq, f_meas), float), -tau_max, tau_max)
        TAU[k] = tau
        if k == n:
            break
        h = dt / substeps

        def rhs(x, tk=t[k]):
            qq, ww = x[:2], x[2:]
            Mq, cq = rehab_dynamics(qq, ww)
            Jq = rehab_jacobian(qq)
            fe = np.asarray(f_ext(tk, rehab_fk(qq), Jq @ ww), float)
            return np.r_[ww, np.linalg.solve(Mq, tau + Jq.T @ fe - cq)]

        x = X[k].copy()
        for _ in range(substeps):
            x = rk4_step(rhs, x, h)
        X[k + 1] = x
    P = np.array([rehab_fk(x[:2]) for x in X])
    V = np.array([rehab_jacobian(x[:2]) @ x[2:] for x in X])
    return dict(t=t, q=X[:, :2], dq=X[:, 2:], p=P, v=V, tau=TAU, f=F)


# ==========================================================================================
# Part A -- the haptic device as a 1-DOF plant
# ==========================================================================================
def handle_mass(model: dict) -> float:
    """Effective translational mass m_h (kg) felt at the handle at theta = 0:
    m_h = (I_pivot + J_rotor * ratio^2) / rh^2, with I_pivot the link inertia about the joint axis
    (= modern_robotics MassMatrix at theta = 0) and the rotor inertia reflected through the
    transmission ratio rs/rp (MR Sec. 8.9.2, apparent inertia G^2 J)."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: handle_mass")


def gravity_stiffness(model: dict) -> float:
    """Translational stiffness k_g (N/m) at the handle caused by gravity on the sector mass below the
    pivot: k_g = (d g(theta)/d theta at 0) / rh^2, with g(theta) = modern_robotics GravityForces."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: gravity_stiffness")


def handle_force_to_current(F, model: dict, k_t: float):
    """Motor current I (A) that produces the handle force F (N) (MR Sec. 11.8, current control of a geared
    motor): joint torque tau = F rh, motor torque tau_m = tau / ratio, I = tau_m / k_t.
    Friction in the transmission is ignored (that is exactly the error MR p. 448 warns about)."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: handle_force_to_current")


def fit_free_decay(t, x, min_fraction: float = 0.1) -> tuple[float, float]:
    """Natural frequency wn (rad/s) and damping ratio zeta of a logged free decay x(t) (rest at x = 0).

    Method: positive peaks (scipy.signal.find_peaks with a minimum spacing of half the dominant period),
    keep peaks above min_fraction of the first one; damped period T_d = mean spacing; logarithmic
    decrement delta = -slope of ln(peak) against peak number (least squares);
    zeta = delta / sqrt(4 pi^2 + delta^2); wn = (2 pi / T_d) / sqrt(1 - zeta^2)."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: fit_free_decay")


def identify_device(Ks, fits) -> dict:
    """Identify the device from free decays at several rendered stiffnesses Ks with fits [(wn, zeta), ...].

    Model of each test:  m_h x'' + b_app x' + (K + k_g) x = 0,  so  wn^2 = K/m_h + k_g/m_h  and
    b_app = 2 zeta wn m_h.  The sample-and-hold of the rendered spring acts like a delay tau_eff, which
    to first order removes damping K tau_eff:  b_app = b_h - tau_eff K.  Two straight-line fits give
    m_h = 1/slope(wn^2 vs K), k_g = intercept * m_h, b_h = intercept(b_app vs K), tau_eff = -slope.
    Returns dict(m_h, k_g, b_h, tau_eff, b_app=[...])."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: identify_device")


# ==========================================================================================
# Part B -- rendering springs, dampers and walls with a sampled-data loop
# ==========================================================================================
def spring_damper_force(x, v, K: float, B: float, x0: float = 0.0, v0: float = 0.0) -> float:
    """Force rendered on the hand by a virtual spring-damper about (x0, v0):  F = -K (x - x0) - B (v - v0)
    (MR Eq. 11.62 without the mass term: impedance control with M = 0, MR p. 446)."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: spring_damper_force")


def wall_force(x, v, K: float, B: float, x_wall: float, side: int = +1) -> float:
    """Unilateral virtual wall occupying side*(x - x_wall) > 0.  Outside the wall F = 0.  Inside,
    F = -K (x - x_wall) - B v, but the wall may only PUSH the hand out (never pull it in): the result is
    clipped so that side*F <= 0."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: wall_force")


def simulate_device(dev: dict, law, T_end: float, x0: float = 0.0, v0: float = 0.0, f_hand=None,
                    delay_steps: int = 0, quantize: bool = True, substeps: int = 10) -> dict:
    """Sampled-data simulation of the haptic device rendering `law`.

    Plant (continuous):  m_h x'' + b_h x' + k_g x = F_applied + f_hand(t, x, v)  with dev["m_h"], dev["b_h"],
    dev["k_g"]; integrated with `substeps` RK4 steps per sample (F_applied held: zero-order hold).
    Controller every T = dev["T"] at t_k = k T:
      1. x_meas = x rounded to the sensor resolution dev["x_res"] (if quantize), else x;
      2. v_est = (x_meas_k - x_meas_{k-1}) / T  (backward difference; v_est = 0 at k = 0);
      3. F_cmd = law(t_k, x_meas, v_est), saturated to +-dev["force_max"];
      4. F_applied during [t_k, t_k+1) is the F_cmd computed delay_steps samples earlier (0 before that).
    Returns dict t, x, v (true state at the samples), x_meas, v_est, F_cmd, F_applied, saturated (bool array)."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: simulate_device")


def static_push_test(dev: dict, K: float, B: float, F_load: float, T_end: float = 2.0) -> dict:
    """Rendered stiffness from a static push: a constant hand force F_load (e.g. a hanging weight) acts on
    the device rendering a spring K, damper B about x0 = 0.  The steady deflection is the mean of x over
    the last 0.2 s; K_rendered = F_load / x_ss.  Returns dict(x_ss, K_rendered, error_pct = 100 (K_rendered
    - K)/K).  (K_rendered includes the small gravity stiffness k_g of the paddle.)"""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: static_push_test")


def closed_loop_matrix(m: float, b: float, k_g: float, K: float, B: float, T: float,
                       delay_steps: int = 0) -> np.ndarray:
    """Exact discrete-time closed-loop matrix of the linear sampled-data loop of simulate_device
    (no quantisation, no saturation).

    Plant ZOH discretisation of  m x'' + b x' + k_g x = F  (A_c = [[0, 1], [-k_g/m, -b/m]], B_c = [0, 1/m]):
    A_d = expm(A_c T), B_d = int_0^T expm(A_c s) ds B_c (use the augmented-matrix exponential).
    Controller c_k = -K x_k - B (x_k - x_{k-1}) / T; applied force F_k = c_{k-d}, d = delay_steps.
    State z_k = [x_k, v_k, x_{k-1}, c_{k-1}, ..., c_{k-d}] (dimension 3 + d)."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: closed_loop_matrix")


def spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def max_stable_K(m: float, b: float, k_g: float, B: float, T: float, delay_steps: int = 0,
                 K_hi: float = 1e5, tol: float = 0.01) -> float:
    """Largest K (N/m) for which closed_loop_matrix(...) has spectral radius < 1 (bisection on [0, K_hi],
    to within tol).  Assumes the loop is stable at K = 0 and loses stability once as K grows."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: max_stable_K")


def first_order_K_limit(b: float, B: float, T: float, delay_steps: int = 0) -> float:
    """First-order (Taylor) estimate of the stiffness limit: the spring force is late by the zero-order
    hold (T/2) plus the loop delay (d T), and a late spring  K x(t - tau) ~ K x - K tau x'  removes damping
    K tau.  The loop stays damped while b + B - K tau > 0:  K_max ~ (b + B) / (T/2 + d T)."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: first_order_K_limit")


def device_energy(dev: dict, sim: dict, K: float, x0: float = 0.0) -> np.ndarray:
    """Total energy of device + rendered spring at the samples: E = 1/2 m_h v^2 + 1/2 (K + k_g)(x - x0)^2
    (k_g about x = 0).  For a passive rendering with no hand E must not grow."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: device_energy")


def oscillation_growth(sim: dict, t_from: float = 0.5) -> float:
    """Ratio of the peak |x| in the last quarter of the run to the peak |x| in [t_from, t_from + quarter):
    < 1 decaying, > 1 growing (a quick stability verdict from a simulation)."""
    t, x = sim["t"], np.abs(sim["x"])
    q = (t[-1] - t_from) / 4
    a = x[(t >= t_from) & (t < t_from + q)].max()
    z = x[t >= t[-1] - q].max()
    return float(z / a) if a > 0 else 0.0


# ==========================================================================================
# Part C -- the rehabilitation robot: apparent inertia, impedance and admittance control
# ==========================================================================================
def task_inertia(q) -> np.ndarray:
    """Task-space (apparent) inertia of the hand, Lambda = J^-T M J^-1  (MR Eq. 8.90), 2 x 2 (kg)."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: task_inertia")


def apparent_mass(q, u) -> float:
    """Mass felt when the hand is pushed along the unit direction u:  m_u = 1 / (u^T Lambda^-1 u)
    (the acceleration along u per unit force along u is u^T Lambda^-1 u)."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: apparent_mass")


def impedance_control_torque(q, dq, p_d, v_d, K, B) -> np.ndarray:
    """Impedance (stiffness + damping) control without inertia shaping or force sensing:
    tau = J^T ( K (p_d - p) + B (v_d - v) )  -- MR Eq. 11.65 with the mass term and the arm-dynamics
    compensation dropped (MR p. 446: 'it is not uncommon to eliminate the mass compensation term ... and to
    set M = 0'); g(q) = 0 in the horizontal plane.  K, B: 2 x 2 matrices (or scalars)."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: impedance_control_torque")


def admittance_control_torque(q, dq, f_meas, p0, M_d, B_d, K_d) -> np.ndarray:
    """Admittance control (MR Sec. 11.7.2): desired hand acceleration from the sensed force,
    a_d = M_d^-1 (f_meas - B_d v - K_d (p - p0))  (MR Eq. 11.66 with the spring about p0); joint acceleration
    q''_d = J^-1 (a_d - J' q') (J is square here, J^dagger = J^-1); inverse dynamics including the sensed
    hand force:  tau = M(q) q''_d + c(q, q') - J^T f_meas.  M_d, B_d, K_d: 2 x 2 matrices or scalars."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("TODO: admittance_control_torque")


def pulse_force(direction, amplitude: float, t_on: float, t_off: float):
    """Hand-force profile: amplitude * unit(direction) for t_on <= t < t_off, else 0 (a call-back for
    simulate_rehab)."""
    u = np.asarray(direction, float) / np.linalg.norm(direction)

    def f(t, p, v):
        return amplitude * u if t_on <= t < t_off else np.zeros(2)
    return f


def one_dof_pulse_response(m: float, b: float, k: float, amplitude: float, t_on: float, t_off: float,
                           t) -> np.ndarray:
    """Displacement of the 1-DOF impedance m x'' + b x' + k x = f(t) (MR Eq. 11.62) from rest for the
    rectangular force pulse, sampled at times t (exact: matrix exponential of the state equation)."""
    A = np.array([[0.0, 1.0], [-k / m, -b / m]])
    t = np.asarray(t, float)
    xs = np.zeros(len(t))
    # response = step(t - t_on) - step(t - t_off) of amplitude
    def step(tau):
        if tau <= 0:
            return 0.0
        # step response x_s(tau) = amplitude/k * (1 - [1 0] expm(A tau) [1 0]^T) - ... use direct formula
        Aug = np.zeros((3, 3))
        Aug[:2, :2] = A
        Aug[1, 2] = amplitude / m
        return (expm(Aug * tau) @ np.array([0.0, 0.0, 1.0]))[0]
    for i, ti in enumerate(t):
        xs[i] = step(ti - t_on) - step(ti - t_off)
    return xs
