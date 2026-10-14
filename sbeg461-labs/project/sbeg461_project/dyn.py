"""Dynamics helpers and a fixed-step simulator built on the official `modern_robotics` library.

Equations of motion (MR Eq. 8.1 and 8.81):
    tau = M(q) qdd + h(q, qd) + J_b^T(q) F_tip,     h = c(q, qd) + g(q)
where F_tip is the wrench the end-effector applies to the environment, expressed in {b}.
A force f_ext that the environment (e.g. a patient's hand) applies to the robot therefore enters
as F_tip = -F_ext,b.
"""
from __future__ import annotations

import modern_robotics as mr
import numpy as np

from .models import Scenario, spatial_inertia_to_mci


# ----------------------------------------------------------------------------- model terms
def mass_matrix(s: Scenario, q) -> np.ndarray:
    return mr.MassMatrix(np.asarray(q, float), s.Mlist, s.Glist, s.Slist)


def gravity(s: Scenario, q, g=None) -> np.ndarray:
    return mr.GravityForces(np.asarray(q, float), s.g if g is None else g, s.Mlist, s.Glist, s.Slist)


def h_vector(s: Scenario, q, qd, g=None) -> np.ndarray:
    """h(q, qd) = c(q, qd) + g(q): inverse dynamics with zero acceleration and zero tip wrench."""
    n = s.n
    return mr.InverseDynamics(np.asarray(q, float), np.asarray(qd, float), np.zeros(n),
                              s.g if g is None else g, np.zeros(6), s.Mlist, s.Glist, s.Slist)


def inverse_dynamics(s: Scenario, q, qd, qdd, Ftip=None, g=None) -> np.ndarray:
    """Recursive Newton-Euler inverse dynamics (MR Sec. 8.3.2)."""
    return mr.InverseDynamics(np.asarray(q, float), np.asarray(qd, float), np.asarray(qdd, float),
                              s.g if g is None else g, np.zeros(6) if Ftip is None else Ftip,
                              s.Mlist, s.Glist, s.Slist)


def forward_dynamics(s: Scenario, q, qd, tau, Ftip=None, g=None) -> np.ndarray:
    """qdd = M^-1 (tau - h - J^T F_tip)   (MR Eq. 8.81)."""
    q = np.asarray(q, float)
    qd = np.asarray(qd, float)
    b = np.asarray(tau, float) - inverse_dynamics(s, q, qd, np.zeros(s.n), Ftip, g)
    return np.linalg.solve(mass_matrix(s, q), b)


def energy(s: Scenario, q, qd, g=None) -> float:
    """Total mechanical energy K + P, K = 1/2 qd^T M qd (MR Eq. 8.55 summed), P = -sum_i m_i g^T p_i,
    p_i = centre of mass of link i (read from G_i, so payloads folded into G_i are included)."""
    g = s.g if g is None else np.asarray(g, float)
    qd = np.asarray(qd, float)
    K = 0.5 * qd @ mass_matrix(s, q) @ qd
    P = 0.0
    Ts = s.link_transforms(q)
    homes = s.com_frames_home()
    for i in range(s.n):
        m, c, _ = spatial_inertia_to_mci(s.Glist[i])
        T = Ts[i + 1] @ homes[i]
        P -= m * g @ (T[:3, :3] @ c + T[:3, 3])
    return float(K + P)


# ----------------------------------------------------------------------------- external forces
def ftip_from_force(s: Scenario, q, f_ext_space, m_ext_space=None) -> np.ndarray:
    """Convert a force f_ext (and optional moment, about the {b} origin) applied BY the environment
    TO the robot's end-effector, both expressed in {s} axes, into MR's F_tip (wrench applied BY the
    robot, expressed in {b}).

    The external wrench about the {s} origin is F_s = (p x f + m, f) and F_b = [Ad_Tsb]^T F_s
    (MR Eq. 3.97)."""
    T = s.fk(q)
    p = T[:3, 3]
    f = np.asarray(f_ext_space, float)
    m = np.zeros(3) if m_ext_space is None else np.asarray(m_ext_space, float)
    F_s = np.r_[np.cross(p, f) + m, f]
    F_b = mr.Adjoint(T).T @ F_s
    return -F_b


# ----------------------------------------------------------------------------- simulator
def simulate(s: Scenario, controller, q0, qd0, T: float, dt: float = 1e-3, *,
             f_ext=None, g_fn=None, saturate: bool = True, integrator: str = "rk4",
             engine: str = "auto", delay_steps: int = 0):
    """Simulate the true plant `s` under `controller` with a zero-order-hold torque.

    controller(t, q, qd) -> tau           (called once per step, before saturation)
    f_ext(t, q, qd)      -> (f, m) or f   external force [and moment] ON the robot tip, in {s}
    g_fn(t)              -> gravity vector in {s} (time-varying; models base tilt / acceleration)
    integrator           "rk4" or "euler" (semi-implicit Euler)
    delay_steps          transport delay of the torque command, in steps of dt (models sensing,
                         computation and communication latency)
    engine               "mr" (modern_robotics, reference), "pinocchio" (fast ABA, same model;
                         cross-checked against mr in tests/test_models.py) or "auto" (pinocchio
                         when installed, else mr)

    Returns a dict of arrays: t, q, qd, tau (applied), tau_cmd (commanded), sat (bool per step).
    """
    N = int(round(T / dt))
    n = s.n
    q, qd = np.array(q0, float), np.array(qd0, float)
    out = dict(t=np.zeros(N + 1), q=np.zeros((N + 1, n)), qd=np.zeros((N + 1, n)),
               tau=np.zeros((N + 1, n)), tau_cmd=np.zeros((N + 1, n)), sat=np.zeros(N + 1, bool))

    def ftip(t, qq, qqd):
        if f_ext is None:
            return None
        r = f_ext(t, qq, qqd)
        if isinstance(r, tuple):
            return ftip_from_force(s, qq, r[0], r[1])
        return ftip_from_force(s, qq, r)

    if engine == "auto":
        from .pin_bridge import HAVE_PINOCCHIO
        engine = "pinocchio" if HAVE_PINOCCHIO else "mr"
    if engine == "pinocchio":
        from .pin_bridge import PinEngine
        fd = PinEngine(s).forward_dynamics
    else:
        def fd(qq, qqd, tau, Ftip, g):
            return forward_dynamics(s, qq, qqd, tau, Ftip, g)

    def acc(t, qq, qqd, tau):
        g = None if g_fn is None else g_fn(t)
        return fd(qq, qqd, tau, ftip(t, qq, qqd), g)

    pipe = [np.zeros(n)] * delay_steps
    for k in range(N + 1):
        t = k * dt
        tau_cmd = np.asarray(controller(t, q, qd), float)
        if delay_steps:
            pipe.append(tau_cmd)
            tau_cmd = pipe.pop(0)
        tau = np.clip(tau_cmd, -s.tau_max, s.tau_max) if saturate else tau_cmd
        out["t"][k], out["q"][k], out["qd"][k] = t, q, qd
        out["tau"][k], out["tau_cmd"][k] = tau, tau_cmd
        out["sat"][k] = bool(np.any(np.abs(tau_cmd) > s.tau_max + 1e-12))
        if k == N:
            break
        if integrator == "euler":
            qd = qd + dt * acc(t, q, qd, tau)
            q = q + dt * qd
        else:
            k1q, k1v = qd, acc(t, q, qd, tau)
            k2q, k2v = qd + 0.5 * dt * k1v, acc(t + dt / 2, q + 0.5 * dt * k1q, qd + 0.5 * dt * k1v, tau)
            k3q, k3v = qd + 0.5 * dt * k2v, acc(t + dt / 2, q + 0.5 * dt * k2q, qd + 0.5 * dt * k2v, tau)
            k4q, k4v = qd + dt * k3v, acc(t + dt, q + dt * k3q, qd + dt * k3v, tau)
            q = q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
            qd = qd + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return out
