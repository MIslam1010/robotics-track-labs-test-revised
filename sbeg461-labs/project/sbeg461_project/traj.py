"""Reference trajectories and feasibility checks (MR Ch. 9).

Each reference returns (q, qd, qdd) or (x, xd, xdd) at time t, so that controllers can use the
feedforward acceleration (computed-torque control needs it, MR Eq. 11.37).
"""
from __future__ import annotations

import numpy as np

from .dyn import inverse_dynamics
from .models import Scenario


# ----------------------------------------------------------------------------- time scalings
def quintic_s(t: float, T: float):
    """Quintic time scaling with zero end velocities and accelerations (MR Sec. 9.2.2.1):
    s = 10 tau^3 - 15 tau^4 + 6 tau^5, tau = t/T. Returns (s, sdot, sddot)."""
    tau = np.clip(t / T, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
    sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
    if t <= 0 or t >= T:
        sd, sdd = 0.0, 0.0
    return s, sd, sdd


def joint_quintic(q0, q1, T: float, t: float):
    """Straight line in joint space with a quintic time scaling (MR Eq. 9.3 with Sec. 9.2.2.1)."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    s, sd, sdd = quintic_s(t, T)
    d = q1 - q0
    return q0 + s * d, sd * d, sdd * d


def min_jerk_line(A, B, T: float, t: float):
    """Straight task-space line from A to B with the quintic (minimum-jerk) time scaling."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    s, sd, sdd = quintic_s(t, T)
    d = B - A
    return A + s * d, sd * d, sdd * d


# ----------------------------------------------------------------------------- scenario references
def gait_reference(t: float, period: float = 1.2):
    """S2: synthetic gait-like hip/knee reference (rad), ILLUSTRATIVE, two Fourier harmonics.

    hip  q1 = 0.15 + 0.33 sin(w t) + 0.04 sin(2 w t)       (about -0.2 .. 0.5 rad)
    knee q2 = 0.62 - 0.45 cos(w t) - 0.15 cos(2 w t)       (0.02 .. 0.94 rad)
    Teams may replace it with a public, recorded gait data set (cite it)."""
    w = 2 * np.pi / period
    q = np.array([0.15 + 0.33 * np.sin(w * t) + 0.04 * np.sin(2 * w * t),
                  0.62 - 0.45 * np.cos(w * t) - 0.15 * np.cos(2 * w * t)])
    qd = np.array([0.33 * w * np.cos(w * t) + 0.08 * w * np.cos(2 * w * t),
                   0.45 * w * np.sin(w * t) + 0.30 * w * np.sin(2 * w * t)])
    qdd = np.array([-0.33 * w**2 * np.sin(w * t) - 0.16 * w**2 * np.sin(2 * w * t),
                    0.45 * w**2 * np.cos(w * t) + 0.60 * w**2 * np.cos(2 * w * t)])
    return q, qd, qdd


def squat_reference(t: float, period: float = 2.4, knee=(0.10, 0.60), lean: float = 0.15):
    """S4: loaded knee-flexion cycle (step-down / squat) with forward trunk lean (rad), ILLUSTRATIVE.

    knee q1 = a + (b - a)(1 - cos(w t))/2 ;  hip q2 = q1 + lean (1 - cos(w t))/2
    The pack's absolute angle is q1 - q2 = -lean (1 - cos w t)/2 (forward lean, clockwise)."""
    a, b = knee
    w = 2 * np.pi / period
    c, sn = np.cos(w * t), np.sin(w * t)
    r = 0.5 * (1 - c)
    rd = 0.5 * w * sn
    rdd = 0.5 * w**2 * c
    q1, q1d, q1dd = a + (b - a) * r, (b - a) * rd, (b - a) * rdd
    return (np.array([q1, q1 + lean * r]), np.array([q1d, q1d + lean * rd]),
            np.array([q1dd, q1dd + lean * rdd]))


# ----------------------------------------------------------------------------- feasibility
def torque_profile(s: Scenario, qs, qds, qdds, g=None) -> np.ndarray:
    """Inverse dynamics along a sampled trajectory; returns an (N, n) torque array."""
    return np.array([inverse_dynamics(s, q, qd, qdd, g=g) for q, qd, qdd in zip(qs, qds, qdds)])


def check_limits(s: Scenario, qds, taus, tol: float = 1e-9) -> dict:
    """Report the worst velocity and torque ratios (|value| / limit) along a trajectory."""
    vr = np.max(np.abs(qds) / s.qdot_max, axis=0)
    tr = np.max(np.abs(taus) / s.tau_max, axis=0)
    return dict(vel_ratio=vr, tau_ratio=tr, ok=bool(np.all(vr <= 1 + tol) and np.all(tr <= 1 + tol)))


def min_quintic_time(s: Scenario, q0, q1, g=None, N: int = 60, T_hi: float = 60.0) -> float:
    """BASELINE: the shortest duration T of a rest-to-rest joint-space quintic from q0 to q1 that
    respects the velocity and torque limits (bisection on T). The team's time-optimal scaling
    (MR Sec. 9.4) must beat the stop-and-go sum of these durations."""
    def feasible(T):
        ts = np.linspace(0, T, N)
        sam = [joint_quintic(q0, q1, T, t) for t in ts]
        qs, qds, qdds = (np.array(x) for x in zip(*sam))
        return check_limits(s, qds, torque_profile(s, qs, qds, qdds, g))["ok"]

    lo, hi = 1e-2, T_hi
    if not feasible(hi):
        raise ValueError("no feasible quintic even at T_hi (static torque limit exceeded?)")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if feasible(mid) else (mid, hi)
    return hi


def stop_and_go_time(s: Scenario, path, g=None) -> float:
    """BASELINE total time to follow a waypoint path with rest-to-rest quintics on every segment."""
    return float(sum(min_quintic_time(s, path[i], path[i + 1], g) for i in range(len(path) - 1)))


# ----------------------------------------------------------------------------- team work
def time_optimal_scaling(s: Scenario, path, **kw):
    """TODO (team): time-optimal time scaling of a smooth path theta(s) subject to
    tau_min <= m(s) sdd + c(s) sd^2 + g(s) <= tau_max (MR Eq. 9.33-9.37) and the joint velocity
    limits. Implement the MR Sec. 9.4.2 algorithm or use `toppra` and explain it.
    Return a function t -> (q, qd, qdd) and the duration T."""
    raise NotImplementedError("time_optimal_scaling: team implementation required")
