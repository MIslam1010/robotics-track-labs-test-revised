"""SBEG 461 Lab 9 -- trajectory library, torque feasibility and time-optimal time scaling (TOPP) on the course arm.
STARTER -- you edit this file. Replace every `raise NotImplementedError` with your code (14 functions).
Run the auto-checks with:  LAB_IMPL=starter python -m pytest -q test_lab.py
Rules: do not call modern_robotics or scipy in this file (the tests use them as ground truth); use numpy,
toppra (only inside `topp`) and the provided helpers (rnea, inverse_dynamics, the loaders, paths and models, ...).

Notation follows Modern Robotics (MR) Ch. 9: a trajectory theta(t) = theta(s(t)) is a path theta(s),
s in [0, 1], composed with a time scaling s(t), t in [0, T] (MR Sec. 9.1, Eqs. 9.1-9.2); theta' = d theta/ds,
theta'' = d^2 theta/ds^2.  Dynamics in MR list form (MR Sec. 8.3.2): Mlist, Glist, Slist, gravity g.
Along a path the dynamics are (MR Eqs. 9.32-9.33)
    m(s) sdd + c(s) sd^2 + g(s) = tau,
    m(s) = M(theta) theta',  c(s) = M(theta) theta'' + theta'^T Gamma(theta) theta',  g(s) = g(theta),
and the actuator limits become L(s, sd) <= sdd <= U(s, sd) (MR Eqs. 9.36-9.37).  Units: SI.

Parts
  A. Point-to-point time scalings: cubic, quintic, trapezoid       MR Sec. 9.2.2, Eqs. 9.9-9.24
     + straight-line joint trajectories                             MR Eqs. 9.3-9.5
  B. Via points: cubic via-point interpolation                      MR Sec. 9.3, Eqs. 9.25-9.29
     + uniform cubic B-spline                                       MR p. 338; Gotor Sec. 7.1.4
  C. Torque feasibility and uniform time scaling                    MR Eqs. 8.50-8.54 (RNEA), 9.30-9.31; Sec. 9.7
  D. Time-optimal time scaling: path dynamics, motion cone,         MR Sec. 9.4, Eqs. 9.32-9.37
     velocity limit curve, TOPP with toppra, TOPP vs quintic
  E. A command file for the course arm (provided)                   course-arm hardware safety checklist

These functions become the trajectory and time-scaling modules of your `sbeg461_dynamics` library; Lab 11 (RRT,
smoothing and time scaling) and the course project use the same names.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EPS_M = 1e-9            # |m_i(s)| below this is treated as a zero-inertia point (MR Sec. 9.4.4)

# ==========================================================================================
# Provided helpers (do not edit)
# ==========================================================================================


def skew(w):
    """[w] for a 3-vector (MR Eq. 3.30)."""
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def TransInv(T):
    R, p = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R.T, -R.T @ p
    return out


def Adjoint(T):
    """[Ad_T] (MR Def. 3.20)."""
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3], Ad[3:, :3], Ad[3:, 3:] = R, skew(p) @ R, R
    return Ad


def ad(V):
    """[ad_V] (MR Eq. 8.38)."""
    out = np.zeros((6, 6))
    out[:3, :3], out[3:, :3], out[3:, 3:] = skew(V[:3]), skew(V[3:]), skew(V[:3])
    return out


def MatrixExp6(S, theta):
    """e^{[S] theta} for a screw axis with ||omega|| = 1 or (omega = 0, ||v|| = 1) (MR Prop. 3.25)."""
    w, v = np.asarray(S[:3], float), np.asarray(S[3:], float)
    T = np.eye(4)
    if np.linalg.norm(w) < 1e-12:
        T[:3, 3] = v * theta
        return T
    W = skew(w)
    R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * W @ W
    G = np.eye(3) * theta + (1 - np.cos(theta)) * W + (theta - np.sin(theta)) * W @ W
    T[:3, :3], T[:3, 3] = R, G @ v
    return T


def rnea(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist):
    """Recursive Newton-Euler inverse dynamics (MR Eqs. 8.50-8.54), same interface as
    modern_robotics.InverseDynamics.  This is the Lab 2/Lab 4 function; you may paste your own."""
    n = len(thetalist)
    Slist = np.asarray(Slist, float)
    Mi = np.eye(4)
    Ai = np.zeros((6, n))
    AdTi = [None] * (n + 1)
    Vi = np.zeros((6, n + 1))
    Vdi = np.zeros((6, n + 1))
    Vdi[3:, 0] = -np.asarray(g, float)
    AdTi[n] = Adjoint(TransInv(Mlist[n]))
    Fi = np.asarray(Ftip, float).copy()
    tau = np.zeros(n)
    for i in range(n):
        Mi = Mi @ Mlist[i]
        Ai[:, i] = Adjoint(TransInv(Mi)) @ Slist[:, i]
        Ti = MatrixExp6(-Ai[:, i], thetalist[i]) @ TransInv(Mlist[i])
        AdTi[i] = Adjoint(Ti)
        Vi[:, i + 1] = AdTi[i] @ Vi[:, i] + Ai[:, i] * dthetalist[i]
        Vdi[:, i + 1] = (AdTi[i] @ Vdi[:, i] + Ai[:, i] * ddthetalist[i]
                         + ad(Vi[:, i + 1]) @ Ai[:, i] * dthetalist[i])
    for i in range(n - 1, -1, -1):
        Fi = (AdTi[i + 1].T @ Fi + Glist[i] @ Vdi[:, i + 1]
              - ad(Vi[:, i + 1]).T @ (Glist[i] @ Vi[:, i + 1]))
        tau[i] = Fi @ Ai[:, i]
    return tau


def _arm_dict(d, Glist, kf, tf):
    return dict(name="course arm", n=6, Slist=np.array(d["Slist"]), M=np.array(d["M"]),
                Mlist=[np.array(m) for m in d["Mlist"]], Glist=Glist,
                g=np.array(d["g"]), qdot_max=kf * np.array(d["qdot_max"]),
                tau_max=tf * np.array(d["tau_max"]), limits=np.array(d["limits"]),
                q_home=np.array(d["q_home"]))


def load_course_arm(payload=True, derated=True):
    """The course arm (Course resources: course robot models; numbers identical, copied into data/course_arm.json).

    Returns a dict with name, n, Slist (6x6), M, Mlist (7 4x4), Glist (6 6x6), g (3,), qdot_max (6,), tau_max (6,),
    limits (6x2), q_home (6,).  payload=True uses the 0.25 kg maximum payload at the tool point.
    derated=True applies the derating of Parts A-C (50 % of rated speed, 60 % of rated torque).
    All inertial data and ratings are ILLUSTRATIVE values."""
    d = json.loads((DATA / "course_arm.json").read_text())
    kf = d["derated_speed_fraction"] if derated else 1.0
    tf = d["derated_current_fraction"] if derated else 1.0
    Glist = [np.array(G) for G in (d["Glist_payload"] if payload else d["Glist"])]
    return _arm_dict(d, Glist, kf, tf)


def course_arm_model(payload_kg=0.0):
    """The course arm with a point-mass payload of `payload_kg` kg at the tool point and its RATED limits
    (Part D applies its own limit fractions).  The payload is added to the last link as in the course robot
    models: G_6 += [Ad_T]^T diag(0, 0, 0, m, m, m) [Ad_T], T = (M_{6,tool})^{-1} (MR Eq. 8.42)."""
    d = json.loads((DATA / "course_arm.json").read_text())
    Glist = [np.array(G) for G in d["Glist"]]
    if payload_kg:
        Ad = Adjoint(TransInv(np.array(d["Mlist"][-1])))
        Glist[-1] = Glist[-1] + Ad.T @ np.diag([0, 0, 0, payload_kg, payload_kg, payload_kg]) @ Ad
    return _arm_dict(d, Glist, 1.0, 1.0)


def load_tasks():
    """Task configurations A, B, C, D, W (rad) and via times (s) of Parts A-C (illustrative)."""
    t = json.loads((DATA / "tasks.json").read_text())
    return {k: (np.array(v) if isinstance(v, list) else v) for k, v in t.items()}


def load_topp_task(name="topp_task.json"):
    """Task data of Parts D-E (waypoints, limit fractions, payload, servo period); illustrative values."""
    return json.loads((DATA / name).read_text())


def export_csv(path, t, Q, Qd):
    """Write t, q1..qn, qd1..qdn (SI units) to a CSV file."""
    n = Q.shape[1]
    header = "t," + ",".join(f"q{i + 1}" for i in range(n)) + "," + ",".join(f"qd{i + 1}" for i in range(n))
    np.savetxt(path, np.column_stack([t, Q, Qd]), delimiter=",", header=header, comments="", fmt="%.6f")


def write_log(path, t, cmd) -> None:
    """Write a command file in the project log format (project starter `hardware.py`): t, cmd_1..cmd_n."""
    t = np.asarray(t, float)
    cmd = np.asarray(cmd, float).reshape(len(t), -1)
    header = ["t"] + [f"cmd_{j + 1}" for j in range(cmd.shape[1])]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(np.column_stack([t, cmd]).tolist())


# ==========================================================================================
# Part A -- point-to-point time scalings (MR Sec. 9.2.2)
# ==========================================================================================


def cubic_time_scaling(T, t):
    """Cubic rest-to-rest time scaling s = 3(t/T)^2 - 2(t/T)^3 (MR Eq. 9.9 with a2 = 3/T^2, a3 = -2/T^3).

    t may be a scalar or an array; it is clipped to [0, T].  Returns (s, sdot, sddot) with the same
    shape as t.  At t = 0 and t = T return the one-sided values (sddot = +-6/T^2)."""
    # TODO: MR Eqs. 9.9-9.13: s = a2 t^2 + a3 t^3 with a2 = 3/T^2, a3 = -2/T^3; differentiate twice.
    #       Clip t to [0, T] first (np.clip).
    raise NotImplementedError("cubic_time_scaling")


def quintic_time_scaling(T, t):
    """Quintic rest-to-rest time scaling s = 10 r^3 - 15 r^4 + 6 r^5, r = t/T, with zero end velocities
    and accelerations (MR Sec. 9.2.2.1, Exercise 9.5; Fig. 9.4).  Returns (s, sdot, sddot)."""
    # TODO: Solve MR Exercise 9.5 (six boundary conditions) or use r = t/T: s = 10 r^3 - 15 r^4 + 6 r^5;
    #       remember the 1/T and 1/T^2 factors of the chain rule.
    raise NotImplementedError("quintic_time_scaling")


def trapezoid_params(v, a):
    """Total time and acceleration time of a trapezoidal time scaling with s-units v (1/s) and a (1/s^2).

    Three-stage profile if v^2/a <= 1:  T = (a + v^2)/(v a), t_a = v/a  (MR p. 334).
    If v^2/a > 1 the cruise speed is never reached (MR Exercise 9.10): bang-bang profile with
    t_a = T/2 = 1/sqrt(a) and peak speed sqrt(a).  Returns (T, t_a, v_peak)."""
    # TODO: MR p. 334: t_a = v/a and T = (a + v^2)/(v a) if v^2/a <= 1; otherwise bang-bang (MR Ex. 9.10, 9.13).
    raise NotImplementedError("trapezoid_params")


def trapezoid_time_scaling(v, a, t):
    """Trapezoidal time scaling (MR Eqs. 9.16-9.24); bang-bang (triangle) if v^2/a > 1.
    Returns (s, sdot, sddot) for scalar or array t (clipped to [0, T])."""
    # TODO: Three phases, MR Eqs. 9.16-9.24 (use v_peak from trapezoid_params so the bang-bang case also
    #       works). np.where helps for arrays.
    raise NotImplementedError("trapezoid_time_scaling")


def trapezoid_from_limits(theta_start, theta_end, qdot_lim, qddot_lim):
    """Provided.  Fastest trapezoidal straight-line motion (MR Eqs. 9.14-9.15): the largest v and a (s-units)
    with |(theta_end - theta_start) v| <= qdot_lim and |(theta_end - theta_start) a| <= qddot_lim, joint by
    joint (joints that do not move impose no limit).  Returns (v, a, T); T uses your trapezoid_params."""
    D = np.abs(np.asarray(theta_end, float) - np.asarray(theta_start, float))
    moving = D > 1e-12
    vl = np.broadcast_to(np.asarray(qdot_lim, float), D.shape)
    al = np.broadcast_to(np.asarray(qddot_lim, float), D.shape)
    v = np.min(vl[moving] / D[moving])
    a = np.min(al[moving] / D[moving])
    T, _, _ = trapezoid_params(v, a)
    return float(v), float(a), float(T)


def joint_trajectory(theta_start, theta_end, T, N, method=5, v=None, a=None):
    """Straight line in joint space theta(s) = theta_start + s (theta_end - theta_start) (MR Eq. 9.3),
    time-scaled with method = 3 (cubic), 5 (quintic) or "trap" (trapezoid; needs v and a and ignores T).

    Returns (t, Q, Qd, Qdd): t (N,), Q/Qd/Qdd (N, n), with Qd = sdot D and Qdd = sddot D (MR Eqs. 9.4-9.5;
    the path is straight so d2theta/ds2 = 0).  The rows of Q equal modern_robotics.JointTrajectory."""
    # TODO: MR Eqs. 9.3-9.5: Q = q0 + s D, Qd = sdot D, Qdd = sddot D (np.outer is handy).
    #       t = np.linspace(0, T, N); for method == "trap" take T from trapezoid_params(v, a).
    raise NotImplementedError("joint_trajectory")


# ==========================================================================================
# Part B -- via points (MR Sec. 9.3) and B-splines
# ==========================================================================================


def via_cubic_coefficients(beta, beta_dot, times):
    """Cubic via-point interpolation of ONE coordinate (MR Eqs. 9.26-9.29).

    beta, beta_dot: (k,) via positions and velocities; times: (k,) increasing via times T_1 = 0 .. T_k.
    Returns a (k-1, 4) array whose row j is (a_j0, a_j1, a_j2, a_j3) for segment j."""
    # TODO: MR Eqs. 9.26-9.29, one row (a_j0, a_j1, a_j2, a_j3) per segment, dT = T_{j+1} - T_j.
    raise NotImplementedError("via_cubic_coefficients")


def heuristic_via_velocities(points, times):
    """Provided.  Via velocities from positions and times (MR p. 337: "a heuristic could be used").
    Rule used in this course (Lecture A), per coordinate: zero at the first and last via; at an
    interior via, the average of the slopes of the two adjacent segments if both slopes have the same
    strict sign, otherwise zero (the via is a turning point, or one side is flat).  points: (k, m); returns (k, m)."""
    P = np.asarray(points, float)
    P = P[:, None] if P.ndim == 1 else P
    Tm = np.asarray(times, float)
    V = np.zeros_like(P)
    for i in range(1, len(P) - 1):
        m1 = (P[i] - P[i - 1]) / (Tm[i] - Tm[i - 1])
        m2 = (P[i + 1] - P[i]) / (Tm[i + 1] - Tm[i])
        V[i] = np.where(m1 * m2 > 0, 0.5 * (m1 + m2), 0.0)
    return V


def via_trajectory(points, velocities, times, t):
    """Evaluate the cubic via-point trajectory of all m coordinates at times t (clipped to [T_1, T_k]).
    points, velocities: (k, m); times: (k,).  Returns (B, Bd, Bdd), each (len(t), m)."""
    # TODO: Find the segment of each t (np.searchsorted, side="right", minus 1, clipped to 0..k-2),
    #       dt = t - T_j, evaluate the cubic of via_cubic_coefficients and its two derivatives, coordinate by coordinate.
    raise NotImplementedError("via_trajectory")


def bspline_basis(u):
    """Provided.  The four uniform cubic B-spline blending functions at local parameter u in [0, 1]:
    b0 = (1-u)^3/6, b1 = (3u^3 - 6u^2 + 4)/6, b2 = (-3u^3 + 3u^2 + 3u + 1)/6, b3 = u^3/6.
    Returns an array of shape (4,) (or (4, len(u)))."""
    u = np.asarray(u, float)
    return np.array([(1 - u) ** 3, 3 * u**3 - 6 * u**2 + 4, -3 * u**3 + 3 * u**2 + 3 * u + 1, u**3]) / 6.0


def uniform_cubic_bspline(ctrl, u):
    """Uniform cubic B-spline with control points ctrl (n, m), n >= 4, at global parameter
    u in [0, n-3] (segment i = floor(u) uses P_i..P_{i+3}, local parameter w = u - i; clamp i to n - 4).
    p = sum_k b_k(w) P_{i+k} with b = bspline_basis(w); dp/du and d2p/du2 use the derivatives of the b_k.
    Returns (p, dp/du, d2p/du2), each (m,).  The curve is C2 and lies in the convex hull of the
    control points (MR p. 338)."""
    # TODO: Segment i = floor(u) (at most n-4), local w = u - i, p = bspline_basis(w) @ P[i:i+4];
    #       differentiate the four b_k(w) by hand for dp/du and d2p/du2 (dw/du = 1).
    raise NotImplementedError("uniform_cubic_bspline")


# ==========================================================================================
# Part C -- torque feasibility (MR Eqs. 9.30-9.31) and uniform time scaling
# ==========================================================================================


def torque_profile(robot, Q, Qd, Qdd):
    """Joint torques tau_k = M(q_k) qdd_k + c(q_k, qd_k) + g(q_k) along a sampled trajectory, by the RNEA.
    robot: dict from load_course_arm (or any model dict with Mlist, Glist, Slist, g).  Returns (N, n)."""
    # TODO: Call rnea(...) with Ftip = 0 for every sample (MR Eq. 9.30: tau = M qdd + c + g).
    raise NotImplementedError("torque_profile")


def check_limits(robot, Qd, Tau, tol=1e-9):
    """Provided.  Worst |velocity|/limit and |torque|/limit per joint along a trajectory.
    Returns dict(vel_ratio=(n,), tau_ratio=(n,), ok=bool)."""
    vr = np.max(np.abs(Qd) / robot["qdot_max"], axis=0)
    tr = np.max(np.abs(Tau) / robot["tau_max"], axis=0)
    return dict(vel_ratio=vr, tau_ratio=tr, ok=bool(np.all(vr <= 1 + tol) and np.all(tr <= 1 + tol)))


def min_time_scale(robot, Q, Qd, Qdd, qddot_max=None):
    """Smallest uniform stretch factor c > 0 that makes a sampled trajectory feasible.

    Stretching time by c (theta_c(t) = theta(t/c)) divides velocities by c and accelerations by c^2, and
    because c(theta, thetadot) is quadratic in thetadot (MR Sec. 9.4, c = thetadot^T Gamma thetadot)
        tau_c = g(theta) + d / c^2,   d = tau(Q, Qd, Qdd) - g(theta)     (MR Sec. 9.7, Hollerbach 1984).
    Velocity limits give c >= max |qd| / qdot_max (robot["qdot_max"]); if qddot_max is given, acceleration
    limits give c >= sqrt(max |qdd| / qddot_max).  Torque limits (robot["tau_max"], symmetric) give, sample by
    sample and joint by joint,
        d > 0:  c^2 >= d / (tau_max - g),      d < 0:  c^2 >= -d / (tau_max + g),
    which needs |g| < tau_max (otherwise the robot cannot even hold the configuration: raise ValueError).

    Returns dict(c_vel, c_acc, c_tau, c = the largest of the three, binding = ("velocity" | "acceleration" |
    "torque", joint index 0-based)); c_acc = 0 if qddot_max is None.  Ties go to velocity, then acceleration.
    For a rest-to-rest trajectory computed with T = 1, c is the minimum feasible duration in seconds."""
    # TODO: G = torques at zero velocity/acceleration; D = full torques - G; apply the two c^2 inequalities of the
    #       docstring sample by sample; c_vel = max |Qd|/qdot_max; c_acc = sqrt(max |Qdd|/qddot_max) if given.
    raise NotImplementedError("min_time_scale")


# ==========================================================================================
# Part D -- time-optimal time scaling (MR Sec. 9.4)
# ==========================================================================================
# Provided: paths and robot models.  Paths are called like toppra paths: path(s, 0) = theta,
# path(s, 1) = theta', path(s, 2) = theta''; `s` may be a float or an array.
try:                                         # toppra paths derive from AbstractGeometricPath
    from toppra.interpolator import AbstractGeometricPath as _PathBase
except Exception:                            # pragma: no cover - toppra missing
    _PathBase = object


class StraightLinePath(_PathBase):
    """theta(s) = q0 + s (q1 - q0), s in [0, 1] (MR Eq. 9.3).  order 0 -> theta, 1 -> theta' = q1 - q0,
    2 -> theta'' = 0."""

    def __init__(self, q0, q1):
        self.q0 = np.asarray(q0, float)
        self.q1 = np.asarray(q1, float)

    def __call__(self, s, order: int = 0):
        s_arr = np.asarray(s, float)
        d = self.q1 - self.q0
        if order == 0:
            return self.q0 + np.multiply.outer(s_arr, d)
        if order == 1:
            return np.broadcast_to(d, s_arr.shape + d.shape).copy()
        return np.zeros(s_arr.shape + d.shape)

    @property
    def dof(self) -> int:
        return len(self.q0)

    @property
    def path_interval(self):
        return np.array([0.0, 1.0])

    @property
    def waypoints(self):
        return np.array([0.0, 1.0]), np.vstack([self.q0, self.q1])


class SemicirclePath(_PathBase):
    """MR Fig. 9.9: x = xc + R cos(pi s), y = yc - R sin(pi s), s in [0, 1] (for the X-Y gantry)."""

    def __init__(self, R=0.3, xc=0.0, yc=0.0):
        self.R, self.xc, self.yc = R, xc, yc

    def __call__(self, s, order: int = 0):
        s = np.asarray(s, float)
        R, a = self.R, np.pi * s
        if order == 0:
            return np.stack([self.xc + R * np.cos(a), self.yc - R * np.sin(a)], -1)
        if order == 1:
            return np.stack([-np.pi * R * np.sin(a), -np.pi * R * np.cos(a)], -1)
        return np.stack([-np.pi**2 * R * np.cos(a), np.pi**2 * R * np.sin(a)], -1)

    @property
    def dof(self) -> int:
        return 2

    @property
    def path_interval(self):
        return np.array([0.0, 1.0])

    @property
    def waypoints(self):
        return np.array([0.0, 1.0]), self(np.array([0.0, 1.0]))


def waypoint_path(waypoints):
    """C2 cubic-spline path through the joint-space waypoints, s_k = k/(K-1), with theta'' = 0 at the
    ends ("natural" spline; toppra.SplineInterpolator).  The robot still starts and ends at rest because
    sdot = 0 there.  (A "clamped" spline, theta' = 0 at the ends, would make m(s) = M theta' = 0 at s = 0
    and 1: zero-inertia points, MR Sec. 9.4.4, which the discretised solver handles poorly.)"""
    import toppra as ta
    W = np.asarray(waypoints, float)
    return ta.SplineInterpolator(np.linspace(0.0, 1.0, len(W)), W, bc_type="natural")


def _trans(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


def mr_2r_model(L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=0.0) -> dict:
    """The MR 2R arm (MR Fig. 8.1, Eqs. 8.9-8.10): point masses at the link ends, gravity g along -y
    (g = 0: horizontal plane, as in MR Exercise 9.25)."""
    Mlist = [_trans([L1, 0, 0]), _trans([L2, 0, 0]), np.eye(4)]
    Glist = [np.diag([0, 0, 0, m1, m1, m1]).astype(float), np.diag([0, 0, 0, m2, m2, m2]).astype(float)]
    Slist = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, -L1, 0]], float).T
    return dict(name="MR 2R", n=2, Mlist=Mlist, Glist=Glist, Slist=Slist, g=np.array([0.0, -g, 0.0]))


def elbow_exo_model() -> dict:
    """Elbow exoskeleton + forearm + 1 kg bottle, vertical plane, theta = 0 with the forearm hanging down.
    Lumped link: m = 3.0 kg, sum(m r) = 0.698 kg m, inertia about the elbow 0.19734 kg m^2 (illustrative)."""
    m = 1.6 + 0.4 + 1.0
    mr_sum = 1.6 * 0.18 + 0.4 * 0.15 + 1.0 * 0.35
    I_elbow = 0.012 + 1.6 * 0.18**2 + 0.002 + 0.4 * 0.15**2 + 1.0 * 0.35**2
    rc = mr_sum / m
    Ic = I_elbow - m * rc**2
    Mlist = [_trans([0, -rc, 0]), _trans([0, -(0.35 - rc), 0])]
    Glist = [np.diag([Ic, Ic, Ic, m, m, m])]
    Slist = np.array([[0, 0, 1, 0, 0, 0]], float).T
    return dict(name="elbow exoskeleton", n=1, Mlist=Mlist, Glist=Glist, Slist=Slist,
                g=np.array([0.0, -9.81, 0.0]))


def gantry_model(Mx=40.0, my=15.0) -> dict:
    """X-Y gantry (two prismatic joints, horizontal plane) of a humanitarian-demining sensor head.
    Joint 1 (x) moves Mx in total (its own carriage Mx - my plus the y axis); joint 2 (y) moves my.
    Illustrative values."""
    Mlist = [np.eye(4), np.eye(4), np.eye(4)]
    Glist = [np.diag([0, 0, 0, Mx - my, Mx - my, Mx - my]).astype(float),
             np.diag([0, 0, 0, my, my, my]).astype(float)]
    Slist = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]], float).T
    return dict(name="gantry", n=2, Mlist=Mlist, Glist=Glist, Slist=Slist, g=np.zeros(3))


def inverse_dynamics(model: dict, q, dq, ddq, gravity: bool = True) -> np.ndarray:
    """The RNEA (`rnea` above) for a model dict; gravity=False sets g = 0."""
    g = model["g"] if gravity else np.zeros(3)
    return rnea(np.asarray(q, float), np.asarray(dq, float), np.asarray(ddq, float), g, np.zeros(6),
                model["Mlist"], model["Glist"], model["Slist"])


def path_dynamics(model: dict, path, s: float):
    """Return (m, c, g), each shape (n,), of MR Eq. 9.33 at path parameter s.

    Use three calls of inverse_dynamics (no new dynamics code):
      g(s) = ID(theta, 0, 0)            with gravity,
      m(s) = ID(theta, 0, theta')       without gravity  (= M theta'),
      c(s) = ID(theta, theta', theta'') without gravity  (= M theta'' + theta'^T Gamma theta').
    """
    # TODO: three calls of inverse_dynamics(model, ...) as in the docstring; return m, c, g
    raise NotImplementedError("path_dynamics")


def accel_limits(m, c, g, sdot: float, tau_min, tau_max):
    """(L, U) of MR Eqs. 9.36-9.37 at one state: the smallest and largest sdd that satisfy every
    tau_min_i <= m_i sdd + c_i sdot^2 + g_i <= tau_max_i.

    If m_i < 0 the two bounds swap.  If |m_i| < EPS_M (zero-inertia point) joint i gives no bound on
    sdd, but if c_i sdot^2 + g_i lies outside [tau_min_i, tau_max_i] the state is inadmissible: return
    (L, U) = (+inf, -inf).  The state is admissible when L <= U.
    """
    # TODO: loop over joints; MR Eq. 9.36 (swap the bounds if m_i < 0); handle |m_i| < EPS_M;
    #       L = max_i L_i, U = min_i U_i (MR Eq. 9.37); start from L = -inf, U = +inf
    raise NotImplementedError("accel_limits")


def velocity_limit(model: dict, path, s: float, tau_min, tau_max, qdot_max=None,
                   sdot_max: float = 100.0, tol: float = 1e-9) -> float:
    """sdot_lim(s): the largest sdot with L(s, sdot) <= U(s, sdot) (the velocity limit curve, MR Sec. 9.4.1),
    combined with the kinematic limit |theta'_i| sdot <= qdot_max_i if qdot_max is given.
    Bisection on sdot in [0, sdot_max] (assumes one admissible interval [0, sdot_lim], MR Sec. 9.4.4).
    Returns sdot_max if no limit is found below it, and 0 if the state (s, 0) is inadmissible."""
    # TODO: path_dynamics once; kinematic bound min_i qdot_max_i/|theta'_i| (skip |theta'_i| <= 1e-12),
    #       then bisection on sdot with accel_limits (admissible: L <= U)
    raise NotImplementedError("velocity_limit")


def topp(model: dict, path, tau_max, qdot_max=None, qddot_max=None, N: int = 200, tau_min=None) -> dict:
    """Time-optimal time scaling with toppra (TOPP-RA, a discretised, reachability-based solver of the
    same problem as MR Sec. 9.4).  Constraints: joint torques tau_min <= tau <= tau_max (default
    tau_min = -tau_max) via inverse_dynamics, and optionally |qdot| <= qdot_max, |qddot| <= qddot_max.
    Grid: N + 1 equally spaced points on [0, 1].  Start and end at rest.
    Returns dict(T, traj, s_grid, sdot_grid) where traj(t, order) gives q, qdot or qddot, and sdot_grid are
    the optimal path velocities on the grid (the phase-plane curve).
    Raises RuntimeError if toppra finds no feasible time scaling."""
    import toppra as ta
    import toppra.algorithm as algo
    import toppra.constraint as constraint
    ta.setup_logging("ERROR")
    # TODO: build toppra constraints: constraint.JointTorqueConstraint(inv_dyn, np.c_[tau_min, tau_max], np.zeros(n))
    #       with inv_dyn(q, qd, qdd) = inverse_dynamics(model, q, qd, qdd), and JointVelocityConstraint(np.c_[-v, v]) /
    #       JointAccelerationConstraint(np.c_[-a, a]) when the limits are given; grid = np.linspace(0, 1, N + 1);
    #       algo.TOPPRA(cons, path, gridpoints=grid, parametrizer="ParametrizeConstAccel"); compute_trajectory(0, 0)
    #       (None -> raise RuntimeError); T = traj.duration; sdot_grid = problem_data.sd_vec
    raise NotImplementedError("topp")


def quintic_traj(path, T: float):
    """Provided.  theta(s(t)) with your quintic s(t); returns a callable traj(t, order) like a toppra
    trajectory (order 0, 1, 2 via MR Eqs. 9.1-9.2).  Sampled on t in [0, 1], quintic_traj(path, 1.0) gives the
    Q, Qd, Qdd that min_time_scale needs for the TOPP-vs-quintic comparison of Part D."""
    def traj(t, order: int = 0):
        s, sd, sdd = quintic_time_scaling(T, t)
        if order == 0:
            return path(s, 0)
        d1 = path(s, 1)
        if order == 1:
            return d1 * np.asarray(sd)[..., None]
        return d1 * np.asarray(sdd)[..., None] + path(s, 2) * np.asarray(sd)[..., None] ** 2
    traj.duration = T
    return traj


def peak_usage(model: dict, traj, T: float, tau_max, qdot_max, N: int = 1000) -> dict:
    """Provided.  Peak |qdot|/qdot_max and |tau|/tau_max per joint, and the peak |jerk| (finite
    differences of qddot) along a trajectory callable traj(t, order) sampled at N + 1 instants on [0, T]."""
    t = np.linspace(0.0, T, N + 1)
    q, qd, qdd = traj(t, 0), traj(t, 1), traj(t, 2)
    tau = np.array([inverse_dynamics(model, a, b, c) for a, b, c in zip(q, qd, qdd)])
    jerk = np.diff(qdd, axis=0) / np.diff(t)[:, None]
    return dict(vel_ratio=np.max(np.abs(qd), 0) / np.asarray(qdot_max),
                tau_ratio=np.max(np.abs(tau), 0) / np.asarray(tau_max),
                jerk_max=float(np.max(np.abs(jerk))), tau=tau, t=t)


# ==========================================================================================
# Part E -- a command file for the course arm (provided)
# ==========================================================================================


def command_table(traj, T: float, dt: float = 0.02, slow: float = 1.0):
    """Provided.  Sample a trajectory for the servo stream.  `slow` >= 1 stretches time uniformly
    (q_cmd(t) = q(t / slow)), which divides all speeds by `slow` and accelerations by slow^2.
    Returns (t, q): t = 0, dt, ..., slow*T (the last sample is exactly slow*T), q shape (len(t), n)."""
    Tc = slow * T
    nstep = int(np.ceil(Tc / dt - 1e-9))
    t = np.minimum(np.arange(nstep + 1) * dt, Tc)
    q = np.asarray(traj(t / slow, 0), float)
    return t, q


def check_command(t, q, model: dict, speed_fraction: float, margin_deg: float = 10.0,
                  tol: float = 0.01) -> dict:
    """Provided.  Safety check of a command table before it goes to the arm (course-arm hardware safety
    checklist, rules 1 and 3; Course resources):
      * within_limits: every q stays at least `margin_deg` (checklist: 10 deg) inside the joint limits;
      * peak_speed_ratio: max over joints and samples of |dq/dt| / (speed_fraction * qdot_max)
        (finite differences of the samples; qdot_max = the RATED speeds of course_arm_model);
        speed_ok = ratio <= 1 + tol;
      * starts_at_rest / ends_at_rest: first and last steps are smaller than 1e-3 rad;
      * ok = all of the above.
    """
    t = np.asarray(t, float)
    q = np.asarray(q, float)
    lim = np.asarray(model["limits"], float)
    m = np.deg2rad(margin_deg)
    within = bool(np.all(q >= lim[:, 0] + m) and np.all(q <= lim[:, 1] - m))
    v = np.diff(q, axis=0) / np.diff(t)[:, None]
    ratio = float(np.max(np.abs(v) / (speed_fraction * np.asarray(model["qdot_max"], float))))
    steps = np.abs(np.diff(q, axis=0)).max(axis=1)
    start_rest = bool(steps[0] < 1e-3)
    end_rest = bool(steps[-1] < 1e-3)
    speed_ok = ratio <= 1.0 + tol
    return dict(ok=within and speed_ok and start_rest and end_rest, within_limits=within,
                peak_speed_ratio=ratio, speed_ok=speed_ok, starts_at_rest=start_rest, ends_at_rest=end_rest)
