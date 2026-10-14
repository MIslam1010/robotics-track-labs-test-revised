"""SBEG 361 Lab 10 -- Resolved-rate control, damped least squares, redundancy and a hexapod (STARTER).

Complete every function marked TODO (15 functions); keep the names and signatures.
The helpers at the top are complete (Labs 2-7). Do NOT import modern_robotics here: the tests compare against it.
Run:  LAB_IMPL=starter python -m pytest -q test_lab.py

Notation follows Lynch & Park, *Modern Robotics* (MR):
  pseudo-inverse  J^+ = J^T (J J^T)^-1 (fat, full rank), (J^T J)^-1 J^T (tall)      (MR Sec. 6.2.2, p. 227)
  inverse velocity kinematics  thetadot = J^+(theta) V_d                             (MR Eq. 6.7, p. 231)
  weighted pseudo-inverse  W^-1 J^T (J W^-1 J^T)^-1  (W = M(theta) on MR p. 232)
  redundancy  thetadot = J^+ xdot + (I - J^+ J) thetadot_0   (null-space projector I - J^+ J)
  damped least squares  J^#(lam) = J^T (J J^T + lam^2 I)^-1 = sum_i sigma_i/(sigma_i^2 + lam^2) v_i u_i^T
  task-space feedforward + P control  V_b = [Ad_{X^-1 X_d}] V_d + k_p X_e,  [X_e] = log(X^-1 X_d)
                                                                                     (MR Eq. 11.16 with K_i = 0, p. 421)
  Newton-Raphson IK in the body frame  theta <- theta + J_b^+ V_b                    (MR Sec. 6.2.2, pp. 228-229)
  Stewart-Gough platform  s_i = |p + R b_i - a_i|,  sdot = J_s^-1 V_s,
  J_s^-1 = [ -n_1 x q_1 ... -n_6 x q_6 ; n_1 ... n_6 ]^T, q_i = a_i                    (MR Secs. 7.1.2, 7.2.1, pp. 247-252)
Twists V = (omega, v): angular part first.
"""
import csv
import json
from pathlib import Path

import numpy as np

TOL = 1e-6  # same threshold as modern_robotics.NearZero
DATA = Path(__file__).resolve().parent / "data"


# =============================================================== helpers (complete; from Labs 2-7)
def vec_to_so3(omega):
    """3x3 skew-symmetric matrix [omega] (MR Eq. 3.30)."""
    w1, w2, w3 = np.asarray(omega, float)
    return np.array([[0.0, -w3, w2], [w3, 0.0, -w1], [-w2, w1, 0.0]])


def so3_to_vec(so3mat):
    m = np.asarray(so3mat, float)
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def vec_to_se3(V):
    """4x4 matrix [V] of a twist V = (omega, v) (MR Eq. 3.71)."""
    V = np.asarray(V, float)
    out = np.zeros((4, 4))
    out[:3, :3] = vec_to_so3(V[:3])
    out[:3, 3] = V[3:]
    return out


def se3_to_vec(se3mat):
    m = np.asarray(se3mat, float)
    return np.r_[m[2, 1], m[0, 2], m[1, 0], m[:3, 3]]


def trans_inv(T):
    """T^-1 = (R^T, -R^T p) (MR Eq. 3.64)."""
    R, p = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def adjoint(T):
    """[Ad_T] = [[R, 0], [[p]R, R]] (MR Def. 3.20)."""
    R, p = T[:3, :3], T[:3, 3]
    out = np.zeros((6, 6))
    out[:3, :3] = R
    out[3:, 3:] = R
    out[3:, :3] = vec_to_so3(p) @ R
    return out


def matrix_exp3(so3mat):
    """Rodrigues' formula (MR Eq. 3.51)."""
    w = so3_to_vec(so3mat)
    th = np.linalg.norm(w)
    if th < TOL:
        return np.eye(3) + so3mat
    K = so3mat / th
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def matrix_exp6(se3mat):
    """e^[V] for [V] in se(3) (MR Eq. 3.88).  For a rotation below TOL the first-order form
    (I + [omega], v) is used (modern_robotics would drop the rotation; Newton iterations on tiny
    corrections need it)."""
    se3mat = np.asarray(se3mat, float)
    w = so3_to_vec(se3mat[:3, :3])
    th = np.linalg.norm(w)
    out = np.eye(4)
    if th < TOL:
        out[:3, :3] += se3mat[:3, :3]
        out[:3, 3] = se3mat[:3, 3]
        return out
    K = se3mat[:3, :3] / th
    v = se3mat[:3, 3] / th
    out[:3, :3] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    out[:3, 3] = (np.eye(3) * th + (1 - np.cos(th)) * K + (th - np.sin(th)) * K @ K) @ v
    return out


def matrix_log3(R):
    """[w]theta = log R, theta in [0, pi] (MR Sec. 3.2.3.3)."""
    R = np.asarray(R, float)
    c = np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
    if c >= 1 - 1e-12:
        return np.zeros((3, 3))
    if c <= -1 + 1e-12:
        if 1 + R[2, 2] > 1e-9:
            w = np.array([R[0, 2], R[1, 2], 1 + R[2, 2]]) / np.sqrt(2 * (1 + R[2, 2]))
        elif 1 + R[1, 1] > 1e-9:
            w = np.array([R[0, 1], 1 + R[1, 1], R[2, 1]]) / np.sqrt(2 * (1 + R[1, 1]))
        else:
            w = np.array([1 + R[0, 0], R[1, 0], R[2, 0]]) / np.sqrt(2 * (1 + R[0, 0]))
        return vec_to_so3(np.pi * w)
    th = np.arccos(c)
    return th / (2 * np.sin(th)) * (R - R.T)


def matrix_log6(T):
    """[V] = log T (MR Sec. 3.3.3.2)."""
    T = np.asarray(T, float)
    R, p = T[:3, :3], T[:3, 3]
    W = matrix_log3(R)
    out = np.zeros((4, 4))
    if np.allclose(W, 0):
        out[:3, 3] = p
        return out
    th = np.linalg.norm(so3_to_vec(W))
    K = W / th
    Ginv = np.eye(3) / th - K / 2 + (1 / th - 0.5 / np.tan(th / 2)) * K @ K
    out[:3, :3] = W
    out[:3, 3] = (Ginv @ p) * th
    return out


def fk_space(M, Slist, theta):
    """T(theta) = e^[S1]th1 ... e^[Sn]thn M (MR Eq. 4.14)."""
    T = np.eye(4)
    for S, th in zip(np.asarray(Slist, float).T, theta):
        T = T @ matrix_exp6(vec_to_se3(S) * th)
    return T @ M


def body_screws(Slist, M):
    """B_i = [Ad_{M^-1}] S_i (MR Eq. 4.16)."""
    return adjoint(trans_inv(M)) @ np.asarray(Slist, float)


def jacobian_space(Slist, theta):
    """J_s(theta) (MR Eq. 5.11) -- Lab 7."""
    Slist = np.asarray(Slist, float)
    J = Slist.copy()
    T = np.eye(4)
    for i in range(1, Slist.shape[1]):
        T = T @ matrix_exp6(vec_to_se3(Slist[:, i - 1]) * theta[i - 1])
        J[:, i] = adjoint(T) @ Slist[:, i]
    return J


def jacobian_body(Blist, theta):
    """J_b(theta) (MR Eq. 5.18) -- Lab 7."""
    Blist = np.asarray(Blist, float)
    n = Blist.shape[1]
    J = Blist.copy()
    T = np.eye(4)
    for i in range(n - 2, -1, -1):
        T = T @ matrix_exp6(vec_to_se3(Blist[:, i + 1]) * -theta[i + 1])
        J[:, i] = adjoint(T) @ Blist[:, i]
    return J


def load_robot(name, path=None):
    """(Slist, M, info) of a model in data/robots.json (numbers identical to the course robot models)."""
    with open(path or DATA / "robots.json") as f:
        d = json.load(f)[name]
    return np.array(d["Slist"], float), np.array(d["M"], float), d


def load_hexapod(path=None):
    """(A, B, info): base joint points a_i in {s} and platform joint points b_i in {b} (6 x 3 each)."""
    with open(path or DATA / "hexapod.json") as f:
        d = json.load(f)
    return np.array(d["a"], float), np.array(d["b"], float), d


def read_joint_log(path):
    """(t, cmd_deg, meas_deg) from a log in the project's hwcompare.py format
    (header t, cmd_1..cmd_n, meas_1..meas_n[, stop]; angles in degrees)."""
    with open(path) as f:
        rows = list(csv.reader(f))
    head, a = rows[0], np.array(rows[1:], float)
    n = sum(h.startswith("cmd_") for h in head)
    return a[:, 0], a[:, 1:1 + n], a[:, 1 + n:1 + 2 * n]


def write_joint_log(path, t, cmd_deg, meas_deg=None):
    """Write a command (and optionally measurement) log in the hwcompare.py format."""
    cmd_deg = np.atleast_2d(cmd_deg)
    meas_deg = cmd_deg if meas_deg is None else np.atleast_2d(meas_deg)
    n = cmd_deg.shape[1]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"cmd_{i+1}" for i in range(n)] + [f"meas_{i+1}" for i in range(n)])
        for k in range(len(t)):
            w.writerow([f"{t[k]:.4f}"] + [f"{x:.4f}" for x in cmd_deg[k]] + [f"{x:.4f}" for x in meas_deg[k]])


# =============================================================== Checkpoint 1 -- inverse velocity kinematics
def pinv_svd(J, tol=1e-10):
    """Moore-Penrose pseudo-inverse from the SVD J = U Sigma V^T: J^+ = V Sigma^+ U^T,
    where Sigma^+ inverts every singular value > tol * sigma_max and sets the others to 0
    (MR Sec. 6.2.2, p. 227: minimum-norm / least-squares solution of J y = z)."""
    # TODO: U, s, Vt = np.linalg.svd(J, full_matrices=False)
    # TODO: invert only the singular values > tol * s[0]; J^+ = V diag(1/s) U^T
    # TODO: do NOT call np.linalg.pinv here (the tests compare with it)
    raise NotImplementedError("pinv_svd")


def dls_inverse(J, lam):
    """Damped least-squares inverse J^#(lam) = J^T (J J^T + lam^2 I)^-1.

    It minimises |J thetadot - V|^2 + lam^2 |thetadot|^2 and equals (J^T J + lam^2 I)^-1 J^T.
    lam = 0 must return the pseudo-inverse (use pinv_svd)."""
    # TODO: lam == 0: return pinv_svd(J)
    # TODO: otherwise J^T (J J^T + lam^2 I)^-1 -- use np.linalg.solve, not inv, if you can
    raise NotImplementedError("dls_inverse")


def dls_singular_gains(J, lam):
    """(sigma, gain): the singular values of J (descending) and the gains sigma/(sigma^2 + lam^2)
    that J^#(lam) applies along each singular direction (1/sigma when lam = 0)."""
    # TODO: s = np.linalg.svd(J, compute_uv=False)
    # TODO: gain = s / (s^2 + lam^2); for lam = 0 the gain is 1/s (inf where s = 0)
    raise NotImplementedError("dls_singular_gains")


def adaptive_lambda(sigma_min, eps, lam_max):
    """Damping that switches on only near a singularity:
    lam = 0 if sigma_min >= eps, else lam = lam_max * sqrt(1 - (sigma_min/eps)^2)."""
    # TODO: one if/else; return a float
    raise NotImplementedError("adaptive_lambda")


# =============================================================== Checkpoint 2 -- redundancy and null space
def null_projector(J):
    """P = I - J^+ J (n x n): projects any joint velocity onto the null space of J."""
    # TODO: I - J^+ J with your pinv_svd
    raise NotImplementedError("null_projector")


def weighted_pinv(J, W):
    """W^-1 J^T (J W^-1 J^T)^-1: the thetadot with J thetadot = V that minimises
    (1/2) thetadot^T W thetadot (W symmetric positive definite; W = mass matrix on MR p. 232)."""
    # TODO: W^-1 J^T (J W^-1 J^T)^-1
    raise NotImplementedError("weighted_pinv")


def redundant_rates(J, xdot, qdot0):
    """thetadot = J^+ xdot + (I - J^+ J) qdot0 (primary task + secondary motion in the null space)."""
    # TODO: two terms: pinv_svd(J) @ xdot and null_projector(J) @ qdot0
    raise NotImplementedError("redundant_rates")


def joint_centering_gradient(theta, limits):
    """Gradient of h(theta) = 1/2 sum_i ((theta_i - c_i)/r_i)^2, c_i = mid-range, r_i = range.
    qdot0 = -k grad h moves the joints towards the middle of their ranges."""
    # TODO: c = mid-range, r = range, gradient = (theta - c) / r^2 (element-wise)
    raise NotImplementedError("joint_centering_gradient")


# =============================================================== Checkpoint 3 -- resolved-rate control
def line_trajectory(T0, displacement, speed, dt):
    """Straight line at constant speed and constant orientation, starting at T0.

    Returns (t, Xd, Vd): times 0, dt, ..., t_f (t_f = |displacement|/speed, last sample exactly at t_f),
    the desired poses X_d(t) (list of 4x4) and the desired BODY twists V_d(t) (N x 6),
    [V_d] = X_d^-1 Xdot_d = (0, R0^T u speed) while moving and 0 at the last sample."""
    # TODO: tf = |d| / speed, n = round(tf/dt) steps, t = linspace(0, n dt, n+1) with t[-1] = tf
    # TODO: X_d(t): orientation R0, position p0 + u * min(speed t, |d|)
    # TODO: V_d = (0, 0, 0, R0^T u speed) for all samples except the last (zero)
    raise NotImplementedError("line_trajectory")


def resolved_rate_sim(Slist, M, theta0, Xd, Vd, dt, kp=0.0, lam=0.0, qdot_max=None,
                      eps=None, lam_max=None):
    """Simulate body-frame resolved-rate control with feedforward and proportional feedback:
        V_b = [Ad_{X^-1 X_d}] V_d + kp * X_e,   [X_e] = log(X^-1 X_d)          (MR Eq. 11.16, K_i = 0)
        thetadot = J_b^#(lam) V_b                                              (MR Eq. 6.7, damped)
    If eps and lam_max are given, lam = adaptive_lambda(sigma_min(J_b), eps, lam_max) at every step.
    If qdot_max is given and some |thetadot_i| > qdot_max_i, scale the WHOLE vector down (keeps its
    direction). Integrate with explicit Euler: theta_{k+1} = theta_k + thetadot_k dt.

    Returns dict with arrays (N = len(Xd)):
      theta (N x n), qdot (N x n; last row zeros), pos_err (N, m), ang_err (N, rad),
      sigma_min (N), lam (N)."""
    # TODO: for k in range(N): X = fk_space(...); Xerr = trans_inv(X) @ Xd[k]; store theta, errors, sigma_min
    # TODO: stop after storing the last sample (no step from it)
    # TODO: V_b = adjoint(Xerr) @ Vd[k] + kp * se3_to_vec(matrix_log6(Xerr))
    # TODO: lam_k = adaptive_lambda(...) if eps is given, else lam; qdot = dls_inverse(J_b, lam_k) @ V_b
    # TODO: rate limit: ratio = max_i |qdot_i| / qdot_max_i; if ratio > 1: qdot /= ratio
    # TODO: theta = theta + qdot * dt
    raise NotImplementedError("resolved_rate_sim")


# =============================================================== Checkpoint 4 -- IK near singularities
def ik_newton(Slist, M, T_d, theta0, lam=0.0, eomg=1e-6, ev=1e-6, max_iter=100):
    """Newton-Raphson IK in the body frame (MR Sec. 6.2.2) with an optional damped step:
        V_b = log(T(theta)^-1 T_d);  theta <- theta + J_b^#(lam) V_b   (lam = 0: MR's J_b^+).
    Stop when |omega_b| < eomg and |v_b| < ev, or after max_iter steps.
    Returns (theta, success, iterations, err) where err is a list of (|omega_b|, |v_b|) for
    every iterate including the initial guess (len = iterations + 1)."""
    # TODO: B = body_screws(Slist, M); V_b = se3_to_vec(matrix_log6(trans_inv(fk_space(...)) @ T_d))
    # TODO: append (|omega_b|, |v_b|) to err BEFORE testing convergence
    # TODO: theta += dls_inverse(jacobian_body(B, theta), lam) @ V_b
    raise NotImplementedError("ik_newton")


# =============================================================== Checkpoint 5 -- hexapod (closed chain)
def hexapod_leg_lengths(p, R, A, B):
    """Leg lengths s_i = |p + R b_i - a_i| of a Stewart-Gough platform (MR Sec. 7.1.2, p. 248)."""
    # TODO: d_i = p + R b_i - a_i (vectorise: (R @ B.T).T), s_i = |d_i|
    raise NotImplementedError("hexapod_leg_lengths")


def hexapod_inverse_jacobian(p, R, A, B):
    """6 x 6 inverse Jacobian J_s^-1 with sdot = J_s^-1 V_s (space twist V_s = (omega_s, v_s)).
    Row i = ( (-n_i x a_i)^T , n_i^T ), n_i = unit vector from a_i to the platform joint
    (MR Sec. 7.2.1, p. 252, with q_i = a_i)."""
    # TODO: n_i = d_i / s_i; row i = np.r_[-np.cross(n_i, a_i), n_i]
    raise NotImplementedError("hexapod_inverse_jacobian")


def hexapod_forward_kinematics(s, A, B, p0, R0, tol=1e-12, max_iter=50):
    """Forward kinematics by Newton-Raphson on the leg lengths: from the guess T = (R0, p0),
    repeat  ds = s - s(T);  V_s = (J_s^-1)^-1 ds;  T <- e^[V_s] T  until |ds| < tol.
    Returns (p, R, iterations)."""
    # TODO: T = (R0, p0); loop: ds = s - hexapod_leg_lengths(...); stop if max|ds| < tol
    # TODO: V = np.linalg.solve(hexapod_inverse_jacobian(...), ds); T = matrix_exp6(vec_to_se3(V)) @ T  (space twist: multiply on the LEFT)
    raise NotImplementedError("hexapod_forward_kinematics")


# =============================================================== Checkpoint 6 -- hardware session (sim first)
def tracking_report(t, cmd_deg, meas_deg, Slist, M):
    """Hardware acceptance numbers from a joint log (degrees), as in project starter/hwcompare.py:
      joint_rms_deg, joint_max_deg (per joint, |cmd - meas|),
      task_rms_mm, task_max_mm (tool point |p(cmd) - p(meas)| by PoE FK, millimetres),
      peak_speed_deg_s (per joint, measured, from np.gradient(meas, t)).
    Returns a dict with these keys."""
    # TODO: e = |cmd - meas| per sample and joint; RMS and max along axis 0
    # TODO: tool points by fk_space of np.deg2rad(cmd[k]) and np.deg2rad(meas[k]); distance in mm
    # TODO: peak speed: np.abs(np.gradient(meas, t, axis=0)).max(axis=0)
    raise NotImplementedError("tracking_report")
