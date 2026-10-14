"""SBEG 361 Lab 7 -- Space, body and analytic Jacobians; finite-difference verification (STARTER).

Fill in every function marked  # TODO  (13 functions).  Do not change names or signatures:
test_lab.py imports them.  Run the auto-checks with

    LAB_IMPL=starter python -m pytest -q test_lab.py

You may NOT call modern_robotics (JacobianSpace, JacobianBody, Adjoint, ...) inside these
functions -- it is what the tests compare against.  The helpers marked "complete" (from your
Labs 2-5 library) may be used freely; replace them by your own versions if you prefer.
When everything passes, copy the functions into your sbeg361_kinematics library (jacobians.py).

Notation follows Lynch & Park, *Modern Robotics* (MR), Section 5.1:
  T(theta) = e^[S1]th1 ... e^[Sn]thn M = M e^[B1]th1 ... e^[Bn]thn          (MR Eqs. 4.14, 4.16)
  V_s = J_s(theta) thetadot,  J_s1 = S1,  J_si = [Ad_{e^[S1]th1...e^[S(i-1)]th(i-1)}] S_i   (MR Eqs. 5.10-5.11)
  V_b = J_b(theta) thetadot,  J_bn = Bn,  J_bi = [Ad_{e^-[Bn]thn...e^-[B(i+1)]th(i+1)}] B_i (MR Eqs. 5.17-5.18)
  J_b = [Ad_Tbs] J_s,  J_s = [Ad_Tsb] J_b                                             (MR Eqs. 5.22-5.23)
  analytic Jacobian for q = (r, x), r = exponential coordinates of R_sb:
  J_a = [[A(r)^-1, 0], [0, R_sb]] J_b,  A(r) = I - (1-cos|r|)/|r|^2 [r] + (|r|-sin|r|)/|r|^3 [r]^2   (MR Eqs. 5.24-5.25)
  velocity of the origin of {b}:  pdot = v_s + w_s x p = R_sb v_b                (MR Eqs. 3.69, 3.73)
"""
import csv
import json
from pathlib import Path

import numpy as np

TOL = 1e-6  # same threshold as modern_robotics.NearZero
DATA = Path(__file__).resolve().parent / "data"


# =============================================================== helpers (complete; from Labs 2-5)
def vec_to_so3(omega):
    """3x3 skew-symmetric matrix [omega] (MR Eq. 3.30)."""
    w1, w2, w3 = np.asarray(omega, float)
    return np.array([[0.0, -w3, w2], [w3, 0.0, -w1], [-w2, w1, 0.0]])


def so3_to_vec(so3mat):
    """omega from [omega] (reads the entries (2,1), (0,2), (1,0))."""
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
    """V = (omega, v) from a 4x4 matrix [V] (entries read as in MR se3ToVec)."""
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
    """6x6 adjoint [Ad_T] = [[R, 0], [[p]R, R]] (MR Def. 3.20)."""
    R, p = T[:3, :3], T[:3, 3]
    out = np.zeros((6, 6))
    out[:3, :3] = R
    out[3:, 3:] = R
    out[3:, :3] = vec_to_so3(p) @ R
    return out


def matrix_exp3(so3mat):
    """Rodrigues' formula e^[w]theta (MR Eq. 3.51)."""
    w = so3_to_vec(so3mat)
    th = np.linalg.norm(w)
    if th < TOL:
        return np.eye(3)
    K = so3mat / th
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def matrix_log3(R):
    """[w]theta = log R, theta in [0, pi] (MR Algorithm, Sec. 3.2.3.3)."""
    R = np.asarray(R, float)
    c = (np.trace(R) - 1) / 2.0
    if c >= 1.0:
        return np.zeros((3, 3))
    if c <= -1.0:
        if abs(1 + R[2, 2]) > TOL:
            w = np.array([R[0, 2], R[1, 2], 1 + R[2, 2]]) / np.sqrt(2 * (1 + R[2, 2]))
        elif abs(1 + R[1, 1]) > TOL:
            w = np.array([R[0, 1], 1 + R[1, 1], R[2, 1]]) / np.sqrt(2 * (1 + R[1, 1]))
        else:
            w = np.array([1 + R[0, 0], R[1, 0], R[2, 0]]) / np.sqrt(2 * (1 + R[0, 0]))
        return vec_to_so3(np.pi * w)
    th = np.arccos(c)
    return th / (2 * np.sin(th)) * (R - R.T)


def matrix_exp6(se3mat):
    """e^[S]theta for a twist matrix (MR Eqs. 3.88-3.89)."""
    se3mat = np.asarray(se3mat, float)
    W, v = se3mat[:3, :3], se3mat[:3, 3]
    th = np.linalg.norm(so3_to_vec(W))
    out = np.eye(4)
    if th < TOL:
        out[:3, 3] = v
        return out
    K = W / th
    out[:3, :3] = matrix_exp3(W)
    G = np.eye(3) * th + (1 - np.cos(th)) * K + (th - np.sin(th)) * K @ K
    out[:3, 3] = G @ (v / th)
    return out


def matrix_log6(T):
    """[S]theta = log T (MR Algorithm, Sec. 3.3.3.2)."""
    R, p = T[:3, :3], T[:3, 3]
    W = matrix_log3(R)
    out = np.zeros((4, 4))
    if np.allclose(W, 0):
        out[:3, 3] = p
        return out
    th = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1, 1))
    Ginv = np.eye(3) - W / 2.0 + (1.0 / th - 1.0 / np.tan(th / 2.0) / 2.0) * (W @ W) / th
    out[:3, :3] = W
    out[:3, 3] = Ginv @ p
    return out


def fk_space(M, Slist, theta):
    """Space-form PoE forward kinematics (MR Eq. 4.14)."""
    T = np.eye(4)
    for i, th in enumerate(np.asarray(theta, float)):
        T = T @ matrix_exp6(vec_to_se3(Slist[:, i] * th))
    return T @ M


def fk_body(M, Blist, theta):
    """Body-form PoE forward kinematics (MR Eq. 4.16)."""
    T = np.array(M, float)
    for i, th in enumerate(np.asarray(theta, float)):
        T = T @ matrix_exp6(vec_to_se3(Blist[:, i] * th))
    return T


def body_screws(Slist, M):
    """B_i = [Ad_{M^-1}] S_i (MR Eq. 4.16)."""
    return adjoint(trans_inv(M)) @ np.asarray(Slist, float)


def load_robot(name, path=None):
    """(Slist, M, info) of a model in data/robots.json (numbers identical to the course robot models)."""
    with open(path or DATA / "robots.json") as f:
        d = json.load(f)[name]
    return np.array(d["Slist"], float), np.array(d["M"], float), d


def load_teleop_csv(path=None):
    """(t, Q) from data/teleop_needle.csv; Q is N x 6 (rad)."""
    with open(path or DATA / "teleop_needle.csv") as f:
        rows = list(csv.reader(f))[1:]
    a = np.array(rows, float)
    return a[:, 0], a[:, 1:]


def fit_slope(hs, errs):
    """Least-squares slope of log(err) against log(h) (the observed order of accuracy)."""
    return float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


# =============================================================== Checkpoint 1 -- geometric Jacobians
def jacobian_space(Slist, theta):
    """Space Jacobian J_s(theta), 6 x n (MR Def. 5.1, Eq. 5.11).

    Column 1 is S_1; column i is [Ad_{T_(i-1)}] S_i with T_(i-1) = e^[S1]th1 ... e^[S(i-1)]th(i-1).
    """
    # TODO: Start with Js = copy of Slist; accumulate T = T @ exp([S_(i-1)] th_(i-1)); column i = Ad_T S_i.
    raise NotImplementedError("jacobian_space")


def jacobian_body(Blist, theta):
    """Body Jacobian J_b(theta), 6 x n (MR Def. 5.4, Eq. 5.18).

    Column n is B_n; column i is [Ad_{e^-[Bn]thn ... e^-[B(i+1)]th(i+1)}] B_i.
    """
    # TODO: Mirror image of jacobian_space: walk from joint n-1 down to 1, accumulating exp(-[B_(i+1)] th_(i+1)).
    raise NotImplementedError("jacobian_body")


def body_from_space(Js, T_sb):
    """J_b = [Ad_{T_bs}] J_s (MR Eq. 5.22)."""
    # TODO: One line with adjoint() and trans_inv().
    raise NotImplementedError("body_from_space")


def space_from_body(Jb, T_sb):
    """J_s = [Ad_{T_sb}] J_b (MR Eq. 5.23)."""
    # TODO: One line with adjoint().
    raise NotImplementedError("space_from_body")


def point_jacobian(Js, p):
    """3 x n Jacobian of the velocity of a point p (in {s}) fixed to the end-effector:
    pdot = v_s + w_s x p = (J_v - [p] J_w) thetadot, where J_w, J_v are the top and bottom rows of J_s."""
    # TODO: pdot = v_s + w_s x p = v_s - [p] w_s; use vec_to_so3(p).
    raise NotImplementedError("point_jacobian")


# =============================================================== Checkpoint 2 -- finite-difference checks
def fd_jacobian_space(fk, theta, h=1e-6, scheme="central"):
    """Numerical space Jacobian from finite differences of T(theta).

    fk: callable theta -> 4x4 T_sb.  For each joint i, approximate dT/dtheta_i by
      forward:  (T(theta + h e_i) - T(theta)) / h
      central:  (T(theta + h e_i) - T(theta - h e_i)) / (2h)
    and take column i = se3_to_vec( dT/dtheta_i  T(theta)^-1 )      ([V_s] = Tdot T^-1, MR Eq. 3.74).
    """
    # TODO: Loop over joints; build dT/dtheta_i with the chosen scheme; multiply by T(theta)^-1 on the RIGHT.
    raise NotImplementedError("fd_jacobian_space")


def fd_jacobian_body(fk, theta, h=1e-6, scheme="central"):
    """Numerical body Jacobian: column i = se3_to_vec( T^-1 dT/dtheta_i ) ([V_b] = T^-1 Tdot, MR Eq. 3.69)."""
    # TODO: As fd_jacobian_space but T(theta)^-1 on the LEFT.
    raise NotImplementedError("fd_jacobian_body")


def log_jacobian_space(fk, theta, h=0.1):
    """Space Jacobian from the matrix log of a finite joint step:
    column i = se3_to_vec( log( T(theta + h e_i) T(theta)^-1 ) ) / h.
    Exact (up to rounding) for any 0 < |h| < pi: moving one joint is a screw motion about J_si."""
    # TODO: Use matrix_log6 and se3_to_vec; divide by h.
    raise NotImplementedError("log_jacobian_space")


def fd_error_study(fk, J_exact, theta, hs, scheme="central"):
    """Max-abs error of fd_jacobian_space(fk, theta, h, scheme) against J_exact for every h in hs."""
    # TODO: One list comprehension over hs calling fd_jacobian_space.
    raise NotImplementedError("fd_error_study")


# =============================================================== Checkpoint 3 -- analytic Jacobian
def a_matrix(r):
    """A(r) with omega_b = A(r) rdot for exponential coordinates r of R_sb (MR Example 5.5)."""
    # TODO: Return I for |r| < TOL; otherwise the MR Example 5.5 formula.
    raise NotImplementedError("a_matrix")


def analytic_jacobian(Jb, T_sb):
    """6 x n analytic Jacobian of q = (r, x): r = exponential coordinates of R_sb, x = p_sb
    (MR Eq. 5.25): J_a = [[A(r)^-1, 0], [0, R_sb]] J_b."""
    # TODO: r = so3_to_vec(matrix_log3(R_sb)); top block A(r)^-1 J_w (use np.linalg.solve), bottom block R_sb J_v.
    raise NotImplementedError("analytic_jacobian")


def minimal_coords(T):
    """q = (r, x): exponential coordinates of the rotation and the position of {b}."""
    return np.r_[so3_to_vec(matrix_log3(T[:3, :3])), T[:3, 3]]


def fd_analytic_jacobian(fk, theta, h=1e-6):
    """Central finite differences of q(theta) = minimal_coords(fk(theta)) (valid for |r| < pi)."""
    # TODO: Central differences of minimal_coords(fk(theta +- h e_i)).
    raise NotImplementedError("fd_analytic_jacobian")


# =============================================================== Checkpoint 4 -- teleoperation record
def tip_velocity_series(Slist, M, Q, Qdot):
    """N x 3 array of tool-tip (origin of {b}) velocities pdot_k = point_jacobian(J_s(q_k), p_k) qdot_k."""
    # TODO: For each sample: p from fk_space, then point_jacobian(jacobian_space(...), p) @ qdot.
    raise NotImplementedError("tip_velocity_series")
