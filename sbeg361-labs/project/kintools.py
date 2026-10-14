"""kintools.py - small kinematics toolkit for the SBEG 361 course project.

Everything follows *Modern Robotics* (Lynch & Park) notation:
twists V = (omega, v), wrenches F = (m, f), screw axes S = (omega, v),
space-form PoE  T(theta) = e^[S1]th1 ... e^[Sn]thn M        (MR Eq. 4.14),
space Jacobian  J_si = Ad_{e^[S1]th1...e^[S(i-1)]th(i-1)}(S_i) (MR Eq. 5.11),
body Jacobian   J_b = [Ad_Tbs] J_s                            (MR Eq. 5.22),
statics         tau = J^T F (J and F in the same frame)       (MR Eq. 5.26),
inverse velocity theta_dot = J^+ V_d                          (MR Eq. 6.7).

The functions wrap the official ``modern_robotics`` library where it already
provides the operation, so your own lab library (sbeg361_kinematics) can be
checked against the same ground truth.

Names differ from your lab library; the closest library functions are
    fk(Slist, M, theta)          = fk_space(M, Slist, theta)    (Lab 5; NOTE the argument order)
    body_screws(Slist, M)        = body_screws(Slist, M)        (Lab 5, same)
    jacobian_space(Slist, theta) = jacobian_space(Slist, theta) (Lab 7, same)
    point_jacobian(Slist, M, theta, p_b)  differs from Lab 7's point_jacobian(Js, p) (p in {s})
    manipulability(J)            ~ manipulability_measures(J)   (Lab 8)
    holding_torques(Slist, ...)  ~ joint_torques(J, F)          (Lab 8)
    dls_pinv(J, lam, W)          ~ pinv_svd / dls_inverse       (Lab 10)
    ik_body(...)                 ~ ik_newton_body / ik_newton   (Labs 9-10; damped, with a task mask)
    within_limits, wrap_to_pi, pose_error                       (Lab 9, same signatures)
"""
from __future__ import annotations

import numpy as np
import modern_robotics as mr

# ----------------------------------------------------------------------------
# Screw axes and transforms
# ----------------------------------------------------------------------------

def screw_revolute(omega, q) -> np.ndarray:
    """Screw axis of a revolute joint: S = (w, -w x q), |w| = 1 (MR Sec. 3.3.2)."""
    w = np.asarray(omega, float)
    w = w / np.linalg.norm(w)
    q = np.asarray(q, float)
    return np.r_[w, -np.cross(w, q)]


def screw_prismatic(v) -> np.ndarray:
    """Screw axis of a prismatic joint: S = (0, v), |v| = 1."""
    v = np.asarray(v, float)
    return np.r_[np.zeros(3), v / np.linalg.norm(v)]


def trans(p=(0, 0, 0), R=None) -> np.ndarray:
    """Homogeneous transform from a rotation R (default I) and position p."""
    R = np.eye(3) if R is None else np.asarray(R, float)
    return mr.RpToTrans(R, np.asarray(p, float))


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def frame_from_z(z, x_hint=(1, 0, 0)) -> np.ndarray:
    """Rotation whose z-axis is z and whose x-axis is as close as possible to x_hint."""
    z = np.asarray(z, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(x_hint, float)
    x = x - (x @ z) * z
    if np.linalg.norm(x) < 1e-9:  # hint parallel to z: pick another
        x = np.array([0.0, 1.0, 0.0]) - z[1] * z
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def body_screws(Slist, M) -> np.ndarray:
    """B_i = [Ad_{M^-1}] S_i (MR Sec. 4.1.3)."""
    return mr.Adjoint(mr.TransInv(M)) @ np.asarray(Slist, float)

# ----------------------------------------------------------------------------
# Forward kinematics and Jacobians
# ----------------------------------------------------------------------------

def fk(Slist, M, theta) -> np.ndarray:
    """Space-form PoE forward kinematics (MR Eq. 4.14)."""
    return mr.FKinSpace(M, np.asarray(Slist, float), np.asarray(theta, float))


def jacobian_space(Slist, theta) -> np.ndarray:
    """Space Jacobian J_s(theta), 6 x n (MR Eq. 5.11)."""
    return mr.JacobianSpace(np.asarray(Slist, float), np.asarray(theta, float))


def jacobian_body_from_space(Slist, M, theta) -> np.ndarray:
    """Body Jacobian J_b = [Ad_{T_bs}] J_s (MR Eq. 5.22), computed from the SPACE screws.

    Named *_from_space so it is not confused with the Lab 7 library function
    jacobian_body(Blist, theta), which takes the body screws B_i (MR Eq. 5.18);
    both give the same J_b: jacobian_body_from_space(Slist, M, th) ==
    jacobian_body(body_screws(Slist, M), th)."""
    T = fk(Slist, M, theta)
    return mr.Adjoint(mr.TransInv(T)) @ jacobian_space(Slist, theta)


def point_jacobian(Slist, M, theta, p_b) -> np.ndarray:
    """3 x n Jacobian of the linear velocity (in {s}) of a point fixed in {b} at p_b.

    Uses v_p = v_s + w_s x p_s with V_s = J_s theta_dot (MR Sec. 5.1.1).
    """
    T = fk(Slist, M, theta)
    p_s = T[:3, :3] @ np.asarray(p_b, float) + T[:3, 3]
    Js = jacobian_space(Slist, theta)
    Jw, Jv = Js[:3], Js[3:]
    return Jv - mr.VecToso3(p_s) @ Jw  # v + w x p  =  v - [p] w


def moved_points(Slist, theta, points_home):
    """Positions of chain points.  points_home is a list of (p_home, j): the point is
    carried by joints 1..j (j = 0: fixed to the base).  A point on the axis of joint i
    is carried by joints 1..i-1 (or 1..i, which gives the same result)."""
    Slist = np.asarray(Slist, float)
    theta = np.asarray(theta, float)
    out = []
    for p, j in points_home:
        T = np.eye(4)
        for i in range(j):
            T = T @ mr.MatrixExp6(mr.VecTose3(Slist[:, i]) * theta[i])
        out.append((T @ np.r_[np.asarray(p, float), 1.0])[:3])
    return np.array(out)


def sample_segments(points, step=0.02):
    """Densely sample the polyline through `points` (for clearance checks)."""
    pts = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for k in range(1, n + 1):
            pts.append(a + (b - a) * k / n)
    return np.array(pts)

# ----------------------------------------------------------------------------
# Manipulability, statics, singularity
# ----------------------------------------------------------------------------

def manipulability(J) -> dict:
    """MR Sec. 5.4 measures for A = J J^T:
    mu1 = sqrt(lmax/lmin) (>= 1, smaller is better), mu2 = lmax/lmin,
    mu3 = sqrt(det A) (larger is better).  Also returns the semi-axes sqrt(lambda_i)."""
    J = np.atleast_2d(np.asarray(J, float))
    A = J @ J.T
    lam = np.clip(np.linalg.eigvalsh(A), 0.0, None)
    lmin, lmax = lam.min(), lam.max()
    mu1 = np.inf if lmin < 1e-14 else np.sqrt(lmax / lmin)
    return {"mu1": mu1, "mu2": mu1 ** 2, "mu3": float(np.sqrt(np.prod(lam))),
            "semi_axes": np.sqrt(lam)}


def sigma_min(J) -> float:
    """Smallest singular value of J (0 at a singularity)."""
    return float(np.linalg.svd(np.asarray(J, float), compute_uv=False).min())


def gravity_wrench_space(p_s, mass, g=9.81) -> np.ndarray:
    """Space-frame wrench F_s = (m, f) = (p x f, f) that the ROBOT must apply at point
    p_s to hold a load of the given mass against gravity (f points up, +z_s).
    With this sign, tau = J_s^T F_s are the joint holding torques (MR Eq. 5.26)."""
    f = np.array([0.0, 0.0, mass * g])  # robot pushes up on the load
    p = np.asarray(p_s, float)
    return np.r_[np.cross(p, f), f]


def holding_torques(Slist, theta, F_s) -> np.ndarray:
    """tau = J_s(theta)^T F_s (MR Eq. 5.26 with J and F both in {s})."""
    return jacobian_space(Slist, theta).T @ np.asarray(F_s, float)

# ----------------------------------------------------------------------------
# Inverse kinematics and inverse velocity kinematics
# ----------------------------------------------------------------------------

def dls_pinv(J, lam=0.0, W=None) -> np.ndarray:
    """Damped (and optionally task-weighted) pseudo-inverse
    J^T (J J^T + lam^2 I)^-1.  lam = 0 gives the Moore-Penrose pseudo-inverse."""
    J = np.asarray(J, float)
    if W is not None:
        J = np.diag(W) @ J
    m = J.shape[0]
    Jp = J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(m)) if lam > 0 else np.linalg.pinv(J)
    if W is not None:
        Jp = Jp @ np.diag(W)
    return Jp


def ik_body(Slist, M, T_d, theta0, mask=None, lam=1e-3, eomg=1e-7, ev=1e-7,
            max_iter=200, limits=None, step_max=0.5):
    """Numerical IK in the body frame (MR Sec. 6.2.2 algorithm, with a damped
    pseudo-inverse) for an optional *reduced task*.

    mask : length-6 0/1 weights on the body twist error V_b = log(T_sb^-1 T_sd)
           (omega_x, omega_y, omega_z, v_x, v_y, v_z).  Example: (1,1,0,1,1,1) leaves the
           rotation about the tool z-axis free (needle / instrument roll).
    limits : n x 2 array; the iterate is clipped to the limits after every step.
    Returns (theta, success, iterations, final_error_twist).
    """
    Slist = np.asarray(Slist, float)
    th = np.array(theta0, float)
    W = np.ones(6) if mask is None else np.asarray(mask, float)
    for k in range(max_iter):
        T = fk(Slist, M, th)
        Vb = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T) @ T_d)) * W
        if np.linalg.norm(Vb[:3]) < eomg and np.linalg.norm(Vb[3:]) < ev:
            return th, True, k, Vb
        Jb = jacobian_body_from_space(Slist, M, th) * W[:, None]
        dth = dls_pinv(Jb, lam) @ Vb
        nrm = np.linalg.norm(dth)
        if nrm > step_max:
            dth *= step_max / nrm
        th = th + dth
        if limits is not None:
            th = np.clip(th, limits[:, 0], limits[:, 1])
    T = fk(Slist, M, th)
    Vb = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(T) @ T_d)) * W
    ok = np.linalg.norm(Vb[:3]) < 1e3 * eomg and np.linalg.norm(Vb[3:]) < 1e3 * ev
    return th, bool(ok), max_iter, Vb


def resolved_rate_step(Slist, M, theta, V_b_des, dt, lam=0.01, mask=None):
    """One step of body-frame resolved-rate control (MR Eq. 6.7, damped):
    theta_dot = J_b^+ V_b,des ; theta <- theta + theta_dot dt.  Returns (theta_new, theta_dot)."""
    W = np.ones(6) if mask is None else np.asarray(mask, float)
    Jb = jacobian_body_from_space(Slist, M, theta) * W[:, None]
    thd = dls_pinv(Jb, lam) @ (np.asarray(V_b_des, float) * W)
    return np.asarray(theta, float) + thd * dt, thd


def pose_error(T, T_d):
    """(angular error [rad], position error [m]) between two poses."""
    R_err = T[:3, :3].T @ T_d[:3, :3]
    ang = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
    return float(ang), float(np.linalg.norm(T[:3, 3] - T_d[:3, 3]))

# ----------------------------------------------------------------------------
# Constraint helpers used by the scenarios
# ----------------------------------------------------------------------------

def point_line_distance(p, a, u) -> float:
    """Distance from point p to the line through a with direction u."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    d = np.asarray(p, float) - np.asarray(a, float)
    return float(np.linalg.norm(d - (d @ u) * u))


def within_limits(theta, limits, margin=0.0) -> bool:
    th = np.asarray(theta, float)
    return bool(np.all(th >= limits[:, 0] + margin) and np.all(th <= limits[:, 1] - margin))


def wrap_to_pi(theta):
    return (np.asarray(theta, float) + np.pi) % (2 * np.pi) - np.pi
