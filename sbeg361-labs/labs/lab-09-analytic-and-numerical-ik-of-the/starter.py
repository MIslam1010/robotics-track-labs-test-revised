"""SBEG 361 Lab 9 -- Analytic and numerical inverse kinematics of the course arm;
all-solutions search (STARTER).

Fill in every function marked  # TODO  (12 functions).  Do not change names or signatures:
test_lab.py imports them.  Run the auto-checks with

    LAB_IMPL=starter python -m pytest -q test_lab.py

You may NOT call modern_robotics (IKinBody, IKinSpace, FKinSpace, ...) inside these functions --
it is what the tests compare against.  The helpers marked "complete" (from your Labs 2-7 library)
may be used freely.  When everything passes, copy the functions into your sbeg361_kinematics
library (ik.py).

Notation follows Lynch & Park, *Modern Robotics* (MR), Chapter 6, and Murray-Li-Sastry (MLS) Ch. 3 Sec. 3:
  forward kinematics   T(theta) = e^[S1]th1 ... e^[S6]th6 M                      (MR Eq. 4.14, p. 140)
  2R planar IK         beta = acos((L1^2+L2^2-x^2-y^2)/(2 L1 L2)),  alpha = acos((x^2+y^2+L1^2-L2^2)/(2 L1 r)),
                       gamma = atan2(y, x);  righty (th1, th2) = (gamma-alpha, pi-beta),
                       lefty (gamma+alpha, beta-pi)                                (MR pp. 218-219)
  Paden-Kahan          PK1  e^[xi th] p = q;  PK2  e^[xi1 th1] e^[xi2 th2] p = q;  PK3 |q - e^[xi th] p| = delta
                       (MLS pp. 99-103, Eqs. 3.17-3.29; Pardos-Gotor Sec. 4.3, pp. 59-63)
  Newton-Raphson       theta_{k+1} = theta_k - (dg/dtheta)^-1 g(theta_k)            (MR Sec. 6.2.1, p. 225)
  body-twist IK        [V_b] = log(T_sb(theta_i)^-1 T_sd),  theta_{i+1} = theta_i + J_b^+(theta_i) V_b
                       stop when |omega_b| <= eps_omega and |v_b| <= eps_v        (MR Sec. 6.2.2, pp. 228-229)
  space-twist IK       V_s = [Ad_Tsb] V_b,  theta_{i+1} = theta_i + J_s^+ V_s        (MR p. 229)
Twists are (omega, v), angular part first.  MLS and Pardos-Gotor write twists linear-first.
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
    """omega from [omega]."""
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
    """V = (omega, v) from a 4x4 matrix [V]."""
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


def rot(axis, angle):
    """Rotation matrix Rot(omega_hat, theta) by Rodrigues' formula (MR Eq. 3.51)."""
    w = np.asarray(axis, float)
    w = w / np.linalg.norm(w)
    K = vec_to_so3(w)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def matrix_exp3(so3mat):
    """Rodrigues' formula e^[w]theta (MR Eq. 3.51)."""
    w = so3_to_vec(so3mat)
    th = np.linalg.norm(w)
    if th < TOL:
        return np.eye(3)
    K = so3mat / th
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def matrix_log3(R):
    """[w]theta = log R, theta in [0, pi] (MR Sec. 3.2.3.3)."""
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
    """[S]theta = log T (MR Sec. 3.3.3.2)."""
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


def jacobian_space(Slist, theta):
    """Space Jacobian J_s(theta) (MR Eq. 5.11) -- your Lab 7 function."""
    Slist = np.asarray(Slist, float)
    theta = np.asarray(theta, float)
    Js = Slist.copy()
    T = np.eye(4)
    for i in range(1, Slist.shape[1]):
        T = T @ matrix_exp6(vec_to_se3(Slist[:, i - 1] * theta[i - 1]))
        Js[:, i] = adjoint(T) @ Slist[:, i]
    return Js


def jacobian_body(Blist, theta):
    """Body Jacobian J_b(theta) (MR Eq. 5.18) -- your Lab 7 function."""
    Blist = np.asarray(Blist, float)
    theta = np.asarray(theta, float)
    n = Blist.shape[1]
    Jb = Blist.copy()
    T = np.eye(4)
    for i in range(n - 2, -1, -1):
        T = T @ matrix_exp6(vec_to_se3(-Blist[:, i + 1] * theta[i + 1]))
        Jb[:, i] = adjoint(T) @ Blist[:, i]
    return Jb


def wrap_to_pi(theta):
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(theta, float) + np.pi) % (2 * np.pi) - np.pi


def pose_error(T, T_d):
    """(rotation error angle [rad], position error [m]) between two poses."""
    R_err = T[:3, :3].T @ T_d[:3, :3]
    ang = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
    return float(ang), float(np.linalg.norm(T[:3, 3] - T_d[:3, 3]))


def within_limits(theta, limits, margin=0.0):
    """True if every joint value lies inside its [lower, upper] limit (minus a margin)."""
    th = np.asarray(theta, float)
    lim = np.asarray(limits, float)
    return bool(np.all(th >= lim[:, 0] + margin) and np.all(th <= lim[:, 1] - margin))


def load_robot(name, path=None):
    """(Slist, M, info) of a model in data/robots.json (numbers identical to the course robot models)."""
    with open(path or DATA / "robots.json") as f:
        d = json.load(f)[name]
    return np.array(d["Slist"], float), np.array(d["M"], float), d


def load_biopsy_targets(path=None):
    """List of dicts {id, entry (3,), axis (3,), note} from data/biopsy_targets.csv."""
    out = []
    with open(path or DATA / "biopsy_targets.csv") as f:
        for row in csv.DictReader(f):
            out.append(dict(id=row["id"], note=row["note"],
                            entry=np.array([float(row[k]) for k in ("ex", "ey", "ez")]),
                            axis=np.array([float(row[k]) for k in ("ax", "ay", "az")])))
    return out


# =============================================================== Checkpoint 1 -- Paden-Kahan subproblems
def pk1(omega, r, p, q):
    """Paden-Kahan subproblem 1 (MLS p. 99-100, Eqs. 3.17-3.19): find theta with e^[xi theta] p = q,
    xi = zero-pitch unit twist with axis direction omega through the point r.

    Returns theta in (-pi, pi].  Raises ValueError if no solution exists (omega.u != omega.v or
    |u'| != |v'|, tolerance 1e-6 m).  If p lies on the axis (u' = 0), every theta works: return 0.0.
    """
    # TODO: u = p - r, v = q - r; project on the plane normal to omega (u' = u - w w^T u); check MLS Eq. 3.19;
    #       theta = atan2(w . (u' x v'), u' . v').
    raise NotImplementedError("pk1")


def pk2(omega1, omega2, r, p, q):
    """Paden-Kahan subproblem 2 (MLS pp. 100-102, Eqs. 3.20-3.25): find (theta1, theta2) with
    e^[xi1 theta1] e^[xi2 theta2] p = q; the two axes (directions omega1, omega2) intersect at r
    and are not parallel.

    Returns a list of 0, 1 or 2 tuples (theta1, theta2).
    """
    # TODO: alpha, beta, gamma^2 from MLS p. 102; z = alpha w1 + beta w2 + gamma (w1 x w2); c = z + r;
    #       then theta2 = pk1(w2, r, p, c) and theta1 = pk1(w1, r, c, q) for each gamma.
    raise NotImplementedError("pk2")


def pk3(omega, r, p, q, delta):
    """Paden-Kahan subproblem 3 (MLS pp. 102-103, Eqs. 3.26-3.29): find theta with
    |q - e^[xi theta] p| = delta.

    Returns a list of 0, 1 or 2 angles in [-pi, pi).
    """
    # TODO: delta'^2 = delta^2 - (w . (p - q))^2; theta0 = atan2(w . (u' x v'), u' . v');
    #       theta = theta0 -+ acos((|u'|^2 + |v'|^2 - delta'^2) / (2 |u'| |v'|))  (MLS Eq. 3.29); wrap to [-pi, pi).
    raise NotImplementedError("pk3")


# =============================================================== Checkpoint 2 -- analytic IK
def ik_2r(x, y, L1, L2):
    """Planar 2R inverse kinematics (MR pp. 218-219).

    Returns [] if (x, y) is outside the annulus L1-L2 <= r <= L1+L2, otherwise
    [righty, lefty] = [(gamma - alpha, pi - beta), (gamma + alpha, beta - pi)] (one tuple if both coincide).
    """
    # TODO: law of cosines (MR p. 218): beta, alpha, gamma = atan2(y, x); return [righty, lefty].
    raise NotImplementedError("ik_2r")


def ik_course_arm(T, geom):
    """All analytic IK solutions of the UR-type course arm (MR Example 4.5 layout).

    geom: dict with W1, W2, L1, L2, H1, H2 [m] and optional 'tool' [m] (tool along z_b).
    Returns a list of up to 8 joint vectors (np.ndarray, 6), each wrapped to [-pi, pi).
    Steps (lab handout, "Derivation of the course-arm IK"):
      1. wrist point p5 = p - (W2 + tool) z_b (intersection of axes 5 and 6);
      2. theta1 from  -sin(th1) p5x + cos(th1) p5y = W1   (2 solutions, shoulder left/right);
      3. theta5 from  cos(th5) = -sin(th1) z_bx + cos(th1) z_by   (2 solutions, wrist flip);
      4. theta6 from row 2 of R' = Rot(z,-th1) R, which equals (s5 c6, -s5 s6, c5);
      5. theta234 from Rot(y, th234) = R' (Rot(z,-th5) Rot(y,th6) R_M)^T;
      6. (theta2, theta3) from planar 2R IK of the joint-4 point; theta4 = theta234 - theta2 - theta3.
    The wrist singularity sin(th5) = 0 (axes 4 and 6 parallel) is handled by setting theta6 = 0.
    """
    # TODO: follow the six steps of the docstring; loop over 2 x theta1, 2 x theta5 and the ik_2r branches.
    #       Use rot(axis, angle) and wrap_to_pi; R_M = [[-1,0,0],[0,0,1],[0,1,0]] is the rotation part of M.
    raise NotImplementedError("ik_course_arm")


# =============================================================== Checkpoint 3 -- Newton-Raphson
def newton_raphson(g, dg, theta0, tol=1e-10, max_iter=50):
    """Newton-Raphson root finding for g(theta) = 0 (MR Sec. 6.2.1), theta scalar or vector.

    g(theta) -> residual (float or array), dg(theta) -> derivative / Jacobian.
    Iteration theta_{k+1} = theta_k - dg(theta_k)^-1 g(theta_k) (np.linalg.solve; pinv if not square).
    Stops when |g(theta_k)| <= tol or after max_iter steps.
    Returns (theta, history) with history = [(k, theta_k, g(theta_k)), ...] starting at k = 0.
    """
    # TODO: loop: store (k, theta, g); stop if |g| <= tol; step = solve(dg, g); theta -= step.
    raise NotImplementedError("newton_raphson")


def ik_newton_body(Blist, M, T_sd, theta0, eomg=1e-3, ev=1e-4, max_iter=20):
    """Body-twist Newton-Raphson IK (MR Sec. 6.2.2, p. 228-229; same stopping rule and default
    iteration limit as modern_robotics.IKinBody).

    Returns (theta, success, table); table rows are dicts
    {i, theta (copy), V_b (6,), w_err = |omega_b|, v_err = |v_b|} for i = 0, 1, ..., last.
    """
    # TODO: V_b = se3_to_vec(matrix_log6(trans_inv(fk_body(M, Blist, th)) @ T_sd)); record the row;
    #       stop if |w_b| <= eomg and |v_b| <= ev (or i == max_iter); else th += pinv(J_b) V_b.
    raise NotImplementedError("ik_newton_body")


def ik_newton_space(Slist, M, T_sd, theta0, eomg=1e-3, ev=1e-4, max_iter=20):
    """Space-twist Newton-Raphson IK (MR p. 229: V_s = [Ad_Tsb] V_b, J_s), as modern_robotics.IKinSpace.

    Returns (theta, success, table) with rows {i, theta, V_s, w_err, v_err}.
    """
    # TODO: as ik_newton_body, but V_s = [Ad_Tsb] V_b and J_s (fk_space, jacobian_space).
    raise NotImplementedError("ik_newton_space")


# =============================================================== Checkpoint 4 -- all-solutions search
def unique_solutions(sols, tol=1e-4):
    """Remove duplicates from a list of joint vectors; two solutions are equal if every wrapped
    joint difference is below tol (rad).  Returns wrapped solutions in first-seen order."""
    # TODO: compare wrap_to_pi(s - o) for every kept o.
    raise NotImplementedError("unique_solutions")


def all_solutions_search(Slist, M, T_sd, n_starts=200, seed=0, eomg=1e-9, ev=1e-9, max_iter=60):
    """Multi-start numerical IK: run ik_newton_body from n_starts initial guesses drawn uniformly
    in [-pi, pi)^n with np.random.default_rng(seed) (one call rng.uniform(-pi, pi, (n_starts, n))).

    Returns (solutions, counts, n_converged): distinct converged solutions (unique_solutions, tol 1e-4),
    how many starts converged to each one (its basin size), and the total number of converged starts.
    """
    # TODO: B = body_screws(Slist, M); starts = rng.uniform(-pi, pi, (n_starts, n)); run ik_newton_body;
    #       group the converged (wrapped) results with tolerance 1e-4 and count the members of each group.
    raise NotImplementedError("all_solutions_search")


def select_solution(sols, theta_now, limits):
    """Return the feasible solution closest to theta_now, or None if no solution is feasible.

    A revolute joint angle and the same angle +- 2 pi give the same pose, so for every joint of a
    solution consider the representatives s_j + 2 pi k (k = -1, 0, 1) that lie inside [lower, upper]
    and keep the one closest to theta_now[j]; a solution is infeasible if some joint has none.
    Among the feasible solutions return the one with the smallest Euclidean distance to theta_now."""
    # TODO: for each joint try s_j - 2 pi, s_j, s_j + 2 pi inside the limits; keep the closest to theta_now[j].
    raise NotImplementedError("select_solution")


# =============================================================== Checkpoint 5 -- biopsy needle guide
def needle_guide_pose(entry, axis, standoff, x_hint=(1.0, 0.0, 0.0)):
    """Desired pose T_sd of the needle-guide frame {b} for a biopsy trajectory.

    z_b = unit vector along the needle direction `axis` (pointing into the patient); the guide tip is
    `standoff` metres before the skin entry point: p = entry - standoff * z_b; x_b is the unit
    projection of x_hint onto the plane normal to z_b (if x_hint is parallel to z_b use (0, 1, 0)),
    y_b = z_b x x_b.
    """
    # TODO: z_b = axis / |axis|; x_b = normalised (x_hint - (x_hint . z_b) z_b); y_b = z_b x x_b; p = entry - standoff z_b.
    raise NotImplementedError("needle_guide_pose")
