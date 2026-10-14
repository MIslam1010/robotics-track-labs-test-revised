"""SBEG 361 Lab 8 -- Statics, manipulability maps of an assistive arm, workspace design (STARTER).

Fill in every function marked  # TODO  (14 functions).  Do not change names or signatures:
test_lab.py imports them.  Run the auto-checks with

    LAB_IMPL=starter python -m pytest -q test_lab.py

You may NOT call modern_robotics inside these functions -- it is what the tests compare against.
The helpers marked "complete" (your Lab 2-7 library: adjoint, FK, jacobian_space/body, plus the 2R
helpers jacobian_2r, ik_2r_given, within_limits, select_rows and the loaders) may be used freely.
When everything passes, copy the functions into your sbeg361_kinematics library (statics.py and
manipulability.py).

Notation follows Lynch & Park, *Modern Robotics* (MR), Sections 5.2-5.4:
  statics (power balance tau^T thetadot = F^T V):  tau = J^T(theta) F,  F = (m, f) moment first      (MR Eq. 5.26)
      tau = J_b^T F_b = J_s^T F_s;  wrench frame change F_s = [Ad_{T_bs}]^T F_b                     (MR Eq. 3.98)
      n = 6 and J invertible:  F = J^-T tau;  Null(J^T) = wrenches resisted with zero joint torque    (MR pp. 188-189)
  singularity: rank J(theta) < min(6, n); rank J_s = rank J_b                                           (MR Sec. 5.3)
  manipulability ellipsoid  qdot^T A^-1 qdot = 1, A = J J^T, semi-axes sqrt(lambda_i(A))               (MR Eq. 5.27)
  force ellipsoid  f^T B^-1 f = 1, B = (J J^T)^-1, semi-axes 1/sqrt(lambda_i(A)), same axes             (MR p. 197)
  measures  mu1 = sqrt(lmax/lmin) >= 1,  mu2 = lmax/lmin = mu1^2,  mu3 = sqrt(det A)                    (MR pp. 196-197)
  planar 2R point Jacobian (tip velocity in the plane)                                                  (MR Eq. 5.1)
Wrenches and twists are 6-vectors with the angular part first.  Units: m, kg, s, rad, N, N m.
"""
import csv
import json
from pathlib import Path

import numpy as np

TOL = 1e-6  # same threshold as modern_robotics.NearZero
DATA = Path(__file__).resolve().parent / "data"
G = np.array([0.0, 0.0, -9.81])


# =============================================================== helpers (complete; from Labs 2-7)
def vec_to_so3(omega):
    """3x3 skew-symmetric matrix [omega] (MR Eq. 3.30)."""
    w1, w2, w3 = np.asarray(omega, float)
    return np.array([[0.0, -w3, w2], [w3, 0.0, -w1], [-w2, w1, 0.0]])


def vec_to_se3(V):
    """4x4 matrix [V] of a twist V = (omega, v) (MR Eq. 3.71)."""
    V = np.asarray(V, float)
    out = np.zeros((4, 4))
    out[:3, :3] = vec_to_so3(V[:3])
    out[:3, 3] = V[3:]
    return out


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


def matrix_exp6(se3mat):
    """e^[S]theta for a twist matrix (MR Eqs. 3.51, 3.88-3.89)."""
    se3mat = np.asarray(se3mat, float)
    W, v = se3mat[:3, :3], se3mat[:3, 3]
    th = np.linalg.norm([W[2, 1], W[0, 2], W[1, 0]])
    out = np.eye(4)
    if th < TOL:
        out[:3, 3] = v
        return out
    K = W / th
    out[:3, :3] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    Gm = np.eye(3) * th + (1 - np.cos(th)) * K + (th - np.sin(th)) * K @ K
    out[:3, 3] = Gm @ (v / th)
    return out


def fk_space(M, Slist, theta):
    """Space-form PoE forward kinematics (MR Eq. 4.14)."""
    T = np.eye(4)
    for i, th in enumerate(np.asarray(theta, float)):
        T = T @ matrix_exp6(vec_to_se3(Slist[:, i] * th))
    return T @ M


def body_screws(Slist, M):
    """B_i = [Ad_{M^-1}] S_i (MR Eq. 4.16)."""
    return adjoint(trans_inv(M)) @ np.asarray(Slist, float)


def jacobian_space(Slist, theta):
    """Space Jacobian J_s(theta) (MR Eq. 5.11) -- your Lab 7 function."""
    Slist = np.asarray(Slist, float)
    Js = Slist.copy()
    T = np.eye(4)
    for i in range(1, Slist.shape[1]):
        T = T @ matrix_exp6(vec_to_se3(Slist[:, i - 1] * theta[i - 1]))
        Js[:, i] = adjoint(T) @ Slist[:, i]
    return Js


def jacobian_body(Blist, theta):
    """Body Jacobian J_b(theta) (MR Eq. 5.18) -- your Lab 7 function."""
    Blist = np.asarray(Blist, float)
    n = Blist.shape[1]
    Jb = Blist.copy()
    T = np.eye(4)
    for i in range(n - 2, -1, -1):
        T = T @ matrix_exp6(vec_to_se3(-Blist[:, i + 1] * theta[i + 1]))
        Jb[:, i] = adjoint(T) @ Blist[:, i]
    return Jb


def jacobian_2r(theta, L1, L2):
    """2 x 2 tip-velocity Jacobian of a planar 2R arm (MR Eq. 5.1)."""
    t1, t2 = theta
    s1, c1, s12, c12 = np.sin(t1), np.cos(t1), np.sin(t1 + t2), np.cos(t1 + t2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]])


def ik_2r_given(x, y, L1, L2, elbow=1):
    """Analytic IK of a planar 2R arm with the base at the origin (law of cosines), given complete.
    One branch only: elbow=+1 gives theta2 >= 0, elbow=-1 gives theta2 <= 0.  Returns None if the point
    is out of reach.  Not the same function as Lab 9's ik_2r(x, y, L1, L2), which returns the list of
    both branches; keep both names in your library."""
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(c2) > 1 + 1e-12:
        return None
    t2 = elbow * np.arccos(np.clip(c2, -1, 1))
    t1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(t2), L1 + L2 * np.cos(t2))
    t1 = (t1 + np.pi) % (2 * np.pi) - np.pi
    return np.array([t1, t2])


def within_limits(theta, limits):
    """True if every joint is inside [low, high] (limits n x 2; None means no limits)."""
    if limits is None:
        return True
    lim = np.asarray(limits, float)
    return bool(np.all(theta >= lim[:, 0] - 1e-12) and np.all(theta <= lim[:, 1] + 1e-12))


def load_robot(name, path=None):
    """(Slist, M, info) of a model in data/robots.json (numbers = Course resources: course robot models)."""
    with open(path or DATA / "robots.json") as f:
        d = json.load(f)[name]
    return np.array(d["Slist"], float), np.array(d["M"], float), d


def load_targets(path=None):
    """N x 2 therapy targets (table coordinates, m) from data/therapy_targets.csv."""
    with open(path or DATA / "therapy_targets.csv") as f:
        rows = list(csv.reader(f))[1:]
    return np.array(rows, float)


def load_feeding_csv(path=None):
    """(t, Q) from data/feeding_path.csv; Q is N x 7 (rad)."""
    with open(path or DATA / "feeding_path.csv") as f:
        rows = list(csv.reader(f))[1:]
    a = np.array(rows, float)
    return a[:, 0], a[:, 1:]


def select_rows(J, rows="all"):
    """'all' -> J; 'angular' -> J_omega (top 3 rows); 'linear' -> J_v (bottom 3 rows) (MR p. 196)."""
    J = np.asarray(J, float)
    if rows == "all":
        return J
    if rows == "angular":
        return J[:3]
    if rows == "linear":
        return J[3:]
    raise ValueError(f"rows must be 'all', 'angular' or 'linear', not {rows!r}")


# =============================================================== Checkpoint 1 -- statics
def joint_torques(J, F):
    """Joint torques/forces that balance an end-effector wrench: tau = J^T F (MR Eq. 5.26).

    J and F must be in the same frame (J_s with F_s, or J_b with F_b).  For a planar arm J may be
    the 2 x n point Jacobian and F the planar tip force (MR Eq. 5.3).
    """
    # TODO: tau = J^T F (one line).
    raise NotImplementedError("joint_torques")


def wrench_from_torques(J, tau, tol=1e-9):
    """End-effector wrench produced by joint torques at equilibrium: F = J^-T tau (MR Eq. 5.4, p. 173; general case §5.2, p. 188).

    Only well posed for a square, non-singular J: raise ValueError otherwise.
    """
    # TODO: check that J is square and non-singular (SVD), raise ValueError otherwise; then solve J^T F = tau.
    raise NotImplementedError("wrench_from_torques")


def body_to_space_wrench(F_b, T_sb):
    """F_s = [Ad_{T_bs}]^T F_b (MR Eq. 3.98)."""
    # TODO: use adjoint() and trans_inv(): F_s = [Ad_{T_bs}]^T F_b.
    raise NotImplementedError("body_to_space_wrench")


def holding_wrench_space(m, p, g=G):
    """Wrench F_s = (p x f, f) that the robot must apply to hold a point mass m at point p (in {s}):
    f = -m g (the opposite of the load's weight), moment about the origin of {s} = p x f."""
    # TODO: f = -m g; return the 6-vector (p x f, f).
    raise NotImplementedError("holding_wrench_space")


def wrench_null_space(J, tol=1e-6):
    """Orthonormal basis (6 x k) of Null(J^T): the wrenches the robot resists with zero joint torque
    and cannot actively generate (MR p. 189).  Columns are left singular vectors of J."""
    # TODO: SVD J = U S V^T; rank = number of singular values > tol; return the last 6 - rank columns of U.
    raise NotImplementedError("wrench_null_space")


# =============================================================== Checkpoint 2 -- ellipsoids and measures
def manipulability_ellipsoid(J):
    """(semi_axes, directions) of the manipulability ellipsoid qdot^T (J J^T)^-1 qdot = 1 (MR Eq. 5.27).

    semi_axes = sqrt(lambda_i(A)), A = J J^T, sorted from longest to shortest; directions[:, i] is the
    unit eigenvector of A for semi_axes[i] (sign arbitrary)."""
    # TODO: A = J J^T; np.linalg.eigh; sort eigenvalues in DESCENDING order (reorder the eigenvectors too); semi-axes = sqrt.
    raise NotImplementedError("manipulability_ellipsoid")


def force_ellipsoid(J, tol=1e-12):
    """(semi_axes, directions) of the force ellipsoid f^T (J J^T) f = 1 (MR p. 197).

    Same directions and ORDER as manipulability_ellipsoid(J); semi_axes = 1/sqrt(lambda_i(A))
    (np.inf where lambda_i <= tol): the longest velocity axis is the shortest force axis."""
    # TODO: reuse manipulability_ellipsoid; semi-axes 1/sqrt(lambda), np.inf where lambda <= tol.
    raise NotImplementedError("force_ellipsoid")


def manipulability_measures(J, tol=1e-12):
    """dict(mu1, mu2, mu3) of A = J J^T (MR pp. 196-197).

    mu1 = sqrt(lmax/lmin) (isotropy, 1 is best), mu2 = lmax/lmin (condition number of A),
    mu3 = sqrt(det A) = product of the semi-axes (volume measure, larger is better).
    At a singularity mu1 = mu2 = inf and mu3 = 0."""
    # TODO: eigenvalues of A = J J^T (np.linalg.eigvalsh); treat lambda <= tol*max(1, lmax) as 0; mu2 = inf at a singularity.
    raise NotImplementedError("manipulability_measures")


# =============================================================== Checkpoint 3 -- singularities
def singularity_report(J, tol=1e-6):
    """dict(rank, sigma_min, lost_twists, null_motions) of a Jacobian (MR Sec. 5.3).

    rank          number of singular values > tol
    sigma_min     smallest of the min(m, n) singular values
    lost_twists   m x (m - rank) orthonormal basis of twists the end-effector cannot produce
                  (= Null(J^T), the wrenches resisted with zero torque)
    null_motions  n x (n - rank) orthonormal basis of joint velocities that give zero twist
    """
    # TODO: np.linalg.svd(J) gives U, s, Vt; rank = sum(s > tol); lost_twists = U[:, rank:], null_motions = Vt[rank:].T.
    raise NotImplementedError("singularity_report")


# =============================================================== Checkpoint 4 -- maps and design (rehab_2r)
def _measure(J, measure):
    if measure == "sigma_min":
        return float(np.linalg.svd(J, compute_uv=False)[-1])
    if measure in ("mu1", "mu2", "mu3"):
        return manipulability_measures(J)[measure]
    raise ValueError(f"unknown measure {measure!r}")


def manipulability_map_2r(L1, L2, xs, ys, measure="mu3", elbow=1, limits=None, base=(0.0, 0.0)):
    """Grid (len(ys) x len(xs)) of a measure of the 2R point Jacobian over table points (x, y).

    The base is at `base`; each point is solved with ik_2r_given(x - bx, y - by, ...).  Entries are NaN
    where the point is out of reach or the IK solution violates `limits`.  measure: 'mu1', 'mu2',
    'mu3' or 'sigma_min'."""
    # TODO: loop over ys (rows) and xs (columns); ik_2r_given(x - bx, y - by, ...); skip None or not within_limits(); _measure(jacobian_2r(...), measure).
    raise NotImplementedError("manipulability_map_2r")


def region_score(L1, L2, points, measure="mu3", elbow=1, limits=None, base=(0.0, 0.0)):
    """(worst, fraction) over a task region: worst = min of mu3 / sigma_min, or max of mu1 / mu2,
    over the reachable points; fraction = share of points reachable within limits (NaN worst if none)."""
    # TODO: same loop as the map but over the list of points; worst = min (mu3, sigma_min) or max (mu1, mu2) over the reachable points.
    raise NotImplementedError("region_score")


def design_search(total, ratios, base_offsets, points, limits=None, elbow=1, measure="mu3"):
    """Grid search of a 2R design: L1 = ratio * total, L2 = total - L1, base at (0, -b).

    Returns (rows, best): rows = list of dicts(ratio, L1, L2, b, worst, fraction) in loop order
    (ratios outer, base_offsets inner); best = the feasible row (fraction == 1) with the best worst
    case (largest for mu3 / sigma_min, smallest for mu1 / mu2), or None if no row is feasible."""
    # TODO: two nested loops (ratios outer, base_offsets inner), base = (0, -b); keep rows with fraction == 1 for the best.
    raise NotImplementedError("design_search")


# =============================================================== Checkpoint 5 -- spatial arms
def measures_along_path(Slist, M, Q, rows="linear"):
    """dict(mu1, mu3, sigma_min) arrays (one value per row of Q) of the BODY Jacobian rows
    `rows` ('linear' = J_v, 'angular' = J_omega, 'all') along a joint path Q (N x n) (MR p. 196)."""
    # TODO: B = body_screws(S, M); for each q: J = select_rows(jacobian_body(B, q), rows); manipulability_measures; smallest singular value.
    raise NotImplementedError("measures_along_path")


def joint_space_map(Slist, M, q0, i, j, gi, gj, measure="sigma_min", rows="all"):
    """Grid (len(gj) x len(gi)) of a measure of the body Jacobian rows `rows` when joints i and j
    (0-based) sweep the values gi and gj and the other joints stay at q0."""
    # TODO: copy q0, set q[i], q[j]; J = select_rows(jacobian_body(B, q), rows); _measure(J, measure).
    raise NotImplementedError("joint_space_map")
