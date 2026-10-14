"""SBEG 361 Lab 4 -- Exponential coordinates on SE(3), wrenches and force-torque sensor frame
changes for a laparoscopic tool and an EOD gripper (STARTER).

Fill in every function marked  # TODO  (11 functions).  Do not change names or signatures:
test_lab.py imports them.  Run the auto-checks with

    LAB_IMPL=starter python -m pytest -q test_lab.py

Do NOT call modern_robotics or scipy inside the TODO functions -- they are what the tests compare
against.  The helpers from Labs 1-3 (vec_to_so3 ... adjoint), the loaders and tip_forces are complete
and may be used.  np.linalg.lstsq is allowed in estimate_payload.

Notation follows Lynch & Park, *Modern Robotics* (MR), Sections 3.3.3 and 3.4:
  S = (omega, v), [S] in se(3)          screw axis, angular part FIRST          (MR Eq. 3.85)
  e^{[S]theta} = [[e^{[w]theta}, G(theta) v], [0, 1]]                         (MR Prop. 3.25, Eq. 3.88)
  G(theta) = I theta + (1 - cos theta)[w] + (theta - sin theta)[w]^2            (MR Eq. 3.87)
  e^{[S]theta} = [[I, v theta], [0, 1]]  if omega = 0, |v| = 1                  (MR Eq. 3.89)
  log: v = G^{-1}(theta) p,  G^{-1} = I/theta - [w]/2 + (1/theta - cot(theta/2)/2)[w]^2
                                                                                (MR Eqs. 3.91-3.92)
  F = (m, f)                             wrench, MOMENT FIRST                   (MR Eq. 3.93)
  V_b^T F_b = V_a^T F_a                  power is frame independent              (MR Eq. 3.94)
  F_b = [Ad_{T_ab}]^T F_a                change of frame of a wrench             (MR Eq. 3.95, Prop. 3.27)
  wrench screw (Poinsot): h = m.f/|f|^2, q = f x m/|f|^2   (MLS Eqs. 2.69-2.70, p. 65; MLS orders (f, tau))

Force-torque sensor convention (as in MR Example 3.28, pp. 108-109): the sensor frame {f} reports
the total wrench of the loads acting on everything distal to the sensor (gravity, contact), plus a
constant electrical offset ("bias"), all expressed in {f}.  Static or slow motion only.
"""
import csv
import json
import math

import numpy as np

TOL = 1e-6  # same threshold as modern_robotics.NearZero
G_S = np.array([0.0, 0.0, -9.81])


# ====================================================================== helpers from Labs 1-3
def vec_to_so3(omega):
    """[omega] (MR Eq. 3.30)."""
    w1, w2, w3 = np.asarray(omega, float)
    return np.array([[0.0, -w3, w2], [w3, 0.0, -w1], [-w2, w1, 0.0]])


def so3_to_vec(so3mat):
    m = np.asarray(so3mat, float)
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def matrix_exp3(so3mat):
    """Rodrigues' formula (MR Eq. 3.51); input [omega]theta."""
    so3mat = np.asarray(so3mat, float)
    wt = so3_to_vec(so3mat)
    theta = np.linalg.norm(wt)
    if theta < TOL:
        return np.eye(3)
    W = so3mat / theta
    return np.eye(3) + math.sin(theta) * W + (1 - math.cos(theta)) * W @ W


def matrix_log3(R):
    """Matrix logarithm on SO(3) (MR Sec. 3.2.3.3); returns [omega]theta with theta in [0, pi]."""
    R = np.asarray(R, float)
    c = (np.trace(R) - 1) / 2.0
    if c >= 1 - 1e-12:
        return np.zeros((3, 3))
    if c <= -1 + 1e-12:
        # theta = pi: MR Eqs. 3.58-3.60 (choose a numerically safe column)
        if 1 + R[2, 2] > TOL:
            w = np.array([R[0, 2], R[1, 2], 1 + R[2, 2]]) / math.sqrt(2 * (1 + R[2, 2]))
        elif 1 + R[1, 1] > TOL:
            w = np.array([R[0, 1], 1 + R[1, 1], R[2, 1]]) / math.sqrt(2 * (1 + R[1, 1]))
        else:
            w = np.array([1 + R[0, 0], R[1, 0], R[2, 0]]) / math.sqrt(2 * (1 + R[0, 0]))
        return vec_to_so3(math.pi * w)
    theta = math.acos(max(-1.0, min(1.0, c)))
    return theta / (2 * math.sin(theta)) * (R - R.T)


def rp_to_trans(R, p):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(p, float)
    return T


def trans_to_rp(T):
    T = np.asarray(T, float)
    return T[:3, :3].copy(), T[:3, 3].copy()


def trans_inv(T):
    """MR Eq. 3.64."""
    R, p = trans_to_rp(T)
    return rp_to_trans(R.T, -R.T @ p)


def vec_to_se3(V):
    """[V] (MR Eq. 3.71)."""
    V = np.asarray(V, float)
    M = np.zeros((4, 4))
    M[:3, :3] = vec_to_so3(V[:3])
    M[:3, 3] = V[3:]
    return M


def se3_to_vec(se3mat):
    m = np.asarray(se3mat, float)
    return np.r_[so3_to_vec(m[:3, :3]), m[:3, 3]]


def adjoint(T):
    """[Ad_T] = [[R, 0], [[p]R, R]] (MR Def. 3.20)."""
    R, p = trans_to_rp(T)
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[3:, :3] = vec_to_so3(p) @ R
    A[3:, 3:] = R
    return A


# ====================================================================== Part A: exp and log on SE(3) (from Lab 3)
# You wrote matrix_exp6, matrix_log6, screw_between and screw_path in Lab 3 (Week 4): paste your tested Lab 3
# versions here and re-run the Checkpoint 1a/1b tests (Lab 4 Steps 1-2). The hints below summarise the method.
def matrix_exp6(se3mat):
    """T = e^{[S]theta} for the 4x4 matrix [S]theta (MR Prop. 3.25, Eqs. 3.88-3.89).

    Works for any element of se(3): theta = |omega theta| if omega != 0, otherwise the pure
    translation case (Eq. 3.89)."""
    # TODO: Split [S]theta into omega*theta and v*theta; if |omega theta| < TOL use Eq. 3.89, else
    # theta = |omega theta|, [w] = [omega theta]/theta, v = (v theta)/theta and Eq. 3.88 with G(theta) of Eq. 3.87.
    # Reuse matrix_exp3 for the rotation block.
    raise NotImplementedError("matrix_exp6")


def matrix_log6(T):
    """[S]theta = log(T) (MR Sec. 3.3.3.2 algorithm, Eqs. 3.91-3.92), theta in [0, pi]."""
    # TODO: Use matrix_log3(R). If it is zero: return [[0, p], [0, 0]] (case a). Otherwise theta = |omega theta|,
    # [w] = log3/theta, G^-1 from Eq. 3.92 (cot = 1/tan), v = G^-1 p, and return [[ [w]theta, v theta ], [0, 0]].
    raise NotImplementedError("matrix_log6")


def screw_between(T1, T2, frame="space"):
    """Unit screw axis S and distance theta that carry T1 to T2.

    frame="space":  T2 = e^{[S]theta} T1   (S expressed in {s}; log of T2 T1^-1)
    frame="body":   T2 = T1 e^{[S]theta}   (S expressed in the moving frame at T1; log of T1^-1 T2)
    S is normalised as in MR Def. 3.24: |omega| = 1, or omega = 0 and |v| = 1.  For T1 = T2 the
    function returns (zeros(6), 0.0)."""
    # TODO: D = T2 T1^-1 (space) or T1^-1 T2 (body); St = se3_to_vec(matrix_log6(D)); theta = |omega part|,
    # or |v part| if omega = 0; return (St/theta, theta).  Raise ValueError for an unknown frame.
    raise NotImplementedError("screw_between")


def screw_path(T1, T2, s):
    """Pose a fraction s in [0, 1] of the way along the constant screw motion from T1 to T2:
    T(s) = e^{[S] s theta} T1 (MR Sec. 3.3.3; this is the straight line in exponential coordinates)."""
    # TODO: Use screw_between(T1, T2, 'space') and matrix_exp6.
    raise NotImplementedError("screw_path")


# ====================================================================== Part B: wrenches
def point_force_wrench(f, r):
    """Wrench F = (r x f, f) of a force f acting at point r, both in the same frame (MR Eq. 3.93)."""
    # TODO: One line: (r x f, f).
    raise NotImplementedError("point_force_wrench")


def change_wrench_frame(F_a, T_ab):
    """F_b = [Ad_{T_ab}]^T F_a (MR Eq. 3.95 / Prop. 3.27): the same wrench, represented in {b}."""
    # TODO: One line, with adjoint() from Lab 2.
    raise NotImplementedError("change_wrench_frame")


def power(V, F):
    """Instantaneous power V^T F (MR Eq. 3.94); V and F must be expressed in the same frame."""
    # TODO: One line; return a Python float.
    raise NotImplementedError("power")


def wrench_to_screw(F):
    """Screw (Poinsot) coordinates of a wrench F = (m, f) (MLS Thm. 2.17, Eqs. 2.69-2.71, p. 65).

    Returns (h, q, s_hat, magnitude):
      f != 0: h = m.f/|f|^2, q = f x m/|f|^2 (point of the axis closest to the origin), s_hat = f/|f|,
              magnitude |f|.  Check: m = q x f + h f (MLS Eq. 2.68 with q x w = -w x q).
      f == 0: pure moment, h = inf, q = None, s_hat = m/|m|, magnitude |m|.
    Raises ValueError for F = 0."""
    # TODO: Two cases: f != 0 and f == 0 (pure moment); ValueError for F = 0.
    raise NotImplementedError("wrench_to_screw")


def gravity_wrench(mass, r_com, R_sf, g_s=G_S):
    """Gravity wrench on a body of mass `mass` whose centre of mass is at r_com in {f},
    expressed in {f}, when the orientation of {f} in {s} is R_sf and gravity is g_s in {s}:
    f_f = mass R_sf^T g_s,  m_f = r_com x f_f."""
    # TODO: f_f = mass R_sf^T g_s, then point_force_wrench(f_f, r_com).
    raise NotImplementedError("gravity_wrench")


def estimate_payload(R_list, F_list, g_s=G_S):
    """Least-squares mass, centre of mass and sensor bias from K >= 3 static readings.

    Model for reading k (sensor orientation R_k = R_sf,k):
        F_k = b + ( c x (R_k^T g_s),  mass R_k^T g_s ),   c = mass * r_com
    which is LINEAR in x = (mass, c, b) (10 unknowns):
        f-rows:  f_k = b_f + mass gamma_k            (gamma_k = R_k^T g_s)
        m-rows:  m_k = b_m - [gamma_k] c
    Returns (mass, r_com, bias) with bias = (b_m, b_f)."""
    # TODO: Build the 6K x 10 matrix A and the 6K vector y as in the docstring, solve with
    # np.linalg.lstsq (allowed here), then r_com = c / mass.
    raise NotImplementedError("estimate_payload")


def compensate(F_meas, mass, r_com, R_sf, bias, g_s=G_S):
    """Contact wrench in {f}: reading minus bias minus the tool's own gravity wrench."""
    # TODO: Reading - bias - gravity_wrench(...).
    raise NotImplementedError("compensate")


# ====================================================================== loaders and analysis (given)
def load_pose_wrench_csv(path):
    """Read a CSV with columns [t,] R11..R33, px, py, pz, mx, my, mz, fx, fy, fz.
    Returns (t or None, list of 4x4 T_sf, K x 6 array of wrenches)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    t = np.array([float(r["t"]) for r in rows]) if "t" in rows[0] else None
    Ts, Fs = [], []
    for r in rows:
        R = np.array([[float(r[f"R{i}{j}"]) for j in (1, 2, 3)] for i in (1, 2, 3)])
        p = [float(r[k]) for k in ("px", "py", "pz")]
        Ts.append(rp_to_trans(R, p))
        Fs.append([float(r[k]) for k in ("mx", "my", "mz", "fx", "fy", "fz")])
    return t, Ts, np.array(Fs)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def tip_forces(Ts, Fs, mass, r_com, bias, T_ft, g_s=G_S):
    """Compensate every reading and express it at the tool tip: returns K x 6 tip wrenches F_t."""
    return np.array([change_wrench_frame(compensate(F, mass, r_com, T[:3, :3], bias, g_s), T_ft)
                     for T, F in zip(Ts, Fs)])
