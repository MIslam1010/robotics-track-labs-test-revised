"""SBEG 361 Lab 3 -- Exponential coordinates and logarithms on SO(3) and SE(3); IMU orientation (STARTER).

Fill in every function marked  # TODO  (21 functions).  Do not change the function names or signatures:
the auto-checks in test_lab.py import them.  Run the tests with

    LAB_IMPL=starter python -m pytest -q test_lab.py

Notation follows Lynch & Park, *Modern Robotics* (MR), Sections 3.2.3, 3.3.2.2-3.3.3 and Appendix B.3.
SO(3) and SE(3) side by side:
  [w]          3x3 skew-symmetric matrix of w in R^3                         (MR Eq. 3.30)
  [S]          4x4 matrix [[ [w], v ], [0, 0]] of S = (w, v), angular first  (MR Eq. 3.85)
  e^[w]theta   Rodrigues: I + sin(t)[w] + (1 - cos t)[w]^2                    (MR Eq. 3.51)
  e^[S]theta   [[e^[w]theta, G(theta) v], [0, 1]]                             (MR Prop. 3.25, Eq. 3.88)
  G(theta)     I theta + (1 - cos theta)[w] + (theta - sin theta)[w]^2          (MR Eq. 3.87)
  e^[S]theta   [[I, v theta], [0, 1]]  if w = 0, |v| = 1                       (MR Eq. 3.89)
  log R        three-case algorithm                                           (MR Sec. 3.2.3.3, Eqs. 3.58-3.61)
  log T        two-case algorithm, v = G^-1(theta) p                          (MR Sec. 3.3.3.2, Eqs. 3.91-3.92)
  G^-1(theta)  I/theta - [w]/2 + (1/theta - cot(theta/2)/2)[w]^2               (MR Eq. 3.92)
  q            unit quaternion (q0, q1, q2, q3), scalar first                 (MR Eqs. B.9-B.12, B.15)

You may NOT call modern_robotics, scipy.spatial.transform or scipy.linalg.expm/logm inside
these functions -- they are what you compare against in the tests.  The helpers from Labs 1-2 at the
top of this file (rp_to_trans ... twist_to_screw, load_pose_track) are complete and may be used.
"""
import numpy as np

TOL = 1e-6  # same threshold as modern_robotics.NearZero


# ====================================================================== given helpers (Labs 1-2)
def rp_to_trans(R, p):
    """Build T = [[R, p], [0, 1]] (MR Eq. 3.62)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(p, float).reshape(3)
    return T


def trans_to_rp(T):
    """Split T into (R, p)."""
    T = np.asarray(T, float)
    return T[:3, :3].copy(), T[:3, 3].copy()


def trans_inv(T):
    """Inverse of T in SE(3) without a general matrix inverse (MR Eq. 3.64)."""
    R, p = trans_to_rp(T)
    return rp_to_trans(R.T, -R.T @ p)


def vec_to_se3(V):
    """[V] = [[ [omega], v ], [0, 0]] of a 6-vector V = (omega, v) (MR Eq. 3.85)."""
    w1, w2, w3, v1, v2, v3 = np.asarray(V, float).reshape(6)
    return np.array([[0.0, -w3, w2, v1],
                     [w3, 0.0, -w1, v2],
                     [-w2, w1, 0.0, v3],
                     [0.0, 0.0, 0.0, 0.0]])


def se3_to_vec(se3mat):
    """6-vector (omega, v) of a 4x4 matrix [V] in se(3)."""
    m = np.asarray(se3mat, float)
    return np.array([m[2, 1], m[0, 2], m[1, 0], m[0, 3], m[1, 3], m[2, 3]])


def twist_to_screw(V):
    """Screw interpretation of a twist or of exponential coordinates S theta (MR Sec. 3.3.2.2, Def. 3.24).

    Returns a dict with keys
      'S'        normalised screw axis (MR Def. 3.24),
      'thetadot' speed (or distance) about/along the axis (>= 0),
      'h'        pitch (np.inf for a pure translation),
      's_hat'    unit axis direction,
      'q'        the point on the axis closest to the frame origin (q = omega x v / |omega|^2),
                 or None for a pure translation.
    Raises ValueError for the zero twist."""
    V = np.asarray(V, float).reshape(6)
    w, v = V[:3], V[3:]
    nw, nv = np.linalg.norm(w), np.linalg.norm(v)
    if nw > TOL:
        return {"S": V / nw, "thetadot": nw, "h": float(w @ v) / nw**2,
                "s_hat": w / nw, "q": np.cross(w, v) / nw**2}
    if nv > TOL:
        return {"S": np.r_[np.zeros(3), v / nv], "thetadot": nv, "h": np.inf,
                "s_hat": v / nv, "q": None}
    raise ValueError("zero twist has no screw axis")


def load_pose_track(path, noisy=False):
    """Load a tracker log (columns t, q0..q3, px, py, pz and the noisy nq0..nq3, npx..npz).

    Returns (t, Ts): t of shape (N,), Ts of shape (N, 4, 4).  Uses YOUR quat_to_rot (Step 5)."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    n = "n" if noisy else ""
    Ts = np.array([rp_to_trans(quat_to_rot([r[f"{n}q{i}"] for i in range(4)]),
                               [r[f"{n}p{c}"] for c in "xyz"]) for r in d])
    return np.asarray(d["t"], float), Ts


# ====================================================================== so(3) <-> R^3
def vec_to_so3(omega):
    """Return the 3x3 skew-symmetric matrix [omega] (MR Eq. 3.30).

    Example: vec_to_so3([1, 2, 3]) -> [[0, -3, 2], [3, 0, -1], [-2, 1, 0]]
    """
    # TODO: build the matrix from the three components of omega (you wrote this in Lab 2: copy it)
    raise NotImplementedError("vec_to_so3")


def so3_to_vec(so3mat):
    """Return the 3-vector omega from a skew-symmetric matrix [omega]."""
    # TODO: read omega back from entries (3,2), (1,3), (2,1) (1-based)
    raise NotImplementedError("so3_to_vec")


def axis_ang3(expc3):
    """Split exponential coordinates  w_hat*theta  into (w_hat, theta), theta >= 0.

    Raise ValueError if theta < TOL (axis undefined)."""
    # TODO
    raise NotImplementedError("axis_ang3")


# ====================================================================== exponentials: SO(3) and SE(3)
def matrix_exp3(so3mat):
    """Matrix exponential of [w_hat]theta in so(3) by Rodrigues' formula (MR Eq. 3.51).

    Hint: theta = ||so3_to_vec(so3mat)||; if theta < TOL return I; otherwise [w_hat] = so3mat/theta.
    """
    # TODO
    raise NotImplementedError("matrix_exp3")


def g_matrix(omega_hat, theta):
    """G(theta) = I theta + (1 - cos theta)[w] + (theta - sin theta)[w]^2 for a unit axis w (MR Eq. 3.87).

    e^{[S]theta} has translation G(theta) v (MR Eq. 3.88)."""
    # TODO: one line with vec_to_so3(omega_hat)
    raise NotImplementedError("g_matrix")


def g_inv_matrix(omega_hat, theta):
    """G^-1(theta) = I/theta - [w]/2 + (1/theta - cot(theta/2)/2)[w]^2, 0 < theta <= pi (MR Eq. 3.92)."""
    # TODO: cot(x) = 1/tan(x)
    raise NotImplementedError("g_inv_matrix")


def matrix_exp6(se3mat):
    """T = e^{[S]theta} for the 4x4 matrix [S]theta (MR Prop. 3.25, Eqs. 3.88-3.89).

    Works for any element of se(3): theta = |omega theta| if omega != 0, otherwise the pure
    translation case (Eq. 3.89)."""
    # TODO: split [S]theta into omega*theta (top-left 3x3 block) and v*theta (top-right column).
    # If |omega theta| < TOL use Eq. 3.89.  Otherwise theta = |omega theta|, w = (omega theta)/theta,
    # v = (v theta)/theta, rotation block = matrix_exp3(top-left block), translation = g_matrix(w, theta) @ v.
    raise NotImplementedError("matrix_exp6")


# ====================================================================== logarithms: SO(3) and SE(3)
def matrix_log3(R):
    """Matrix logarithm [w_hat]theta of R in SO(3), theta in [0, pi] (MR Sec. 3.2.3.3).

    Implement all three cases of the MR algorithm:
      (a) R = I               -> zero matrix
      (b) tr R = -1 (theta=pi) -> Eqs. 3.58 / 3.59 / 3.60 (use the first feasible one)
      (c) otherwise           -> theta = arccos((tr R - 1)/2), [w_hat] = (R - R^T)/(2 sin theta)
    Use a small tolerance (e.g. 1e-12) when comparing (tr R - 1)/2 with +1 and -1.
    """
    # TODO
    raise NotImplementedError("matrix_log3")


def matrix_log6(T):
    """[S]theta = log(T) (MR Sec. 3.3.3.2 algorithm, Eqs. 3.91-3.92), theta in [0, pi]."""
    # TODO: (R, p) = trans_to_rp(T); so3 = matrix_log3(R).
    # Case (a), so3 = 0 (use np.allclose(so3, 0, atol=TOL)): return [[0, p], [0, 0]].
    # Case (b): theta = |so3_to_vec(so3)|, w = so3_to_vec(so3)/theta, v = g_inv_matrix(w, theta) @ p,
    # return [[so3, v theta], [0, 0]].
    raise NotImplementedError("matrix_log6")


# ====================================================================== screw motions
def screw_between(T1, T2, frame="space"):
    """Unit screw axis S and distance theta that carry T1 to T2.

    frame="space":  T2 = e^{[S]theta} T1   (S expressed in {s}; log of T2 T1^-1)
    frame="body":   T2 = T1 e^{[S]theta}   (S expressed in the moving frame at T1; log of T1^-1 T2)
    S is normalised as in MR Def. 3.24: |omega| = 1, or omega = 0 and |v| = 1.  For T1 = T2 the
    function returns (zeros(6), 0.0).  Raise ValueError for an unknown frame."""
    # TODO: D = T2 T1^-1 (space) or T1^-1 T2 (body); St = se3_to_vec(matrix_log6(D)); theta = |omega part|,
    # or |v part| if that is < TOL; if both are < TOL return (np.zeros(6), 0.0); else return (St/theta, theta).
    raise NotImplementedError("screw_between")


def screw_path(T1, T2, s):
    """Pose a fraction s in [0, 1] of the way along the constant screw motion from T1 to T2:
    T(s) = e^{[S] s theta} T1 (MR Sec. 3.3.3; a straight line in exponential coordinates)."""
    # TODO: use screw_between(T1, T2, "space"), vec_to_se3 and matrix_exp6.
    raise NotImplementedError("screw_path")


def screw_motion_profile(Ts):
    """Cumulative rotation angle and advance along the screw axis of a tracked screw motion.

    Ts has shape (N, 4, 4).  For every step k = 1..N-1 take (S_k, dtheta_k) = screw_between(Ts[k-1], Ts[k],
    "space").  The angle is SIGNED: positive about the reference direction w_ref = the omega of the step with
    the LARGEST dtheta, negative for a step whose omega points the other way (omega_k . w_ref < 0: the screw is
    backed out).  (The largest step is used because, near standstill, tracker noise dominates small steps.)
    The advance of step k is h_k * (signed dtheta_k) with the pitch h_k = omega_k . v_k of S_k.
    Steps without motion (dtheta < TOL) add nothing.  Raise ValueError if a step translates without
    rotating (|omega_k| = 0 but dtheta >= TOL): that is not a screw insertion.
    Returns (theta, d), both of shape (N,), starting at 0."""
    # TODO: first compute all steps (and raise ValueError for a pure translation); w_ref = omega of the step with
    # the largest dtheta; then loop again: sign = +1 if omega_k . w_ref >= 0 else -1; theta[k] = theta[k-1] + sign*dtheta, d[k] = d[k-1] + (omega_k . v_k)*sign*dtheta.
    raise NotImplementedError("screw_motion_profile")


# ====================================================================== quaternions (MR App. B.3)
def rot_to_quat(R):
    """Unit quaternion q = (q0, q1, q2, q3) of R, with q0 >= 0 (MR Eqs. B.10-B.11).

    Note: printed Eq. B.11 has a typo in its last entry; it should read r21 - r12.
    For q0 <= 0.5 (theta >= 120 deg) build q from the matrix log instead (Eq. B.9),
    because dividing by 4*q0 becomes ill-conditioned near theta = pi.
    """
    # TODO
    raise NotImplementedError("rot_to_quat")


def quat_to_rot(q):
    """Rotation matrix of a unit quaternion (MR Eq. B.12). Normalise q first."""
    # TODO
    raise NotImplementedError("quat_to_rot")


def quat_mul(q, p):
    """Quaternion of the product R_q R_p (MR Eq. B.15)."""
    # TODO
    raise NotImplementedError("quat_mul")


def quat_conj(q):
    """Conjugate (= inverse for a unit quaternion): the rotation R_q^T."""
    # TODO
    raise NotImplementedError("quat_conj")


# ====================================================================== IMU orientation
def integrate_gyro_exp(R0, omega_b, dt):
    """Integrate body-frame gyro samples with the exponential update
        R_{k+1} = R_k exp([omega_b,k] dt)
    omega_b has shape (N, 3). Return an array of shape (N+1, 3, 3) whose first entry is R0."""
    # TODO
    raise NotImplementedError("integrate_gyro_exp")


def integrate_gyro_euler(R0, omega_b, dt):
    """First-order (forward Euler) update R_{k+1} = R_k (I + [omega_b,k] dt) -- for comparison.
    Same input/output shapes as integrate_gyro_exp."""
    # TODO
    raise NotImplementedError("integrate_gyro_euler")


def orientation_error(R_cur, R_des):
    """Exponential coordinates w_hat*theta (in the current body frame) of the rotation that takes
    R_cur to R_des, i.e. R_des = R_cur exp([w_hat] theta)."""
    # TODO
    raise NotImplementedError("orientation_error")


def rotation_angle(R1, R2):
    """Angle (rad, in [0, pi]) of the relative rotation R1^T R2."""
    # TODO
    raise NotImplementedError("rotation_angle")


def knee_flexion(R_thigh, R_shank):
    """Signed knee flexion angle (rad): y-component of log(R_thigh^T R_shank).

    Convention for this lab: both IMUs have their y-axis along the knee flexion axis,
    so R_shank = R_thigh Rot(y_hat, kappa)."""
    # TODO
    raise NotImplementedError("knee_flexion")
