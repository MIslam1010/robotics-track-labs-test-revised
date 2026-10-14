"""SBEG 361 Lab 2 -- Angular velocities and twists (STARTER).

Fill in every function marked  # TODO.  Do not change names or signatures: test_lab.py
imports them.  Run the auto-checks with

    LAB_IMPL=starter python -m pytest -q test_lab.py

You may NOT call modern_robotics, scipy.spatial.transform or np.linalg.inv inside these
functions -- they are what the tests compare against.  Everything in the GIVEN section (your
Lab 1 SE(3) functions and transform tree, the loaders and the analysis helpers) is complete and
may be used freely; paste your own Lab 1 versions over it if you prefer.

Notation follows Lynch & Park, *Modern Robotics* (MR), Sections 3.2.2 and 3.3.2:
  [omega_s] = Rdot R^-1, [omega_b] = R^-1 Rdot                          (MR Eqs. 3.37-3.38, Prop. 3.9)
  omega_s = R omega_b,  R [omega] R^T = [R omega]                       (MR Eq. 3.35, Prop. 3.8)
  V = (omega, v)               twist, angular part FIRST                 (MR Eqs. 3.70, 3.74)
  [V] = [[omega] v; 0 0]       4x4 se(3) matrix                          (MR Eq. 3.71)
  [V_b] = T^-1 Tdot, [V_s] = Tdot T^-1                                  (MR Eqs. 3.81-3.82)
  [Ad_T] = [[R 0]; [[p]R R]],  [Ad_T V] = T [V] T^-1                    (MR Def. 3.20, Eqs. 3.83-3.84)
  velocity of a body point x (coordinates in the same frame as V): omega x x + v   (MR Eq. 3.73, Sec. 3.3.2)
  screw {q, s_hat, h}: V = (s_hat, -s_hat x q + h s_hat) thetadot       (MR Sec. 3.3.2.2)

Transform tree convention (identical to ROS 2 tf2): an edge parent -> child stores
T_parent_child, the pose of the child frame expressed in the parent frame.
lookup(target, source) returns T_target_source (tf2: buffer.lookup_transform(target, source, t)).
"""
import json
import math

import numpy as np

TOL = 1e-6  # same threshold as modern_robotics.NearZero


# ================================================================ GIVEN (from Lab 1)
def vec_to_so3(omega):
    """3x3 skew-symmetric matrix [omega] (MR Eq. 3.30)."""
    w1, w2, w3 = np.asarray(omega, float)
    return np.array([[0.0, -w3, w2], [w3, 0.0, -w1], [-w2, w1, 0.0]])


def so3_to_vec(so3mat):
    """omega from [omega]."""
    m = np.asarray(so3mat, float)
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def rp_to_trans(R, p):
    """Build T = [[R, p], [0, 1]] (MR Eq. 3.62)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(p, float)
    return T


def trans_to_rp(T):
    """Split T into (R, p)."""
    T = np.asarray(T, float)
    return T[:3, :3].copy(), T[:3, 3].copy()


def trans_inv(T):
    """Inverse of T in SE(3) without a general matrix inverse (MR Eq. 3.64)."""
    R, p = trans_to_rp(T)
    return rp_to_trans(R.T, -R.T @ p)


def is_se3(T, tol=1e-6):
    """True if T is a 4x4 element of SE(3): R^T R = I, det R = +1, last row (0, 0, 0, 1)."""
    T = np.asarray(T, float)
    if T.shape != (4, 4):
        return False
    R = T[:3, :3]
    return (np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1) < tol
            and np.allclose(T[3], [0, 0, 0, 1], atol=tol))


def transform_point(T, x):
    """Return T x = R x + p for one point (shape (3,)) or many points (shape (N, 3)) (MR Eq. 3.65)."""
    R, p = trans_to_rp(T)
    x = np.asarray(x, float)
    return x @ R.T + p


def quat_xyzw_to_rot(q_xyzw):
    """Rotation matrix from a tf2/ROS quaternion (x, y, z, w) -- SCALAR LAST.

    Quaternions are taught in Week 4 (MR App. B.3, scalar FIRST there); this converter is given so that the
    tf2-style JSON files can be read. The quaternion is normalised first (MR Eq. B.12)."""
    x, y, z, w = np.asarray(q_xyzw, float) / np.linalg.norm(q_xyzw)
    return np.array([
        [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (w * y + x * z)],
        [2 * (w * z + x * y), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (w * x + y * z), w * w - x * x - y * y + z * z]])


class TransformTree:
    """A tree of named frames; each edge parent -> child stores T_parent_child (Lab 1, given).

    Like tf2, every frame has at most one parent, so there is exactly one path between any two
    connected frames.  lookup(target, source) = (T_common_target)^-1 T_common_source."""

    def __init__(self):
        self.parent = {}   # child -> parent
        self.T = {}        # child -> T_parent_child

    @property
    def frames(self):
        return set(self.parent) | set(self.parent.values())

    def add(self, parent, child, T_parent_child):
        """Add (or update, if the same parent->child edge exists) a transform (ValueError if invalid)."""
        T = np.asarray(T_parent_child, float)
        if parent == child:
            raise ValueError("a frame cannot be its own parent")
        if not is_se3(T):
            raise ValueError(f"transform {parent}->{child} is not in SE(3)")
        if child in self.parent and self.parent[child] != parent:
            raise ValueError(f"frame '{child}' already has parent '{self.parent[child]}'")
        if child in self._ancestors(parent):
            raise ValueError(f"edge {parent}->{child} would create a cycle")
        self.parent[child] = parent
        self.T[child] = T.copy()

    def _ancestors(self, frame):
        """[frame, parent, grandparent, ..., root]."""
        chain = [frame]
        while chain[-1] in self.parent:
            chain.append(self.parent[chain[-1]])
        return chain

    def _to_ancestor(self, frame, ancestor):
        """T_ancestor_frame, composed down the chain ancestor -> ... -> frame."""
        T = np.eye(4)
        f = frame
        while f != ancestor:
            T = self.T[f] @ T
            f = self.parent[f]
        return T

    def _common(self, target, source):
        """Lowest common ancestor of two frames (KeyError / LookupError if impossible)."""
        for f in (target, source):
            if f not in self.frames:
                raise KeyError(f"unknown frame '{f}'")
        up_s = self._ancestors(source)
        common = next((f for f in self._ancestors(target) if f in up_s), None)
        if common is None:
            raise LookupError(f"frames '{target}' and '{source}' are not connected")
        return common

    def path(self, target, source):
        """List of frames from target to source through their lowest common ancestor."""
        common = self._common(target, source)
        up_t, up_s = self._ancestors(target), self._ancestors(source)
        return up_t[:up_t.index(common) + 1] + up_s[:up_s.index(common)][::-1]

    def lookup(self, target, source):
        """T_target_source: pose of `source` expressed in `target` (tf2 lookup_transform order)."""
        common = self._common(target, source)
        return trans_inv(self._to_ancestor(target, common)) @ self._to_ancestor(source, common)


def transform_from_tf(translation, rotation_xyzw):
    """T from a tf2-style (translation, quaternion x,y,z,w) pair."""
    return rp_to_trans(quat_xyzw_to_rot(rotation_xyzw), translation)


def load_tree_json(path):
    """Build a TransformTree from a JSON file holding a list of tf2-style static transforms:
       {"transforms": [{"parent": .., "child": .., "translation": [x,y,z],
                        "rotation_xyzw": [qx,qy,qz,qw]}, ...]}"""
    with open(path) as fh:
        data = json.load(fh)
    tree = TransformTree()
    for e in data["transforms"]:
        tree.add(e["parent"], e["child"], transform_from_tf(e["translation"], e["rotation_xyzw"]))
    return tree


def load_track_csv(path):
    """Read a tracked pose sequence. Columns: t, r11..r33 (row-major), px, py, pz.
    Returns (t, Ts) with Ts of shape (N, 4, 4)."""
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    t = d[:, 0]
    Ts = np.tile(np.eye(4), (len(t), 1, 1))
    Ts[:, :3, :3] = d[:, 1:10].reshape(-1, 3, 3)
    Ts[:, :3, 3] = d[:, 10:13]
    return t, Ts


def axis_angle_between(a, b):
    """Angle (rad) between two direction vectors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.acos(max(-1.0, min(1.0, c)))


def point_line_distance(x, q, s_hat):
    """Distance from point x to the line through q with direction s_hat."""
    s = np.asarray(s_hat, float) / np.linalg.norm(s_hat)
    d = np.asarray(x, float) - np.asarray(q, float)
    return float(np.linalg.norm(d - (d @ s) * s))


# ================================================================ Step 1: angular velocity
def angular_velocities(R, Rdot):
    """Return (omega_s, omega_b) of a rotating frame R = R_sb(t) with time derivative Rdot.

    [omega_s] = Rdot R^-1 = Rdot R^T (space frame), [omega_b] = R^-1 Rdot = R^T Rdot (body frame)
    (MR Eqs. 3.37-3.38).  R^-1 = R^T for a rotation matrix -- no general inverse is needed."""
    # TODO: [omega_s] = Rdot R^T and [omega_b] = R^T Rdot (MR Eqs. 3.37-3.38); read the vectors with so3_to_vec
    raise NotImplementedError("angular_velocities")


# ================================================================ Step 2: se(3) and the adjoint
def vec_to_se3(V):
    """4x4 matrix [V] of the twist V = (omega, v) (MR Eq. 3.71)."""
    # TODO: top-left block [omega] (vec_to_so3), top-right column v, bottom row zeros
    raise NotImplementedError("vec_to_se3")


def se3_to_vec(se3mat):
    """Twist V = (omega, v) from its 4x4 matrix [V]."""
    # TODO: inverse of vec_to_se3
    raise NotImplementedError("se3_to_vec")


def adjoint(T):
    """6x6 adjoint [Ad_T] = [[R, 0], [[p]R, R]] (MR Def. 3.20)."""
    # TODO: MR Def. 3.20: blocks R, 0, [p]R, R
    raise NotImplementedError("adjoint")


# ================================================================ Step 3: twists and point velocities
def twists_from_T_Tdot(T, Tdot):
    """Return (V_s, V_b) from a configuration T_sb and its time derivative (MR Eqs. 3.81-3.82)."""
    # TODO: [V_s] = Tdot T^-1, [V_b] = T^-1 Tdot (MR Eqs. 3.81-3.82); use trans_inv and se3_to_vec
    raise NotImplementedError("twists_from_T_Tdot")


def point_velocity(V, x):
    """Velocity omega x x + v of the body point whose coordinates are x, for a twist V = (omega, v).

    V and x must be expressed in the SAME frame {a}; the result is the velocity of that point in {a}
    coordinates.  With {a} = {s}: xdot_s = omega_s x x_s + v_s (from [V_s] = Tdot T^-1, MR Eqs. 3.72-3.74);
    with {a} = {b}: R^T xdot_s = omega_b x x_b + v_b.  x may be one point (3,) or many points (N, 3)."""
    # TODO: omega x x + v with omega = V[:3], v = V[3:]; np.cross works row by row for an (N, 3) array
    raise NotImplementedError("point_velocity")


def twists_from_samples(Ts, dt):
    """Estimate (V_s, V_b) at every interior sample of a pose sequence by central differences.

    Ts: array (N, 4, 4) of T_sb(t_k) sampled every dt seconds.
    Returns two arrays of shape (N-2, 6): twist k corresponds to sample k+1.
    Tdot_k ~ (T_{k+1} - T_{k-1}) / (2 dt).  The numerical T^-1 Tdot is only approximately in
    se(3), so the angular part is read from its skew-symmetric part (M - M^T)/2."""
    # TODO: central difference Tdot_k = (T_{k+1}-T_{k-1})/(2 dt); read omega from the skew part (M - M^T)/2
    raise NotImplementedError("twists_from_samples")


# ================================================================ Step 4: screws
def screw_to_twist(q, s_hat, h, thetadot=1.0):
    """Twist V = (s_hat thetadot, (-s_hat x q + h s_hat) thetadot) of a screw {q, s_hat, h}
    (MR Sec. 3.3.2.2).  Use h = np.inf for a pure translation along s_hat (then q is ignored)."""
    # TODO: normalise s_hat; h = inf -> (0, s_hat*thetadot); else (s_hat, -s_hat x q + h s_hat)*thetadot
    raise NotImplementedError("screw_to_twist")


def twist_to_screw(V):
    """Screw interpretation of a twist (MR Sec. 3.3.2.2, Def. 3.24).

    Returns a dict with keys
      'S'        normalised screw axis (MR Def. 3.24),
      'thetadot' speed about/along the axis (>= 0),
      'h'        pitch (np.inf for a pure translation),
      's_hat'    unit axis direction,
      'q'        the point on the axis closest to the frame origin (q = omega x v / |omega|^2),
                 or None for a pure translation.
    Raises ValueError for the zero twist."""
    # TODO: case |omega| > TOL: thetadot = |omega|, h = omega.v/|omega|^2, q = omega x v/|omega|^2; case omega = 0: h = inf, q = None; zero twist -> ValueError
    raise NotImplementedError("twist_to_screw")
