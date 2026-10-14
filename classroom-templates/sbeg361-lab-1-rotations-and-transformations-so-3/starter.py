"""SBEG 361 Lab 1 -- Rotations and transformations: SO(3)/SE(3) property tests and transform trees (STARTER).

Fill in every function marked  # TODO.  Do not change function names or signatures:
the auto-checks in test_lab.py import this file with   LAB_IMPL=starter python -m pytest -q

Notation follows Lynch & Park, *Modern Robotics* (MR), Sections 3.1-3.3.1, rotations and rigid motions
side by side:
  * R_sb in SO(3): orientation of {b} in {s} (columns = x_b, y_b, z_b written in {s})     (MR Def. 3.1)
  * T_sb = (R_sb, p_sb) in SE(3): configuration of {b} in {s}                              (MR Def. 3.13, Eq. 3.62)
  * Rot(w_hat, theta): rotation operator about the unit axis w_hat by theta                (MR p. 72)
  * inverses: R^-1 = R^T (MR Prop. 3.3);  T^-1 = (R^T, -R^T p)                             (MR Eq. 3.64)
  * fixed vs body frame: R R_sb / R_sb R (MR Eqs. 3.23-3.24);  T T_sb / T_sb T              (MR Eqs. 3.66-3.67)
  * planar frames (P, p), P in SO(2), and their 3x3 SE(2) matrices                         (MR Eqs. 3.8-3.9, 3.63)

Transform tree convention (identical to ROS 2 tf2): an edge parent -> child stores T_parent_child, the pose of
the child frame expressed in the parent frame. lookup(target, source) returns T_target_source
(tf2: buffer.lookup_transform(target, source, t)).

Use NumPy only: do NOT call modern_robotics, scipy.spatial.transform or np.linalg.inv inside this file -- they
are what the tests compare your answers with.  The functions marked (Given.) are complete and may be used.
"""
import json

import numpy as np

# ---------------------------------------------------------------------------
# Part A -- building R and T
# ---------------------------------------------------------------------------

def rot_x(theta):
    """Rot(x_hat, theta) as a 3x3 NumPy array (theta in radians).  (Given as an example.)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


def rot_y(theta):
    """Rot(y_hat, theta) as a 3x3 NumPy array (theta in radians).  See MR p. 72."""
    # TODO: build the matrix; watch the sign of the sin terms (it differs from rot_x and rot_z)
    raise NotImplementedError("TODO: rot_y")


def rot_z(theta):
    """Rot(z_hat, theta) as a 3x3 NumPy array (theta in radians).  See MR p. 72."""
    # TODO
    raise NotImplementedError("TODO: rot_z")


def rot_axis_angle(omega_hat, theta):
    """Rot(omega_hat, theta) from the explicit formula on MR p. 72.

    omega_hat must be a unit 3-vector: raise ValueError if | ||omega_hat|| - 1 | > 1e-6.
    Hint: write c = cos(theta), s = sin(theta), v = 1 - c and fill the 9 entries.
    """
    # TODO
    raise NotImplementedError("TODO: rot_axis_angle")


def rp_to_trans(R, p):
    """Build T = [[R, p], [0, 1]] (MR Eq. 3.62)."""
    # TODO: start from np.eye(4) and fill the R block and the p column
    raise NotImplementedError("TODO: rp_to_trans")


def trans_to_rp(T):
    """Split T into (R, p)."""
    # TODO: return copies of the top-left 3x3 block and the top-right 3x1 column
    raise NotImplementedError("TODO: trans_to_rp")


# ---------------------------------------------------------------------------
# Part B -- group property tests: SO(3) and SE(3) side by side
# ---------------------------------------------------------------------------

def so3_errors(R):
    """Return (orth_err, det) for a 3x3 matrix (raise ValueError if R is not 3x3).

    orth_err = Frobenius norm of R^T R - I   (MR Eq. 3.19)   -> np.linalg.norm
    det      = det(R)                         (MR Eq. 3.21 requires +1)
    Return plain Python floats.
    """
    # TODO
    raise NotImplementedError("TODO: so3_errors")


def is_rotation(R, tol=1e-3):
    """True if R is (numerically) in SO(3): shape 3x3, ||R^T R - I||_F < tol and |det R - 1| < tol.
    Return False (do not raise) for a wrong shape."""
    # TODO
    raise NotImplementedError("TODO: is_rotation")


def rot_inv(R):
    """Inverse of a rotation matrix (MR Proposition 3.3).  Do NOT call np.linalg.inv."""
    # TODO
    raise NotImplementedError("TODO: rot_inv")


def project_to_so3(M):
    """Closest rotation matrix to M in the Frobenius norm (SVD method).

    M = U S V^T  ->  R = U diag(1, 1, d) V^T with d = det(U V^T), so that det R = +1 always.
    """
    # TODO: np.linalg.svd returns U, S, Vt (Vt is already V transposed)
    raise NotImplementedError("TODO: project_to_so3")


def is_se3(T, tol=1e-6):
    """True if T is a 4x4 element of SE(3): R^T R = I, det R = +1, last row (0, 0, 0, 1).
    Return False (do not raise) for a wrong shape."""
    # TODO: check shape, R^T R = I, det R = +1 and the last row (np.allclose with atol=tol)
    raise NotImplementedError("TODO: is_se3")


def trans_inv(T):
    """Inverse of T in SE(3) without a general matrix inverse (MR Eq. 3.64)."""
    # TODO: use MR Eq. 3.64 (and your trans_to_rp / rp_to_trans) -- do NOT call np.linalg.inv
    raise NotImplementedError("TODO: trans_inv")


# ---------------------------------------------------------------------------
# Part C -- the three uses of R and T (MR Sections 3.2.1, 3.3.1)
# ---------------------------------------------------------------------------

def change_frame(R_ab, p_b):
    """Change the reference frame of a vector: return p_a (subscript cancellation rule)."""
    # TODO
    raise NotImplementedError("TODO: change_frame")


def transform_point(T, x):
    """Return T x = R x + p for one point (shape (3,)) or many points (shape (N, 3)) (MR Eq. 3.65)."""
    # TODO: R x + p; for an (N,3) array use x @ R.T + p (this form also works for one point)
    raise NotImplementedError("TODO: transform_point")


def rotate_in_fixed(R_sb, R):
    """Rotate {b} by R about an axis expressed in the fixed frame {s} (MR Eq. 3.23)."""
    # TODO
    raise NotImplementedError("TODO: rotate_in_fixed")


def rotate_in_body(R_sb, R):
    """Rotate {b} by R about an axis expressed in the body frame {b} (MR Eq. 3.24)."""
    # TODO
    raise NotImplementedError("TODO: rotate_in_body")


def transform_in_fixed(T_sb, T):
    """Displace {b} by T = (R, p) expressed in the fixed frame {s} (MR Eq. 3.66)."""
    # TODO: the SE(3) twin of rotate_in_fixed
    raise NotImplementedError("TODO: transform_in_fixed")


def transform_in_body(T_sb, T):
    """Displace {b} by T = (R, p) expressed in the body frame {b} (MR Eq. 3.67)."""
    # TODO: the SE(3) twin of rotate_in_body
    raise NotImplementedError("TODO: transform_in_body")


def planar_to_se2(P, p):
    """3x3 SE(2) matrix [[P, p], [0, 0, 1]] of a planar frame (P, p) (MR Def. 3.14, Eq. 3.63)."""
    # TODO: start from np.eye(3), as in rp_to_trans
    raise NotImplementedError("TODO: planar_to_se2")


def planar_compose(P, p, Q, q):
    """Compose planar frames: given {b} in {s} as (P, p) and {c} in {b} as (Q, q),
    return {c} in {s} as the tuple (R, r)  (MR Eqs. 3.8-3.9)."""
    # TODO
    raise NotImplementedError("TODO: planar_compose")


# ---------------------------------------------------------------------------
# Part D -- the transform tree (tf2-like)
# ---------------------------------------------------------------------------

def quat_xyzw_to_rot(q_xyzw):
    """Rotation matrix from a tf2/ROS quaternion (x, y, z, w) -- SCALAR LAST.  (Given.)

    MR App. B.3 writes q = (q0, q1, q2, q3) with the scalar first; tf2 and SciPy store
    (x, y, z, w) = (q1, q2, q3, q0).  The quaternion is normalised first, then MR Eq. B.12 is applied.
    Unit quaternions are taught in Week 4 (Lab 3); here the function only reads the JSON files."""
    x, y, z, w = np.asarray(q_xyzw, float) / np.linalg.norm(q_xyzw)
    return np.array([
        [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (w * y + x * z)],
        [2 * (w * z + x * y), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (w * x + y * z), w * w - x * x - y * y + z * z]])


class TransformTree:
    """A tree of named frames; each edge parent -> child stores T_parent_child.

    Like tf2, every frame has at most one parent, so there is exactly one path between any two
    connected frames.  lookup(target, source) composes transforms along that path:
        T_target_source = (T_common_target)^-1 T_common_source  (via the lowest common ancestor).
    """

    def __init__(self):
        self.parent = {}   # child -> parent
        self.T = {}        # child -> T_parent_child

    @property
    def frames(self):
        return set(self.parent) | set(self.parent.values())

    def add(self, parent, child, T_parent_child):
        """Add (or update, if the same parent->child edge exists) a transform.

        Raise ValueError if `child` already has a different parent, if parent == child, if
        T is not in SE(3), or if the edge would create a cycle."""
        # TODO: validate (parent != child, is_se3, single parent, no cycle via self._ancestors(parent)), then store
        raise NotImplementedError("TODO: TransformTree.add")

    def _ancestors(self, frame):
        """[frame, parent, grandparent, ..., root].  (Given.)"""
        chain = [frame]
        while chain[-1] in self.parent:
            chain.append(self.parent[chain[-1]])
        return chain

    def _to_ancestor(self, frame, ancestor):
        """T_ancestor_frame, composed down the chain ancestor -> ... -> frame."""
        # TODO: walk up from frame to ancestor, left-multiplying by self.T[f] (subscript cancellation)
        raise NotImplementedError("TODO: TransformTree._to_ancestor")

    def _common(self, target, source):
        """Lowest common ancestor of two frames (KeyError / LookupError if impossible).  (Given.)"""
        for f in (target, source):
            if f not in self.frames:
                raise KeyError(f"unknown frame '{f}'")
        up_s = self._ancestors(source)
        common = next((f for f in self._ancestors(target) if f in up_s), None)
        if common is None:
            raise LookupError(f"frames '{target}' and '{source}' are not connected")
        return common

    def path(self, target, source):
        """List of frames from target to source through their lowest common ancestor.  (Given.)"""
        common = self._common(target, source)
        up_t, up_s = self._ancestors(target), self._ancestors(source)
        return up_t[:up_t.index(common) + 1] + up_s[:up_s.index(common)][::-1]

    def lookup(self, target, source):
        """T_target_source: pose of `source` expressed in `target` (tf2 lookup_transform order)."""
        # TODO: common = self._common(target, source); combine two _to_ancestor results (one inverted)
        raise NotImplementedError("TODO: TransformTree.lookup")


def transform_from_tf(translation, rotation_xyzw):
    """T from a tf2-style (translation, quaternion x,y,z,w) pair.  (Given; uses your rp_to_trans.)"""
    return rp_to_trans(quat_xyzw_to_rot(rotation_xyzw), translation)


def load_tree_json(path):
    """Build a TransformTree from a JSON file holding a list of tf2-style static transforms:
       {"transforms": [{"parent": .., "child": .., "translation": [x,y,z],
                        "rotation_xyzw": [qx,qy,qz,qw]}, ...]}
    (Given; uses your TransformTree.add.)"""
    with open(path) as fh:
        data = json.load(fh)
    tree = TransformTree()
    for e in data["transforms"]:
        tree.add(e["parent"], e["child"], transform_from_tf(e["translation"], e["rotation_xyzw"]))
    return tree


# ---------------------------------------------------------------------------
# Part E -- applications: tracker data cleaning and surgical navigation
# ---------------------------------------------------------------------------

def load_tracker_csv(path):
    """Read the tracker file (given). Returns an (N, 3, 3) array; each row of the file holds
    sample_id, r11, r12, r13, r21, ..., r33 (row-major)."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 1:10].reshape(-1, 3, 3)


def classify_samples(Rs, tol=1e-3):
    """Boolean NumPy array: True where sample k is a valid rotation matrix (use is_rotation)."""
    # TODO
    raise NotImplementedError("TODO: classify_samples")


def tool_orientations(R_sm_list, R_mt):
    """Orientation of the tool-tip frame {t} in the tracker frame {s} for every sample.
    Return an (N, 3, 3) array.  Which matrix goes first?  Use subscript cancellation."""
    # TODO
    raise NotImplementedError("TODO: tool_orientations")


def load_track_csv(path):
    """Read a tracked pose sequence (given). Columns: t, r11..r33 (row-major), px, py, pz.
    Returns (t, Ts) with Ts of shape (N, 4, 4)."""
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    t = d[:, 0]
    Ts = np.tile(np.eye(4), (len(t), 1, 1))
    Ts[:, :3, :3] = d[:, 1:10].reshape(-1, 3, 3)
    Ts[:, :3, 3] = d[:, 10:13]
    return t, Ts
