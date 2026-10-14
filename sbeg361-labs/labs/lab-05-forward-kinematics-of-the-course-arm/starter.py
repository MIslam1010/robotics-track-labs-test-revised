"""SBEG 361 Lab 5 -- Forward kinematics of the course arm by the product of exponentials (STARTER).

Fill in every function marked  # TODO.  Do not change names or signatures: test_lab.py imports them.
Run the auto-checks with

    LAB_IMPL=starter python -m pytest -q test_lab.py

You may NOT call modern_robotics, scipy or np.linalg.inv / np.linalg.pinv / scipy.linalg.expm inside
these functions -- they are what the tests compare against.  Functions marked "(complete helper)"
and the helpers from Labs 1-3 at the top may be used freely.  These functions join your
`sbeg361_kinematics` library (module fk.py); Lab 6 (DH) and Lab 7 (Jacobians) reuse them.

Notation follows Lynch & Park, *Modern Robotics* (MR), Chapter 4:
  space form   T(theta) = e^[S1]th1 ... e^[Sn]thn M        (MR Eq. 4.14)
  body form    T(theta) = M e^[B1]th1 ... e^[Bn]thn        (MR Eq. 4.16), B_i = [Ad_{M^-1}] S_i
  revolute     S = (omega, -omega x q), |omega| = 1;  prismatic  S = (0, v), |v| = 1   (MR Sec. 4.1.1)
  exp on se(3) MR Prop. 3.25, Eqs. 3.88-3.89
  URDF         joint origin = pose of the child-link frame in the parent-link frame at q = 0;
               rpy = roll about x, pitch about y, yaw about z, all about FIXED axes:
               R = Rot(z, yaw) Rot(y, pitch) Rot(x, roll)   (MR Sec. 4.2)
Hardware log format (identical to the project's hwcompare.py):
  t, cmd_1..cmd_n, meas_1..meas_n[, stop]   -- t in s, angles in SERVO DEGREES.
  Model angles:  q_model = q_servo - servo_zero_offset   (course convention; offsets in data/servo_offsets.json).
"""
import csv
import json
import math
import xml.etree.ElementTree as ET

import numpy as np

TOL = 1e-6  # same threshold as modern_robotics.NearZero


# =========================================================== helpers from Labs 1-3 (complete)
def vec_to_so3(omega):
    """[omega] (MR Eq. 3.30)."""
    w1, w2, w3 = np.asarray(omega, float)
    return np.array([[0.0, -w3, w2], [w3, 0.0, -w1], [-w2, w1, 0.0]])


def vec_to_se3(V):
    """[V] = [[omega] v; 0 0] (MR Eq. 3.71)."""
    V = np.asarray(V, float)
    m = np.zeros((4, 4))
    m[:3, :3] = vec_to_so3(V[:3])
    m[:3, 3] = V[3:]
    return m


def trans_inv(T):
    """T^-1 = (R^T, -R^T p) (MR Eq. 3.64)."""
    T = np.asarray(T, float)
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def adjoint(T):
    """[Ad_T] = [[R, 0], [[p]R, R]] (MR Def. 3.20)."""
    T = np.asarray(T, float)
    R, p = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[3:, 3:] = R
    A[3:, :3] = vec_to_so3(p) @ R
    return A


def rot_axis(axis, angle):
    """Rotation about a coordinate axis 'x', 'y' or 'z' (MR Eq. 3.16 family)."""
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# =========================================================== Step 1 -- screw axes and exp
def screw_axis(omega=None, q=None, v=None):
    """Screw axis of a one-DOF joint at the zero configuration (MR Sec. 4.1.1).

    Revolute: give the unit axis `omega` and any point `q` on the axis -> S = (omega, -omega x q).
    Prismatic: give only the unit direction `v` -> S = (0, v).
    Raises ValueError if the direction is not a unit vector or the arguments are inconsistent."""
    # TODO: revolute: check |omega| = 1 and return np.r_[omega, -cross(omega, q)]; prismatic: np.r_[0,0,0, v]; raise ValueError for bad input
    raise NotImplementedError("screw_axis")


def screw_exp(S, theta):
    """e^{[S] theta} for a screw axis S (MR Prop. 3.25, Eq. 3.88; Eq. 3.89 if omega = 0).

    Same result as Lab 3's matrix_exp6(se3mat) called as matrix_exp6(vec_to_se3(S) * theta);
    here the axis and the angle are passed separately."""
    # TODO: omega ~ 0 -> Eq. 3.89 (I, v*theta). Otherwise R by Rodrigues (Eq. 3.51) and p = G(theta) v with G from Eq. 3.87
    raise NotImplementedError("screw_exp")


# =========================================================== Step 2 -- PoE forward kinematics
def fk_space(M, Slist, theta):
    """Space-form PoE: T = e^[S1]th1 ... e^[Sn]thn M (MR Eq. 4.14). Slist is 6 x n."""
    # TODO: multiply e^[S_i]theta_i from i = 1 to n (left to right), then M on the RIGHT (Eq. 4.14)
    raise NotImplementedError("fk_space")


def fk_body(M, Blist, theta):
    """Body-form PoE: T = M e^[B1]th1 ... e^[Bn]thn (MR Eq. 4.16). Blist is 6 x n."""
    # TODO: start from M, then multiply e^[B_i]theta_i on the RIGHT for i = 1..n (Eq. 4.16)
    raise NotImplementedError("fk_body")


def body_screws(Slist, M):
    """B_i = [Ad_{M^-1}] S_i for every column (MR Eq. 4.16)."""
    # TODO: one line with adjoint() and trans_inv()
    raise NotImplementedError("body_screws")


def course_arm_screws(W1, W2, L1, L2, H1, H2, tool=0.0):
    """(M, Slist) of the course arm from its geometry (UR-type layout of MR Example 4.5).

    Zero configuration: arm stretched along +x_s, {s} on the joint-1 axis at the base, z_s up.
    Joint axes at home (direction, one point on the axis):
      1: +z through (0, 0, 0)            2: +y through (0, 0, H1)
      3: +y through (L1, 0, H1)          4: +y through (L1+L2, 0, H1)
      5: -z through (L1+L2, W1, 0)       6: +y through (L1+L2, 0, H1-H2)
    {b} at home: origin (L1+L2, W1+W2, H1-H2), x_b = -x_s, y_b = z_s, z_b = y_s;
    a straight tool of length `tool` moves {b} along its own z_b."""
    # TODO: list the six (omega_i, q_i) pairs from the docstring, build Slist with screw_axis, then M; tool: M @ Trans(0,0,tool)
    raise NotImplementedError("course_arm_screws")


# =========================================================== Step 3 -- URDF -> PoE
def read_urdf_joints(path):
    """Parse the <joint> elements of a URDF file (complete helper).

    Returns a list of dicts with keys name, type, parent, child, xyz, rpy, axis, limit
    (limit = (lower, upper) or None).  Missing <origin>/<axis> take the URDF defaults
    (zero, and (1, 0, 0))."""
    root = ET.parse(path).getroot()
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        a = j.find("axis")
        lim = j.find("limit")
        xyz = [float(x) for x in o.get("xyz", "0 0 0").split()] if o is not None else [0.0] * 3
        rpy = [float(x) for x in o.get("rpy", "0 0 0").split()] if o is not None else [0.0] * 3
        axis = [float(x) for x in a.get("xyz").split()] if a is not None else [1.0, 0.0, 0.0]
        limit = None
        if lim is not None and lim.get("lower") is not None:
            limit = (float(lim.get("lower")), float(lim.get("upper")))
        joints.append(dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                           child=j.find("child").get("link"), xyz=np.array(xyz), rpy=np.array(rpy),
                           axis=np.array(axis), limit=limit))
    return joints


def rpy_to_rot(rpy):
    """URDF roll-pitch-yaw (fixed axes x, y, z): R = Rot(z, yaw) Rot(y, pitch) Rot(x, roll)."""
    # TODO: use rot_axis; the ORDER matters (fixed axes x, then y, then z)
    raise NotImplementedError("rpy_to_rot")


def urdf_chain(joints, tip_link):
    """Joints from the root link to `tip_link`, in order from the base (complete helper)."""
    by_child = {j["child"]: j for j in joints}
    chain, link = [], tip_link
    while link in by_child:
        j = by_child[link]
        chain.append(j)
        link = j["parent"]
    if not chain:
        raise ValueError(f"no joint leads to link '{tip_link}'")
    return chain[::-1]


def poe_from_urdf(joints, tip_link):
    """Space-form PoE model of the chain that ends at `tip_link`.

    Walk the chain from the base with all joint values zero, composing the joint origins:
    T_s,child = T_s,parent @ (rpy_to_rot(rpy), xyz).  For each movable joint the axis in {s} is
    omega = R_s,child @ axis (axis is given in the child frame), the joint frame origin lies on the
    axis, so S = (omega, -omega x p) (revolute/continuous) or (0, R axis) (prismatic).
    Fixed joints only move the frame.  M = T_s,tip at zero.
    Returns (M, Slist [6 x n], names [n], limits [n x 2] with +-inf where none)."""
    # TODO: loop over urdf_chain(joints, tip_link); compose T = T @ (rpy_to_rot(rpy), xyz); skip 'fixed' after composing; omega = R @ axis
    raise NotImplementedError("poe_from_urdf")


# =========================================================== Step 4 -- simulation vs hardware
def load_joint_log(path):
    """Read a joint log 't, cmd_i, meas_i[, stop]' (servo degrees) (complete helper)."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    header = [h.strip() for h in rows[0]]
    data = np.array([[float(x) for x in r] for r in rows[1:]])
    col = {h: i for i, h in enumerate(header)}
    n = sum(h.startswith("cmd_") for h in header)
    return dict(t=data[:, col["t"]],
                cmd=data[:, [col[f"cmd_{i + 1}"] for i in range(n)]],
                meas=data[:, [col[f"meas_{i + 1}"] for i in range(n)]],
                stop=data[:, col["stop"]] if "stop" in col else None)


def load_offsets(path):
    """Servo zero offsets in degrees from a calibration JSON file (complete helper)."""
    with open(path) as fh:
        return np.array(json.load(fh)["servo_zero_offset_deg"], float)


def servo_to_model(q_servo_deg, offsets_deg):
    """Model joint angles in RADIANS: q_model = deg2rad(q_servo - offset). Works on [n] or [N x n]."""
    # TODO: subtract the offset (degrees), then np.deg2rad
    raise NotImplementedError("servo_to_model")


def tool_points(M, Slist, Q, p_tool=(0.0, 0.0, 0.0)):
    """Positions in {s} (m) of the tool point p_tool (given in {b}) for each row of Q (radians)."""
    # TODO: homogeneous point np.r_[p_tool, 1]; one fk_space call per row of Q
    raise NotImplementedError("tool_points")


def compare_log(log, M, Slist, offsets_deg, p_tool=(0.0, 0.0, 0.0)):
    """Commanded vs measured motion, using only the samples BEFORE the stop command.

    Returns a dict with
      joint_rms_deg, joint_max_deg, joint_mean_deg   [n] statistics of e = meas - cmd (degrees)
      task_err_mm                                   [N'] |p(meas) - p(cmd)| of the tool point
      task_rms_mm, task_max_mm                       floats
      n_samples                                      N' (samples used)"""
    # TODO: find k0 (first stop sample), e = meas - cmd, convert both to model radians, tool_points for both, distance in mm
    raise NotImplementedError("compare_log")


def tip_errors_mm(M, Slist, Q, p_meas_mm, p_tool=(0.0, 0.0, 0.0)):
    """Distance (mm) between the FK-predicted tool point and the externally measured one, per pose."""
    # TODO: 1e3 * tool_points(...) minus the measured points, row-wise norm
    raise NotImplementedError("tip_errors_mm")


def fit_joint_offset(M, Slist, Q, p_meas_mm, joint, p_tool=(0.0, 0.0, 0.0),
                     search_deg=5.0, tol_deg=1e-4):
    """Zero-offset error delta (degrees) of ONE joint (0-based index) that best explains the
    measured tool points: minimise sum_k |p(q_k + delta e_joint) - p_meas_k|^2 over
    |delta| <= search_deg (golden-section search).  Returns (delta_deg, rms_after_mm)."""
    # TODO: cost(delta) = mean squared tip error with Q[:, joint] += deg2rad(delta); golden-section search on [-search_deg, search_deg]
    raise NotImplementedError("fit_joint_offset")
