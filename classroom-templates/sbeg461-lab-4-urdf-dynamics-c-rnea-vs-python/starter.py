"""SBEG 461 Lab 4 -- URDF dynamics, RNEA timing (Python vs C++) and the constrained 2R arm.
STARTER -- you edit this file. Replace every `raise NotImplementedError` with your code.
Run the auto-checks with:  LAB_IMPL=starter python -m pytest -q test_lab.py

Notation follows Modern Robotics (MR): twists V = (omega, v), wrenches F = (m, f), screw axes S,
Mlist = [M01, M12, ..., M_{n,n+1}] (link CoM frames at theta = 0), Glist = [G1, ..., Gn] (6x6 spatial
inertias in the CoM frames, MR Eq. 8.32), Slist = 6 x n space screw axes.

Parts
  A. URDF -> (Mlist, Glist, Slist)                         MR Sec. 8.8, Sec. 4.2
  B. Recursive Newton-Euler inverse dynamics (your own)     MR Eqs. 8.50-8.54
     + timing helper; export for the C++ program in cpp/
  C. Task-space dynamics                                    MR Eqs. 8.89-8.91
  D. Constrained dynamics of the 2R arm in a channel        MR Sec. 8.7, Eqs. 8.93-8.100, Example 8.5
     + Baumgarte stabilisation (Featherstone Sec. 8.3, beyond MR)
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ==========================================================================================
# Provided helpers (do not edit) -- small SE(3) toolkit, so the lab does not
# depend on modern_robotics inside your own code. The tests use modern_robotics as ground truth.
# ==========================================================================================


def skew(w):
    """[w] for a 3-vector (MR Eq. 3.30)."""
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def rpy_to_R(rpy):
    """URDF fixed-axis roll-pitch-yaw: R = Rot(z, yaw) Rot(y, pitch) Rot(x, roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def RpToTrans(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def TransInv(T):
    R, p = T[:3, :3], T[:3, 3]
    return RpToTrans(R.T, -R.T @ p)


def Adjoint(T):
    """[Ad_T] (MR Def. 3.20)."""
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


def ad(V):
    """[ad_V] (MR Eq. 8.38)."""
    w, v = V[:3], V[3:]
    out = np.zeros((6, 6))
    out[:3, :3] = skew(w)
    out[3:, :3] = skew(v)
    out[3:, 3:] = skew(w)
    return out


def MatrixExp6(S, theta):
    """e^{[S] theta} for a screw axis S with ||omega|| = 1 or (omega = 0, ||v|| = 1) (MR Prop. 3.25)."""
    w, v = np.asarray(S[:3], float), np.asarray(S[3:], float)
    if np.linalg.norm(w) < 1e-12:
        return RpToTrans(np.eye(3), v * theta)
    W = skew(w)
    R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * W @ W
    Gv = (np.eye(3) * theta + (1 - np.cos(theta)) * W + (theta - np.sin(theta)) * W @ W) @ v
    return RpToTrans(R, Gv)


def read_urdf_chain(path):
    """Parse a serial-chain URDF into a list of joints ordered from the base to the tip.

    Each entry is a dict with keys
      name, type ('revolute' | 'continuous' | 'fixed'),
      T_origin : 4x4 pose of the child-link frame in the parent-link frame when the joint is at zero,
      axis     : unit 3-vector, the joint axis expressed in the child-link frame (zeros for 'fixed'),
      child    : name of the child link,
      mass     : float (0.0 if the child link has no <inertial>),
      T_com    : 4x4 pose of the child link's inertial (CoM) frame in the child-link frame,
      I        : 3x3 rotational inertia about the CoM, expressed in the inertial frame.
    """
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            links[link.get("name")] = (0.0, np.eye(4), np.zeros((3, 3)))
            continue
        org = inertial.find("origin")
        xyz = np.fromstring(org.get("xyz", "0 0 0"), sep=" ") if org is not None else np.zeros(3)
        rpy = np.fromstring(org.get("rpy", "0 0 0"), sep=" ") if org is not None else np.zeros(3)
        m = float(inertial.find("mass").get("value"))
        a = {k: float(inertial.find("inertia").get(k, 0.0)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
        Ic = np.array([[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]], [a["ixz"], a["iyz"], a["izz"]]])
        links[link.get("name")] = (m, RpToTrans(rpy_to_R(rpy), xyz), Ic)
    joints = {j.find("parent").get("link"): j for j in root.findall("joint")}
    children = {j.find("child").get("link") for j in root.findall("joint")}
    base = next(name for name in links if name not in children)
    chain, parent = [], base
    while parent in joints:
        j = joints[parent]
        org = j.find("origin")
        xyz = np.fromstring(org.get("xyz", "0 0 0"), sep=" ") if org is not None else np.zeros(3)
        rpy = np.fromstring(org.get("rpy", "0 0 0"), sep=" ") if org is not None else np.zeros(3)
        jtype = j.get("type")
        axis = np.zeros(3)
        if jtype != "fixed":
            ax = j.find("axis")
            axis = np.fromstring(ax.get("xyz"), sep=" ") if ax is not None else np.array([1.0, 0, 0])
            axis = axis / np.linalg.norm(axis)
        child = j.find("child").get("link")
        m, T_com, Ic = links[child]
        chain.append(dict(name=j.get("name"), type=jtype, T_origin=RpToTrans(rpy_to_R(rpy), xyz),
                          axis=axis, child=child, mass=m, T_com=T_com, I=Ic))
        parent = child
    return chain


def export_for_cpp(model_path, states_path, Mlist, Glist, Slist, states):
    """Write the plain-text files read by cpp/main.cpp.

    model file : line 1 = n; then n+1 matrices M_{i-1,i} (4 rows of 4); n matrices G_i (6 rows of 6);
                 then n screw axes S_i (one row of 6 each).
    states file: one state per line, 3n numbers (theta, thetadot, thetaddot).
    """
    n = len(Glist)
    with open(model_path, "w") as f:
        f.write(f"{n}\n")
        for M in Mlist:
            np.savetxt(f, M, fmt="%.17g")
        for G in Glist:
            np.savetxt(f, G, fmt="%.17g")
        np.savetxt(f, np.asarray(Slist).T, fmt="%.17g")
    np.savetxt(states_path, np.asarray(states), fmt="%.17g")


def load_states(csv_path):
    """Read states_arm6.csv -> (N, 18) array."""
    return np.loadtxt(csv_path, delimiter=",", skiprows=1)


# ==========================================================================================
# Part A -- URDF -> MR dynamic parameters (MR Sec. 8.8)
# ==========================================================================================


def urdf_to_mr(path):
    """Return (Mlist, Glist, Slist) for the serial chain in a URDF file.

    Frames: {0} = base_link frame; {i} = CoM (inertial) frame of moving link i; {n+1} = frame of the last
    link reached through fixed joints (the tool frame). All poses at theta = 0.
      M_{i-1,i} = T_{0,i-1}^{-1} T_{0,i}   (Mlist has n + 1 entries),
      G_i = diag(I_i, m_i I_3)            (MR Eq. 8.32, I_i about the CoM in the inertial frame),
      S_i = (omega_i, -omega_i x q_i)     (omega_i = joint axis in {0}, q_i = joint-frame origin in {0}).
    Moving links are those that are the child of a revolute/continuous joint. Fixed joints after the last
    moving joint only move the tool frame (their links are assumed massless).
    """
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("urdf_to_mr")


# ==========================================================================================
# Part B -- RNEA (MR Eqs. 8.50-8.54) and timing
# ==========================================================================================


def rnea(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist):
    """Joint torques tau = M(theta) thetaddot + h(theta, thetadot) + J^T Ftip by the recursive
    Newton-Euler algorithm (same interface as modern_robotics.InverseDynamics).

    Forward:  T_{i,i-1} = e^{-[A_i] theta_i} M_{i,i-1}                         (8.50)
              V_i = Ad_{T_{i,i-1}} V_{i-1} + A_i thetadot_i                       (8.51)
              Vdot_i = Ad_{T_{i,i-1}} Vdot_{i-1} + ad_{V_i} A_i thetadot_i + A_i thetaddot_i   (8.52)
    Backward: F_i = Ad^T_{T_{i+1,i}} F_{i+1} + G_i Vdot_i - ad^T_{V_i} G_i V_i   (8.53)
              tau_i = F_i^T A_i                                                    (8.54)
    with V_0 = 0, Vdot_0 = (0, -g) (gravity as a fictitious base acceleration) and F_{n+1} = Ftip.
    """
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("rnea")


def time_per_call(fn, arg_tuples, repeats=3):
    """Median over `repeats` runs of the mean wall-clock time (seconds) of fn(*args) over arg_tuples.
    Use time.perf_counter()."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("time_per_call")


def benchmark(urdf_path, csv_path, cpp_exe=None, n_states=200):
    """Provided: time your rnea, modern_robotics.InverseDynamics, Pinocchio and (optionally) the C++ program.
    Returns a dict name -> microseconds per call."""
    import subprocess
    import tempfile

    Mlist, Glist, Slist = urdf_to_mr(urdf_path)
    X = load_states(csv_path)[:n_states]
    g, Ftip = np.array([0, 0, -9.81]), np.zeros(6)
    args = [(x[:6], x[6:12], x[12:], g, Ftip, Mlist, Glist, Slist) for x in X]
    out = {"python_own_rnea": 1e6 * time_per_call(rnea, args)}
    try:
        import modern_robotics as mr
        out["modern_robotics"] = 1e6 * time_per_call(mr.InverseDynamics, args)
    except ImportError:
        pass
    try:
        import pinocchio as pin
        model = pin.buildModelFromUrdf(str(urdf_path))
        data = model.createData()
        pargs = [(model, data, x[:6], x[6:12], x[12:]) for x in X]
        out["pinocchio_cpp_via_python"] = 1e6 * time_per_call(pin.rnea, pargs)
    except ImportError:
        pass
    if cpp_exe is not None and Path(cpp_exe).exists():
        with tempfile.TemporaryDirectory() as d:
            mp, sp, tp = Path(d) / "model.txt", Path(d) / "states.txt", Path(d) / "tau.txt"
            export_for_cpp(mp, sp, Mlist, Glist, Slist, X)
            res = subprocess.run([str(cpp_exe), str(mp), str(sp), str(tp), "2000"], capture_output=True,
                                 text=True, check=True)
            for line in res.stdout.splitlines():
                if line.startswith("us_per_call"):
                    out["cpp_eigen_own_rnea"] = float(line.split()[1])
    return out


# ==========================================================================================
# Part C -- Task-space dynamics (MR Sec. 8.6)
# ==========================================================================================


def task_space_dynamics(M, h, J, Jdot, dthetalist):
    """Lambda = J^-T M J^-1 (8.90) and eta = J^-T h - Lambda Jdot J^-1 V with J^-1 V = thetadot (8.91).
    J must be square and invertible."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("task_space_dynamics")


# ==========================================================================================
# Part D -- The MR 2R arm (Fig. 8.1) and constrained dynamics (Sec. 8.7)
# ==========================================================================================


def twor_dynamics(theta, dtheta, m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=0.0):
    """M(theta) and h(theta, thetadot) = c + g of the MR 2R arm with point masses at the link ends
    (MR Eqs. 8.9-8.10); gravity of magnitude g acts along -y."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("twor_dynamics")


def twor_jacobian(theta, dtheta, L1=1.0, L2=1.0):
    """Tip Jacobian J (for (xdot, ydot)) and its time derivative Jdot (MR Example 8.5)."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("twor_jacobian")


def channel_constraint(theta, dtheta, L1=1.0, L2=1.0):
    """Tip in a frictionless vertical channel x = const (MR Example 8.5).
    Holonomic constraint L1 c1 + L2 c12 = x_c  ->  A(theta) thetadot = 0 with A = d x / d theta (1 x 2).
    Returns (A, Adot), both of shape (1, 2)."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("channel_constraint")


def constrained_forward_dynamics(M, h, A, Adot, dtheta, tau):
    """Solve tau = M thetaddot + h + A^T lambda with Adot thetadot + A thetaddot = 0 (MR Eqs. 8.93-8.98).
    Returns (thetaddot (n,), lam (k,))."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("constrained_forward_dynamics")


def constrained_inverse_dynamics(M, h, A, ddtheta, lam):
    """tau = M thetaddot + h + A^T lambda (MR Eq. 8.93); thetaddot must satisfy (8.95)."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("constrained_inverse_dynamics")


def constraint_projector(M, A):
    """P = I - A^T (A M^-1 A^T)^-1 A M^-1 (MR Eq. 8.100), rank n - k."""
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("constraint_projector")


def simulate_channel_2r(theta0, dtheta0, tau_fun, T, dt, x_c=1.0, alpha=0.0, beta=0.0,
                        m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=0.0):
    """RK4 simulation of the 2R arm whose tip is held in the channel x = x_c.

    tau_fun(t, theta, dtheta) -> joint torques (2,).
    Constraint error e = L1 c1 + L2 c12 - x_c, edot = A thetadot. With alpha = beta = 0 the acceleration
    constraint is MR Eq. 8.95; with alpha, beta > 0 it becomes edd + 2 alpha edot + beta^2 e = 0
    (Baumgarte stabilisation, Featherstone Sec. 8.3), i.e. the lambda equation (8.98) gets the extra term
    + 2 alpha edot + beta^2 e on the right-hand side.

    Returns dict with arrays t (N+1,), theta (N+1, 2), dtheta (N+1, 2), lam (N+1,), e (N+1,).
    """
    # TODO: implement (see the docstring and lab_handout.md)
    raise NotImplementedError("simulate_channel_2r")
