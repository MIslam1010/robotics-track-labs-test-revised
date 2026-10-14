"""SBEG 46X Lab 1 -- Software remote centre of motion (RCM): constrained resolved-rate control of the course arm
holding a laparoscope.  STARTER CODE.

Fill in every function marked # TODO (13 functions; keep the names and signatures), then run
    LAB_IMPL=starter python -m pytest -q test_lab.py
in your Lab code folder.  The helpers marked "(Provided.)" are complete.  When all tests pass, copy the functions
into module `rcm.py` of your course library `sbeg46x_medrob` (later labs and the project reuse them).

Notation follows Lynch & Park, *Modern Robotics* (MR):
  space-form PoE  T_sb(theta) = e^[S1]th1 ... e^[S6]th6 M                       (MR Eq. 4.14)
  space Jacobian  V_s = J_s(theta) thetadot,  V_s = (w_s, v_s)                  (MR Eq. 5.11)
  velocity of a point p fixed to the moving body: pdot = v_s + w_s x p = [-[p]  I] V_s   (from MR Eq. 3.73, p. 97)
  inverse velocity kinematics thetadot = J^+ V_d                                 (MR Eq. 6.7, p. 231)
  damped least squares  A^# = A^T (A A^T + lam^2 I)^-1
Twists V = (omega, v): angular part first.

Software RCM (this lab).  The tool frame {b} is at the laparoscope tip, z_b along the shaft towards the tip.  The
trocar port p_port is fixed in {s}.  The shaft point closest to the port is p_c = p_tip - s z, s = z.(p_tip - p_port)
(the insertion depth beyond the port).  The RCM constraint asks that the material point of the shaft at p_c moves
only ALONG the shaft:  N^T (v_s + w_s x p_c) = k_rcm N^T (p_port - p_c),  N = [n1 n2] perpendicular to z.
Together with the tip task  v_tip = pdot_d + k_p (p_d - p_tip)  this gives 5 equations for 6 joint rates;
the remaining freedom (roll about the shaft, mixed with the arm's self-motion) is resolved by the minimum-norm
(damped) pseudo-inverse.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

from course_models import course_arm_kinematics

# ---------------------------------------------------------------- kinematics backend (provided)
# FK and space Jacobians come from YOUR SBEG 361 library `sbeg361_kinematics` (fk.py from SBEG 361 Lab 5,
# jacobians.py from SBEG 361 Lab 7).  modern_robotics is used ONLY as a fallback.
try:
    from sbeg361_kinematics.fk import fk_space                # T = e^[S1]th1 ... e^[Sn]thn M   (MR Eq. 4.14)
    from sbeg361_kinematics.jacobians import jacobian_space   # J_s(theta), 6 x n               (MR Eq. 5.11)
    KIN_BACKEND = "sbeg361_kinematics"
except ImportError:
    import modern_robotics as _mr

    def fk_space(M, Slist, theta):
        """Fallback for sbeg361_kinematics.fk.fk_space: mr.FKinSpace (MR Eq. 4.14)."""
        return _mr.FKinSpace(np.asarray(M, float), np.asarray(Slist, float), np.asarray(theta, float))

    def jacobian_space(Slist, theta):
        """Fallback for sbeg361_kinematics.jacobians.jacobian_space: mr.JacobianSpace (MR Eq. 5.11)."""
        return _mr.JacobianSpace(np.asarray(Slist, float), np.asarray(theta, float))

    KIN_BACKEND = "modern_robotics (fallback)"

try:                                                          # SBEG 461 library: only checked, not needed here
    import sbeg461_dynamics  # noqa: F401
    DYN_BACKEND = "sbeg461_dynamics"
except ImportError:
    DYN_BACKEND = "not found (not needed in Lab 1; keep importable for the project)"

DATA = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------- provided helpers
def environment_report() -> dict:
    """Versions and backends of your lab environment (Checkpoint 0).  (Provided.)"""
    import scipy
    import modern_robotics
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                modern_robotics=getattr(modern_robotics, "__version__", "installed"),
                kinematics=KIN_BACKEND, dynamics=DYN_BACKEND)


def load_task(path=None) -> dict:
    """Load the lab task (port, start configuration, tip path) from data/lab1_task.json.  (Provided.)"""
    with open(path or DATA / "lab1_task.json") as fh:
        task = json.load(fh)
    task["p_port"] = np.asarray(task["p_port"], float)
    task["theta0"] = np.asarray(task["theta0"], float)
    return task


def skew(p) -> np.ndarray:
    """[p], the 3 x 3 skew-symmetric matrix with [p] x = p x x (MR Eq. 3.30).  (Provided.)"""
    x, y, z = np.asarray(p, float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def min_jerk(tau):
    """Minimum-jerk time scaling s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 and ds/dtau (MR Sec. 9.2.2, quintic, p. 332).
    (Provided.)"""
    tau = float(np.clip(tau, 0.0, 1.0))
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    ds = 30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4
    return s, ds


def port_frame(z_axis) -> np.ndarray:
    """Rotation R_sp of a 'port frame' {p}: z_p = initial shaft direction, x_p horizontal.  (Provided.)"""
    z = np.asarray(z_axis, float) / np.linalg.norm(z_axis)
    x = np.cross([0.0, 0.0, 1.0], z)
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def make_trajectory(task, p_tip0, z0):
    """Desired tip path from the task file.  (Provided.)

    The path is written in the port frame {p} (origin at the port, z_p along the initial shaft direction z0, in m):
      {"type": "line",   "to": [x, y, z], "T": duration}          straight line, minimum-jerk timing
      {"type": "circle", "radius": r, "turns": k, "T": duration}  circle about the current point's z_p-axis
                                                                    projection (centre = start - r x_p)
    Returns (traj, T_total) with traj(t) -> (p_d, pdot_d) in {s}.
    """
    R = port_frame(z0)
    o = np.asarray(task["p_port"], float)
    segs, t0 = [], 0.0
    cur = R.T @ (np.asarray(p_tip0, float) - o)
    for seg in task["path"]:
        T = float(seg["T"])
        if seg["type"] == "line":
            a, b = cur.copy(), np.asarray(seg["to"], float)
            segs.append(("line", t0, T, a, b))
            cur = b
        elif seg["type"] == "circle":
            r, k = float(seg["radius"]), float(seg["turns"])
            centre = cur - np.array([r, 0.0, 0.0])
            segs.append(("circle", t0, T, centre, (r, k)))
        else:
            raise ValueError(seg["type"])
        t0 += T

    def traj(t):
        for kind, ts, T, a, b in segs:
            if t <= ts + T or (kind, ts, T) == segs[-1][:3]:
                s, ds = min_jerk((t - ts) / T)
                if kind == "line":
                    pp, vp = a + s * (b - a), ds / T * (b - a)
                else:
                    r, k = b
                    ang, dang = 2 * np.pi * k * s, 2 * np.pi * k * ds / T
                    pp = a + r * np.array([np.cos(ang), np.sin(ang), 0.0])
                    vp = r * dang * np.array([-np.sin(ang), np.cos(ang), 0.0])
                return o + R @ pp, R @ vp
        raise RuntimeError("unreachable")

    return traj, t0


# ---------------------------------------------------------------- Part A: geometry of the tool and the port
def tool_pose(robot, theta) -> np.ndarray:
    """T_sb(theta) of the laparoscope tip frame (MR Eq. 4.14), using the kinematics backend above."""
    # TODO: call fk_space(M, Slist, theta) from the kinematics backend; return a 4 x 4 float array.
    raise NotImplementedError("TODO: tool_pose")


def shaft(T):
    """Tip position p_tip and unit shaft direction z (towards the tip) from T_sb (both 3-vectors)."""
    # TODO: p_tip is the position column of T, z is the third column of R (z_b).
    raise NotImplementedError("TODO: shaft")


def rcm_geometry(T, p_port) -> dict:
    """Port geometry of the shaft for the tool pose T.

    Returns a dict with
      depth : s = z.(p_tip - p_port), insertion depth of the tip beyond the port along the shaft (m)
      p_c   : p_tip - s z, the shaft point closest to the port
      e     : p_port - p_c, the RCM error vector (perpendicular to z)
      dist  : |e|, the distance from the port to the shaft line (m)
    """
    # TODO: s = z . (p_tip - p_port);  p_c = p_tip - s z;  e = p_port - p_c;  dist = |e|.
    raise NotImplementedError("TODO: rcm_geometry")


def perp_basis(z) -> np.ndarray:
    """3 x 2 matrix N = [n1 n2] of orthonormal vectors perpendicular to the unit vector z (right-handed n1, n2, z)."""
    # TODO: pick a helper vector a not parallel to z, n1 = a x z / |a x z|, n2 = z x n1.
    raise NotImplementedError("TODO: perp_basis")


# ---------------------------------------------------------------- Part B: Jacobians
def point_jacobian(Js, p) -> np.ndarray:
    """3 x n Jacobian of the point p of the moving body (p in {s} coordinates):  pdot = [-[p]  I] J_s thetadot."""
    # TODO: pdot = v_s + w_s x p = -[p] w_s + v_s, so J_p = [-[p]  I] J_s (use skew()).
    raise NotImplementedError("TODO: point_jacobian")


def rcm_jacobian(robot, theta, p_port):
    """Stacked 5 x 6 task Jacobian of the software RCM.

    Rows 1-3: tip velocity  J_tip = [-[p_tip] I] J_s.
    Rows 4-5: lateral velocity of the shaft point at the port  N^T J_c,  J_c = [-[p_c] I] J_s,  N = perp_basis(z).
    Returns (A, info) with info = dict(T=T_sb, N=N, geometry=rcm_geometry(T, p_port)).
    """
    # TODO: J_s from jacobian_space; stack point_jacobian at p_tip (3 rows) and N^T point_jacobian at p_c (2 rows).
    raise NotImplementedError("TODO: rcm_jacobian")


def dls_inverse(A, lam: float = 0.0) -> np.ndarray:
    """Damped least-squares inverse A^T (A A^T + lam^2 I)^-1; lam = 0 gives the pseudo-inverse A^+ (MR Eq. 6.7)."""
    # TODO: lam == 0: np.linalg.pinv(A); otherwise A^T (A A^T + lam^2 I)^-1.
    raise NotImplementedError("TODO: dls_inverse")


# ---------------------------------------------------------------- Part C: controllers
def rcm_rates(robot, theta, p_port, p_d, pdot_d, kp=5.0, k_rcm=10.0, lam=0.0) -> np.ndarray:
    """Software-RCM resolved-rate law (6 joint rates):
        thetadot = A^#(lam) [ pdot_d + kp (p_d - p_tip) ;  k_rcm N^T e ],   e = p_port - p_c.
    """
    # TODO: b = [pdot_d + kp (p_d - p_tip),  k_rcm N^T e]  (5-vector);  thetadot = dls_inverse(A, lam) @ b.
    raise NotImplementedError("TODO: rcm_rates")


def tip_only_rates(robot, theta, p_d, pdot_d, kp=5.0, lam=0.0) -> np.ndarray:
    """Naive resolved rate WITHOUT the RCM rows: thetadot = J_tip^#(lam) [pdot_d + kp (p_d - p_tip)]."""
    # TODO: same as rcm_rates but with only the 3 tip rows (no RCM rows).
    raise NotImplementedError("TODO: tip_only_rates")


def scale_rates(qdot, qdot_max) -> np.ndarray:
    """Uniformly scale qdot so that |qdot_i| <= qdot_max_i for every joint (keeps the direction of motion)."""
    # TODO: ratio = max_i |qdot_i| / qdot_max_i; divide qdot by ratio only if ratio > 1.
    raise NotImplementedError("TODO: scale_rates")


def simulate(robot, task, controller="rcm", dt=0.005, kp=5.0, k_rcm=10.0, lam=0.0) -> dict:
    """Simulate the tip path of `task` with forward-Euler integration theta <- theta + dt * thetadot.

    controller = "rcm" (rcm_rates) or "tip" (tip_only_rates).  Joint rates are limited with scale_rates
    (robot["qdot_max"]).  Returns a dict of arrays (one row per step, k = 0..K):
      t, theta (K+1 x 6), p_tip, p_d (K+1 x 3), tip_err, rcm_err, depth (K+1,)
    """
    # TODO: loop k = 0..K: read traj(t), log tip error, RCM distance and depth, then theta += dt * scale_rates(qdot).
    raise NotImplementedError("TODO: simulate")


# ---------------------------------------------------------------- Part D: fulcrum effect and motion scaling
def fulcrum_ratio(L, depth) -> float:
    """Hand-held instrument pivoting at the port: lateral tip motion / lateral handle motion = -depth / (L - depth)
    (minus sign: the tip moves opposite to the hand).  L = shaft length from handle to tip."""
    # TODO: similar triangles about the port: tip moves depth/(L - depth) times the hand, in the opposite direction.
    raise NotImplementedError("TODO: fulcrum_ratio")


def scaled_tip_command(d_leader, scale, R_sv=None) -> np.ndarray:
    """Motion scaling: tip displacement in {s} = scale * R_sv d_leader, where d_leader is the leader (hand)
    displacement in the view (camera) frame {v} and R_sv the camera orientation in {s} (identity if None)."""
    # TODO: return scale * R_sv @ d_leader (R_sv = identity when None).
    raise NotImplementedError("TODO: scaled_tip_command")


if __name__ == "__main__":
    for k, v in environment_report().items():           # Checkpoint 0: works before any TODO is done
        print(f"{k:16s} {v}")
    try:
        robot = course_arm_kinematics()
        task = load_task()
        for ctrl in ("rcm", "tip"):
            log = simulate(robot, task, controller=ctrl)
            print(f"{ctrl:4s}: max tip error {1e3 * log['tip_err'].max():.3f} mm, "
                  f"max RCM error {1e3 * log['rcm_err'].max():.3f} mm")
    except NotImplementedError as exc:
        print("Not done yet:", exc)
