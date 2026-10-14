"""Robot models used in Lab 8 -- copied from the shared course robot models (Course resources) (numbers identical).

  arm_exo_2r()   Arm + exoskeleton 2R in the sagittal plane (shoulder flexion, elbow flexion), SBEG 461 Weeks 1-2.
                 Lab 8 uses its ELBOW joint with the shoulder held with the upper arm hanging (q1 = -pi/2):
                 the elbow-flexion exoskeleton joint of a seated arm-rehabilitation session.
  load_exo_2r()  Load-carriage exoskeleton 2R (knee, hip), SBEG 461 project scenario 4 (defence, protective/logistic).
                 Lab 8 (optional Experiment E) uses its KNEE joint with the hip locked at q2 = 0 (squat assist).

The lab must be self-contained, so only the few helpers these models need are copied as well, plus `joint_model`,
which turns one joint of a model into the 1-DOF parameters of MR Eq. 11.21 (inertia about the joint from
modern_robotics MassMatrix, gravity torque from modern_robotics GravityForces).
ILLUSTRATIVE VALUES: geometry and inertias are teaching values, not data of a real product or person.
Notation: Modern Robotics (MR) -- screw axes S = (w, v), Mlist / Glist as in MR Sec. 8.3.2 (Eq. 8.32).
"""
from __future__ import annotations

import modern_robotics as mr
import numpy as np

G0 = 9.81


# ---------------------------------------------------------------- helpers (from the shared course robot models)
def screw_revolute(w, q, h: float = 0.0) -> np.ndarray:
    """Screw axis S = (w, -w x q + h w) of a joint about unit axis w through point q (MR Sec. 3.3.2)."""
    w = np.asarray(w, float)
    q = np.asarray(q, float)
    return np.r_[w, -np.cross(w, q) + h * w]


def trans(p=(0.0, 0.0, 0.0), R=None) -> np.ndarray:
    return mr.RpToTrans(np.eye(3) if R is None else np.asarray(R, float), np.asarray(p, float))


def body_inertia(m: float, Ib) -> np.ndarray:
    """G = diag(I_b, m I3) in a CoM frame (MR Eq. 8.32)."""
    G = np.zeros((6, 6))
    G[:3, :3] = np.diag(np.asarray(Ib, float))
    G[3:, 3:] = m * np.eye(3)
    return G


def mlist_from_absolute(com_frames, M_ee) -> list:
    frames = [np.eye(4)] + list(com_frames) + [M_ee]
    return [mr.TransInv(frames[i]) @ frames[i + 1] for i in range(len(frames) - 1)]


def planar_chain(L, m, r, Izz, Ixx_factor=0.2, g=(0, -G0, 0)) -> dict:
    """Planar n-R chain in the vertical x-y plane (gravity -y), joints along z, zero pose along +x;
    CoM r_i from joint i, Izz about the CoM; Glist = diag(Ixx_factor*Izz, Izz, Izz, m, m, m)."""
    n = len(L)
    jx = np.r_[0.0, np.cumsum(L)]
    coms = [trans([jx[i] + r[i], 0, 0]) for i in range(n)]
    M = trans([jx[n], 0, 0])
    S = np.array([[0, 0, 1, 0, -jx[i], 0] for i in range(n)], float).T
    G = [np.diag([Ixx_factor * Izz[i], Izz[i], Izz[i], m[i], m[i], m[i]]) for i in range(n)]
    return dict(Slist=S, M=M, Mlist=mlist_from_absolute(coms, M), Glist=G, g=np.asarray(g, float),
                params=dict(L=tuple(L), m=tuple(m), r=tuple(r), Izz=tuple(Izz)))


# ---------------------------------------------------------------- models
def arm_exo_2r() -> dict:
    """from the shared course robot models (Course resources): arm_exo_2r.
    Upper arm + cuff (L 0.30 m, 2.6 kg, CoM 0.13 m, Izz 0.025), forearm + hand + cuff (L 0.35 m, 2.0 kg, CoM 0.17 m,
    Izz 0.020); vertical plane, gravity -y.  Joint limits [-0.5, 2.6], [0, 2.4] rad; rated torques 30, 15 N m."""
    r = planar_chain([0.30, 0.35], [2.6, 2.0], [0.13, 0.17], [0.025, 0.020])
    r.update(key="arm_exo_2r", limits=np.array([[-0.5, 2.6], [0.0, 2.4]]), tau_max=np.array([30.0, 15.0]))
    return r


def load_exo_2r() -> dict:
    """from the shared course robot models (Course resources): load_exo_2r.
    Soldier load-carriage exoskeleton, stance leg (knee, hip) with a 30 kg pack on a 4 kg hip frame; shank strut
    clamped (bench model).  Joint limits [0, 1.6], [-0.3, 1.8] rad; rated torques 200, 150 N m."""
    Ls, Lt = 0.45, 0.45
    m1, I1 = 2.5, 0.045
    m2, I2 = 34.0, 1.20
    pack_com = np.array([-0.15, 0.25, 0.0])
    knee, hip = np.array([0, Ls, 0.0]), np.array([0, Ls + Lt, 0.0])
    S = np.c_[screw_revolute([0, 0, 1], knee), screw_revolute([0, 0, -1], hip)]
    M = trans(hip + np.array([0, 0.50, 0]))
    com = [trans(knee + np.array([0, Lt / 2, 0])), trans(hip + pack_com)]
    G = [body_inertia(m1, [I1, 0.005, I1]), body_inertia(m2, [I2, 0.6, I2])]
    return dict(key="load_exo_2r", Slist=S, M=M, Mlist=mlist_from_absolute(com, M), Glist=G,
                g=np.array([0, -G0, 0.0]), limits=np.array([[0.0, 1.6], [-0.3, 1.8]]),
                tau_max=np.array([200.0, 150.0]),
                params=dict(L=(Ls, Lt), pack_mass_nominal=30.0, frame_mass=4.0, pack_com=pack_com))


# ---------------------------------------------------------------- one joint of a model (MR Eq. 11.21)
JOINTS = {
    # name: (model function, index of the controlled joint, configuration of the locked joint(s))
    "elbow": (arm_exo_2r, 1, np.array([-np.pi / 2, 0.0])),   # upper arm hanging, elbow controlled
    "knee": (load_exo_2r, 0, np.array([0.0, 0.0])),          # hip locked at 0, knee controlled
}


def _q(name: str, theta: float) -> np.ndarray:
    _, j, q0 = JOINTS[name]
    q = q0.copy()
    q[j] = theta
    return q


def joint_inertia(name: str, theta: float = 0.5) -> float:
    """Inertia about the controlled joint, M(theta) = MassMatrix[j, j] (modern_robotics), kg m^2.
    Constant for both joints used here (it depends only on the locked joint)."""
    f, j, _ = JOINTS[name]
    r = f()
    return float(mr.MassMatrix(_q(name, theta), r["Mlist"], r["Glist"], r["Slist"])[j, j])


def joint_gravity(name: str, theta: float) -> float:
    """Gravity torque g(theta) at the controlled joint (modern_robotics GravityForces), N m.
    For the elbow it equals m g r sin(theta) = 3.3354 sin(theta) (MR Eq. 11.21 with the angle from the vertical)."""
    f, j, _ = JOINTS[name]
    r = f()
    return float(mr.GravityForces(_q(name, theta), r["g"], r["Mlist"], r["Glist"], r["Slist"])[j])


def joint_model(name: str) -> dict:
    """1-DOF parameters of a course-model joint: inertia M, rated torque and the mechanical joint limits."""
    f, j, _ = JOINTS[name]
    r = f()
    return dict(name=name, M=joint_inertia(name), tau_rated=float(r["tau_max"][j]),
                limits=tuple(float(v) for v in r["limits"][j]))


def gravity_coefficients(name: str) -> tuple:
    """(g_s, g_c) with g(theta) = g_s sin(theta) + g_c cos(theta): exact for one rigid body turning about a
    horizontal axis (the locked joints make the moving part one rigid body).  Taken from GravityForces at
    theta = pi/2 and 0; the lab simulator uses this closed form because it is called ~10^4 times per run."""
    return joint_gravity(name, np.pi / 2), joint_gravity(name, 0.0)
