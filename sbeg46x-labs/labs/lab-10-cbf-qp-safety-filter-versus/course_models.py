"""Robot models used in Lab 10 -- copied from the shared course robot models (Course resources) (numbers identical).

  arm_exo_2r()   Arm + exoskeleton 2R in the sagittal plane (shoulder flexion, elbow flexion), SBEG 461 Weeks 1-2.
                 Lab 10 (as Lab 8) uses its ELBOW joint with the upper arm hanging (q1 = -pi/2): the elbow-flexion
                 exoskeleton joint of a seated arm-rehabilitation session.
  load_exo_2r()  Load-carriage exoskeleton 2R (knee, hip), SBEG 461 project scenario 4 (defence, protective/logistic).
                 Lab 10 (optional Experiment D) uses its KNEE joint with the hip locked at q2 = 0 (squat assist).
  hapkit()       Hapkit-type 1-DOF haptic paddle.  Lab 10 Experiment C uses it in handle coordinates x = r_h theta
                 (hapkit_linear(): handle mass m_h, gravity stiffness k_g), as in the course project devices.

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


# ---------------------------------------------------------------- Hapkit-type haptic device
def hapkit() -> dict:
    """from the shared course robot models (Course resources): hapkit.
    Hapkit-type 1-DOF haptic paddle; handle pivots about z_s, {b} at the handle grip r_h = 0.090 m above the
    pivot; sector radius r_s = 0.073 m, motor pulley r_p = 0.0042 m (ratio ~17.4); sector 0.10 kg with its CoM
    0.010 m below the pivot; rotor inertia 1.0e-6 kg m^2.  Derated limits: handle force <= 2 N, range +-40 mm."""
    rh, rs, rp = 0.090, 0.073, 0.0042
    S = screw_revolute([0, 0, 1], [0, 0, 0]).reshape(6, 1)
    M = trans([0, rh, 0])
    com = [trans([0, -0.010, 0])]
    G = [body_inertia(0.10, [2.0e-4, 1.0e-4, 1.5e-4])]
    th_max = 0.040 / rh
    return dict(key="hapkit", Slist=S, M=M, Mlist=mlist_from_absolute(com, M), Glist=G,
                g=np.array([0, -G0, 0.0]), limits=np.array([[-th_max, th_max]]), tau_max=np.array([2.0 * rh]),
                params=dict(rh=rh, rs=rs, rp=rp, ratio=rs / rp, rotor_inertia=1.0e-6,
                            reflected_rotor_inertia=1.0e-6 * (rs / rp) ** 2, force_max=2.0, x_range=0.040,
                            loop_rate_hz=1000.0))


def hapkit_linear() -> dict:
    """Handle-coordinate model of the Hapkit, x = r_h theta (small angles):  m_h x'' + b_h x' + k_g x = F + F_ext.
    m_h = (I_pivot + reflected rotor inertia) / r_h^2 from modern_robotics MassMatrix; k_g = (gravity torque
    stiffness at theta = 0) / r_h^2 from modern_robotics GravityForces (the sector hangs below the pivot, so
    gravity centres the handle).  b_h is not in the shared model: 0.15 N s/m is the SBEG 461 Lab 8 reference value
    (re-identify it on your device)."""
    r = hapkit()
    p = r["params"]
    rh = p["rh"]
    I_pivot = float(mr.MassMatrix([0.0], r["Mlist"], r["Glist"], r["Slist"])[0, 0])
    eps = 1e-6
    k_theta = float(mr.GravityForces([eps], r["g"], r["Mlist"], r["Glist"], r["Slist"])[0]) / eps
    return dict(m_h=(I_pivot + p["reflected_rotor_inertia"]) / rh ** 2, k_g=k_theta / rh ** 2, b_h=0.15,
                force_max=p["force_max"], x_range=p["x_range"], T=1.0 / p["loop_rate_hz"], rh=rh)
