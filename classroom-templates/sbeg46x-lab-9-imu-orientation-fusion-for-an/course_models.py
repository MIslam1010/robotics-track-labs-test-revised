"""Robot model used in Lab 9 -- copied from the shared course robot models (Course resources) (numbers identical).

  gait_exo_2r()  Hip-knee gait exoskeleton 2R (SBEG 461 project scenario S2): swing leg on a test stand, leg along
                 -y_s at q = 0 (hanging), hip flexion about +z_s, knee flexion about -z_s, gravity along -y_s.
                 Lab 9 uses its THIGH segment (joint 1, the hip): the thigh cuff carries a 6-axis IMU.

The lab must be self-contained, so the few helpers the model needs are copied as well, plus `imu_pose`, which places
the thigh IMU frame {b} with the product of exponentials (MR Eq. 4.14) and `imu_specific_force`, which gives the ideal
accelerometer reading of that IMU from modern_robotics.
ILLUSTRATIVE VALUES: geometry and inertias are teaching values, not data of a real product or person.
Notation: Modern Robotics (MR) -- screw axes S = (w, v), T_sb = e^{[S1] theta} M_imu.
"""
from __future__ import annotations

import modern_robotics as mr
import numpy as np

G0 = 9.81

# IMU on the thigh cuff (illustrative): r_imu below the hip axis, sensor frame {b} parallel to {s} when the leg
#   hangs (q1 = 0): x_b forward (anterior, perpendicular to the thigh), y_b up the thigh toward the hip,
#   z_b = hip axis (+z_s).  So R_sb = Rot(z, theta) and at rest f_b = (g sin(theta), g cos(theta), 0)
#   (same frame as the Week 10 lecture notes).
R_IMU = 0.20
R_MOUNT = np.eye(3)


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


def gait_exo_2r() -> dict:
    """SBEG 461 project S2: hip-knee exoskeleton + leg (swing leg on a test stand), leg along -y
    at q = 0; hip flexion about +z, knee flexion about -z."""
    L1, L2 = 0.42, 0.43
    m1, c1, I1 = 9.0, 0.18, 0.16
    m2, c2, I2 = 5.8, 0.24, 0.10
    S = np.c_[screw_revolute([0, 0, 1], [0, 0, 0]), screw_revolute([0, 0, -1], [0, -L1, 0])]
    M = trans([0, -(L1 + L2), 0])
    com = [trans([0, -c1, 0]), trans([0, -(L1 + c2), 0])]
    G = [body_inertia(m1, [I1, 0.03, I1]), body_inertia(m2, [I2, 0.01, I2])]
    return dict(key="gait_exo_2r", name="Hip-knee gait exoskeleton 2R", Slist=S, M=M,
                Mlist=mlist_from_absolute(com, M), Glist=G, g=np.array([0.0, -G0, 0.0]),
                limits=[[-0.5, 1.2], [0.0, 1.6]], qdot_max=[8.0, 10.0], tau_max=[60.0, 45.0],
                params=dict(L=(L1, L2), gear_ratio=80.0, rotor_inertia=1.5e-5))


def imu_home(r_imu: float = R_IMU) -> np.ndarray:
    """M_imu: pose of the thigh IMU frame {b} in {s} at q = 0 (on the thigh, r_imu below the hip)."""
    return trans([0.0, -r_imu, 0.0], R_MOUNT)


def imu_pose(theta_hip: float, r_imu: float = R_IMU) -> np.ndarray:
    """T_sb(theta) = e^{[S1] theta} M_imu (MR Eq. 4.14 with the hip joint only)."""
    S1 = gait_exo_2r()["Slist"][:, :1]
    return mr.FKinSpace(imu_home(r_imu), S1, [theta_hip])


def imu_specific_force(theta, dtheta, ddtheta, r_imu: float = R_IMU) -> np.ndarray:
    """Ideal accelerometer reading f_b = R_sb^T (a_s - g_s) of the thigh IMU (no noise).  T_sb comes from
    modern_robotics (FKinSpace); for a pure rotation about the fixed hip axis through the {s} origin the IMU
    acceleration is a_s = alpha x p + w x (w x p) with w = w_1 dtheta, alpha = w_1 ddtheta."""
    T = imu_pose(theta, r_imu)
    R, p = mr.TransToRp(T)
    S1 = gait_exo_2r()["Slist"][:, 0]
    w = S1[:3] * dtheta
    alpha = S1[:3] * ddtheta
    a_s = np.cross(alpha, p) + np.cross(w, np.cross(w, p))      # hip axis passes through the {s} origin
    g_s = gait_exo_2r()["g"]
    return R.T @ (a_s - g_s)
