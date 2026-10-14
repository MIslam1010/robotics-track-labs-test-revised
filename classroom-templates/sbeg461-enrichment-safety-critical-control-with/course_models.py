"""Robot model used in the safety-filter enrichment lab -- copied from the shared course robot models
(Course resources: "Course robot models and hardware safety checklist"), numbers identical.

  rehab_2r()  planar end-effector rehabilitation robot in a horizontal plane (461 project scenario S1),
              the same robot as in Lab 8 (Week 9) and the Week 13 tele-rehabilitation case study.

The lab must be self-contained, so the few helpers this model needs are copied as well.
ILLUSTRATIVE VALUES: geometry and inertias are teaching values, not data of a real product.
Notation: Modern Robotics (MR) -- screw axes S = (w, v), Mlist / Glist as in MR Sec. 8.3.2 (Eq. 8.32).
"""
from __future__ import annotations

import modern_robotics as mr
import numpy as np

G0 = 9.81


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


def rod_inertia_x(m: float, L: float, r: float = 0.03) -> np.ndarray:
    """Uniform solid cylinder (radius r) of length L along the CoM-frame x axis."""
    Ia = 0.5 * m * r**2
    It = m * (3 * r**2 + L**2) / 12.0
    return body_inertia(m, [Ia, It, It])


def mlist_from_absolute(com_frames, M_ee) -> list:
    frames = [np.eye(4)] + list(com_frames) + [M_ee]
    return [mr.TransInv(frames[i]) @ frames[i + 1] for i in range(len(frames) - 1)]


def rehab_2r() -> dict:
    """Shared course model rehab_2r (461 project S1).
    Planar end-effector rehabilitation robot in a horizontal plane (gravity along -z, so g(theta) = 0).
    L1 = L2 = 0.35 m, m1 = 2.4 kg, m2 = 1.6 kg (uniform rods, radius 0.03 m); joint limits
    theta1 in [0, pi], theta2 in [0.2, 2.8] rad; rated torques (20, 12) N m; rated speeds 3 rad/s."""
    L1, L2, m1, m2 = 0.35, 0.35, 2.4, 1.6
    S = np.c_[screw_revolute([0, 0, 1], [0, 0, 0]), screw_revolute([0, 0, 1], [L1, 0, 0])]
    M = trans([L1 + L2, 0, 0])
    com = [trans([L1 / 2, 0, 0]), trans([L1 + L2 / 2, 0, 0])]
    return dict(key="rehab_2r", Slist=S, M=M, Mlist=mlist_from_absolute(com, M),
                Glist=[rod_inertia_x(m1, L1), rod_inertia_x(m2, L2)], g=np.array([0, 0, -G0]),
                limits=np.array([[0.0, np.pi], [0.2, 2.8]]), qdot_max=np.array([3.0, 3.0]),
                tau_max=np.array([20.0, 12.0]), q_home=np.array([0.9, 1.6]),
                params=dict(L=(L1, L2), m=(m1, m2)))
