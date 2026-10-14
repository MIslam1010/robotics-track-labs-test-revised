"""Robot models used in Lab 8 -- copied from the shared course robot models (Course resources) (numbers identical).

  hapkit()    from the shared course robot models (Course resources): hapkit      (Hapkit-type 1-DOF haptic paddle)
  rehab_2r()  from the shared course robot models (Course resources): rehab_2r    (planar end-effector rehabilitation robot,
                                                                 461 project scenario S1)

The lab must be self-contained, so only the few helpers these two models need are copied as well.
ILLUSTRATIVE VALUES: geometry and inertias are teaching values, not data of a real product.
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


def rod_inertia_x(m: float, L: float, r: float = 0.03) -> np.ndarray:
    """Uniform solid cylinder (radius r) of length L along the CoM-frame x axis."""
    Ia = 0.5 * m * r**2
    It = m * (3 * r**2 + L**2) / 12.0
    return body_inertia(m, [Ia, It, It])


def mlist_from_absolute(com_frames, M_ee) -> list:
    frames = [np.eye(4)] + list(com_frames) + [M_ee]
    return [mr.TransInv(frames[i]) @ frames[i + 1] for i in range(len(frames) - 1)]


# ---------------------------------------------------------------- models
def hapkit() -> dict:
    """from the shared course robot models (Course resources): hapkit.
    Hapkit-type 1-DOF haptic paddle.  Handle pivots about z_s; {b} at the handle grip, rh = 0.090 m above
    the pivot; sector radius rs = 0.073 m, motor pulley rp = 0.0042 m (ratio rs/rp ~ 17.4).  Handle
    displacement x ~ rh*theta.  Derated limits: handle force <= 2 N, handle range +-40 mm, 1 kHz loop."""
    rh, rs, rp = 0.090, 0.073, 0.0042
    S = screw_revolute([0, 0, 1], [0, 0, 0]).reshape(6, 1)
    M = trans([0, rh, 0])
    com = [trans([0, -0.010, 0])]                    # sector mass slightly below the pivot
    G = [body_inertia(0.10, [2.0e-4, 1.0e-4, 1.5e-4])]
    th_max = 0.040 / rh
    return dict(key="hapkit", Slist=S, M=M, Mlist=mlist_from_absolute(com, M), Glist=G,
                g=np.array([0, -G0, 0.0]), limits=np.array([[-th_max, th_max]]), qdot_max=np.array([20.0]),
                tau_max=np.array([2.0 * rh]),
                params=dict(rh=rh, rs=rs, rp=rp, ratio=rs / rp, rotor_inertia=1.0e-6,
                            reflected_rotor_inertia=1.0e-6 * (rs / rp) ** 2,
                            force_max=2.0, x_range=0.040, loop_rate_hz=1000.0))


def rehab_2r() -> dict:
    """from the shared course robot models (Course resources): rehab_2r (461 project S1).
    Planar end-effector rehabilitation robot in a horizontal plane (gravity along -z, so g(theta) = 0)."""
    L1, L2, m1, m2 = 0.35, 0.35, 2.4, 1.6
    S = np.c_[screw_revolute([0, 0, 1], [0, 0, 0]), screw_revolute([0, 0, 1], [L1, 0, 0])]
    M = trans([L1 + L2, 0, 0])
    com = [trans([L1 / 2, 0, 0]), trans([L1 + L2 / 2, 0, 0])]
    return dict(key="rehab_2r", Slist=S, M=M, Mlist=mlist_from_absolute(com, M),
                Glist=[rod_inertia_x(m1, L1), rod_inertia_x(m2, L2)], g=np.array([0, 0, -G0]),
                limits=np.array([[0.0, np.pi], [0.2, 2.8]]), qdot_max=np.array([3.0, 3.0]),
                tau_max=np.array([20.0, 12.0]), q_home=np.array([0.9, 1.6]),
                params=dict(L=(L1, L2), m=(m1, m2), path_A=(0.0, 0.35), path_B=(0.0, 0.55)))
