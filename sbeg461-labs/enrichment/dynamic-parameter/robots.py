"""Robot models for the parameter-identification enrichment lab: the course arm and the EOD arm 3R.

Copied from the shared course robot models (Course resources: "Course robot models and hardware safety
checklist") so that this folder is self-contained. The numbers are IDENTICAL to the shared models; the
inertial data are ILLUSTRATIVE VALUES (the nominal "CAD" model that identification is meant to correct).

Every model is a dict in Modern Robotics list form (MR Sec. 8.3.2):
    n, Slist (6 x n), M, Mlist = [M_01, ..., M_n,n+1] (link frames {i} at the link CoMs at home),
    Glist = [G_1, ..., G_n] (spatial inertias in {i}, MR Eq. 8.32), g (gravity in {s}),
    limits (n x 2, rad), qdot_max (rad/s), tau_max (N m), q_home.
Units: m, kg, s, rad, N m.
"""
from __future__ import annotations

import modern_robotics as mr
import numpy as np

G0 = 9.81
DEG = np.pi / 180.0


def trans(p=(0.0, 0.0, 0.0), R=None) -> np.ndarray:
    """Homogeneous transform with rotation R (identity by default) and position p."""
    return mr.RpToTrans(np.eye(3) if R is None else np.asarray(R, float), np.asarray(p, float))


def body_inertia(m: float, Ib) -> np.ndarray:
    """G = diag(I_b, m I3) in a CoM frame (MR Eq. 8.32). Ib: 3-vector (principal) or 3x3."""
    Ib = np.diag(np.asarray(Ib, float)) if np.ndim(Ib) == 1 else np.asarray(Ib, float)
    G = np.zeros((6, 6))
    G[:3, :3] = Ib
    G[3:, 3:] = m * np.eye(3)
    return G


def box_inertia(m: float, a: float, b: float, c: float) -> np.ndarray:
    """Solid box with edges a, b, c along the CoM-frame x, y, z axes."""
    return body_inertia(m, [m * (b**2 + c**2) / 12, m * (a**2 + c**2) / 12, m * (a**2 + b**2) / 12])


def mlist_from_absolute(com_frames, M_ee) -> list:
    """MR's Mlist from the absolute home CoM frames M_{s,i} and the end-effector home frame."""
    frames = [np.eye(4)] + list(com_frames) + [M_ee]
    return [mr.TransInv(frames[i]) @ frames[i + 1] for i in range(len(frames) - 1)]


# ------------------------------------------------------------------------------------------- course arm
COURSE_ARM_GEOMETRY = dict(W1=0.0634, W2=0.0456, L1=0.110, L2=0.096, H1=0.1312, H2=0.0751)
COURSE_ARM_LIMITS_DEG = [(-165, 165)] * 5 + [(-175, 175)]
COURSE_ARM_RATED_SPEED_DEG_S = 160.0       # illustrative; re-measure at commissioning
COURSE_ARM_TAU_RATED = [1.8, 3.0, 1.5, 0.6, 0.6, 0.4]   # N m, illustrative smart-servo ratings
# link: (mass kg, CoM at home in {s} [m], box edges a, b, c [m] along x_s, y_s, z_s) -- ILLUSTRATIVE
COURSE_ARM_LINKS = [
    (0.20, (0.000, 0.010, 0.105), (0.050, 0.060, 0.060)),   # 1 shoulder housing (turns with J1)
    (0.16, (0.055, 0.030, 0.1312), (0.130, 0.040, 0.040)),  # 2 upper arm
    (0.12, (0.158, 0.005, 0.1312), (0.120, 0.035, 0.035)),  # 3 forearm
    (0.08, (0.206, 0.040, 0.1312), (0.035, 0.050, 0.035)),  # 4 wrist 1 (J5 servo)
    (0.08, (0.206, 0.0634, 0.090), (0.035, 0.035, 0.050)),  # 5 wrist 2 (J6 servo)
    (0.03, (0.206, 0.100, 0.0561), (0.040, 0.015, 0.040)),  # 6 flange
]


def course_arm() -> dict:
    """THE course arm: desktop UR-type 6R (myCobot-280 class), MR Example 4.5 layout (pp. 145-146)."""
    g_ = COURSE_ARM_GEOMETRY
    W1, W2, L1, L2, H1, H2 = (g_[k] for k in ("W1", "W2", "L1", "L2", "H1", "H2"))
    S = np.array([[0, 0, 1, 0, 0, 0],
                  [0, 1, 0, -H1, 0, 0],
                  [0, 1, 0, -H1, 0, L1],
                  [0, 1, 0, -H1, 0, L1 + L2],
                  [0, 0, -1, -W1, L1 + L2, 0],
                  [0, 1, 0, H2 - H1, 0, L1 + L2]], float).T
    M = np.array([[-1, 0, 0, L1 + L2], [0, 0, 1, W1 + W2], [0, 1, 0, H1 - H2], [0, 0, 0, 1]], float)
    coms = [trans(c) for _, c, _ in COURSE_ARM_LINKS]
    return dict(name="course arm", key="course_arm", n=6, Slist=S, M=M,
                limits=np.array(COURSE_ARM_LIMITS_DEG, float) * DEG,
                qdot_max=np.full(6, COURSE_ARM_RATED_SPEED_DEG_S * DEG),
                tau_max=np.array(COURSE_ARM_TAU_RATED, float),
                q_home=np.array([0.0, -0.6, 1.4, -2.4, -1.57, 0.0]),
                Mlist=mlist_from_absolute(coms, M),
                Glist=[box_inertia(m, *e) for m, _, e in COURSE_ARM_LINKS],
                g=np.array([0.0, 0.0, -G0]))


# ------------------------------------------------------------------------------------------- EOD arm 3R
EOD_3R_TAU_RATED = (155.0, 70.0, 15.0)   # N m, illustrative


def planar_chain(L, m, r, Izz, Ixx_factor=0.2, g=(0, -G0, 0)) -> dict:
    """Planar n-R chain in the vertical x-y plane (gravity -y), joints along z, zero pose along +x;
    CoM r_i from joint i, Izz about the CoM; Glist = diag(Ixx_factor*Izz, Izz, Izz, m, m, m)."""
    n = len(L)
    jx = np.r_[0.0, np.cumsum(L)]
    coms = [trans([jx[i] + r[i], 0, 0]) for i in range(n)]
    M = trans([jx[n], 0, 0])
    S = np.array([[0, 0, 1, 0, -jx[i], 0] for i in range(n)], float).T
    G = [np.diag([Ixx_factor * Izz[i], Izz[i], Izz[i], m[i], m[i], m[i]]) for i in range(n)]
    return dict(n=n, Slist=S, M=M, Mlist=mlist_from_absolute(coms, M), Glist=G, g=np.asarray(g, float),
                L=tuple(L), m=tuple(m), r=tuple(r), Izz=tuple(Izz))


def eod_3r() -> dict:
    """Vertical-plane shoulder / elbow / wrist-pitch arm of a tracked EOD robot; link 3 = wrist + gripper +
    2 kg reference package (SBEG 461 Weeks 1-3). ILLUSTRATIVE VALUES."""
    r = planar_chain([0.60, 0.50, 0.15], [8.0, 5.0, 3.5], [0.30, 0.25, 0.10], [0.24, 0.10, 0.008])
    r.update(name="EOD arm 3R", key="eod_3r",
             limits=np.array([[-1.0, 2.0], [-2.6, 2.6], [-2.0, 2.0]]),
             qdot_max=None, tau_max=np.array(EOD_3R_TAU_RATED), q_home=np.zeros(3))
    return r


ROBOTS = {"course_arm": course_arm, "eod_3r": eod_3r}


def get(key: str) -> dict:
    return ROBOTS[key]()
