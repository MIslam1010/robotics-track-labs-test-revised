"""The SBEG course arm (desktop UR-type 6R, myCobot-280 class) with its illustrative dynamics.

# from the shared course robot models: course_arm (and the helpers trans, body_inertia, box_inertia,
# transform_inertia, mlist_from_absolute, add_payload). Numbers are IDENTICAL to the shared model;
# copied here so that the lab folder is self-contained.

Kinematics: MR Example 4.5 layout (pp. 145-146) scaled down, W1 = 0.0634, W2 = 0.0456, L1 = 0.110,
L2 = 0.096, H1 = 0.1312, H2 = 0.0751 m. Rated speed, torques, masses and inertias are ILLUSTRATIVE
VALUES; replace them with the commissioned / identified values published on the course site.
"""
from __future__ import annotations

import modern_robotics as mr
import numpy as np

G0 = 9.81
DEG = np.pi / 180.0

COURSE_ARM_GEOMETRY = dict(W1=0.0634, W2=0.0456, L1=0.110, L2=0.096, H1=0.1312, H2=0.0751)
COURSE_ARM_LIMITS_DEG = [(-165, 165)] * 5 + [(-175, 175)]
COURSE_ARM_RATED_SPEED_DEG_S = 160.0       # illustrative; re-measure at commissioning
COURSE_ARM_TAU_RATED = [1.8, 3.0, 1.5, 0.6, 0.6, 0.4]   # N m, illustrative smart-servo ratings
COURSE_ARM_PAYLOAD_MAX = 0.25              # kg, illustrative
# hardware safety checklist values (Course resources: "Course robot models and hardware safety checklist")
COURSE_ARM_SAFETY = dict(
    soft_limit_margin_deg=10.0,          # software joint limits >= 10 deg inside the hardware limits
    keepout_extra_m=0.30,                # taped keep-out radius = reach + 0.30 m
    speed_fraction_first_run=0.25,       # first run of any new motion, and all of 361 session H1
    speed_fraction=0.50,                 # otherwise
    payload_max_kg=0.5 * COURSE_ARM_PAYLOAD_MAX,   # 0.125 kg unless the department approves an exception
)
# link: (mass kg, CoM at home in {s} [m], box edges a, b, c [m] along x_s, y_s, z_s) -- ILLUSTRATIVE
COURSE_ARM_LINKS = [
    (0.20, (0.000, 0.010, 0.105), (0.050, 0.060, 0.060)),   # 1 shoulder housing (turns with J1)
    (0.16, (0.055, 0.030, 0.1312), (0.130, 0.040, 0.040)),  # 2 upper arm
    (0.12, (0.158, 0.005, 0.1312), (0.120, 0.035, 0.035)),  # 3 forearm
    (0.08, (0.206, 0.040, 0.1312), (0.035, 0.050, 0.035)),  # 4 wrist 1 (J5 servo)
    (0.08, (0.206, 0.0634, 0.090), (0.035, 0.035, 0.050)),  # 5 wrist 2 (J6 servo)
    (0.03, (0.206, 0.100, 0.0561), (0.040, 0.015, 0.040)),  # 6 flange
]


def trans(p=(0.0, 0.0, 0.0), R=None) -> np.ndarray:
    return mr.RpToTrans(np.eye(3) if R is None else np.asarray(R, float), np.asarray(p, float))


def body_inertia(m: float, Ib) -> np.ndarray:
    """G = diag(I_b, m I3) in a CoM frame (MR Eq. 8.32)."""
    Ib = np.diag(np.asarray(Ib, float)) if np.ndim(Ib) == 1 else np.asarray(Ib, float)
    G = np.zeros((6, 6))
    G[:3, :3] = Ib
    G[3:, 3:] = m * np.eye(3)
    return G


def box_inertia(m: float, a: float, b: float, c: float) -> np.ndarray:
    return body_inertia(m, [m * (b**2 + c**2) / 12, m * (a**2 + c**2) / 12, m * (a**2 + b**2) / 12])


def transform_inertia(G_b: np.ndarray, T_ba: np.ndarray) -> np.ndarray:
    """G_a = [Ad_Tba]^T G_b [Ad_Tba] (MR Eq. 8.42)."""
    Ad = mr.Adjoint(T_ba)
    return Ad.T @ G_b @ Ad


def mlist_from_absolute(com_frames, M_ee) -> list:
    frames = [np.eye(4)] + list(com_frames) + [M_ee]
    return [mr.TransInv(frames[i]) @ frames[i + 1] for i in range(len(frames) - 1)]


def course_arm(tool: float = 0.0, payload: float = 0.0) -> dict:
    """THE course arm. Returns Slist, M, limits, qdot_max, tau_max, q_home, Mlist, Glist, g.
    `payload` (kg) adds a point mass at the tool point (same rule as the shared model)."""
    g_ = COURSE_ARM_GEOMETRY
    W1, W2, L1, L2, H1, H2 = (g_[k] for k in ("W1", "W2", "L1", "L2", "H1", "H2"))
    S = np.array([[0, 0, 1, 0, 0, 0],
                  [0, 1, 0, -H1, 0, 0],
                  [0, 1, 0, -H1, 0, L1],
                  [0, 1, 0, -H1, 0, L1 + L2],
                  [0, 0, -1, -W1, L1 + L2, 0],
                  [0, 1, 0, H2 - H1, 0, L1 + L2]], float).T
    M = np.array([[-1, 0, 0, L1 + L2], [0, 0, 1, W1 + W2], [0, 1, 0, H1 - H2], [0, 0, 0, 1]], float)
    M = M @ trans((0, 0, tool))
    coms = [trans(c) for _, c, _ in COURSE_ARM_LINKS]
    Glist = [box_inertia(m, *e) for m, _, e in COURSE_ARM_LINKS]
    Mlist = mlist_from_absolute(coms, M)
    if payload:
        T_np = Mlist[-1]                      # payload CoM = tool point, in the last-link CoM frame
        Glist[-1] = Glist[-1] + transform_inertia(body_inertia(payload, [0.0, 0.0, 0.0]), mr.TransInv(T_np))
    return dict(name="course arm", n=6, Slist=S, M=M,
                limits=np.array(COURSE_ARM_LIMITS_DEG, float) * DEG,
                qdot_max=np.full(6, COURSE_ARM_RATED_SPEED_DEG_S * DEG),
                tau_max=np.array(COURSE_ARM_TAU_RATED, float),
                q_home=np.array([0.0, -0.6, 1.4, -2.4, -1.57, 0.0]),
                Mlist=Mlist, Glist=Glist, g=np.array([0.0, 0.0, -G0]),
                payload_max=COURSE_ARM_PAYLOAD_MAX, derated_speed_fraction=0.5,
                derated_current_fraction=0.6)
