"""Robot and tool models used to generate the Lab 4 data (SBEG 361, Week 5).

Copied from the course robot models (Course resources: Course robot models and hardware
safety checklist) so that the lab folder is self-contained.
The numbers are IDENTICAL to the shared models; do not edit them here.

  ur5_flange()        from the course models: ur5() / _ur_type_kinematics(**UR5_SOFTWARE)
                      (MR Example 4.5, pp. 145-146, full-precision MR-software values)
  LAPAROSCOPE         from the course models: ur5_laparoscope() -- 0.30 m, 0.6 kg tool,
                      CoM 0.10 m from the flange along z_b (illustrative values)
  eod_6r_ugv()        from the course models: eod_6r_ugv() -- SBEG 361 project scenario S4
                      (illustrative values)

You do NOT need this file for the lab (the poses are given in the data files; forward
kinematics by the product of exponentials is Week 6).  It is used by make_data.py.
Units: m, kg, s, rad.
"""
import numpy as np

G0 = 9.81
DEG = np.pi / 180.0

# from the course robot models: UR5_SOFTWARE
UR5_SOFTWARE = dict(W1=0.10915, W2=0.0823, L1=0.425, L2=0.39225, H1=0.089159, H2=0.09465)

# from the course robot models: ur5_laparoscope() (tool parameters)
LAPAROSCOPE = dict(L_tool=0.30, m_tool=0.6, com_from_flange=0.10)


def screw_revolute(w, q, h=0.0):
    """S = (w, -w x q + h w)  (MR Sec. 3.3.2.2).  From the course models: screw_revolute."""
    w = np.asarray(w, float)
    return np.r_[w, -np.cross(w, np.asarray(q, float)) + h * w]


def trans(p=(0.0, 0.0, 0.0), R=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(p, float)
    return T


def ur5_flange():
    """(Slist, M) of the UR5 with {b} at the tool flange.
    From the course models: _ur_type_kinematics(**UR5_SOFTWARE), tool = 0."""
    W1, W2, L1, L2, H1, H2 = (UR5_SOFTWARE[k] for k in ("W1", "W2", "L1", "L2", "H1", "H2"))
    S = np.array([[0, 0, 1, 0, 0, 0],
                  [0, 1, 0, -H1, 0, 0],
                  [0, 1, 0, -H1, 0, L1],
                  [0, 1, 0, -H1, 0, L1 + L2],
                  [0, 0, -1, -W1, L1 + L2, 0],
                  [0, 1, 0, H2 - H1, 0, L1 + L2]], float).T
    M = np.array([[-1, 0, 0, L1 + L2], [0, 0, 1, W1 + W2], [0, 1, 0, H1 - H2], [0, 0, 0, 1]], float)
    return S, M


def eod_6r_ugv(shoulder=(0.10, 0.20), L1=0.80, L2=0.75, Lg=0.25):
    """(Slist, M, wrist_point) of the UGV-mounted EOD 6R arm, {s} on the turret base, {b} at the
    gripper.  From the course models: eod_6r_ugv() (SBEG 361 project scenario S4)."""
    a, h = shoulder
    sh = np.array([a, 0, h])
    el = sh + [L1, 0, 0]
    wr = el + [L2, 0, 0]
    w = [(0, 0, 1), (0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0)]
    q = [np.zeros(3), sh, el, wr, wr, wr]
    S = np.column_stack([screw_revolute(wi, qi) for wi, qi in zip(w, q)])
    return S, trans(wr + [Lg, 0, 0]), wr
