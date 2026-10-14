"""Reference data of the SBEG 361 course arm, copied for Hardware rotation 1 (Week 8) so that the lab is
self-contained.

# from the shared course robot models (Course resources): COURSE_ARM_GEOMETRY, COURSE_ARM_LIMITS_DEG,
#      COURSE_ARM_RATED_SPEED_DEG_S, COURSE_ARM_LINKS, _ur_type_kinematics, course_arm (kinematics,
#      link masses / CoMs / box inertias, gravity), box_inertia, mlist_from_absolute

The numbers are IDENTICAL to the shared model (checked by test_lab.py when the shared folder is present).
Rated speed, joint limits, masses and inertias are ILLUSTRATIVE VALUES; replace them with the values published
on the course site once the arms are commissioned.

This file is the ground truth for the tests and for the emulator `sim_arm.py`.  Do NOT import it in starter.py.
"""
import numpy as np

# Desktop UR-type 6R (myCobot-280 class), layout of MR Example 4.5 (printed pp. 145-146).
COURSE_ARM_GEOMETRY = dict(W1=0.0634, W2=0.0456, L1=0.110, L2=0.096, H1=0.1312, H2=0.0751)  # m
COURSE_ARM_LIMITS_DEG = [(-165, 165)] * 5 + [(-175, 175)]      # model angles, illustrative
COURSE_ARM_RATED_SPEED_DEG_S = 160.0                            # illustrative
ENC_RES_DEG = 360.0 / 4096                                      # 12-bit encoders (0.088 deg)
G0 = 9.81
# link: (mass kg, CoM at home in {s} [m], box edges a, b, c [m] along x_s, y_s, z_s) -- ILLUSTRATIVE
COURSE_ARM_LINKS = [
    (0.20, (0.000, 0.010, 0.105), (0.050, 0.060, 0.060)),   # 1 shoulder housing (turns with J1)
    (0.16, (0.055, 0.030, 0.1312), (0.130, 0.040, 0.040)),  # 2 upper arm
    (0.12, (0.158, 0.005, 0.1312), (0.120, 0.035, 0.035)),  # 3 forearm
    (0.08, (0.206, 0.040, 0.1312), (0.035, 0.050, 0.035)),  # 4 wrist 1 (J5 servo)
    (0.08, (0.206, 0.0634, 0.090), (0.035, 0.035, 0.050)),  # 5 wrist 2 (J6 servo)
    (0.03, (0.206, 0.100, 0.0561), (0.040, 0.015, 0.040)),  # 6 flange
]


def course_arm_reference(tool=0.0):
    """(M, Slist, axis_points) of the course arm; `tool` moves {b} along z_b (pointer / needle guide: 0.06 m)."""
    g = COURSE_ARM_GEOMETRY
    W1, W2, L1, L2, H1, H2 = g["W1"], g["W2"], g["L1"], g["L2"], g["H1"], g["H2"]
    S = np.array([[0, 0, 1, 0, 0, 0],
                  [0, 1, 0, -H1, 0, 0],
                  [0, 1, 0, -H1, 0, L1],
                  [0, 1, 0, -H1, 0, L1 + L2],
                  [0, 0, -1, -W1, L1 + L2, 0],
                  [0, 1, 0, H2 - H1, 0, L1 + L2]], float).T
    M = np.array([[-1, 0, 0, L1 + L2], [0, 0, 1, W1 + W2], [0, 1, 0, H1 - H2], [0, 0, 0, 1]], float)
    Tt = np.eye(4)
    Tt[2, 3] = tool
    M = M @ Tt
    q = np.array([(0, 0, 0), (0, 0, H1), (L1, 0, H1), (L1 + L2, 0, H1), (L1 + L2, W1, 0),
                  (L1 + L2, 0, H1 - H2)], float)
    return M, S, q


def _box_inertia(m, a, b, c):
    G = np.zeros((6, 6))
    G[:3, :3] = np.diag([m * (b**2 + c**2) / 12, m * (a**2 + c**2) / 12, m * (a**2 + b**2) / 12])
    G[3:, 3:] = m * np.eye(3)
    return G


def _trans_inv(T):
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3], Ti[:3, 3] = R.T, -R.T @ p
    return Ti


def course_arm_dynamics():
    """(Mlist, Glist, g) in MR list form (MR Sec. 8.3.2): CoM frames parallel to {s} at home (illustrative)."""
    M, _, _ = course_arm_reference()
    frames = [np.eye(4)]
    for _, c, _ in COURSE_ARM_LINKS:
        T = np.eye(4)
        T[:3, 3] = c
        frames.append(T)
    frames.append(M)
    Mlist = [_trans_inv(frames[i]) @ frames[i + 1] for i in range(len(frames) - 1)]
    Glist = [_box_inertia(m, *e) for m, _, e in COURSE_ARM_LINKS]
    return Mlist, Glist, np.array([0.0, 0.0, -G0])
