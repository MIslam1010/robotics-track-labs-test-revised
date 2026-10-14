"""Robot model used in SBEG 46X Lab 5 -- a self-contained copy of the course arm (numbers identical).

# from the shared course robot models (Course resources): course_arm  (kinematics, limits, encoder resolution)

The course arm is the low-cost desktop UR-type 6R arm of the myCobot-280 class that you used in SBEG 361, SBEG 461
and Lab 1 of this course (layout of the UR5 in Modern Robotics (MR) Example 4.5, pp. 145-146, scaled down).
In this lab the arm carries a straight digitising pointer on its flange.  The pointer tip offset p_tip in the flange
frame {b} is NOT given: you find it by pivot calibration (Part A).

ILLUSTRATIVE VALUES: the pointer and its nominal length are teaching values, not data of a real product.
Robot data and the hardware safety checklist: "Course resources: Course robot models and hardware safety checklist".
Notation: MR -- screw axes S = (w, v) angular part first, space-form PoE T = e^[S1]th1 ... e^[S6]th6 M (MR Eq. 4.14).
Units in this file: metres and radians (the lab code converts positions to millimetres).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- course arm geometry (m)
W1, W2, L1, L2, H1, H2 = 0.0634, 0.0456, 0.110, 0.096, 0.1312, 0.0751

LIMITS_DEG = np.array([(-165, 165)] * 5 + [(-175, 175)], float)       # hardware joint limits
SOFT_LIMIT_MARGIN_DEG = 10.0                                           # safety checklist rule 1
RATED_SPEED_DEG_S = 160.0                                              # illustrative, all joints
FIRST_RUN_SPEED_FRACTION = 0.25                                        # safety checklist rule 3
ENCODER_BITS = 12                                                      # 4096 counts per turn (0.088 deg)
ENC_RES = 2 * np.pi / 2 ** ENCODER_BITS
Q_HOME = np.array([0.0, -0.6, 1.4, -2.4, -1.57, 0.0])                  # "ready" configuration (rad)

POINTER_NOMINAL_LENGTH = 0.070   # m, nominal length of the 3-D-printed digitising pointer (illustrative)


def course_arm_kinematics() -> dict:
    """Course arm PoE model of the FLANGE frame {b} (tool = 0).

    Returns a dict with
        Slist  : 6 x 6 space screw axes (columns S_i = (w_i, v_i), metres)      (MR Example 4.5 form)
        M      : 4 x 4 home pose of the flange frame {b} in the base frame {s} (metres)
        limits : 6 x 2 software joint limits (rad) = hardware limits minus the 10 deg safety margin
        qdot_max : 6 joint-speed limits (rad/s) = 25 % of the rated 160 deg/s (first-run rule)
    """
    S = np.array([[0, 0, 1, 0, 0, 0],
                  [0, 1, 0, -H1, 0, 0],
                  [0, 1, 0, -H1, 0, L1],
                  [0, 1, 0, -H1, 0, L1 + L2],
                  [0, 0, -1, -W1, L1 + L2, 0],
                  [0, 1, 0, H2 - H1, 0, L1 + L2]], float).T
    M = np.array([[-1, 0, 0, L1 + L2],
                  [0, 0, 1, W1 + W2],
                  [0, 1, 0, H1 - H2],
                  [0, 0, 0, 1]], float)
    soft = np.deg2rad(LIMITS_DEG - np.array([-1, 1]) * SOFT_LIMIT_MARGIN_DEG)
    qdot_max = np.full(6, np.deg2rad(FIRST_RUN_SPEED_FRACTION * RATED_SPEED_DEG_S))
    return dict(Slist=S, M=M, limits=soft, qdot_max=qdot_max)
