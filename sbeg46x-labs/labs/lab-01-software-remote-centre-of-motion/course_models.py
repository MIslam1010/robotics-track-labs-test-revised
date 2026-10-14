"""Robot model used in SBEG 46X Lab 1 -- a self-contained copy of the course arm (numbers identical).

# from the shared course robot models (Course resources): course_arm  (kinematics and rated speed only)

The course arm is the low-cost desktop UR-type 6R arm of the myCobot-280 class that you used in SBEG 361 and SBEG 461
(layout of the UR5 in Modern Robotics (MR) Example 4.5, pp. 145-146, scaled down).  In this lab it holds a straight
laparoscope tool of length TOOL_LENGTH along the flange axis z_b, so the tool frame {b} sits at the laparoscope tip
with z_b pointing along the shaft towards the tip (MR: `course_arm(tool=L)` moves {b} a distance L along z_b).

ILLUSTRATIVE VALUES: the rated speed and the 0.12 m teaching laparoscope are teaching values, not data of a real
product.  Robot data and the hardware safety checklist: "Course resources: Course robot models and hardware safety
checklist".  Notation: MR -- screw axes S = (w, v) angular part first, space-form PoE T = e^[S1]th1 ... e^[S6]th6 M
(MR Eq. 4.14).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- course arm geometry (m)
W1, W2, L1, L2, H1, H2 = 0.0634, 0.0456, 0.110, 0.096, 0.1312, 0.0751

LIMITS_DEG = np.array([(-165, 165)] * 5 + [(-175, 175)], float)       # hardware joint limits
SOFT_LIMIT_MARGIN_DEG = 10.0                                           # safety checklist rule 1
RATED_SPEED_DEG_S = 160.0                                              # illustrative, all joints
FIRST_RUN_SPEED_FRACTION = 0.25                                        # safety checklist rule 3
Q_HOME = np.array([0.0, -0.6, 1.4, -2.4, -1.57, 0.0])                  # "ready" configuration (rad)

TOOL_LENGTH = 0.12   # m, straight teaching laparoscope (illustrative; a clinical laparoscope is ~0.3 m or longer)


def course_arm_kinematics(tool: float = TOOL_LENGTH) -> dict:
    """Course arm PoE model with a straight tool of length `tool` along z_b.

    Returns a dict with
        Slist  : 6 x 6 space screw axes (columns S_i = (w_i, v_i))           (MR Example 4.5 form)
        M      : 4 x 4 home pose of the tool frame {b} (at the tool tip)
        limits : 6 x 2 software joint limits (rad) = hardware limits minus the 10 deg safety margin
        qdot_max : 6 joint-speed limits (rad/s) = 25 % of the rated 160 deg/s (first-run rule)
        tool   : the tool length (m)
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
    M = M.copy()
    M[:3, 3] += tool * M[:3, 2]          # M @ Trans(0, 0, tool)
    soft = np.deg2rad(LIMITS_DEG - np.array([-1, 1]) * SOFT_LIMIT_MARGIN_DEG)
    qdot_max = np.full(6, np.deg2rad(FIRST_RUN_SPEED_FRACTION * RATED_SPEED_DEG_S))
    return dict(Slist=S, M=M, limits=soft, qdot_max=qdot_max, tool=float(tool))
