"""Reference data of the SBEG 361 course arm, copied for Lab 5 so that the lab is self-contained.

Contents: COURSE_ARM_GEOMETRY, COURSE_ARM_LIMITS_DEG, COURSE_ARM_RATED_SPEED_DEG_S and
course_arm_reference() (kinematic part of the instructors' shared course-arm model).

The numbers are IDENTICAL to the shared model (checked by test_lab.py on the instructors' copy;
the check is skipped in the student copy).  Rated speed and joint limits are ILLUSTRATIVE VALUES; replace them with the values
published on the course site once the arms are commissioned.

This file is the ground truth for the tests.  Students do NOT edit it and do NOT import it in
starter.py: in Step 2 you derive the same screw axes yourself from the geometry.
"""
import numpy as np

# Desktop UR-type 6R (myCobot-280 class), layout of MR Example 4.5 (printed pp. 145-146).
COURSE_ARM_GEOMETRY = dict(W1=0.0634, W2=0.0456, L1=0.110, L2=0.096, H1=0.1312, H2=0.0751)  # m
COURSE_ARM_LIMITS_DEG = [(-165, 165)] * 5 + [(-175, 175)]
COURSE_ARM_RATED_SPEED_DEG_S = 160.0  # illustrative


def course_arm_reference(tool=0.0):
    """(M, Slist, axis_points) of the course arm; `tool` moves {b} along z_b (needle guide: 0.06 m)."""
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
