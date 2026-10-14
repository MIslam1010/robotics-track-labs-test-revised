"""Course arm and lab robots for SBEG 361 Lab 6 (self-contained copy).

# from the course robot models (Course resources): course_arm  (kinematics only; numbers identical)
The course arm is a low-cost desktop UR-type 6R arm of the myCobot-280 class with the layout of the UR5 in
MR Example 4.5 (pp. 145-146), scaled down.  Rated values are ILLUSTRATIVE (see the Lab 5 handout).

# The RCM laparoscope holder (Workshop 7 Problem 2) is problem data for this week (illustrative values),
# not a library robot.
"""
import numpy as np

# ---------------------------------------------------------------- course arm (PoE, MR Eq. 4.14)
W1, W2, L1, L2, H1, H2 = 0.0634, 0.0456, 0.110, 0.096, 0.1312, 0.0751   # m

SLIST = np.array([[0, 0, 1, 0, 0, 0],
                  [0, 1, 0, -H1, 0, 0],
                  [0, 1, 0, -H1, 0, L1],
                  [0, 1, 0, -H1, 0, L1 + L2],
                  [0, 0, -1, -W1, L1 + L2, 0],
                  [0, 1, 0, H2 - H1, 0, L1 + L2]], float).T          # 6 x 6, columns S_i = (w_i, v_i)

M_HOME = np.array([[-1, 0, 0, L1 + L2],
                   [0, 0, 1, W1 + W2],
                   [0, 1, 0, H1 - H2],
                   [0, 0, 0, 1]], float)                              # T_sb at theta = 0

LIMITS = np.deg2rad(np.array([(-165, 165)] * 5 + [(-175, 175)], float))
Q_HOME = np.array([0.0, -0.6, 1.4, -2.4, -1.57, 0.0])                  # "ready" configuration (rad)
ENC_RES = 2 * np.pi / 4096                                             # 12-bit encoders (0.088 deg)

# ---------------------------------------------------------------- RCM laparoscope holder (illustrative)
# Joint 1: yaw about +z_s through the port; joint 2: pitch about +y_s through the port;
# joint 3: insertion (prismatic) along the shaft; joint 4: roll about the shaft.
# At q = 0 the shaft points straight down (-z_s) and the tip is RCM_TIP0 below the port.
RCM_PORT = np.array([0.45, 0.10, 0.30])   # m, remote centre of motion (trocar port) in {s}
RCM_TIP0 = 0.10                           # m, tip depth below the port at q3 = 0
