"""Robot model used in Lab 11 -- copied from the shared course robot models (Course resources: Course robot models and
hardware safety checklist), numbers identical.

  ur5_laparoscope()  UR5 (MR Example 4.5 kinematics with the full-precision values of the MR software:
                     W1 = 0.10915, W2 = 0.0823, L1 = 0.425, L2 = 0.39225, H1 = 0.089159, H2 = 0.09465 m) holding a
                     0.30 m laparoscope (endoscope) along z_b of the flange.  The end-effector frame {b} is at the
                     scope TIP; in Lab 11 the endoscope camera frame {c} is {b} (optical axis = z_b, a 0-degree scope).
                     Kinematics only (the lab commands joint velocities).

The lab must be self-contained, so the model is written out here.
ILLUSTRATIVE VALUES for the tool: teaching values, not the data of a real product.
Notation: Modern Robotics (MR) -- screw axes S = (w, v) (MR Sec. 3.3.2), home pose M, body screws B = [Ad_{M^-1}] S.
"""
from __future__ import annotations

import modern_robotics as mr
import numpy as np

UR5_GEOMETRY = dict(W1=0.10915, W2=0.0823, L1=0.425, L2=0.39225, H1=0.089159, H2=0.09465)


def ur5_laparoscope() -> dict:
    """UR5 + 0.30 m laparoscope ({b} at the scope tip).  Returns Slist, Blist (6 x 6, axes as columns), M,
    q_home (the start pose of Lab 11: scope pointing down, z_b ~ -z_s), qdot_max (rad/s, all joints), limits."""
    g = UR5_GEOMETRY
    W1, W2, L1, L2, H1, H2 = g["W1"], g["W2"], g["L1"], g["L2"], g["H1"], g["H2"]
    L_tool = 0.30
    S = np.array([[0, 0, 1, 0, 0, 0],
                  [0, 1, 0, -H1, 0, 0],
                  [0, 1, 0, -H1, 0, L1],
                  [0, 1, 0, -H1, 0, L1 + L2],
                  [0, 0, -1, -W1, L1 + L2, 0],
                  [0, 1, 0, H2 - H1, 0, L1 + L2]], float).T
    M = np.array([[-1, 0, 0, L1 + L2], [0, 0, 1, W1 + W2], [0, 1, 0, H1 - H2], [0, 0, 0, 1]], float)
    M = M @ mr.RpToTrans(np.eye(3), np.array([0.0, 0.0, L_tool]))
    return dict(key="ur5_laparoscope", name="UR5 + 0.30 m laparoscope", Slist=S,
                Blist=mr.Adjoint(mr.TransInv(M)) @ S, M=M,
                q_home=np.array([2.6, -2.5, 2.3, -1.4, -1.57, 0.0]), qdot_max=1.0,
                limits=np.tile([-2 * np.pi, 2 * np.pi], (6, 1)), params=dict(L_tool=L_tool, **g),
                illustrative=True)
