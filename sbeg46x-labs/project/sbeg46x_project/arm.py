"""The course arm (kinematics, limits and a servo model) for Scenarios 2, 4 and 5.

The numbers are copied unchanged from the shared course robot models (Course resources: "Course robot
models and hardware safety checklist", model ``course_arm``): a desktop UR-type 6R arm of the
myCobot-280 class with the layout of the UR5 of MR Example 4.5, scaled down. Rated speed and torque
are ILLUSTRATIVE VALUES; use the identified values published on the course site for the hardware run.

Notation as in *Modern Robotics*: space screw axes S_i = (w_i, v_i), T(q) = e^[S1]q1 ... e^[S6]q6 M
(MR Eq. 4.14). The tool frame {b} is at the tool tip and z_b points along the tool shaft.
"""
from __future__ import annotations

import numpy as np
import modern_robotics as mr

# from the shared course robot models: course_arm (numbers identical)
GEOMETRY = dict(W1=0.0634, W2=0.0456, L1=0.110, L2=0.096, H1=0.1312, H2=0.0751)
LIMITS_DEG = np.array([(-165, 165)] * 5 + [(-175, 175)], float)     # hardware limits
SOFT_LIMIT_MARGIN_DEG = 10.0                                         # checklist rule 1
RATED_SPEED = np.radians(160.0)                                      # rad/s, all joints
TAU_RATED = np.array([1.8, 3.0, 1.5, 0.6, 0.6, 0.4])                # N m
ENC_RES = 2 * np.pi / 4096                                           # 12-bit encoders
SPEED_FRACTION = 0.50            # checklist rule 3 (0.25 on the first run of any new motion)
SPEED_FRACTION_FIRST_RUN = 0.25
CURRENT_FRACTION = 0.60          # derated servo current
PAYLOAD_MAX = 0.125              # kg (50 % of the rated 0.25 kg), checklist rule 4
Q_HOME = np.array([0.0, -0.6, 1.4, -2.4, -1.57, 0.0])               # ready configuration


def screw_list(tool: float = 0.0):
    """Space screw axes (6 x 6, columns) and home pose M of the course arm with a straight tool of
    length ``tool`` (m) along z_b (MR Example 4.5 layout)."""
    W1, W2, L1, L2, H1, H2 = (GEOMETRY[k] for k in ("W1", "W2", "L1", "L2", "H1", "H2"))
    S = np.array([[0, 0, 1, 0, 0, 0],
                  [0, 1, 0, -H1, 0, 0],
                  [0, 1, 0, -H1, 0, L1],
                  [0, 1, 0, -H1, 0, L1 + L2],
                  [0, 0, -1, -W1, L1 + L2, 0],
                  [0, 1, 0, H2 - H1, 0, L1 + L2]], float).T
    M = np.array([[-1, 0, 0, L1 + L2], [0, 0, 1, W1 + W2], [0, 1, 0, H1 - H2], [0, 0, 0, 1]], float)
    M[:3, 3] += M[:3, 2] * tool
    return S, M


def soft_limits() -> np.ndarray:
    """Software joint limits (rad): hardware limits minus the 10 deg margin of checklist rule 1."""
    lim = LIMITS_DEG.copy()
    lim[:, 0] += SOFT_LIMIT_MARGIN_DEG
    lim[:, 1] -= SOFT_LIMIT_MARGIN_DEG
    return np.radians(lim)


def fk(q, tool: float = 0.0) -> np.ndarray:
    """T_sb(q) of the tool tip (MR Eq. 4.14)."""
    S, M = screw_list(tool)
    return mr.FKinSpace(M, S, np.asarray(q, float))


def fk_tool(q, T_ft) -> np.ndarray:
    """Tool pose T_sb = T_sf(q) T_ft for a general tool transform T_ft (flange -> tool tip), e.g.
    the result of a pivot calibration."""
    return fk(q, 0.0) @ np.asarray(T_ft, float)


def ik_tool(T_goal, q0, T_ft, eomg: float = 1e-7, ev: float = 1e-8):
    """Numerical IK for a general tool transform: solves T_sf(q) = T_goal T_ft^-1."""
    return ik(np.asarray(T_goal, float) @ mr.TransInv(np.asarray(T_ft, float)), q0, 0.0, eomg, ev)


def jacobian_space(q, tool: float = 0.0) -> np.ndarray:
    S, _ = screw_list(tool)
    return mr.JacobianSpace(S, np.asarray(q, float))


def point_jacobian(q, p) -> np.ndarray:
    """3 x 6 Jacobian of the velocity (in {s}) of the point of the last link that is at position p
    (in {s}) now: p_dot = v_s + w_s x p = [-[p]  I] J_s q_dot (MR Eq. 3.73 and the space Jacobian,
    MR Eq. 5.11)."""
    Js = jacobian_space(q, 0.0)
    return -mr.VecToso3(np.asarray(p, float)) @ Js[:3] + Js[3:]


def ik(T_goal, q0, tool: float = 0.0, eomg: float = 1e-6, ev: float = 1e-7):
    """Numerical inverse kinematics (Newton-Raphson, MR §6.2, ``IKinSpace``) from the initial guess q0.
    Returns (q, success). Check the result against the soft limits yourself."""
    S, M = screw_list(tool)
    q, ok = mr.IKinSpace(S, M, np.asarray(T_goal, float), np.asarray(q0, float), eomg, ev)
    q = (np.asarray(q) + np.pi) % (2 * np.pi) - np.pi
    return q, bool(ok)


def line_pose(p_tip, axis, roll_ref=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Tool pose with the tip at ``p_tip`` and z_b along the unit vector ``axis`` (pointing into the
    patient / object). x_b is chosen perpendicular to ``axis`` and as close as possible to
    ``roll_ref`` x axis (the roll about the tool axis is free for a symmetric tool)."""
    z = np.asarray(axis, float) / np.linalg.norm(axis)
    r = np.asarray(roll_ref, float)
    x = np.cross(r, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return mr.RpToTrans(np.column_stack([x, y, z]), np.asarray(p_tip, float))


def shaft_distance(T, point) -> float:
    """Distance (m) from ``point`` to the tool-shaft line through the tip along z_b. For a remote
    centre of motion (RCM) at a port, this is the RCM error."""
    p, z = T[:3, 3], T[:3, 2]
    d = np.asarray(point, float) - p
    return float(np.linalg.norm(d - np.dot(d, z) * z))


def segment_point_distance(a, b, point) -> float:
    """Distance (m) from ``point`` to the segment a-b (e.g. the tool shaft between the tip and a
    point 0.10 m behind it). Used for forbidden-region and keep-out checks."""
    a, b, c = (np.asarray(v, float) for v in (a, b, point))
    ab = b - a
    s = np.clip(np.dot(c - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(a + s * ab - c))


def quantise(q) -> np.ndarray:
    """12-bit encoder quantisation of joint angles."""
    return np.round(np.asarray(q, float) / ENC_RES) * ENC_RES


# ---------------------------------------------------------------------------------- servo model
class ServoModel:
    """Model of the arm's position-controlled smart servos, one per joint: a second-order loop
    q_ddot = wn^2 (q_c - q) - 2 zeta wn q_dot fed by the command q_c delayed by ``delay`` s, with the
    command rate-limited to ``speed_fraction`` x rated speed (the servo firmware refuses faster
    moves). ILLUSTRATIVE VALUES; identify wn, zeta and delay on your unit in hardware session 1
    from a step response (they are not the same as the 461 project's values unless you measure
    the same)."""

    def __init__(self, wn: float = 30.0, zeta: float = 0.8, delay: float = 0.010,
                 speed_fraction: float = SPEED_FRACTION, n: int = 6):
        self.wn, self.zeta, self.delay, self.n = wn, zeta, delay, n
        self.vmax = speed_fraction * RATED_SPEED

    def simulate(self, t, q_cmd, q0=None):
        """Simulate the response to the sampled command q_cmd (N x n) on the uniform grid t.
        Returns dict(t, q, qd, q_cmd_applied)."""
        t = np.asarray(t, float)
        q_cmd = np.asarray(q_cmd, float).reshape(len(t), -1)
        dt = float(t[1] - t[0])
        k = int(round(self.delay / dt))
        qc = np.vstack([np.repeat(q_cmd[:1], k, axis=0), q_cmd])[:len(t)]
        q = np.zeros_like(qc)
        qd = np.zeros_like(qc)
        q[0] = qc[0] if q0 is None else q0
        ql = q[0].copy()                       # rate-limited command
        for i in range(1, len(t)):
            ql = ql + np.clip(qc[i] - ql, -self.vmax * dt, self.vmax * dt)
            a = self.wn**2 * (ql - q[i - 1]) - 2 * self.zeta * self.wn * qd[i - 1]
            qd[i] = qd[i - 1] + a * dt         # semi-implicit Euler
            q[i] = q[i - 1] + qd[i] * dt
        return dict(t=t, q=q, qd=qd, q_cmd_applied=qc)


# ----------------------------------------------------------------------------- reference helpers
def quintic(q0, q1, T, t):
    """Quintic (fifth-order polynomial) time scaling s(t) = 10 tau^3 - 15 tau^4 + 6 tau^5 (MR §9.2.2.1) of the joint-space
    line q0 -> q1. Returns (q, qd, qdd) at time t."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    tau = np.clip(t / T, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T if 0 < t < T else 0.0
    sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2 if 0 < t < T else 0.0
    return q0 + s * (q1 - q0), sd * (q1 - q0), sdd * (q1 - q0)


def quintic_duration(q0, q1, vmax, amax=None) -> float:
    """Shortest duration of the quintic q0 -> q1 that respects |qd| <= vmax (peak 1.875 |dq|/T)
    and, if given, |qdd| <= amax (peak 5.7735 |dq|/T^2) on every joint."""
    dq = np.abs(np.asarray(q1, float) - np.asarray(q0, float))
    T = float(np.max(1.875 * dq / vmax))
    if amax is not None:
        T = max(T, float(np.max(np.sqrt(5.7735 * dq / amax))))
    return T
