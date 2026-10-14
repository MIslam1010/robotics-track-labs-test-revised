"""Scenario-specific task models: human/environment forces, RCM geometry, base disturbance.
All values are ILLUSTRATIVE and come from Scenario.params."""
from __future__ import annotations

import modern_robotics as mr
import numpy as np

from .models import Scenario, trans
from .traj import min_jerk_line


# ----------------------------------------------------------------------------- S1 rehabilitation
def patient_force_s1(s: Scenario, pulse_start: float = 0.8, active: bool = True):
    """Force of the patient's hand ON the robot handle (in {s}): a voluntary spring-damper toward
    the patient's intended minimum-jerk path A->B, plus a lateral (+x) involuntary pulse."""
    p = s.params
    A, B, T = p["path_A"], p["path_B"], p["T_reach"]

    def f(t, q, qd):
        Tsb = s.fk(q)
        x = Tsb[:3, 3]
        v = Tsb[:3, :3] @ (s.jacobian_body(q)[3:] @ qd)
        F = np.zeros(3)
        if active:
            xd, vd, _ = min_jerk_line(A, B, T, t)
            F[:2] = p["patient_K"] * (xd - x[:2]) + p["patient_B"] * (vd - v[:2])
        if pulse_start <= t < pulse_start + p["pulse_T"]:
            F[0] += p["pulse_F"]
        return F
    return f


def planar_ik(s: Scenario, x, y, elbow_sign=+1) -> np.ndarray:
    """Closed-form IK of a planar 2R arm (link lengths from params['L'])."""
    L1, L2 = s.params["L"]
    c2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(c2) > 1:
        raise ValueError("target out of reach")
    q2 = elbow_sign * np.arccos(c2)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([q1, q2])


# ----------------------------------------------------------------------------- S3 surgical RCM
def rcm_pose(p_tip, p_rcm, x_hint=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Tool-tip pose whose shaft (z_b axis, pointing from the RCM toward the tip) passes through
    the remote centre of motion p_rcm. x_b is the projection of x_hint orthogonal to z_b."""
    p_tip, p_rcm = np.asarray(p_tip, float), np.asarray(p_rcm, float)
    z = p_tip - p_rcm
    z = z / np.linalg.norm(z)
    x = np.asarray(x_hint, float) - (np.asarray(x_hint, float) @ z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return trans(p_tip, np.c_[x, y, z])


def rcm_error(T_sb: np.ndarray, p_rcm) -> float:
    """Distance (m) from the RCM point to the tool-shaft line (through the tip, along z_b)."""
    p, z = T_sb[:3, 3], T_sb[:3, 2]
    d = np.asarray(p_rcm, float) - p
    return float(np.linalg.norm(d - (d @ z) * z))


def insertion_reference(s: Scenario, t: float, T_ins: float = 4.0):
    """S3 task reference: insert along the vertical shaft from the pre-insertion point to the
    working depth (quintic over T_ins), then trace a horizontal circle of radius scan_radius about
    the inserted tip while pivoting about the RCM. Returns the desired tool-tip pose X_d(t)."""
    p = s.params
    top = p["p_rcm"] + np.array([0, 0, p["pre_insertion_height"]])
    bottom = p["p_rcm"] - np.array([0, 0, p["insertion_depth"]])
    if t <= T_ins:
        tip, _, _ = min_jerk_line(top, bottom, T_ins, t)
        return trans(tip, p["R_pre"])
    tau = t - T_ins
    w = 2 * np.pi / p["scan_period"]
    ramp = min(1.0, tau / 1.0)              # 1 s radial ramp so the start is continuous
    r = p["scan_radius"] * ramp
    tip = bottom + np.array([r * np.cos(w * tau) - r, r * np.sin(w * tau), 0.0])
    return rcm_pose(tip, p["p_rcm"], x_hint=p["R_pre"][:, 0])


# ----------------------------------------------------------------------------- S5 EOD base motion
def base_gravity_s5(s: Scenario, slope_deg=None, heave_amp=None, heave_freq=None):
    """Effective gravity in the deck frame {s}: g_eff(t) = R^T g_world - a_deck(t).
    Slope = static pitch of the vehicle about the deck y axis; heave = vertical deck acceleration
    a(t) = A sin(2 pi f t). Rotation rates of the deck are neglected (state this assumption)."""
    p = s.params
    beta = np.radians(p["slope_deg"] if slope_deg is None else slope_deg)
    A = p["heave_amp"] if heave_amp is None else heave_amp
    f = p["heave_freq"] if heave_freq is None else heave_freq
    R = mr.MatrixExp3(mr.VecToso3(np.array([0, 1.0, 0]) * beta))
    g_deck = R.T @ np.array([0, 0, -9.81])

    def g_fn(t):
        return g_deck - np.array([0, 0, A * np.sin(2 * np.pi * f * t)])
    return g_fn


# ----------------------------------------------------------------------------- S6 contact
def wall_force_s6(s: Scenario, K_wall: float, x_wall=None, B_wall=None):
    """Unilateral spring-damper wall at x = x_wall (e.g. the casualty's torso): force ON the robot
    tip f_x = -max(0, K (x - x_w) + B xdot) when x > x_w (it can push, never pull)."""
    p = s.params
    xw = p["wall_x"] if x_wall is None else x_wall
    Bw = p["wall_B"] if B_wall is None else B_wall

    def f(t, q, qd):
        Tsb = s.fk(q)
        x = Tsb[0, 3]
        if x <= xw:
            return np.zeros(3)
        v = (Tsb[:3, :3] @ (s.jacobian_body(q)[3:] @ qd))[0]
        return np.array([-max(0.0, K_wall * (x - xw) + Bw * v), 0.0, 0.0])
    return f


def planar_pose_ik(s: Scenario, x, y, phi, q0) -> np.ndarray:
    """Numerical IK (MR Sec. 6.2, IKinSpace) for a planar pose (x, y, phi) of a planar 3R arm."""
    R = mr.MatrixExp3(mr.VecToso3(np.array([0, 0, 1.0]) * phi))
    q, ok = mr.IKinSpace(s.Slist, s.M, trans([x, y, 0], R), np.asarray(q0, float), 1e-10, 1e-10)
    if not ok:
        raise RuntimeError("IK did not converge")
    return q
