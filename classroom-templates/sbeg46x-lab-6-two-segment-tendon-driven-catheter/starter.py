"""SBEG 46X Lab 6 -- constant-curvature (CC) kinematics of a two-segment tendon-driven catheter: arc parameters,
tendon-length and tendon-tension maps, the tip Jacobian and resolved-rate tip control, all checked against the
matrix exponential of Modern Robotics.
STARTER CODE -- fill in every function marked TODO (in the order A1 ... C4).
Run the auto-checks with:  LAB_IMPL=starter python -m pytest -q test_lab.py   (or copy this file to your own module name)

These functions become the `continuum` module of your cumulative library `sbeg46x_medrob`.

Notation and units: SI (m, rad, N).  Twists V = (omega, v), angular first (MR Ch. 3).  T = e^[V] uses
mr.MatrixExp6 (MR Eq. 3.88 when ||omega|| = 1; MatrixExp6 handles any V).

Conventions used everywhere in this file
  * One CC segment of length ell bends in a plane through its base z-axis.  Arc parameters (kappa, phi, ell):
    curvature kappa >= 0, bending-plane angle phi measured from the base x-axis, length ell.  Bending angle
    theta = kappa * ell.
  * Bending-angle vector theta_v = (theta_x, theta_y) = theta (cos phi, sin phi) (bold theta in the notes): it is
    zero for a straight segment and has no singularity there
    (phi is undefined only for the arc parameters, not for theta_v).
  * Twist-free CC map (Webster & Jones 2010):  R = Rot(z, phi) Rot(y, theta) Rot(z, -phi),
    p = ((1 - cos theta)/kappa) (cos phi, sin phi, 0) + (sin theta / kappa) z_hat.
    The same transform is the exponential of the constant body twist V = (-theta_y, theta_x, 0, 0, 0, ell):
    T = e^[V]  (unit-speed travel along the backbone, angular rate kappa about the axis (-sin phi, cos phi, 0)).
  * Tendon i of a segment runs parallel to the backbone at radius d and angle sigma_i in the (untwisted)
    cross-section frame.  Its length in the segment is
    l_i = ell - d (theta_v . s_i),  s_i = (cos sigma_i, sin sigma_i).
    A tendon that ends at segment 2 also runs through segment 1 at the same angle.
  * Tendon tension t_i >= 0 (a tendon can only pull).  With bending energy U = sum_j EI_j |theta_v_j|^2 / (2 ell_j)
    the equilibrium (virtual work, MLS Ch. 6 Sec. 4) is  (EI_j / ell_j) theta_v_j = d sum_{i crossing j} t_i s_i.
  * Catheter joint vector q = (z_ins, theta_1x, theta_1y, theta_2x, theta_2y): insertion of the steerable
    section out of the sheath along the sheath z-axis, then segment 1, then segment 2.
    T_tip(q) = T_base Trans(0, 0, z_ins) e^[V1] e^[V2].
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import modern_robotics as mr

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ==========================================================================================
# Provided (do not edit): data files and small helpers
# ==========================================================================================
def load_catheter() -> dict:
    """Catheter parameters from data/catheter.json (illustrative values).  Keys: ell (2,), d (2,),
    sigma_deg (2 x 3), EI (2,), theta_max, z_range (2,), T_base (4 x 4), as numpy arrays where useful."""
    c = json.loads((DATA / "catheter.json").read_text())
    for k in ("ell", "d", "sigma_deg", "EI", "z_range", "T_base"):
        c[k] = np.array(c[k], dtype=float)
    c["sigma"] = np.deg2rad(c["sigma_deg"])
    return c


def load_csv(name: str) -> dict:
    """Columns of data/<name> as numpy arrays (header row = names)."""
    with open(DATA / name, newline="") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], np.array(rows[1:], dtype=float)
    return {h: body[:, i] for i, h in enumerate(head)}


def vee(se3mat) -> np.ndarray:
    """6-vector (omega, v) of a 4x4 matrix whose top-left block is (approximately) skew-symmetric."""
    return np.asarray(mr.se3ToVec(np.asarray(se3mat)), dtype=float)


def trans_z(z: float) -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = z
    return T


# ==========================================================================================
# Part A -- one constant-curvature segment
# ==========================================================================================
def arc_transform(kappa: float, phi: float, ell: float) -> np.ndarray:
    """A1. Robot-independent map: arc parameters -> T in SE(3) (closed form, twist-free).
    kappa = 0 must return a straight segment (pure translation ell along z)."""
    # TODO A1
    raise NotImplementedError("arc_transform")


def bend_to_arc(theta_v, ell: float):
    """A2. Bending-angle vector theta_v = (theta_x, theta_y) -> (kappa, phi, theta).  For theta_v = 0 return phi = 0."""
    # TODO A2
    raise NotImplementedError("bend_to_arc")


def arc_to_bend(kappa: float, phi: float, ell: float) -> np.ndarray:
    """A2 (inverse). (kappa, phi, ell) -> theta_v."""
    # TODO A2
    raise NotImplementedError("arc_to_bend")


def segment_twist(theta_v, ell: float) -> np.ndarray:
    """A3. Constant body twist V of the segment such that T = e^[V] (MR exponential), V = (omega, v)."""
    # TODO A3
    raise NotImplementedError("segment_twist")


def segment_transform_exp(theta_v, ell: float) -> np.ndarray:
    """A3. T of the segment from the MR matrix exponential of segment_twist."""
    # TODO A3
    raise NotImplementedError("segment_transform_exp")


def tendon_lengths(theta_v, ell: float, d: float, sigma) -> np.ndarray:
    """A4. Robot-specific map: lengths of the tendons (angles sigma, radius d) inside one segment."""
    # TODO A4
    raise NotImplementedError("tendon_lengths")


def bend_from_tendons(dl, d: float, sigma) -> np.ndarray:
    """A5. Inverse robot-specific map: bending-angle vector theta_v from tendon length CHANGES dl_i = l_i - ell
    (least squares; exact when the dl come from tendon_lengths).  For three tendons at 120 deg this is
    theta_v = -(2 / (3 d)) sum_i dl_i s_i."""
    # TODO A5
    raise NotImplementedError("bend_from_tendons")


def tension_allocation(m, d: float, sigma) -> np.ndarray:
    """A6. Non-negative tendon tensions t (one per tendon) with d sum t_i s_i = m and min(t) = 0.
    m is the 2-vector bending moment required in the cross-section.  Equally spaced tendons (n >= 3):
    minimum-norm solution t_i = (2 / (n d)) s_i . m, then add the same constant to every tendon (it adds no
    moment) so that the smallest tension is zero."""
    # TODO A6
    raise NotImplementedError("tension_allocation")


def segment_tensions(theta_v, ell: float, d: float, sigma, EI: float) -> np.ndarray:
    """A6. Tensions that hold one segment at bending-angle vector theta_v: (EI / ell) theta_v = d sum t_i s_i."""
    # TODO A6
    raise NotImplementedError("segment_tensions")


# ==========================================================================================
# Part B -- the two-segment catheter
# ==========================================================================================
def split_q(q):
    q = np.asarray(q, dtype=float)
    return q[0], q[1:3], q[3:5]


def catheter_fk(q, cat: dict) -> np.ndarray:
    """B1. Tip pose T_tip = T_base Trans(0,0,z_ins) e^[V1] e^[V2]."""
    # TODO B1
    raise NotImplementedError("catheter_fk")


def catheter_backbone(q, cat: dict, n: int = 20) -> np.ndarray:
    """B1 (plot helper). (2n + 2) x 3 backbone points from the sheath exit to the tip."""
    z, theta_v1, theta_v2 = split_q(q)
    pts = [cat["T_base"][:3, 3]]
    T0 = cat["T_base"] @ trans_z(z)
    pts.append(T0[:3, 3])
    for j, (theta_v, ell) in enumerate(((theta_v1, cat["ell"][0]), (theta_v2, cat["ell"][1]))):
        for s in np.linspace(0, 1, n + 1)[1:]:
            pts.append((T0 @ segment_transform_exp(np.asarray(theta_v) * s, ell * s))[:3, 3])
        T0 = T0 @ segment_transform_exp(theta_v, ell)
    return np.array(pts)


def catheter_tendon_lengths(q, cat: dict) -> np.ndarray:
    """B2. Path lengths of the 6 tendons (3 ending at segment 1, then 3 ending at segment 2) measured from the
    sheath exit; segment-2 tendons pass through segment 1 at their own angle.  Insertion does not change them
    (the tendons slide with the catheter)."""
    # TODO B2
    raise NotImplementedError("catheter_tendon_lengths")


def catheter_tendon_jacobian(cat: dict) -> np.ndarray:
    """B2. 6 x 5 matrix dl/dq (constant: the tendon map is linear in theta_v).  Its transpose is the MLS coupling
    matrix P (MLS Ch. 6 Sec. 4) restricted to the bending coordinates."""
    # TODO B2
    raise NotImplementedError("catheter_tendon_jacobian")


def bend_from_catheter_tendons(dl, cat: dict) -> np.ndarray:
    """B2 (inverse). (theta_v1, theta_v2) as a 4-vector from the 6 tendon length changes l - l(straight)."""
    # TODO B2
    raise NotImplementedError("bend_from_catheter_tendons")


def catheter_tensions(q, cat: dict) -> np.ndarray:
    """B3. Non-negative tensions (6,) that hold the catheter at q (no external load).  Segment 2 is held by
    its own tendons; those tendons also bend segment 1, so the segment-1 tendons supply only the rest:
    (EI2/ell2) theta_v2 = d2 sum t_(2,i) s_(2,i);
    (EI1/ell1) theta_v1 = d1 sum t_(1,i) s_(1,i) + d2 sum t_(2,i) s_(2,i)."""
    # TODO B3
    raise NotImplementedError("catheter_tensions")


def dexp_left(V, terms: int = 40) -> np.ndarray:
    """B4 helper. Left Jacobian of the SE(3) exponential, sum_k ad_V^k / (k+1)!  (6 x 6, ad = mr.ad).
    (d/dt e^[V]) e^-[V] = [dexp_left(V) Vdot]."""
    # TODO B4
    raise NotImplementedError("dexp_left")


def catheter_jacobian_space(q, cat: dict) -> np.ndarray:
    """B4. 6 x 5 space Jacobian J_s(q): V_s = J_s(q) qdot, where [V_s] = Tdot T^-1 (MR Sec. 5.1 definition)."""
    # TODO B4
    raise NotImplementedError("catheter_jacobian_space")


def tip_position_jacobian(q, cat: dict) -> np.ndarray:
    """B5. 3 x 5 Jacobian of the tip position p(q) in the space frame: pdot = [-[p]  I] V_s."""
    # TODO B5
    raise NotImplementedError("tip_position_jacobian")


# ==========================================================================================
# Part C -- resolved-rate tip control
# ==========================================================================================
def clamp_q(q, cat: dict) -> np.ndarray:
    """C1. Enforce limits: z_ins in z_range, |theta_v_j| <= theta_max (scale theta_v_j radially)."""
    # TODO C1
    raise NotImplementedError("clamp_q")


def dls_inverse(J, lam: float) -> np.ndarray:
    """C1. Damped least-squares inverse J^T (J J^T + lam^2 I)^-1 (lam = 0 gives the pseudoinverse)."""
    # TODO C1
    raise NotImplementedError("dls_inverse")


def resolved_rate_step(q, p_des, pdot_des, cat: dict, dt: float, K: float = 20.0, lam: float = 1e-3,
                       W=None) -> np.ndarray:
    """C2. One step of resolved-rate position control
    qdot = W J^T (J W J^T + lam^2 I)^-1 (pdot_des + K (p_des - p)),   q <- clamp(q + qdot dt).
    W (5,) are joint weights (default: 1 for every coordinate)."""
    # TODO C2
    raise NotImplementedError("resolved_rate_step")


def go_to(q0, p_goal, cat: dict, dt: float = 0.01, K: float = 5.0, steps: int = 400, lam: float = 1e-3,
          W=None) -> dict:
    """C3. Regulate the tip to a fixed point (pdot_des = 0).  Returns {'q': (steps+1, 5), 'err': (steps+1,)}."""
    # TODO C3
    raise NotImplementedError("go_to")


def track_path(q0, t, P, cat: dict, K: float = 20.0, lam: float = 1e-3, W=None) -> dict:
    """C4. Track a sampled tip path P (N x 3) at times t (N,) with feed-forward pdot from finite differences.
    Returns {'q': (N, 5), 'p': (N, 3), 'err': (N,), 'tendon_dl': (N, 6)} where tendon_dl = l(q) - l(straight)."""
    # TODO C4
    raise NotImplementedError("track_path")
