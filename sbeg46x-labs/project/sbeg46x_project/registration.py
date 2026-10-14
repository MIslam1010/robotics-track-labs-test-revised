"""Tool calibration and point-based rigid registration (Scenarios 2, 4 and 5).

Provided: transform helpers, fiducial-localisation-error (FLE) simulation, FRE and TRE of a given
registration. TEAM TODO (port your Lab 5 code): ``pivot_calibration``, ``paired_point_registration``
(SVD, Arun, Huang & Blostein 1987; Corke §14.3.9) and ``tre_prediction`` (Fitzpatrick, West & Maurer
1998).

Conventions: points are rows of an (N, 3) array in metres. A registration is a 4 x 4 transform
T_ab that maps points given in frame {b} (e.g. the phantom's CAD frame) to frame {a} (e.g. the
arm's base frame {s}): p_a = R p_b + t.
"""
from __future__ import annotations

import numpy as np


def apply(T, P) -> np.ndarray:
    """Apply the 4 x 4 transform T to the (N, 3) points P (or one 3-vector)."""
    P = np.asarray(P, float)
    return (P @ T[:3, :3].T + T[:3, 3]) if P.ndim == 2 else T[:3, :3] @ P + T[:3, 3]


def rot_z(a) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def pose(R=None, p=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = p
    return T


def fre(T, P_b, P_a) -> float:
    """Fiducial registration error: RMS over the N fiducials of |T p_b - p_a| (m)."""
    d = apply(T, P_b) - np.asarray(P_a, float)
    return float(np.sqrt(np.mean(np.sum(d**2, axis=1))))


def tre(T_est, T_true, p_b) -> float:
    """Target registration error at the target p_b (given in {b}): |T_est p_b - T_true p_b| (m)."""
    return float(np.linalg.norm(apply(T_est, p_b) - apply(T_true, p_b)))


def localise(P_true, sigma, rng) -> np.ndarray:
    """Simulated fiducial localisation: true positions plus isotropic Gaussian error with standard
    deviation ``sigma`` (m) per axis, so FLE_rms = sqrt(3) sigma."""
    P_true = np.asarray(P_true, float)
    return P_true + sigma * rng.standard_normal(P_true.shape)


# ----------------------------------------------------------------------------------- TEAM TODO
def pivot_calibration(R_list, p_list):
    """TEAM TODO (port Lab 5). Pivot calibration: given flange orientations R_i and positions p_i
    (in {s}) recorded while the tool tip rests in a fixed divot, solve R_i t + p_i = p_pivot in the
    least-squares sense. Return (t, p_pivot, rms_residual): the tip offset t in the flange frame,
    the divot position in {s} and the RMS residual (m)."""
    raise NotImplementedError("pivot_calibration: port your Lab 5 code")


def paired_point_registration(P_b, P_a):
    """TEAM TODO (port Lab 5). Least-squares rigid transform T_ab with p_a ~ R p_b + t from N >= 3
    paired points (SVD method with the reflection check). Return the 4 x 4 T_ab."""
    raise NotImplementedError("paired_point_registration: port your Lab 5 code")


def tre_prediction(P_b, p_target_b, fle_rms):
    """TEAM TODO (port Lab 5). Expected RMS TRE at p_target_b for isotropic FLE (RMS fle_rms) and the
    fiducial configuration P_b (Fitzpatrick, West & Maurer 1998):
    TRE^2 ~ FLE^2 / N (1 + 1/3 sum_k d_k^2 / f_k^2), with d_k the distance of the target from
    principal axis k of the fiducial configuration and f_k the RMS distance of the fiducials from it."""
    raise NotImplementedError("tre_prediction: port your Lab 5 code")
