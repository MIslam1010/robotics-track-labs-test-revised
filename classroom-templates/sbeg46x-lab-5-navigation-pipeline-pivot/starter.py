"""SBEG 46X Lab 5 -- navigation pipeline: pivot calibration of a digitising pointer on the course arm, paired-point
rigid registration of a phantom (SVD), and navigation accuracy (FLE, FRE, TRE): prediction vs Monte Carlo vs measurement.
STARTER CODE -- fill in every function marked TODO (in the order A1 ... D2).
Run the auto-checks with:  LAB_IMPL=starter python -m pytest -q test_lab.py   (or copy this file to your own module name)

These functions become the `registration` module of your cumulative library `sbeg46x_medrob`.

Notation and units
  * Positions in MILLIMETRES, angles in radians, rotation matrices R in SO(3), homogeneous transforms T in SE(3)
    (MR Sec. 3.3.1, pp. 87-95).  T_ab is the pose of frame {b} in frame {a}; p_a = T_ab p_b.
  * Frames: {s} robot base (course arm), {b} flange, {I} image (CT) of the phantom, {C} optical-tracker camera,
    {P} patient reference marker.
  * Pivot calibration: the pointer tip is fixed at the pivot while the flange takes poses (R_i, p_i):
        R_i p_tip + p_i = p_piv           (p_tip in {b}, p_piv in {s})
  * Registration: fiducials x_k in {I} and the same fiducials y_k in {s};  y_k ~ R x_k + t,  T_sI = (R, t).
  * FRE = RMS residual over the fiducials; FLE = fiducial localisation error (RMS magnitude);
    TRE = error at a target that was NOT used in the registration.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import modern_robotics as mr

import course_models as cm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ==========================================================================================
# Provided (do not edit): data files and small helpers
# ==========================================================================================
def load_phantom() -> dict:
    """Phantom description from data/phantom.json (illustrative values). Arrays are returned as numpy arrays."""
    d = json.loads((DATA / "phantom.json").read_text())
    for k in ("fiducials_image_mm", "check_points_image_mm"):
        d[k] = np.array(d[k], float)
    d["targets_image_mm"] = {k: np.array(v, float) for k, v in d["targets_image_mm"].items()}
    return d


def load_joints(name: str) -> tuple[list[str], np.ndarray]:
    """A joint log of data/ (columns: label, q1 ... q6 in rad). Returns (labels, N x 6 array)."""
    with open(DATA / name) as fh:
        rows = list(csv.DictReader(fh))
    labels = [r["label"] for r in rows]
    Q = np.array([[float(r[f"q{i}"]) for i in range(1, 7)] for r in rows])
    return labels, Q


def rot_to_T(R, t) -> np.ndarray:
    """4 x 4 homogeneous transform from R (3 x 3) and t (3,)  (MR Eq. 3.62)."""
    return mr.RpToTrans(np.asarray(R, float), np.asarray(t, float))


# ==========================================================================================
# Part A -- tool calibration with the course arm
# ==========================================================================================
def flange_pose_mm(q) -> np.ndarray:
    """A1. Pose T_sb of the flange frame {b} in the base frame {s} for joint angles q (rad), with the
    position in MILLIMETRES. Use the space-form PoE of the course arm (MR Eq. 4.14) from course_models."""
    # TODO A1
    raise NotImplementedError("A1: flange_pose_mm")


def pivot_calibration(Rs, ps):
    """A2. Pivot calibration by linear least squares.

    Rs: list of K rotation matrices R_i of the tool (flange) frame; ps: list of K positions p_i (mm).
    Every pose satisfies R_i p_tip + p_i = p_piv; stack them as [R_i  -I] [p_tip; p_piv] = -p_i.
    Returns (p_tip (3,), p_piv (3,), rms) where rms is the RMS length of the 3-D residuals (mm)."""
    # TODO A2
    raise NotImplementedError("A2: pivot_calibration")


def pivot_from_joint_log(Q):
    """A3. Pivot calibration from a joint log Q (K x 6, rad) recorded while the pointer tip stays in the pivot divot.
    Returns the same tuple as pivot_calibration."""
    # TODO A3
    raise NotImplementedError("A3: pivot_from_joint_log")


def tip_position_mm(q, p_tip) -> np.ndarray:
    """A4. Position (mm) of the pointer tip in {s} for joint angles q and the calibrated offset p_tip (mm, in {b})."""
    # TODO A4
    raise NotImplementedError("A4: tip_position_mm")


# ==========================================================================================
# Part B -- paired-point rigid registration
# ==========================================================================================
def register_points(X, Y) -> np.ndarray:
    """B1. Least-squares rigid registration (Arun, Huang & Blostein 1987, with the reflection fix).

    X, Y: N x 3 corresponding points (N >= 3, not collinear). Finds R in SO(3) and t minimising
    sum_k |y_k - (R x_k + t)|^2 and returns T = (R, t) as a 4 x 4 matrix (so T maps X-coordinates to Y-coordinates).
    Steps: centroids, H = sum x~_k y~_k^T, H = U S V^T, R = V diag(1, 1, det(V U^T)) U^T, t = y_bar - R x_bar."""
    # TODO B1
    raise NotImplementedError("B1: register_points")


def apply_T(T, P) -> np.ndarray:
    """B2. Apply a homogeneous transform to points: P is (3,) or N x 3; returns the same shape."""
    # TODO B2
    raise NotImplementedError("B2: apply_T")


def fre(X, Y, T) -> float:
    """B3. Fiducial registration error: RMS over the N fiducials of |y_k - T x_k| (mm)."""
    # TODO B3
    raise NotImplementedError("B3: fre")


# ==========================================================================================
# Part C -- error prediction (Fitzpatrick, West & Maurer 1998) and Monte Carlo
# ==========================================================================================
def fre_expected(N: int, fle_rms: float) -> float:
    """C1. Expected RMS FRE for N fiducials with isotropic FLE of RMS magnitude fle_rms:
    <FRE^2> = (1 - 2/N) <FLE^2>."""
    # TODO C1
    raise NotImplementedError("C1: fre_expected")


def fle_from_fre(fre_value: float, N: int) -> float:
    """C2. Estimate the RMS FLE from one observed FRE by inverting C1 (a rough estimate from a single registration)."""
    # TODO C2
    raise NotImplementedError("C2: fle_from_fre")


def tre_predict(X, target, fle_rms: float) -> float:
    """C3. Predicted RMS TRE at `target` (Fitzpatrick, West & Maurer 1998, isotropic FLE):

        <TRE^2(r)> = <FLE^2>/N * (1 + 1/3 * sum_{k=1..3} d_k^2 / f_k^2)

    with the principal axes of the fiducial configuration (eigenvectors of sum x~ x~^T, x~ = x - x_bar),
    d_k = distance of the target from principal axis k, f_k = RMS distance of the fiducials from axis k."""
    # TODO C3
    raise NotImplementedError("C3: tre_predict")


def monte_carlo(X, targets, fle_rms: float, n: int = 2000, seed: int = 0):
    """C4. Monte-Carlo simulation of registration error.

    In each of n trials, add isotropic Gaussian noise of RMS magnitude fle_rms (std fle_rms/sqrt(3) per axis) to
    the measured copy: Y = X + noise; register X -> Y with register_points; record the FRE and, for every
    target r, the TRE |T r - r| (the true transform is the identity, so any motion of r is error).
    targets: (3,) or M x 3. Returns (fre_samples (n,), tre_samples (n, M)). Use np.random.default_rng(seed)."""
    # TODO C4
    raise NotImplementedError("C4: monte_carlo")


def best_subset(X, target, k: int, fle_rms: float):
    """C5. Fiducial configuration: among all subsets of k fiducials (k >= 3) of X, find the one with the smallest
    predicted TRE at `target`. Returns (indices as a sorted tuple, predicted TRE). Ties: the first in
    itertools.combinations order."""
    # TODO C5
    raise NotImplementedError("C5: best_subset")


# ==========================================================================================
# Part D -- frame chains and the full pipeline
# ==========================================================================================
def chain_robot_from_image(T_CR, T_CP, T_PI) -> np.ndarray:
    """D1. Optical-tracker chain: camera {C} sees the robot-base marker (T_CR = pose of {s} in {C}) and the patient
    reference (T_CP); registration gives T_PI (image in patient-reference coordinates).
    Return T_RI = T_CR^-1 T_CP T_PI, the image frame in robot-base coordinates."""
    # TODO D1
    raise NotImplementedError("D1: chain_robot_from_image")


def navigation_pipeline(n_mc: int = 2000, seed: int = 0) -> dict:
    """D2. Run the whole pipeline on the data files and return a dict with
        p_tip, p_piv, pivot_rms                     (A3 on data/pivot_joints.csv)
        Y_fid                                       (N x 3 digitised fiducials in {s}, A4 on data/fiducial_joints.csv)
        T_sI, fre                                   (B1, B3: image fiducials -> digitised fiducials)
        fle_est                                     (C2)
        tre_pred   {name: predicted TRE}            (C3 at every target of phantom.json, with fle_est)
        tre_mc     {name: RMS TRE of C4}            (C4 with fle_est, n_mc trials, seed)
        fre_mc_rms                                  (RMS of the C4 FRE samples)
        targets_s  {name: target position in {s}}
        check_err  (M,) distances |T_sI c_I - c_s| at the check points (digitised with data/check_joints.csv)"""
    # TODO D2
    raise NotImplementedError("D2: navigation_pipeline")
